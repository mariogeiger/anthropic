# Changelog

## 0.6.0

The conversation type now holds the wire order instead of rebuilding it. Pure
restructuring: every request body this crate emits is byte-identical to 0.5.0,
verified by serializing all three openings on both revisions and diffing.

### Breaking: the opening is an argument to `Context::new`

`Context::new()` followed by `with_system(...)` let the opening be installed at
any moment, including after messages had been appended, and `system` sat as a
field beside `tools` and `messages` as though the three were peers. The API says
otherwise: the prefix is hashed `tools`, then `system`, then `messages`, and a
system instruction that applies from the start is *not* a message — a
`{"role": "system"}` entry "cannot be the first entry in `messages`".

- `context::Opening` is new, with one variant per legitimate opening:
  `Opening::None`, `Opening::Instruction(String)`, and
  `Opening::CachedInstruction { text, slot, ttl }`. Three and not two, because the
  API documents `system` as optional, so "no opening" is a state the type records
  rather than an unset field. Constructors `Opening::instruction` and
  `Opening::cached_instruction` mirror the two former builders.
- `Context::new` takes it: `Context::new(opening)`. `Context::with_system` and
  `Context::with_system_cached` are gone.
- `Context::new` is now **infallible**. `with_system_cached` returned
  `Result<_, AnchorError>` because a slot might already be occupied; on a
  conversation that did not exist a moment ago it cannot be, so the error is gone
  by construction rather than ignored. `AnchorError` remains for
  `with_tools_cached`, its only remaining source.
- `Context::opening()` reads the instruction back, for a caller replaying a
  conversation or tracing a cache miss to the prefix it is measured against.
- `Context`'s field order is now `tools`, `system`, `messages` — the wire order.
- `Context::default()` is `Context::new(Opening::None)`.

**Migration.** Mechanical, three cases:

```text
Context::new()                                    → Context::new(Opening::None)
Context::new().with_system(text)                  → Context::new(Opening::instruction(text))
Context::new().with_system_cached(slot, text, ttl)? → Context::new(Opening::cached_instruction(text, slot, ttl))
```

Note the argument order in the cached form: text first, then slot, then TTL, so
the three constructors read the same way. The `?` or `.unwrap()` on the cached
form is dropped — it no longer returns a `Result`. Import `anthropic::context::Opening`.

### Three new `compile_fail` proofs

The opening cannot be installed late (`with_system` does not exist, E0599) and is
not an assignable field (`system` is private, E0616). Each was checked to fail for
its stated reason, alongside the existing proof that `messages` is private.
`every_opening_reaches_its_documented_wire_shape` asserts all three wire shapes:
absent field, bare string, one-element block array.

### Unchanged, deliberately

The sequence rules stay exactly where they were: `push_system` refuses the first
position and an assistant predecessor, `Request::new` refuses a wrong successor and
a model without the feature. A typestate encoding of those rules was designed and
prototyped — the placement rules are a regular language, recognized by a four-state
DFA that matched all ten live measurements — and deliberately not adopted, because
the one real consumer builds its message list in a loop and a loop cannot carry a
compile-time state. The analysis is recorded here so the option is not re-derived
from scratch.

## 0.5.0

Mid-conversation system messages are not a second spelling of the top-level
`system` field. They are disjoint by position and limited by model, and the crate
now enforces the second of those and writes down both.

### Mid-conversation system messages are refused on a model that does not accept one

The documentation states the feature is available on Fable 5, Mythos 5, Opus 4.8
and Opus 5, and "not available on Claude Sonnet 5; use the top-level `system`
field instead". The crate enforced every *placement* rule already and none of the
*availability* rule, so a Sonnet 5 request carrying a mid-conversation system
message serialized happily and earned a 400.

- `ModelId::accepts_mid_conversation_system_message()` is new: a closed list, so
  adding a model states its answer rather than inheriting a guess. Mid-conversation
  tool changes share the availability and need no second predicate, because a
  `tool_addition` or `tool_removal` block can only travel inside a system message.
- `RequestError::MidConversationSystemMessageUnsupported { model, at }` is new.
  `Request::new` returns it, naming the model and the index of the entry to remove.
- Its doc comment justifies why this is a runtime refusal rather than an
  unrepresentable combination: it pairs the model with the conversation, and those
  are separate types precisely so one conversation can be sent to several models.
  Parameterizing `Context` by model would buy one 400 and cost that. `SOUL.md` now
  states the same boundary as a principle.

### Why both positions exist, written down

The two forms produce identical behaviour and identical token counts, which is why
the question comes up. The `system` module now records that they are nonetheless
disjoint: by placement, because a system message may not be first and the
top-level field is the only legal home for an instruction that applies from the
start; and by model, because the top-level field is the only portable one. Both
rules are cited to the Anthropic documentation.

### Placement rules: no change in behaviour, one test per rule

The audit found no placement gap. Every documented rule was already enforced —
not first, must follow a user turn (`tool_result` blocks included), must end the
array or precede an assistant turn, never between a `tool_use` and its
`tool_result`, and consecutive system messages treated as one section. What was
missing was a test naming each rule, so a regression now names the rule it broke.
The "never between a `tool_use` and its `tool_result`" case is the existing
`AfterAssistant` refusal, because the only entry between them is the assistant
turn holding the `tool_use`.

A `compile_fail` doctest now proves `Context::messages` is private, which is what
makes `push_system` the only door and its checks unavoidable. It fails with
E0616, verified.

### A gateway's 200 is not evidence of legality

`AGENTS.md` recorded only the case where a deployment refuses what the API
documents. The reverse direction matters as much and is not symmetric: our gateway
*accepts* a leading system message, a documented 400, and `count_tokens` validates
placement not at all. It does enforce the successor rule, so the permissiveness is
specific rather than general — which is exactly why a placement or per-model rule
can only be established from the documentation, never from a successful probe.
`tests/live_api.rs` gained
`live_report_the_gateway_is_more_permissive_than_the_documented_placement_rules`,
which reports the status rather than asserting one, so an endpoint that changes its
mind in either direction becomes visible without a false failure.

The Sonnet 5 rule cannot be checked live on this gateway: only the two Opus models
are reachable there, so that rule rests on the documentation alone.

## 0.4.0

The crate's mission is now written down — *represent the entire Anthropic Messages
API faithfully in types* — and this release closes most of the gap between that
claim and the code. Everything below was measured: probed against a live endpoint,
or read from the published stable schema where a deployment refuses it.

### Documents: `SOUL.md` and `AGENTS.md` replace `CLAUDE.md`

`CLAUDE.md` held two jobs in one file. `SOUL.md` now states the mission and the
stable design principles; `AGENTS.md` holds the working instructions — required
checks, versioning and changelog rules, where things live, what is out of scope,
and how to verify against the live API. Nothing was dropped. The numbered section
references in the source named CLAUDE.md's headings, and each now names the
principle it invokes instead.

### Breaking: `Message` is an enum whose variant is the role

The three roles do not admit the same content: a system message takes text and
tool changes only, and the API answers `role 'system' supports text,
tool_addition, and tool_removal blocks only` for anything else. A struct with a
public `role` beside a public `content` let that rejection be written.

- `context::Message` is now `Message::User(Vec<ContentBlock>)`,
  `Message::Assistant(Vec<ContentBlock>)`, and
  `Message::System(Vec<SystemBlock>)`. `Message::role()` derives the wire value,
  and the serializer writes it from there, so it cannot disagree with the content.
- Readers: `role()`, `content()` for the two `ContentBlock` roles, and
  `system_content()` for the system role. `content()` returns `None` on a system
  message, which is the whole point of the split.
- `values::Role` gains `System`.
- **Migration.** `Message { role: Role::User, content: blocks }` becomes
  `Message::User(blocks)`. `msg.role` becomes `msg.role()`, and `msg.content`
  becomes `msg.content()` — an `Option<&[ContentBlock]>`, so `msg.content()
  .unwrap_or_default()` where a system message is impossible by construction. The
  `push_user*`/`push_assistant*`/`push_tool_result` methods are unchanged.
- Two `compile_fail` doctests prove both halves: neither a `role` field nor a
  system message holding an ordinary `ContentBlock` compiles. Each was verified to
  fail for its stated reason.

### Breaking: content blocks move to `block`

`context.rs` had reached 1,341 lines and two missions. The blocks a caller sends
now live in `anthropic::block`, leaving `context` to the cache-safe conversation.

- **Migration.** `anthropic::context::{ContentBlock, TextBlock, ImageBlock,
  ImageSource, ToolUseBlock, ToolResultBlock, ToolResultContent, ToolResultItem,
  ThinkingBlock, RedactedThinkingBlock}` become `anthropic::block::{…}`.
  `Context`, `CacheSlot`, `CacheControl`, and `Tool` stay in `context`.

### Mid-conversation system messages

An instruction that arrives partway through a conversation had nowhere to go:
rewriting the system prompt costs the whole cache, and a user turn makes the model
read it as something the user said. The API's answer is a `{"role": "system"}`
entry inside `messages`, which sits after the cached prefix.

- New `anthropic::system` module: `SystemBlock` (text, tool addition, tool
  removal) and `ToolReference` (`tool_reference`, `mcp_tool_reference`,
  `mcp_toolset_reference`).
- `Context::push_system` and `push_system_text` append one, returning
  `SystemMessageError` for the placement rules decidable at append time: empty
  content, first position, and following an assistant turn.
- `Request::new` checks the remaining rule — a system message must end `messages`
  or precede an assistant turn — because appending a user turn after a legal
  system message makes it illegal, so only the finished history knows.
  `RequestError::SystemMessageNotFollowedByAssistant` reports the chain's last
  message, which is the position to change.
- Measured: *two system messages in a row are accepted*, contrary to both the old
  `CLAUDE.md` §7 note and the API's own wording. A live test holds that in place.
- Cache breakpoints land on a system message's inner text block, which is where
  the API accepts one: `cache_control on mid_conv_system is not supported; set it
  on an inner content block instead`.

### `tool_addition` and `tool_removal`, which our gateway refuses

Implemented from the documented schema, where they are beta types carrying a tool
reference. **The NVIDIA inference gateway rejects them with a 400** — `Input tag
'tool_removal' … does not match any of the expected tags` — with and without the
beta header. The crate's mission is the API rather than one deployment, so they are
expressed and the refusal is stated here and in their documentation. No test
requires the gateway to accept them.

Their value, where they work: a tool withdrawn this way leaves `tools`
byte-identical, so the tools cache stays warm — the same asymmetry that makes
`ToolChoice::None` cheaper than sending no tools.

### Documents, search results, and citations

- New `anthropic::document` module. `DocumentBlock` is material the caller
  supplies (PDF, plain text, URL, Files API id); `SearchResultBlock` is material a
  search returned, whose source and title are required because a result whose
  origin is unknown cannot be cited usefully. The API counts them separately, as
  `document_index` versus `search_result_index`.
- `Citation` decodes all five documented kinds, and an unmodeled kind is a variant
  meaning ignore me.
- `content::BlockDelta::Citations` replaces `citations_delta`'s former
  `Unmodeled` reading. `StreamedBlock::Text` gains a `citations` field.
  `Settled::citations()` and `Response::citations()` flatten them for a caller
  rendering footnotes.
- Citations are opt-in: a request that never mentioned them is byte-identical to
  one built before this release, so its cache prefix still matches.
- `TextBlock` now serializes with its own `type`, because three of the four
  positions holding one require it — measured: a search result whose blocks omit
  it is `search_result.content.0.type: Field required`. `ContentBlock::Text` is
  the exception, since its enum writes the tag.
- `tests/captured/citations.sse` is a real cited stream: four text blocks, two
  grounded by character range. Note the frame order it records — a block's
  `citations_delta` arrives *before* the `text_delta`s it grounds.

### Per-call parameters

- `service_tier`, as `values::ServiceTier` (`auto`, `standard_only`). A scalar the
  API accepts unconditionally with a documented default, so it is a plain
  always-emitted field.
- `metadata.user_id`, as `request::EndUserId` — a newtype enforcing the documented
  512-character bound, counted in characters as JSON Schema counts it, and
  carrying Anthropic's warning that nothing identifying belongs there.
- `output_config.format`, as `request::OutputFormat::json_schema`. `output_config`
  now appears whenever *either* effort or format is present: Haiku 4.5 takes no
  effort but does take a format.
- `top_p` and `top_k` remain absent, and now for a measured reason: every modeled
  model answers `` `top_p` is deprecated for this model ``.

### Tool declarations

`Tool` gains `defer_loading`, `strict`, and `input_examples`, each emitted only
when set — the field is rendered into the prompt, so emitting a default the caller
never asked for writes a different cache key.

`Request::new` refuses a request whose every tool is deferred, which the API
states as `At least one tool must have defer_loading=false`; that is a relation
across the list, so no single tool's type can carry it. `Context::tools()` is now
readable, which is what the check needs.

Measured: `defer_loading` is accepted; **`strict` and `input_examples` are refused
by our gateway** (`tools.0.custom.strict: Extra inputs are not permitted`). Both
are in the documented stable schema, so both stay.

### Images

`ImageBlock` gains the oversize policy, as `values::ImageOversize`. The API's
default silently rescales an image larger than the model accepts, so the model
observes dimensions the caller never chose; `ImageOversize::Error` asks to be told
instead. Absent stays absent, because naming a policy and inheriting one are
different requests. **Measured: our gateway refuses the `transformations` object**
this becomes; it is in the documented stable schema, so it stays.

### Response side

- `usage.service_tier`, as `values::ServedTier` (`standard`, `priority`, `batch`).
  A deliberately different vocabulary from the request's: `batch` is reported but
  never asked for, `auto` asked for but never reported. It merges by keeping the
  first frame that named one rather than by maximum — it is a fact about the
  request, not a counter.
- `usage.server_tool_use`, as `usage::ServerToolUsage`: web search and web fetch
  request counts, billed per call rather than per token.
- `stop_details`, as `stream::RefusalDetails` with `values::RefusalCategory`
  (`cyber`, `bio`, `frontier_llm`, `reasoning_extraction`, `general_harms`). It
  sits beside the stop reason on `Response`, `MessageDelta`, and
  `Outcome::Stopped`, because a refusal is a message the server finished sending.
- **Migration.** `Outcome::Stopped` gains a `refusal` field; a struct pattern over
  it needs `..` or the new field. `Settled::refusal()` reads it.

### Still missing, in priority order

Server-side tool *declarations*: web search, web fetch, code execution, bash, text
editor, memory, tool search, and the computer and browser toolsets, plus MCP
toolsets and the `container`/skills parameters they need. Their result blocks
already decode as `Unmodeled` rather than failing, and a caller that does not
declare them never sees one. Also absent: `connector_text` and `container_upload`
blocks, `ToolResultBlockParam`'s `toolset_name` and its document, search-result and
tool-reference content forms, `allowed_callers`, `eager_input_streaming`,
`inference_geo`, `user_profile_id`, top-level `cache_control`, `display` on the
legacy fixed-budget thinking form, and `thinking`/`tool_choice`/`output_config` on
`count_tokens`.

## 0.3.0

Wire vocabulary is typed everywhere, not just in the places a caller was expected
to go through. `Message.role` was a `pub role: &'static str` whose doc comment
claimed an unknown role could not reach the wire; it could, because a public
string field accepts any string. The same defect appeared in five other places.
This release replaces every "held by a private constructor or a doc comment" with
"held by the type".

### Breaking: `Message::role` is a `Role`

- `values::Role` is a new enum with exactly the two roles a `messages[]` entry may
  carry: `Role::User` (`"user"`) and `Role::Assistant` (`"assistant"`). It is
  re-exported at the crate root, has `as_str`/`from_str` like the other wire
  enums, and serializes to its wire string.
- `context::Message::role` is now `Role` instead of `&'static str`. The field
  stays public, because a closed enum has no invalid value to write — which is
  what makes the impossibility structural rather than documentary. A
  `compile_fail` doctest on the `context` module proves both halves: neither a
  nonexistent variant nor a bare `"wizard"` compiles.
- **Migration.** Matching or comparing a role: `msg.role == "user"` becomes
  `msg.role == Role::User`, and `match msg.role { "user" => … }` becomes
  `match msg.role { Role::User => … }`. Needing the string: `msg.role` becomes
  `msg.role.as_str()`. Constructing a `Message` literal: `role: "assistant"`
  becomes `role: Role::Assistant`. The `push_user*`/`push_assistant*`/
  `push_tool_result` methods are unchanged and remain the ordinary path.
- `Role` has no `System` variant, deliberately. Anthropic does accept a
  `{"role": "system"}` entry inside `messages[]` — the gateway lists it among
  accepted block types as `mid_conv_system` — but only on some models, and only
  under placement rules the API enforces with a 400: it may not be first, must
  follow a user turn or an assistant turn ending in a server tool use, must be
  last or be followed by an assistant turn, and may not be adjacent to another
  one. A `Role::System` this crate cannot place correctly would make those
  rejections *writable*, which inverts the enum's purpose. Support remains the
  separate work CLAUDE.md §7 describes: a `push_system` that upholds the
  placement rules, plus a shared system-content type and a cache-slot position
  for it.

### Breaking: `Request` and `CountRequest` fields are private, with readers

`Request::new` documented itself as "the single construction path, so the checks
can't be bypassed", while `pub model` and `pub max_tokens` let a caller assign
straight past them. `max_tokens` must lie in `1..=` *this model's* maximum output
and Haiku 4.5's legacy thinking budget must stay below it — cross-field
invariants no single type carries, so the constructor must be the only way in.

- `Request` fields are private. Readers: `context()`, `model()`, `max_tokens()`,
  `stop_sequences()`, `is_streamed()`, `tool_choice()`.
- The two builders that collided with those reader names are renamed to the
  crate's `with_*` idiom: `stop_sequences(v)` → `with_stop_sequences(v)` and
  `tool_choice(c)` → `with_tool_choice(c)`. `streamed()` is unchanged.
- `CountRequest` fields are private, with `context()` and `model()` readers.
- **Migration.** Reading: `req.max_tokens` becomes `req.max_tokens()`, and
  likewise for the others. Building: `.stop_sequences(v)` becomes
  `.with_stop_sequences(v)`, `.tool_choice(c)` becomes `.with_tool_choice(c)`.
  Writing a field after construction was never sound and is now impossible;
  build a new `Request` instead.

### Breaking: `YearMonth` is a year plus a `Month`

`pub month: u8` documented as "1 through 12" was a range nothing enforced.

- `model::Month` is a new twelve-variant enum, so there is no invalid month to
  construct and nothing to validate. `Month::ALL` is calendar order and
  `Month::from_number` maps an ordinal back, `None` outside `1..=12`.
- `YearMonth` has private fields with `year()`, `month()`, and `month_number()`
  readers, and `YearMonth::new(year, month)`. It now derives `Ord`: the fields are
  declared most-significant first, so the lexicographic order Rust derives *is*
  chronological order, and two cutoffs compare with `<` directly.
- **Migration.** `cutoff.year` becomes `cutoff.year()`, and `cutoff.month`
  becomes `cutoff.month_number()` for the integer or `cutoff.month()` for the
  enum. `YearMonth { year: 2026, month: 1 }` becomes
  `YearMonth::new(2026, Month::January)`.

### Breaking: `ImageSource::Base64::media_type` is an `ImageMediaType`

The variant's field was a `&'static str` that only `ImageMediaType::as_str` ever
filled, so an unsupported media type was writable by hand. It now holds the enum.
`ImageSource::base64(ImageMediaType::Png, data)` is unchanged; only the field's
type changed, which matters when matching on the variant.

### Every API enum serializes to its own wire string

`api_enum!` now generates `Serialize` alongside `as_str`, so a wire struct holds
the enum rather than a string obtained from it. This removed the last four
internal `&'static str` wire fields: `CacheControl.kind`/`.ttl`, the private
thinking-config `type` fields, and the system prompt's text-block `type` (now the
new one-variant `values::TextBlockType`). The bytes on the wire are unchanged.

### Audited and left open, with reasons

- `usage::Usage`, `usage::CacheCreation`, `usage::OutputTokensDetails`,
  `response::Response`, `response::ApiError`, `stream::MessageStart`,
  `stream::MessageDelta`, `stream::StreamedError` — decoded inbound records whose
  fields are plain counters and strings the server chose. Every `u64` is a valid
  count and every `String` a valid identifier, so there is no invalid value to
  exclude. `Usage::cache_creation_is_consistent` reports a server-side
  inconsistency rather than refusing it, which §6 requires.
- `model::Pricing` — two `u32` cent amounts, both meaningful at any value. Its
  caveats are about which price list it reflects, which no type can settle.
- `settle::Settled` stays `#[non_exhaustive]` with public fields: the invariant is
  "this came from a finished stream", which the missing struct literal already
  enforces, and reading the fields is the point of holding one.
- `settle::ToolCall`, `context::Tool`, `context::TextBlock` and the other outbound
  block structs — public fields are free-form content by nature (a tool name, a
  schema, some text). Their one constrained field, `cache_control`, is already
  crate-private and reachable only through a `CacheSlot`.
- `content::ToolInput` keeps a private `String` and a `from_wire` constructor. The
  bytes are deliberately unvalidated: `input_json_delta` sends partial JSON, so
  "not yet parseable" is a normal intermediate state and `decode` is where it
  becomes a failure.

## 0.2.0

The response half of the wire. The crate decoded nothing before this release; a
consumer had to hand-match raw JSON, which is where the API's shape gets
re-derived badly and where a stream that stopped early reads as one that
finished.

### Streaming

- `stream::StreamEvent` decodes one Server-Sent Event payload into a typed
  event. Modeled: `message_start`, `content_block_start`,
  `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`,
  and `error`. `ping` and any event Anthropic adds later decode to
  `StreamEvent::Unmodeled`, which is ignorable and never an error — the
  versioning policy permits new event types, so failing on one would break on a
  server release.
- `content::BlockDelta` covers all four delta kinds: `text_delta`,
  `thinking_delta`, `signature_delta`, and `input_json_delta`. Unrecognized
  kinds are `Unmodeled`, as are unrecognized block kinds
  (`content::StreamedBlock::Unmodeled`), which is what server tools produce.
- `content::ToolInput` holds tool input as bytes, because `input_json_delta`
  sends *partial* JSON that parses only once concatenated. Decoding is a
  separate step that can fail per call rather than per message, and the bytes
  survive so the call can still be answered.
- `frame::data_payload` extracts the payload from one line of an SSE body,
  handling the framing space the grammar allows.

### Settling

- `settle::Settling` accumulates events; `settle::Settled` is a finished
  message. They are different types on purpose: a truncated stream cannot be
  read as a complete one. `Settling::settle` is the only bridge and returns
  `SettleError::Truncated` for a stream that never reached a terminal event.
  `Settled` is `#[non_exhaustive]`, so the struct literal is unavailable outside
  the crate and a finished message can only come from a finished stream.
- `message_delta` is *not* terminal even though it carries the `stop_reason`.
  Only `message_stop` and `error` are. `SettleError::Truncated::had_stop_reason`
  reports the case where the model finished its turn and the stream still broke,
  which is precisely where trusting the stop reason reports success wrongly.

### Response and usage

- `response::Response` decodes a non-streamed body, with `text`, `thinking` and
  `tool_calls` reading exactly as their `Settled` counterparts do, so switching a
  request between streaming and not leaves the reading code alone.
- `response::ApiError` decodes an error body, keeping the raw `error.type`
  beside the parsed `ErrorType` so a type outside the documented set stays
  legible.
- `usage::Usage` carries `input_tokens`, `output_tokens`,
  `cache_read_input_tokens`, `cache_creation_input_tokens`, the TTL split
  (`cache_creation.ephemeral_5m_input_tokens` /
  `ephemeral_1h_input_tokens`), and `output_tokens_details.thinking_tokens`.
  `total_input_tokens`, `cache_hit_rate` and `cache_creation_is_consistent`
  answer "did my cache work", which is otherwise unanswerable: a prefix below
  the model's minimum is cached silently and never errors.
- `Usage::merge_cumulative` is a pointwise maximum, which is the exact merge for
  counters Anthropic documents as cumulative — commutative, associative,
  idempotent, with zero as identity. So frame order does not matter, duplicates
  do not matter, and a later frame that omits a field cannot zero it. Not
  hypothetical: the observed `message_stop` carries only two counters, and a
  last-writer-wins merge would discard both cache numbers.

### Request

- `tool_choice::ToolChoice` types `auto`, `any`, `tool { name }` and `none`,
  with `disable_parallel_tool_use`. The type documents that changing it
  invalidates the *message* cache while leaving the tools and system caches
  valid — the asymmetry exists because the value is rendered near the messages,
  and it is the first thing Anthropic's troubleshooting list says to hold
  constant. `none` carries no parallel-use flag: with no calls permitted there
  is nothing to parallelize and the API refuses the combination.
- `Request::stream` and `Request::streamed()` express the `stream` flag, which
  the request type could not previously say at all.

### Crate

- `#![deny(missing_docs)]`, with a reason-for-existing on every public item. The
  `api_enum` macro now takes per-variant documentation.
- Split into files with one mission each: `frame` (SSE envelope and frame
  errors), `content` (blocks and deltas), `stream` (events), `settle`
  (accumulation), `response` (non-streamed body), `usage` (token counts).
- The model types moved out of `request` into a new `model` module, which had
  grown past the size a single file should hold. `request::Model` and its
  siblings are re-exported, so no existing path changed.
- `CLAUDE.md` §6 is rewritten: the crate is no longer request-only. §7 records
  what mid-conversation system messages would take.

### Tests

- `tests/captured/` holds verbatim bodies from a live Opus 5 endpoint: a forced
  tool call whose input arrives in six partial-JSON fragments, thinking under
  both `summarized` and `omitted` display, and a cache write and read of the
  same 1043-token prefix. `tests/captured_streams.rs` decodes them, and checks
  that *no* prefix of any of them settles before its terminal frame.

### Deliberately out of scope

- Mid-conversation tool changes (`tool_addition`, `tool_removal`). Measured
  against the gateway: rejected with a 400, `Input tag 'tool_removal' … does not
  match any of the expected tags`. Not added.
- Mid-conversation system messages (`{"role": "system"}` in `messages`) *are*
  accepted by the gateway and are a plausible later addition, but they are
  blocked structurally: system content is one private top-level `SystemPrompt`
  with no block form, and the cache slots would need a fourth location whose
  flow key sits among the messages. See `CLAUDE.md` §7.

## 0.1.0

Request-side bindings for the Messages API: a type per model carrying only the
parameters that model accepts, validated newtypes, cache breakpoints in four
named slots, and the token-counting sibling endpoint.
