# Changelog

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
