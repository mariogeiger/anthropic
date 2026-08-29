# Changelog

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
