# Repository instructions

Rust bindings for the Anthropic Messages API. [`SOUL.md`](SOUL.md) states the
mission and the design principles; obey it, and read it before changing a type.
This file is how to work here.

## Required checks

Before every commit:

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test`
- `cargo doc --no-deps` with no warnings

`#![deny(missing_docs)]` is on, and it applies inside macro expansions exactly as
outside them. Document *why* an item exists, not what it is: the wire shape is
already in the type, so the doc comment's job is the reason, the constraint, or
the failure it prevents.

Prove every impossibility claim with a `compile_fail` doctest, and check it fails
for the stated reason rather than incidentally — a typo compiles-fails too.
`cargo test` runs doctests, so a `compile_fail` that starts passing is caught.

## Where things live

Outbound:

- `context` — cache-safe, append-only conversation state; the cache slots.
- `model` — one type per model, plus the documented per-model constants.
- `request` — per-call parameters and the two request bodies.
- `tool_choice` — whether, and which, tool the model must call.

Inbound:

- `frame` — the SSE envelope and the vocabulary of a broken frame.
- `content` — the content blocks a message is made of, and their deltas.
- `stream` — one streamed frame becomes one typed event.
- `settle` — a stream becomes a finished message, or does not.
- `response` — a non-streamed response body, and an error body.
- `usage` — what a request cost, and what the cache did.

Shared:

- `values` — the enums mirroring closed API vocabularies, re-exported at the root.

Give each file one bounded mission and split it before 1,000 lines. `model` was
split out of `request` for exactly that reason.

## Versioning and the changelog

Add every user-visible change to `CHANGELOG.md` under a new version heading, and
bump `Cargo.toml` in the same commit. A breaking change gets a `### Breaking:`
section naming the old and new spelling and a **Migration.** paragraph a reader
can follow mechanically — the existing entries are the pattern to match.

Land each coherent piece as its own commit. Keep `README.md` true in the same
commit that changes what it describes.

Record where our deployment disagrees with the documented API, in **both**
directions, because the disagreement is not symmetric.

A gateway that *refuses* something the API documents does not remove it: the
crate still expresses the feature, and the changelog and the item's documentation
say plainly that it is refused today. Never write a test that requires the
gateway to accept it.

A gateway that *accepts* something the API forbids proves nothing. A 200 is not
evidence of legality, so a placement or per-model rule can only be established
from the documentation, never from a successful probe. Measured 2026-08-30: our
gateway accepts a leading `{"role": "system"}` message, which the documented API
answers with a 400, and `/v1/messages/count_tokens` validates placement not at all
— it accepted every illegal arrangement tried. The gateway *does* enforce the
successor rule, so its permissiveness is specific rather than general, which is
exactly why a probe cannot be trusted to map it. Use live probes to measure what
the API *rejects* and how it words the rejection; that direction is informative.

## Deliberately out of scope

- Any HTTP client, retry policy, or reconnection. Callers bring their own.
- Endpoints that are not the Messages API: Batches, Files, Models, Skills,
  Managed Agents, Admin. `count_tokens` is in scope because it takes the same
  conversation state.
- Consumer-specific policy: where a caller's stable prefix ends, which model to
  pick, how to render a conversation. That belongs in the consumer.
- Response-side convenience that re-derives what a caller can read from the
  types. Static lookup tables for API-documented wire values *are* in scope:
  an enum's `from_str` (the inverse of `as_str`) and the documented HTTP-status →
  `ErrorType` mapping are pure `match` on a primitive, which is wire vocabulary
  rather than a parser.

## Verifying against the live API

A captured real frame beats an invented fixture. `tests/captured/` holds verbatim
bodies from a live endpoint, and `tests/captured_streams.rs` decodes them and
checks that no prefix of any of them settles before its terminal frame. Add to
that directory whenever a new frame shape is observed.

`tests/live_api.rs` is gated and hits a real endpoint. Live tests consume
credentials and external capacity, so run them only when a claim actually needs
the wire — acceptance of a parameter, the exact wording of a rejection, the shape
of a frame. Cap `max_tokens` on acceptance-only cases so a probe costs a token or
two.

Probing what the API accepts is best done with a deliberately invalid tag: the
server answers with the complete list of tags it expects at that position, which
is a measurement rather than a guess. Reading a rejection's exact wording is how
the per-model differences in `model` were established.

Never print, log, or commit a credential.
