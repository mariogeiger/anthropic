# CLAUDE.md

Design notes for the `anthropic` crate — Rust bindings for the Messages API.

The crate enforces one idea: *make invalid requests and broken caches unrepresentable in the type system*, so the compiler catches what the API would otherwise reject with a 400.

## 1. Prompt caching is hard to break by construction

Conversation state is append-only. Once content has been committed, its bytes are frozen — there is no API for rewriting history, because rewriting history silently invalidates the prompt cache.

Cache breakpoints live in a fixed, named set of slots that mirrors the provider's limit one-to-one, so it is impossible to request more breakpoints than the API accepts. Moving a breakpoint is a metadata-only operation on the slot; the underlying content never shifts. Removing a breakpoint only clears metadata.

All ordering and placement rules the API enforces at request time (TTL ordering, no two breakpoints on the same position with different TTLs, breakpoints that must not be moved once anchored) are checked *before* the mutation commits. A bad call returns an error instead of corrupting state.

Before adding any operation that mutates conversation state, convince yourself it cannot invalidate a previous cache prefix.

## 2. Unrepresentable requests are unrepresentable

Each Claude model accepts a different subset of request parameters, and the API returns 400 for invalid combinations. Model-specific parameters are carried on model-specific types, so a parameter a given model rejects simply does not exist in that model's configuration.

Mutually exclusive settings (for example, two sampling modes the server treats as one-or-the-other) are expressed as sum types, not as independent optional fields that the caller must remember to keep in sync.

If a knob is model-specific, it belongs on the model-specific type, not on the shared request type. Adding support for another model means a new model-specific type carrying only its accepted parameters — never widening an existing one.

An invariant is held by the type or it is not held. A doc comment saying a field is "set by" some method, or a private constructor that checks what a public field lets a caller assign around, is a convention the compiler does not know about — and this crate has already shipped exactly that mistake. Two rules follow, and they are not in tension:

*A closed API vocabulary is an enum, never a string.* A `&'static str` field accepts any string; a field of an enum with no invalid variant accepts only what the API does. So the enum goes in the field, and the string comes out of it at serialization time. A public field of such an enum is perfectly safe, and preferable — it keeps pattern matching available for free.

*A cross-field invariant means private fields plus readers.* Where validity is a relation between fields (`max_tokens` against *this model's* maximum, a thinking budget against `max_tokens`), no single field's type can carry it, so the checking constructor must be the only way in and the fields must not be assignable afterwards. Where there is no such relation, do not hide the field: readers that guard nothing cost pattern matching and buy nothing.

The `compile_fail` doctest is how a claim of impossibility gets tested. A claim only a comment makes is the failure mode this section exists to prevent.

## 3. Model runtime behavior, not HTTP field presence

Types describe what the model actually *sees*, not which JSON fields happen to appear on the wire. Optional fields represent real runtime distinctions — something is present or not, configured or not — never "the field was omitted from the JSON."

When the wire format offers multiple shapes for the same runtime concept (a bare string vs. a one-element array, for example), the type models the single runtime concept and the serializer picks the shape. Callers should not have to think about wire-format variants.

Defaults come from the provider's documentation. The crate does not invent its own defaults or normalize values on the caller's behalf.

Scalar parameters that the API accepts unconditionally — those with a documented server-side default — are modeled as plain (non-`Option`) fields whose `Default::default()` mirrors the value the API documents as its default. They are *always* emitted on the wire. The crate never relies on server-side defaulting via field omission: emitting explicitly makes the request body a complete record of what the model sees, and shields callers from silent behavior changes if the provider's defaults shift. Omission is reserved for the runtime-distinction case above (e.g. `thinking` off vs on), not for "the value happens to equal the default."

## 4. Conversation state vs. per-call parameters

Conversation state — system prompt, tools, message history, cache breakpoints — is stable across turns and lives in its own type. Per-call parameters — the model, token limits, stop sequences — live on the request type, which borrows the conversation state.

This split makes it natural to reuse the same conversation with different models or sampling settings, and it keeps the cache-safety invariants of §1 on a type that exists specifically to uphold them.

Auxiliary endpoints (for example, token counting) follow the same pattern: same conversation state, different per-call shape.

## 5. Explicit serialization, no omit-if-default

Serialization emits whatever the value represents. There is no "omit if equal to default" optimization and no hidden normalization — reading a request value tells you exactly what the model will see. Scalar parameters with a documented server-side default are still emitted explicitly, with their `Default::default()` set to the value the API documents (see §3).

The one kind of omission the crate uses is for optional fields that are genuinely absent at runtime (see §3). An absent optional is a real runtime absence, not a default elided on the wire.

## 6. Scope

Bindings for both halves of the wire: the crate produces a serializable request body and decodes what comes back. No HTTP client, no retry logic, no reconnection. Callers bring their own HTTP stack and hand the bytes over.

Decoding is in scope for the same reason the request half is: a consumer that hand-matches raw JSON re-derives the API's shape badly, and the failure modes that matter — a stream that stopped early, a cache that silently did nothing — are exactly the ones a type can rule out. So the inbound side is held to the outbound side's standard.

Three rules keep it honest:

*Unknown is not broken.* Anthropic's versioning policy permits new event types, and adding one is a compatible change. An unrecognized event, content-block kind, or delta kind is therefore a variant that means "ignore me", never an error. What *is* an error is a frame that contradicts the schema: not JSON, not an object, no `type`, a field of the wrong type, or a `usage` object that will not deserialize.

*Incomplete is a different type from complete.* A truncated stream must not be readable as a finished message. `Settling` accumulates and cannot yield a message; `Settled` is a finished message and cannot take more events; `Settling::settle` is the only bridge and fails on a stream that never reached a terminal event. `#[non_exhaustive]` on `Settled` is what makes that claim structural rather than decorative — a caller cannot write the struct literal, so a finished message can only come from a finished stream. Note that `message_delta` carries the `stop_reason` but is *not* terminal: only `message_stop` and `error` are.

*Cache accounting is not optional detail.* A cached prefix below the model's minimum is a silent no-op, and the only evidence is `usage`. So usage decodes rather than being skipped, and the merge across frames is a pointwise maximum — the join of a product lattice of counters — so a later frame that omits a field cannot zero it.

Static lookup tables for API-documented wire values are part of the wire vocabulary: enum `from_str` (inverse of `as_str`), and the documented HTTP-status-code → `ErrorType` mapping. Both are pure `match` on a primitive.

Out of scope, deliberately: mid-conversation tool changes (`tool_addition`, `tool_removal`). Measured against the gateway, they are rejected with a 400. Mid-conversation system messages (`{"role": "system"}` appended to `messages`) are accepted and are a plausible later addition; see §7 for what they would take.

The crate tracks the current Claude tiers. Older models are not wired up by default, but adding them is a normal extension — follow the per-model-type approach in §2.

## 7. Mid-conversation system messages

Anthropic supports appending a `{"role": "system"}` message to `messages` on Fable 5, Opus 5, and Opus 4.8, adding an instruction partway through a conversation *without* invalidating the system or message caches — the cached prefix is untouched because nothing before it changed. The gateway accepts it. This crate cannot express it yet, and the obstruction is structural rather than a missing field:

- `Message::role` is a `Role` of exactly `User` and `Assistant`, so `"system"` is
  not a value the type can hold. That guard is worth keeping; the addition is a
  `push_system` beside the others plus a `Role::System` variant introduced
  *together* with the placement checks that make it legal, not a public string.
- The blocker is `SystemPrompt`: system content is one private top-level struct with two wire shapes, so "the system prompt" and "a system message in the history" are different types today. Supporting both means system content becoming a shared block type that either position can hold.
- Cache slots then need a fourth `SlotLocation`, and `flow_key` a position for it. A mid-conversation system message sits *in* the message sequence, so its flow key belongs with the messages, not with the top-level system anchor — getting that wrong would let TTL-ordering validation pass on a request the API rejects.
- It is per-model: Sonnet 5 refuses it. Under §2 that makes it a per-model-type capability, not a widening of the shared `Context`.

Doing it properly is a change to §1's invariants, which is why it is not a drive-by.

## 8. Details

Work in the main branch.
