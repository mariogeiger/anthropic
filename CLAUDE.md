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

Bindings only. The crate produces a serializable request body for the Messages API and its token-counting sibling, and nothing else — no HTTP client, no response parser, no retry logic, no streaming decoder. Callers bring their own HTTP stack.

The crate tracks the current Claude tiers. Older models are not wired up by default, but adding them is a normal extension — follow the per-model-type approach in §2.
