# Soul

Represent the entire Anthropic Messages API faithfully in types.

That is why this crate exists. Not a convenience wrapper, not the subset one
consumer happens to need: the whole wire, in both directions, as types that
admit exactly what the API admits. A request the API would answer with a 400
should not compile, and a response the API can send should decode.

## The mission, stated precisely

*Faithfully* is the load-bearing word, and it cuts three ways.

**Complete.** Every field, block, event and enumerated value the documented API
carries has a place here. A gap is a defect, not a scope decision. Where the
crate does not yet cover something, that is recorded as unfinished work rather
than defended as a boundary.

**Exact.** The types admit what the API admits and nothing more. A closed set of
wire strings is an enum, a range-checked number is a newtype, and a combination
the API refuses is a combination that cannot be written.

**Honest.** The crate reports the API, not a gateway's opinion of it. Anthropic's
documented surface is the specification. Where a deployment refuses something the
API documents, the crate still expresses it and says so in the documentation and
the changelog — a caller can then discover the refusal for itself, which is the
truth, instead of finding the feature missing, which is not.

## Both directions, no transport

The crate produces a serializable request body and decodes what comes back. No
HTTP client, no retry logic, no reconnection. Callers bring their own stack.

Decoding is in scope for the same reason the request half is: a consumer that
hand-matches raw JSON re-derives the API's shape badly, and the failure modes
that matter — a stream that stopped early, a cache that silently did nothing —
are exactly the ones a type can rule out.

## Design principles

### A type per model, so a refused combination cannot be built

Each Claude model accepts a different subset of parameters, and the API returns
400 for a combination it does not take. Model-specific parameters therefore live
on model-specific types: a parameter a given model rejects does not exist on that
model's type. Adding support for another model means a new type carrying only its
accepted parameters — never widening an existing one.

The subsets are not cosmetically different. Thinking is always on for Fable 5 and
has no off state; on Opus 4.8 "off" is an *omitted* `thinking` field; on Opus 5
and Sonnet 5 an omitted field leaves thinking *on*, so off must be stated. Opus 5
makes the accepted effort range depend on whether thinking is on, which is why
its effort lives *inside* its thinking state rather than beside it.

Mutually exclusive settings are sum types, never two optional fields a caller
must keep in sync.

Where the type stops is worth stating, because it is a principle and not an
excuse. A model-specific *parameter* lives on the model's type, so the bad
request is unwritable. A refused combination that pairs the model with the
*conversation* is a different shape: the two are deliberately separate types so
one conversation can be sent to several models, and making the pairing
unrepresentable would parameterize the conversation by model and destroy exactly
that. A mid-conversation system message on Sonnet 5 is that shape, so it is a
typed refusal in `Request::new` that names the model. When impossibility would
cost more generality than the 400 it prevents, refuse at the one point where both
facts are known, and say in a doc comment why the type could not carry it.

### No invented defaults, no normalization

Defaults come from the provider's documentation. The crate does not invent its
own and does not normalize values on the caller's behalf.

### A documented server-side default is a plain, always-emitted field

A scalar parameter the API accepts unconditionally and documents a default for is
a plain (non-`Option`) field whose `Default::default()` mirrors the documented
value, and it is *always* emitted. Emitting explicitly makes the request body a
complete record of what the model sees, and shields callers from a silent
behavior change if the provider's defaults shift.

Serialization is explicit throughout: no omit-if-default, no hidden
normalization. Reading a request value tells you exactly what the model will see.

### Omission is reserved for genuine runtime absence

An `Option` means something is really present or really absent at runtime —
thinking off versus on, a stop sequence given or not. It never means "the value
happened to equal the default". Types describe what the model *sees*, not which
JSON fields appear on the wire; where the wire offers several shapes for one
runtime concept, the type models the concept and the serializer picks the shape.

Two deliberate exceptions, both transport rather than content: `stream`, which
the model never sees, and `tool_choice`, whose absence must stay byte-identical
to a request that never mentioned it or the message cache key moves.

### An invariant is held by the type, or it is not held

A doc comment saying a field is "set by" some method, or a private constructor
that checks what a public field lets a caller assign around, is a convention the
compiler does not know about — and this crate has shipped exactly that mistake.
Two rules follow, and they are not in tension.

*A closed API vocabulary is an enum, never a string.* A `&'static str` field
accepts any string; a field of an enum with no invalid variant accepts only what
the API does. So the enum goes in the field and the string comes out of it at
serialization time. A public field of such an enum is safe, and preferable — it
keeps pattern matching available for free.

*A cross-field invariant means private fields plus readers.* Where validity is a
relation between fields (`max_tokens` against *this model's* maximum, a thinking
budget against `max_tokens`), no single field's type can carry it, so the
checking constructor must be the only way in and the fields must not be
assignable afterwards. Where there is no such relation, do not hide the field:
readers that guard nothing cost pattern matching and buy nothing.

### An impossibility claim is tested, not asserted

The `compile_fail` doctest is how a claim of impossibility gets tested. A claim
only a comment makes is the failure mode this principle exists to prevent, and
each such doctest must fail for the reason it states rather than incidentally.

### Prompt caching is hard to break by construction

Conversation state is append-only. Once content has been committed its bytes are
frozen — there is no API for rewriting history, because rewriting history
silently invalidates the prompt cache.

Cache breakpoints live in a fixed, named set of slots that mirrors the provider's
limit one-to-one, so requesting more breakpoints than the API accepts is not a
runtime error but an unwritable program. Moving a breakpoint is a metadata-only
operation on the slot; the underlying content never shifts. Removing one only
clears metadata.

Every placement rule the API enforces at request time — TTL ordering, no two
breakpoints on one position with different TTLs, anchors that must not move once
set — is checked *before* the mutation commits. A bad call returns an error
instead of corrupting state.

`cache_control` is unreachable from outside the crate: it has no public
constructor, no public fields, and every slot holding one is crate-private. The
only way to place a breakpoint is through a named slot, which keeps slot
bookkeeping consistent with content.

Before adding any operation that mutates conversation state, convince yourself it
cannot invalidate a previous cache prefix.

### Conversation state and per-call parameters are different types

Conversation state — system prompt, tools, message history, cache breakpoints —
is stable across turns and lives in its own type. Per-call parameters — the
model, token limits, stop sequences — live on the request type, which borrows it.

One conversation can then be sent to different models or sampling settings
without touching the type that upholds the cache invariants. Auxiliary endpoints
such as token counting follow the same pattern: same conversation state,
different per-call shape.

### Unknown is not broken; incomplete is a different type from complete

Anthropic's versioning policy permits new event types, and adding one is a
compatible change. An unrecognized event, content-block kind, or delta kind is
therefore a variant meaning "ignore me", never an error. What *is* an error is a
frame that contradicts the schema: not JSON, not an object, no `type`, a field of
the wrong type, or a `usage` object that will not deserialize.

A truncated stream must not be readable as a finished message. The accumulator
cannot yield a message and the finished message cannot take more events; the only
bridge between them fails on a stream that never reached a terminal event, and
`#[non_exhaustive]` makes that structural rather than decorative. Note that
`message_delta` carries the `stop_reason` but is *not* terminal.

### Cache accounting is not optional detail

A cached prefix below the model's minimum is a silent no-op, and the only evidence
is `usage`. So usage decodes rather than being skipped, and merging across frames
is a pointwise maximum — the join of a product lattice of counters — so a later
frame that omits a field cannot zero it.
