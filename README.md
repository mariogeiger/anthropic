# anthropic

Rust bindings for the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) — type-safe, cache-safe, and just the right amount opinionated.

**Scope:** bindings only. No HTTP client, no response parser. You bring `reqwest` (or whatever you like), this crate hands you a `Serialize` request body.

**Supported models:** the three latest Claude tiers — `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`. Older models are intentionally not modeled; use an older tag of this crate if you need them.

## What's inside

| Module | What it gives you |
|---|---|
| [`values`] | Enums + constants that mirror API JSON values (`as_str()` out, `from_str()` in). Stop reasons, error types, cache TTLs, thinking modes, usage field names. |
| [`context`] | `Context` — a cache-safe, append-only conversation state. Holds `system` + `tools` + `messages` + up to 4 named cache slots. |
| [`request`] | `Request` (per-call params + `Context`, serializes to `POST /v1/messages` body) and `CountRequest` (for `POST /v1/messages/count_tokens`). Plus `Model` — an enum whose variants only carry parameters the underlying model actually accepts. |

## Minimal example

```rust
use anthropic::{
    API_BASE, HEADER_API_KEY, HEADER_VERSION, MESSAGES_PATH, VERSION,
    context::Context,
    request::{Model, Opus4_7Effort, Request},
    ThinkingDisplay,
};

let mut ctx = Context::new().with_system("you are helpful");
ctx.push_user_text("hello");

let model = Model::opus_4_7()
    .with_effort(Opus4_7Effort::Xhigh)
    .with_adaptive_thinking(ThinkingDisplay::Summarized);

let body = serde_json::to_value(Request::new(&ctx, model, 1024))?;

// Then POST with whichever HTTP client you like:
let resp = reqwest::Client::new()
    .post(format!("{API_BASE}{MESSAGES_PATH}"))
    .header(HEADER_API_KEY, std::env::var("ANTHROPIC_API_KEY")?)
    .header(HEADER_VERSION, VERSION)
    .json(&body)
    .send()
    .await?;
```

## Design philosophy

**Model the runtime behavior, not HTTP field presence.** Types here describe what the model actually sees, not which JSON fields happen to appear on the wire. An `Option<T>` in these types always maps to a real runtime distinction — we never use `Option` just to mirror wire-format optionality.

**Unrepresentable requests are unrepresentable.** Each Claude model accepts a different subset of request parameters: Opus 4.7 rejects `temperature`, Haiku 4.5 rejects `output_config.effort`. The `Model` enum carries this at the type level — you cannot construct a `Model::Opus4_7` with a `temperature` field because the `Opus4_7` struct doesn't have one. The API returns 400 for invalid combinations; this crate catches them at compile time.

**Prompt caching is hard to break.** `Context` is append-only — no `&mut` access to past messages, so committed bytes stay frozen. Cache breakpoints live in 4 named slots (`CacheSlot::S0..S3`), mapping 1:1 to Anthropic's limit. `roll_cache` moves a slot to the current tail without touching past content. The ordering rule (1h breakpoints before 5m) is validated before mutation, so a bad call errors instead of corrupting state.

## Context with prompt caching

```rust
use anthropic::{CacheTtl, context::{CacheSlot, Context}};

// Anchor the system prompt in slot S0 with a 1h TTL:
let mut ctx = Context::new()
    .with_system_cached(CacheSlot::S0, "long stable system prompt", CacheTtl::OneHour);

ctx.push_user_text("turn 1");
ctx.push_assistant_text("reply 1");
ctx.push_user_text("turn 2");

// Roll slot S3 to the current tail each turn for a rolling 5m breakpoint:
ctx.roll_cache(CacheSlot::S3, CacheTtl::FiveMinutes)?;
```

Slots live on the struct; bytes don't move. The serialized request contains exactly what Anthropic expects (`cache_control` on the right content blocks, in the right TTL order).

## Counting tokens

```rust
use anthropic::{COUNT_TOKENS_PATH, request::{CountRequest, Model}};

let body = serde_json::to_value(CountRequest::new(&ctx, Model::sonnet_4_6()))?;
// POST to `{API_BASE}{COUNT_TOKENS_PATH}` — same headers as /messages.
```

`CountRequest` accepts a `Model` for symmetry with `Request`, but only serializes the `model` ID (the count-tokens endpoint ignores `max_tokens`, `temperature`, `thinking`, `output_config`, and `stop_sequences`).

## Adding this crate

```toml
[dependencies]
anthropic = { git = "https://github.com/<your-user>/anthropic" }
```

Crates.io is on the to-do list.

## License

TBD — add before publishing.
