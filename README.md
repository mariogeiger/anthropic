# anthropic

Rust bindings for the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) — type-safe, cache-safe, both directions.

Bring your own HTTP client. This crate hands you a `Serialize` request body and decodes what comes back: a streaming decoder, a response decoder, and the usage counts that prove your prompt cache is working.

Currently modeled: `claude-fable-5`, `claude-opus-5`, `claude-opus-4-8`, `claude-sonnet-5`, `claude-sonnet-4-6`, `claude-haiku-4-5`.

## What the types make impossible

- **A truncated stream cannot be read as a finished one.** `Settling` accumulates and has no method that yields a message; `Settled` is a finished message and has no method that takes more events. `Settling::settle` is the only bridge, and it fails on a stream that never reached `message_stop`. `Settled` is `#[non_exhaustive]`, so you cannot write the struct literal to get around it.
- **A new server event cannot break the decoder.** Anthropic's versioning policy permits new event types. An unknown event, block kind, or delta kind is a variant that means *ignore me*, never an error. A frame that contradicts the schema still is one.
- **A silent cache failure cannot hide.** A cached prefix below the model's minimum caches nothing and returns no error. `Usage::cache_hit_rate` is how you notice. Merging usage across frames is a pointwise maximum, so a later frame that reports fewer fields cannot zero the cache counts an earlier one gave you.
- **A stop reason cannot be mistaken for the end of a stream.** `message_delta` carries `stop_reason` but is not terminal; only `message_stop` and `error` are. A stream cut off in between fails to settle and says the stop reason had already arrived.
- **A parameter a model rejects does not exist on that model's type.** Sonnet 4.6 has no `xhigh` effort; Opus 5 with thinking off has no `xhigh` or `max`; Haiku 4.5 has no effort at all.
- **A value outside an API vocabulary cannot be named.** Every closed set of wire strings is an enum that serializes to its own string, so there is no `&'static str` field to put an unknown value in. A month is one of twelve variants, not a `u8` documented as 1–12; a temperature is a newtype, not an `f32` documented as 0–1.
- **A role cannot carry content that role refuses.** `Message` is an enum whose variant *is* the role, so a system message holds only the text and tool changes the API accepts there — not an image, and not a tool result. There is no `role` field to set independently of the content.
- **A checked constructor cannot be bypassed.** `Request` and `CountRequest` hold private fields behind readers, so `max_tokens` cannot be reassigned past the range check `Request::new` runs against the model's maximum.
- **A system message cannot sit where the API forbids one.** `Context::messages` is private, so `push_system` is the only way one enters a conversation, and it refuses the first position outright — that position belongs to the top-level `system` field. `Request::new` refuses the rest: a placement whose successor is wrong, and a model that does not accept a mid-conversation system message at all, which today means Sonnet 5.

## What the crate covers

Both request bodies (`/v1/messages` and `/v1/messages/count_tokens`), the
non-streamed response, the SSE stream, and the error body. On the way out: a type
per model, the four cache breakpoints, tools with their search and validation
flags, `tool_choice`, stop sequences, `service_tier`, `metadata`, structured
output, images, documents, search results, and mid-conversation system messages
including tool additions and removals. On the way in: every documented event and
delta kind, content blocks, citations, refusal details, and the full usage
breakdown.

Not covered, deliberately: no HTTP client, and no endpoint but Messages — Batches,
Files, Models, and Skills are separate APIs. Server-side tools (web search, web
fetch, code execution, computer and browser use, MCP toolsets) are not yet
declarable; their blocks decode as `Unmodeled` rather than failing.

## Sending a request

```rust
use anthropic::{
    API_BASE, HEADER_API_KEY, HEADER_VERSION, MESSAGES_PATH, VERSION,
    context::Context,
    request::{Model, Request},
};

let mut ctx = Context::new().with_system("you are helpful");
ctx.push_user_text("hello");

let body = serde_json::to_value(Request::new(&ctx, Model::opus_5(), 1024)?.streamed())?;

reqwest::Client::new()
    .post(format!("{API_BASE}{MESSAGES_PATH}"))
    .header(HEADER_API_KEY, std::env::var("ANTHROPIC_API_KEY")?)
    .header(HEADER_VERSION, VERSION)
    .json(&body)
    .send()
    .await?;
```

## Reading a stream

```rust
use anthropic::frame::data_payload;
use anthropic::settle::Settling;

let mut settling = Settling::new();
for line in body.lines() {
    if let Some(payload) = data_payload(line) {
        // Returns the decoded event too, so you can render deltas as they land.
        settling.consume_payload(payload)?;
    }
}

// The only way to get a finished message.
let settled = settling.settle()?;
println!("{}", settled.text());
println!("stopped because {:?}", settled.stop_reason());
println!("cache hit rate {:?}", settled.usage.cache_hit_rate());

for call in settled.tool_calls() {
    // Input is bytes until you ask: `input_json_delta` sends partial JSON, and
    // a malformed call should not cost you the whole message.
    println!("{} {:?}", call.name, call.input.decode());
}
```

## Reading a non-streamed response

```rust
use anthropic::response::Response;

let response = Response::decode(&raw_body)?;
println!("{}", response.text());
println!("{} tokens read from cache", response.usage.cache_read_input_tokens);
```

## Install

```toml
[dependencies]
anthropic = { git = "https://github.com/mariogeiger/anthropic" }
```

## Design

See [SOUL.md](SOUL.md) for the mission and the design principles.

## License

[MIT](LICENSE).
