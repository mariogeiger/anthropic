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

See [CLAUDE.md](CLAUDE.md) for the design philosophy.

## License

[MIT](LICENSE).
