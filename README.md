# anthropic

Rust bindings for the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) — type-safe, cache-safe, bindings-only.

Bring your own HTTP client. This crate hands you a `Serialize` request body; you POST it.

Currently modeled: `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`.

## Example

```rust
use anthropic::{
    API_BASE, HEADER_API_KEY, HEADER_VERSION, MESSAGES_PATH, VERSION,
    context::Context,
    request::{Model, Request},
};

let mut ctx = Context::new().with_system("you are helpful");
ctx.push_user_text("hello");

let body = serde_json::to_value(Request::new(&ctx, Model::opus_4_7(), 1024))?;

reqwest::Client::new()
    .post(format!("{API_BASE}{MESSAGES_PATH}"))
    .header(HEADER_API_KEY, std::env::var("ANTHROPIC_API_KEY")?)
    .header(HEADER_VERSION, VERSION)
    .json(&body)
    .send()
    .await?;
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
