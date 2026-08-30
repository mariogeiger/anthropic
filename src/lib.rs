//! Anthropic Messages API bindings: the typed wire in both directions.
//!
//! Outbound, a request whose invalid forms do not compile: a struct per model
//! carrying only the parameters that model accepts, validated newtypes, and cache
//! breakpoints in a fixed set of slots that mirrors the API's limit one-to-one.
//! Inbound, a streaming decoder and a response decoder that make a truncated
//! stream unreadable as a finished one.
//!
//! See [`SOUL.md`](https://github.com/mariogeiger/anthropic/blob/main/SOUL.md)
//! for the mission and the design rules the whole crate follows.
//!
//! # Outbound
//!
//! * [`context`] — cache-safe, append-only conversation state.
//! * [`model`] — one type per model, carrying only the parameters it accepts.
//! * [`request`] — per-call parameters and the `/v1/messages` body.
//! * [`tool_choice`] — whether, and which, tool the model must call.
//!
//! # Inbound
//!
//! * [`frame`] — the Server-Sent Events envelope, and what a broken frame is.
//! * [`content`] — the content blocks a message is made of, and their deltas.
//! * [`stream`] — one streamed frame becomes one typed event.
//! * [`settle`] — a stream becomes a finished message, or does not.
//! * [`response`] — a non-streamed response body.
//! * [`usage`] — what a request cost, and what the cache did.
//!
//! # Shared
//!
//! * [`values`] — the enums that mirror API JSON values, re-exported at the root.
//!
//! # Reading a stream
//!
//! ```
//! use anthropic::frame::data_payload;
//! use anthropic::settle::{Outcome, Settling};
//!
//! // Whatever your HTTP client hands you, line by line.
//! let body = concat!(
//!     "event: message_start\n",
//!     r#"data: {"type":"message_start","message":{"id":"msg_1","model":"claude-opus-5","content":[],"#,
//!     r#""usage":{"input_tokens":36,"cache_read_input_tokens":1043,"output_tokens":1}}}"#, "\n",
//!     "\n",
//!     "event: content_block_start\n",
//!     r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#, "\n",
//!     "\n",
//!     "event: content_block_delta\n",
//!     r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"21"}}"#, "\n",
//!     "\n",
//!     "event: message_delta\n",
//!     r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":45}}"#, "\n",
//!     "\n",
//!     "event: message_stop\n",
//!     r#"data: {"type":"message_stop"}"#, "\n",
//! );
//!
//! let mut settling = Settling::new();
//! for line in body.lines() {
//!     if let Some(payload) = data_payload(line) {
//!         settling.consume_payload(payload)?;
//!     }
//! }
//!
//! // The only way to get a finished message. A stream cut off before
//! // `message_stop` fails here instead of returning a half answer.
//! let settled = settling.settle()?;
//! assert_eq!(settled.text(), "21");
//! assert!(matches!(settled.outcome, Outcome::Stopped { .. }));
//! assert_eq!(settled.usage.cache_read_input_tokens, 1_043);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]

pub mod content;
pub mod context;
pub mod frame;
pub mod model;
pub mod request;
pub mod response;
pub mod settle;
pub mod stream;
pub mod tool_choice;
pub mod usage;
pub mod values;

pub use values::*;

/// The first-party API's origin.
pub const API_BASE: &str = "https://api.anthropic.com";
/// Path of the Messages endpoint.
pub const MESSAGES_PATH: &str = "/v1/messages";
/// Path of the token-counting endpoint.
pub const COUNT_TOKENS_PATH: &str = "/v1/messages/count_tokens";
/// `anthropic-version` header value, required on every request.
pub const VERSION: &str = "2023-06-01";
/// Name of the API-key header.
pub const HEADER_API_KEY: &str = "x-api-key";
/// Name of the required version header.
pub const HEADER_VERSION: &str = "anthropic-version";
/// Name of the header that opts into beta features.
pub const HEADER_BETA: &str = "anthropic-beta";
