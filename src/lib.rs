//! Anthropic Messages API bindings: the typed wire in both directions.
//!
//! Outbound, a request whose invalid forms do not compile: a struct per model
//! carrying only the parameters that model accepts, validated newtypes, and cache
//! breakpoints in a fixed set of slots that mirrors the API's limit one-to-one.
//! Inbound, a streaming decoder and a response decoder that make a truncated
//! stream unreadable as a finished one; [`settle::Settling`] shows a stream read
//! end to end.
//!
//! * Outbound: [`block`], [`context`], [`document`], [`model`], [`request`],
//!   [`system`], [`tool_choice`].
//! * Both directions: [`web_search`].
//! * Inbound: [`frame`], [`content`], [`stream`], [`settle`],
//!   [`input_transformation`], [`response`], [`prompt_cache`], [`usage`].
//! * Shared: [`values`], re-exported at the root.
//!
//! See [`SOUL.md`](https://github.com/mariogeiger/anthropic/blob/main/SOUL.md)
//! for the mission and the design rules the whole crate follows.

#![deny(missing_docs)]

pub mod block;
pub mod content;
pub mod context;
pub mod document;
pub mod frame;
pub mod input_transformation;
pub mod model;
pub mod prompt_cache;
pub mod request;
pub mod response;
pub mod settle;
pub mod stream;
pub mod system;
mod tool;
pub mod tool_choice;
pub mod usage;
pub mod values;
pub mod web_search;

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
