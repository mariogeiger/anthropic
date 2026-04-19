//! Anthropic Messages API bindings.

pub mod context;
pub mod request;
pub mod values;

pub use values::*;

pub const API_BASE: &str = "https://api.anthropic.com";
pub const MESSAGES_PATH: &str = "/v1/messages";
pub const COUNT_TOKENS_PATH: &str = "/v1/messages/count_tokens";
/// `anthropic-version` header value, required on every request.
pub const VERSION: &str = "2023-06-01";
pub const HEADER_API_KEY: &str = "x-api-key";
pub const HEADER_VERSION: &str = "anthropic-version";
pub const HEADER_BETA: &str = "anthropic-beta";
