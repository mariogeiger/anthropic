//! Enums mirroring API JSON values. `as_str()` is outbound; `from_str()` / the
//! HTTP-status table are pure `match`-on-primitive lookup tables — wire
//! vocabulary under §6, not a response parser.

// Base: enum + `as_str()`. Optional `roundtrip` prefix adds `from_str()`.
macro_rules! api_enum {
    (@base $name:ident { $($variant:ident => $s:literal),* $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name { $($variant),* }
        impl $name {
            pub fn as_str(self) -> &'static str {
                match self { $($name::$variant => $s),* }
            }
        }
    };
    (roundtrip $name:ident { $($variant:ident => $s:literal),* $(,)? }) => {
        api_enum! { @base $name { $($variant => $s),* } }
        impl $name {
            #[allow(clippy::should_implement_trait)]
            pub fn from_str(s: &str) -> Option<Self> {
                match s {
                    $($s => Some($name::$variant),)*
                    _ => None,
                }
            }
        }
    };
    ($name:ident { $($variant:ident => $s:literal),* $(,)? }) => {
        api_enum! { @base $name { $($variant => $s),* } }
    };
}

pub(crate) use api_enum;

api_enum! { ImageMediaType {
    Jpeg => "image/jpeg", Png => "image/png", Gif => "image/gif", Webp => "image/webp",
}}

// `thinking.type`. `Enabled` is the legacy fixed-budget form (deprecated on Sonnet 4.6,
// removed on Opus 4.7); `Adaptive` is the only form currently emitted by `Request`.
api_enum! { ThinkingType { Enabled => "enabled", Adaptive => "adaptive" } }

// Opus 4.7 `thinking.display`. Default `Omitted` = thinking streams but text is empty.
api_enum! { ThinkingDisplay { Summarized => "summarized", Omitted => "omitted" } }

api_enum! { roundtrip StopReason {
    EndTurn => "end_turn",
    MaxTokens => "max_tokens",
    StopSequence => "stop_sequence",
    ToolUse => "tool_use",
    PauseTurn => "pause_turn",
    Refusal => "refusal",
}}

api_enum! { roundtrip ErrorType {
    InvalidRequest => "invalid_request_error",
    Authentication => "authentication_error",
    Billing => "billing_error",
    Permission => "permission_error",
    NotFound => "not_found_error",
    RequestTooLarge => "request_too_large",
    RateLimit => "rate_limit_error",
    Api => "api_error",
    Timeout => "timeout_error",
    Overloaded => "overloaded_error",
}}

impl ErrorType {
    /// Documented HTTP-status-code → `ErrorType` mapping. Pure lookup table,
    /// in-scope under §6; not a response parser.
    pub fn from_status(status: u16) -> Option<Self> {
        Some(match status {
            400 => Self::InvalidRequest,
            401 => Self::Authentication,
            402 => Self::Billing,
            403 => Self::Permission,
            404 => Self::NotFound,
            413 => Self::RequestTooLarge,
            429 => Self::RateLimit,
            500 => Self::Api,
            504 => Self::Timeout,
            529 => Self::Overloaded,
            _ => return None,
        })
    }
}

api_enum! { CacheControlType { Ephemeral => "ephemeral" } }

api_enum! { CacheTtl { FiveMinutes => "5m", OneHour => "1h" } }
