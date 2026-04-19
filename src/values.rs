//! Enums mirroring API JSON values. Outbound only: `as_str()` feeds the
//! request serializer. §6 ("bindings only — no response parser") means
//! there is no `from_str` / HTTP-status mapping / usage-field parser.

macro_rules! api_enum {
    ($name:ident { $($variant:ident => $s:literal),* $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name { $($variant),* }
        impl $name {
            pub fn as_str(self) -> &'static str {
                match self { $($name::$variant => $s),* }
            }
        }
    };
}

api_enum! { ImageMediaType {
    Jpeg => "image/jpeg", Png => "image/png", Gif => "image/gif", Webp => "image/webp",
}}

// `thinking.type`. `Enabled` is the legacy fixed-budget form (deprecated on Sonnet 4.6,
// removed on Opus 4.7); `Adaptive` is the only form currently emitted by `Request`.
api_enum! { ThinkingType { Enabled => "enabled", Adaptive => "adaptive" } }

// Opus 4.7 `thinking.display`. Default `Omitted` = thinking streams but text is empty.
api_enum! { ThinkingDisplay { Summarized => "summarized", Omitted => "omitted" } }

api_enum! { StopReason {
    EndTurn => "end_turn",
    MaxTokens => "max_tokens",
    StopSequence => "stop_sequence",
    ToolUse => "tool_use",
    PauseTurn => "pause_turn",
    Refusal => "refusal",
}}

api_enum! { ErrorType {
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

api_enum! { CacheControlType { Ephemeral => "ephemeral" } }

api_enum! { CacheTtl { FiveMinutes => "5m", OneHour => "1h" } }
