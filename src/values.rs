//! Enums and constants that mirror Anthropic API JSON values exactly
//! (`as_str()` for outbound, `from_str()` for inbound).
//!
//! These types are the vocabulary of the Messages API — they don't carry
//! behavior, they just make the string values type-safe to reference.

macro_rules! api_enum_as_str {
    ($(#[$meta:meta])* $vis:vis enum $name:ident { $($variant:ident => $s:literal),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        $vis enum $name {
            $($variant),*
        }

        impl $name {
            pub fn as_str(self) -> &'static str {
                match self {
                    $($name::$variant => $s),*
                }
            }
        }
    };
}

macro_rules! api_enum_roundtrip {
    ($(#[$meta:meta])* $vis:vis enum $name:ident { $($variant:ident => $s:literal),* $(,)? }) => {
        api_enum_as_str! {
            $(#[$meta])*
            $vis enum $name { $($variant => $s),* }
        }

        impl $name {
            #[allow(clippy::should_implement_trait)]
            pub fn from_str(s: &str) -> Option<Self> {
                match s {
                    $($s => Some($name::$variant)),*
                    ,
                    _ => None,
                }
            }
        }
    };
}

// ── Image source ──────────────────────────────────────────────────────────────

api_enum_as_str! {
    pub enum ImageMediaType {
        Jpeg => "image/jpeg",
        Png => "image/png",
        Gif => "image/gif",
        Webp => "image/webp",
    }
}

// ── Thinking config ───────────────────────────────────────────────────────────

// The `type` field inside the `thinking` request object. `Enabled` is the
// legacy fixed-budget form (`{type: "enabled", budget_tokens: N}`), deprecated
// on Sonnet 4.6 and removed on Opus 4.7. `Adaptive` is the only form currently
// emitted by `crate::request::Request`.
api_enum_as_str! {
    pub enum ThinkingType {
        Enabled => "enabled",
        Adaptive => "adaptive",
    }
}

// Controls what is returned in the `thinking` content block on Opus 4.7. The
// API default is `Omitted` (thinking blocks stream but their text is empty);
// opt into `Summarized` to get visible thinking progress.
api_enum_as_str! {
    pub enum ThinkingDisplay {
        Summarized => "summarized",
        Omitted => "omitted",
    }
}

// ── Stop reasons ─────────────────────────────────────────────────────────────

/// Values of the `stop_reason` field in responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Model reached a natural stopping point.
    EndTurn,
    /// `max_tokens` was reached.
    MaxTokens,
    /// A custom stop sequence was generated.
    StopSequence,
    /// Model invoked one or more tools.
    ToolUse,
    /// A long-running turn was paused.
    PauseTurn,
    /// Streaming classifier refused to continue generation. Added 2025 with Claude 4.
    Refusal,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::EndTurn => "end_turn",
            StopReason::MaxTokens => "max_tokens",
            StopReason::StopSequence => "stop_sequence",
            StopReason::ToolUse => "tool_use",
            StopReason::PauseTurn => "pause_turn",
            StopReason::Refusal => "refusal",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "end_turn" => Some(StopReason::EndTurn),
            "max_tokens" => Some(StopReason::MaxTokens),
            "stop_sequence" => Some(StopReason::StopSequence),
            "tool_use" => Some(StopReason::ToolUse),
            "pause_turn" => Some(StopReason::PauseTurn),
            "refusal" => Some(StopReason::Refusal),
            _ => None,
        }
    }
}

// ── Error types ───────────────────────────────────────────────────────────────

api_enum_roundtrip! {
    /// Values of `error.type` in error response bodies.
    pub enum ErrorType {
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
    }
}

impl ErrorType {
    /// Guess the error type from an HTTP status code.
    pub fn from_status(status: u16) -> Option<Self> {
        match status {
            400 => Some(ErrorType::InvalidRequest),
            401 => Some(ErrorType::Authentication),
            402 => Some(ErrorType::Billing),
            403 => Some(ErrorType::Permission),
            404 => Some(ErrorType::NotFound),
            413 => Some(ErrorType::RequestTooLarge),
            429 => Some(ErrorType::RateLimit),
            500 => Some(ErrorType::Api),
            504 => Some(ErrorType::Timeout),
            529 => Some(ErrorType::Overloaded),
            _ => None,
        }
    }
}

// ── Prompt caching ────────────────────────────────────────────────────────────

api_enum_as_str! {
    pub enum CacheControlType {
        Ephemeral => "ephemeral",
    }
}

/// TTL values accepted by the ephemeral cache control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTtl {
    FiveMinutes,
    OneHour,
}

impl CacheTtl {
    pub fn as_str(self) -> &'static str {
        match self {
            CacheTtl::FiveMinutes => "5m",
            CacheTtl::OneHour => "1h",
        }
    }

    /// Choose the appropriate TTL bucket for a given number of seconds.
    pub fn from_secs(secs: u32) -> Self {
        if secs <= 300 {
            CacheTtl::FiveMinutes
        } else {
            CacheTtl::OneHour
        }
    }
}

// ── Usage field names ─────────────────────────────────────────────────────────

/// JSON field names in the `usage` response object.
pub mod usage_fields {
    pub const INPUT_TOKENS: &str = "input_tokens";
    pub const OUTPUT_TOKENS: &str = "output_tokens";
    pub const CACHE_CREATION_INPUT_TOKENS: &str = "cache_creation_input_tokens";
    pub const CACHE_READ_INPUT_TOKENS: &str = "cache_read_input_tokens";
}
