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

// `thinking.type`. `Enabled` is the legacy fixed-budget form: emitted for Haiku 4.5,
// removed on Opus 4.7+ / Sonnet 5 (deprecated on Sonnet 4.6). `Adaptive` is emitted for
// Fable 5, Opus 4.8, and Sonnet 5. `Disabled` is the explicit thinking-off form emitted
// for Sonnet 5, where an omitted `thinking` field would instead leave adaptive on.
api_enum! { ThinkingType { Enabled => "enabled", Adaptive => "adaptive", Disabled => "disabled" } }

// Opus 4.7+ `thinking.display`. Default `Omitted` = thinking streams but text is empty.
api_enum! { ThinkingDisplay { Summarized => "summarized", Omitted => "omitted" } }

api_enum! { roundtrip StopReason {
    EndTurn => "end_turn",
    MaxTokens => "max_tokens",
    StopSequence => "stop_sequence",
    ToolUse => "tool_use",
    PauseTurn => "pause_turn",
    Refusal => "refusal",
    ModelContextWindowExceeded => "model_context_window_exceeded",
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `from_str` is the documented inverse of `as_str` (§6): every wire string
    /// must round-trip, and unknown strings must return `None`.
    #[test]
    fn stop_reason_roundtrips() {
        for r in [
            StopReason::EndTurn,
            StopReason::MaxTokens,
            StopReason::StopSequence,
            StopReason::ToolUse,
            StopReason::PauseTurn,
            StopReason::Refusal,
            StopReason::ModelContextWindowExceeded,
        ] {
            assert_eq!(StopReason::from_str(r.as_str()), Some(r));
        }
        // The variant added for Sonnet 4.5+ / Opus 4.5+ / Haiku 4.5 — the tiers
        // this crate models all emit it by default.
        assert_eq!(StopReason::from_str("model_context_window_exceeded"), Some(StopReason::ModelContextWindowExceeded));
        assert_eq!(StopReason::from_str("not_a_stop_reason"), None);
    }

    #[test]
    fn error_type_roundtrips() {
        for e in [
            ErrorType::InvalidRequest,
            ErrorType::Authentication,
            ErrorType::Billing,
            ErrorType::Permission,
            ErrorType::NotFound,
            ErrorType::RequestTooLarge,
            ErrorType::RateLimit,
            ErrorType::Api,
            ErrorType::Timeout,
            ErrorType::Overloaded,
        ] {
            assert_eq!(ErrorType::from_str(e.as_str()), Some(e));
        }
        assert_eq!(ErrorType::from_str("nonsense"), None);
    }

    #[test]
    fn error_type_from_status() {
        assert_eq!(ErrorType::from_status(400), Some(ErrorType::InvalidRequest));
        assert_eq!(ErrorType::from_status(402), Some(ErrorType::Billing));
        assert_eq!(ErrorType::from_status(403), Some(ErrorType::Permission));
        assert_eq!(ErrorType::from_status(504), Some(ErrorType::Timeout));
        assert_eq!(ErrorType::from_status(529), Some(ErrorType::Overloaded));
        assert_eq!(ErrorType::from_status(418), None);
    }
}
