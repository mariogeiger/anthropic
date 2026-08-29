//! Enums mirroring API JSON values. `as_str()` is outbound; `from_str()` / the
//! HTTP-status table are pure `match`-on-primitive lookup tables — wire
//! vocabulary under §6, not a response parser.

/// Declares an enum that mirrors a closed set of API JSON string values.
///
/// The generated `as_str` is the outbound direction: one arm per variant, no
/// allocation, no formatting. The optional `roundtrip` prefix adds `from_str`,
/// the documented inverse — a pure `match` on `&str` and therefore wire
/// vocabulary rather than a response parser (§6).
///
/// Every variant takes its own doc comment, because `#![deny(missing_docs)]`
/// applies inside a macro expansion exactly as it does outside one.
macro_rules! api_enum {
    (@base $(#[$enum_doc:meta])* $name:ident { $($(#[$doc:meta])* $variant:ident => $s:literal),* $(,)? }) => {
        $(#[$enum_doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name { $($(#[$doc])* $variant),* }
        impl $name {
            /// The string this value takes on the wire.
            pub fn as_str(self) -> &'static str {
                match self { $($name::$variant => $s),* }
            }
        }
    };
    (roundtrip $(#[$enum_doc:meta])* $name:ident { $($(#[$doc:meta])* $variant:ident => $s:literal),* $(,)? }) => {
        api_enum! { @base $(#[$enum_doc])* $name { $($(#[$doc])* $variant => $s),* } }
        impl $name {
            /// The value a wire string names, or `None` for one this crate does
            /// not know. The documented inverse of [`Self::as_str`].
            #[allow(clippy::should_implement_trait)]
            pub fn from_str(s: &str) -> Option<Self> {
                match s {
                    $($s => Some($name::$variant),)*
                    _ => None,
                }
            }
        }
    };
    ($(#[$enum_doc:meta])* $name:ident { $($(#[$doc:meta])* $variant:ident => $s:literal),* $(,)? }) => {
        api_enum! { @base $(#[$enum_doc])* $name { $($(#[$doc])* $variant => $s),* } }
    };
}

pub(crate) use api_enum;

api_enum! {
    /// The `media_type` of a base64 image source. The four formats the API accepts.
    ImageMediaType {
        /// JPEG.
        Jpeg => "image/jpeg",
        /// PNG.
        Png => "image/png",
        /// GIF.
        Gif => "image/gif",
        /// WebP.
        Webp => "image/webp",
    }
}

api_enum! {
    /// The `thinking.type` of a request. Which forms a model accepts is
    /// model-specific, which is why the per-model types in [`crate::request`]
    /// choose between them rather than exposing this one.
    ThinkingType {
        /// The legacy fixed-budget form, `{type: "enabled", budget_tokens: N}`.
        /// Emitted for Haiku 4.5; removed on Opus 4.7+ and Sonnet 5, deprecated
        /// on Sonnet 4.6.
        Enabled => "enabled",
        /// The adaptive form, where the model chooses how much to think.
        /// Emitted for Fable 5, Opus 5, Opus 4.8, and Sonnet 5.
        Adaptive => "adaptive",
        /// The explicit off form. Needed where an omitted `thinking` field would
        /// instead leave adaptive thinking on, as on Opus 5 and Sonnet 5.
        Disabled => "disabled",
    }
}

api_enum! {
    /// The `thinking.display` of a request: whether reasoning text is sent back.
    /// Opus 4.7 and later.
    ThinkingDisplay {
        /// A condensed summary of the reasoning arrives as `thinking_delta`
        /// events.
        Summarized => "summarized",
        /// The default. Thinking blocks still stream — opening, taking one
        /// `signature_delta`, and closing — but carry no reasoning text, so
        /// [`crate::content::StreamedBlock::Thinking`] arrives with an empty
        /// `thinking` field and a real signature.
        Omitted => "omitted",
    }
}

api_enum! {
    roundtrip
    /// Why generation stopped.
    ///
    /// Every variant means the protocol completed, which is why
    /// [`crate::settle::Outcome::Stopped`] carries all of them: a refusal and a
    /// truncated answer are both messages the server finished sending. What the
    /// caller does next is what differs.
    StopReason {
        /// The model finished its turn.
        EndTurn => "end_turn",
        /// The `max_tokens` budget ran out. The answer is cut short but usable.
        MaxTokens => "max_tokens",
        /// One of the request's `stop_sequences` appeared. Which one is reported
        /// alongside.
        StopSequence => "stop_sequence",
        /// The model wants a tool run. The turn continues once the results are
        /// sent back.
        ToolUse => "tool_use",
        /// A long-running turn was paused and can be continued by sending the
        /// message back unchanged.
        PauseTurn => "pause_turn",
        /// The model declined to answer.
        Refusal => "refusal",
        /// The conversation outgrew the model's context window.
        ModelContextWindowExceeded => "model_context_window_exceeded",
    }
}

api_enum! {
    roundtrip
    /// The `error.type` of an error body or a streamed `error` event.
    ///
    /// A closed set that a gateway may exceed, so both
    /// [`crate::response::ApiError`] and [`crate::stream::StreamedError`] keep the
    /// raw string beside the parsed value rather than losing an unknown type.
    ErrorType {
        /// The request was malformed. HTTP 400.
        InvalidRequest => "invalid_request_error",
        /// The credential was missing or wrong. HTTP 401.
        Authentication => "authentication_error",
        /// The account cannot be billed. HTTP 402.
        Billing => "billing_error",
        /// The credential may not do this. HTTP 403.
        Permission => "permission_error",
        /// No such resource. HTTP 404.
        NotFound => "not_found_error",
        /// The request body exceeded the size limit. HTTP 413.
        RequestTooLarge => "request_too_large",
        /// A rate limit was hit. HTTP 429.
        RateLimit => "rate_limit_error",
        /// An unexpected server-side failure. HTTP 500.
        Api => "api_error",
        /// The request timed out server-side. HTTP 504.
        Timeout => "timeout_error",
        /// Capacity is exhausted. HTTP 529, or a mid-stream `error` event where
        /// the request had already been accepted.
        Overloaded => "overloaded_error",
    }
}

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

api_enum! {
    /// The `cache_control.type` of a breakpoint.
    CacheControlType {
        /// The only type the API currently supports.
        Ephemeral => "ephemeral",
    }
}

api_enum! {
    /// How long a cache entry lives.
    ///
    /// Both refresh for free on a hit; they differ in write price and in the gap
    /// they survive. Mixing them in one request is allowed, but every 1-hour
    /// breakpoint must precede every 5-minute one — an ordering
    /// [`crate::context::Context`] validates before it commits a change.
    CacheTtl {
        /// Five minutes, at 1.25× the base input price to write. The default.
        FiveMinutes => "5m",
        /// One hour, at 2× the base input price to write.
        OneHour => "1h",
    }
}

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
