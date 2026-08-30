//! Enums mirroring API JSON values. `as_str()` is outbound; `from_str()` / the
//! HTTP-status table are pure `match`-on-primitive lookup tables — wire
//! vocabulary, not a response parser.
//!
//! Every one of these enums serializes to its own wire string, so a wire struct
//! anywhere in the crate holds the enum rather than the string: a `&'static str`
//! field accepts any string, an enum field only a value the API accepts.

/// Declares an enum that mirrors a closed set of API JSON string values.
///
/// The generated `as_str` is the outbound direction: one arm per variant, no
/// allocation, no formatting. The optional `roundtrip` prefix adds `from_str`,
/// the documented inverse — a pure `match` on `&str` and therefore wire
/// vocabulary rather than a response parser.
///
/// Every one of these enums *is* a wire string, so every one serializes as that
/// string, through the same `as_str` the outbound direction already uses. That is
/// what lets a wire struct hold the enum itself instead of a `&'static str`
/// obtained from it: a `&'static str` field is writable with any string at all,
/// while a field of a closed enum has no invalid value to write.
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
        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(self.as_str())
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
    roundtrip
    /// The `role` of one `messages[]` entry.
    ///
    /// Exactly the two roles the Messages API accepts on a message a caller
    /// sends, so [`crate::context::Message::role`] can be a public field without
    /// being a hole: a closed enum has no invalid value to write, which is what
    /// makes "an unknown role cannot reach the wire" a fact about the type rather
    /// than a promise in a doc comment.
    ///
    /// All three roles a `messages[]` entry may carry, `system` included: a
    /// mid-conversation instruction is a message with that role. See
    /// [`crate::system`] for why it exists and
    /// [`crate::context::Context::push_system`] for the placement rules.
    ///
    /// The role is *derived* from a [`crate::context::Message`] rather than
    /// assignable on one, because the three roles do not admit the same content —
    /// a system message takes text and tool changes and nothing else. So
    /// [`crate::context::Message`] is an enum whose variant carries the content its
    /// role admits, and this enum is what that variant reports.
    Role {
        /// `user`: the caller's turn. Tool results go here too, which is where the
        /// API expects them.
        User => "user",
        /// `assistant`: the model's own turn, replayed back into the conversation.
        Assistant => "assistant",
        /// `system`: an instruction added partway through, after the cached prefix.
        System => "system",
    }
}

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
    /// in scope as wire vocabulary; not a response parser.
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
    /// The `type` of a content block that carries text. One variant, because
    /// `text` is the only block type a system prompt may hold — the API accepts
    /// text there and nothing else.
    TextBlockType {
        /// The only type a system prompt block takes.
        Text => "text",
    }
}

api_enum! {
    /// The `type` of a block inside a mid-conversation system message.
    ///
    /// Exactly the three the API accepts there, which is a *different* set from
    /// the top-level system prompt's single `text` — hence a second enum rather
    /// than a widened [`TextBlockType`]. See [`crate::system::SystemBlock`].
    SystemBlockType {
        /// An instruction.
        Text => "text",
        /// A tool offered from this point on.
        ToolAddition => "tool_addition",
        /// A tool withdrawn from this point on.
        ToolRemoval => "tool_removal",
    }
}

api_enum! {
    /// How a [`crate::system::ToolChangeBlock`] names the tool it changes.
    ///
    /// Three ways, because a tool's identity depends on where it came from; see
    /// [`crate::system::ToolReference`].
    ToolReferenceType {
        /// A tool the caller declared directly in `tools`.
        Tool => "tool_reference",
        /// One tool of an MCP server's toolset.
        McpTool => "mcp_tool_reference",
        /// Every tool of an MCP server's toolset.
        McpToolset => "mcp_toolset_reference",
    }
}

api_enum! {
    /// Which service tier a request may be served from.
    ///
    /// A capacity choice, not a content one: the model sees the same prompt
    /// either way, and `usage.service_tier` reports which tier actually served
    /// it — which need not be the one asked for.
    ServiceTier {
        /// Priority capacity where the account has it, standard otherwise. The
        /// documented default.
        Auto => "auto",
        /// Standard capacity only, never priority.
        StandardOnly => "standard_only",
    }
}

api_enum! {
    roundtrip
    /// Which tier actually served a request, as `usage.service_tier` reports it.
    ///
    /// A different set from [`ServiceTier`], which is what a caller may *ask*
    /// for: `batch` is never requested on this endpoint but is reported by one,
    /// and `auto` is a request for priority-or-standard rather than a tier a
    /// response can have been served from. Two vocabularies, so two enums.
    ServedTier {
        /// Standard capacity.
        Standard => "standard",
        /// Priority capacity.
        Priority => "priority",
        /// The Message Batches API.
        Batch => "batch",
    }
}

api_enum! {
    /// What the server does with an image larger than the model accepts.
    ImageOversize {
        /// Scale it down to fit, changing the dimensions the model observes
        /// without saying so. The documented default.
        Downsize => "downsize",
        /// Refuse the request with a 400 naming the image's dimensions and the
        /// largest that would fit, so nothing is silently rescaled.
        Error => "error",
    }
}

api_enum! {
    /// The `type` of an [`crate::request::OutputFormat`].
    OutputFormatType {
        /// The only form the API currently supports.
        JsonSchema => "json_schema",
    }
}

api_enum! {
    roundtrip
    /// Why the model refused, as `stop_details.category` names it.
    ///
    /// A refusal is a completed message with [`StopReason::Refusal`]; this is the
    /// policy area that triggered it. Every category can be triggered by benign
    /// work, so it identifies the classifier rather than accusing the caller.
    RefusalCategory {
        /// Could enable cyber harm. Benign security work can trigger it.
        Cyber => "cyber",
        /// Could enable biological harm. Benign life-sciences work can trigger it.
        Bio => "bio",
        /// Could assist development of competing models, which Anthropic's
        /// commercial terms restrict. Benign machine-learning work can trigger it.
        FrontierLlm => "frontier_llm",
        /// Asked the model to reproduce its internal reasoning as answer text.
        /// Adaptive thinking is the supported way to get reasoning.
        ReasoningExtraction => "reasoning_extraction",
        /// Some other area judged harmful.
        GeneralHarms => "general_harms",
    }
}

api_enum! {
    /// The `media_type` of a base64 document source. One variant: a base64
    /// document is a PDF.
    DocumentMediaType {
        /// PDF.
        Pdf => "application/pdf",
    }
}

api_enum! {
    /// The `media_type` of a plain-text document source.
    PlainTextMediaType {
        /// Plain text.
        Text => "text/plain",
    }
}

api_enum! {
    roundtrip
    /// The `type` of a [`crate::document::Citation`], which is to say how
    /// precisely it points at what it cites.
    CitationType {
        /// A character range of a plain-text document.
        CharLocation => "char_location",
        /// A page range of a PDF.
        PageLocation => "page_location",
        /// A block range of a structured document.
        ContentBlockLocation => "content_block_location",
        /// A block range of a search result.
        SearchResultLocation => "search_result_location",
        /// A result of the API's own web search.
        WebSearchResultLocation => "web_search_result_location",
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

    /// `from_str` is the documented inverse of `as_str`: every wire string
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
    fn role_roundtrips_and_rejects_everything_else() {
        for r in [Role::User, Role::Assistant, Role::System] {
            assert_eq!(Role::from_str(r.as_str()), Some(r));
        }
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
        assert_eq!(Role::System.as_str(), "system");
        assert_eq!(Role::from_str("wizard"), None);
        // Serializing is the same string `as_str` gives.
        assert_eq!(serde_json::to_value(Role::Assistant).unwrap(), "assistant");
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
