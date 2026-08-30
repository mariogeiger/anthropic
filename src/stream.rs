//! One streamed frame becomes one typed event.
//!
//! A streaming response is a sequence of Server-Sent Events, each carrying a
//! JSON object whose `type` field names it — the same name the `event:` line
//! carries. This module turns one such payload into a [`StreamEvent`];
//! [`crate::settle`] turns a sequence of them into a finished message.
//!
//! # The documented event flow
//!
//! `message_start` opens the message with empty content. Then, per content
//! block, a `content_block_start`, some `content_block_delta` events, and a
//! `content_block_stop`. Then one or more `message_delta` events carrying the
//! stop reason and cumulative usage, and finally a single `message_stop`. A
//! `ping` may appear anywhere, and an `error` may replace the rest.
//!
//! The blocks and deltas themselves live in [`crate::content`], which is where
//! reassembly is defined; this module is only the envelope.
//!
//! # Why an unknown event is not an error
//!
//! Anthropic's versioning policy states that new event types may be added and
//! that client code should handle unknown ones gracefully. A decoder that errors
//! on an event it has never seen is a decoder a routine server release breaks.
//! So the unrecognized case is a variant — [`StreamEvent::Unmodeled`] — and never
//! a [`FrameError`]. That variant also covers `ping`, which exists only to hold
//! the connection open. "Well-formed, nothing to do here" is one situation, so it
//! is one variant.
//!
//! # What *is* an error
//!
//! Bytes that are not JSON, a payload that is not an object, a missing `type`, a
//! field whose type contradicts the schema, and a `usage` object that will not
//! deserialize. Those are broken frames, not new ones. See [`FrameError`].

use serde_json::Value;

use crate::content::{BlockDelta, StreamedBlock};
use crate::frame::{FrameError, decode_usage, optional_string, require, require_str, require_u32};
use crate::usage::Usage;
use crate::values::{ErrorType, RefusalCategory, StopReason};

// ── Message-level payloads ───────────────────────────────────────────────────

/// The `message` object `message_start` carries.
///
/// Its content array is always empty — the blocks arrive as their own events — so
/// this is the message's identity and its opening usage, nothing more.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageStart {
    /// The message's identifier, worth logging for Anthropic support.
    pub id: String,
    /// The model that answered, which may be named more precisely than the one
    /// asked for: a gateway routing `aws/anthropic/bedrock-claude-opus-5` reports
    /// `claude-opus-5` here.
    pub model: String,
    /// The opening usage.
    ///
    /// Anthropic documents `message_start` as where the cache counts appear, and
    /// they are already final at this point: caching happens on input, before a
    /// single token is generated. So a caller learns whether its cache worked
    /// from the *first* frame, without waiting for the answer.
    pub usage: Usage,
}

/// Why the model refused, where it did.
///
/// A refusal is a *completed* message carrying [`StopReason::Refusal`], so this
/// sits beside the stop reason rather than being an error: the protocol worked and
/// the model declined. The category names the classifier that fired, not a
/// judgement about the caller — Anthropic documents that benign work in each area
/// can trigger it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefusalDetails {
    /// Which policy area fired. `None` where the API named one this crate does not
    /// know, which keeps a newly added category from costing the caller the frame.
    pub category: Option<RefusalCategory>,
    /// The `category` string exactly as sent, so an unknown one stays legible.
    pub raw_category: String,
    /// A human-readable explanation, where one was given. Anthropic documents this
    /// text as not guaranteed stable, so log it rather than matching on it.
    pub explanation: Option<String>,
}

impl RefusalDetails {
    /// One `stop_details` object, decoded, or `None` where it was absent, null, or
    /// of a kind other than `refusal`.
    ///
    /// Infallible in the same way [`StreamedError::decode`] is: a malformed
    /// `stop_details` beside a real stop reason must not cost the caller the
    /// message, so an unreadable one reads as absent.
    pub(crate) fn decode(details: Option<&Value>) -> Option<Self> {
        let details = details?.as_object()?;
        if details.get("type").and_then(Value::as_str) != Some("refusal") {
            return None;
        }
        let raw_category = details.get("category").and_then(Value::as_str).unwrap_or_default().to_owned();
        Some(RefusalDetails {
            category: RefusalCategory::from_str(&raw_category),
            raw_category,
            explanation: details.get("explanation").and_then(Value::as_str).map(str::to_owned),
        })
    }
}

/// The `message_delta` payload: how the message ended, and what it cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageDelta {
    /// Why generation stopped. `None` when the API named a reason this crate does
    /// not know, which keeps a newly added stop reason from failing a frame whose
    /// content is perfectly usable.
    pub stop_reason: Option<StopReason>,
    /// Which stop sequence matched, on [`StopReason::StopSequence`].
    pub stop_sequence: Option<String>,
    /// Why the model refused, on [`StopReason::Refusal`]. `None` otherwise.
    pub refusal: Option<RefusalDetails>,
    /// Cumulative usage, per Anthropic's documentation. Merged into what
    /// `message_start` reported — see [`Usage::merge_cumulative`].
    pub usage: Usage,
}

/// An error the API reported inside the stream.
///
/// `kind` is the parsed vocabulary and `raw_kind` the string it came from, so a
/// type Anthropic adds later stays legible instead of vanishing. Anthropic's own
/// example is a mid-stream `overloaded_error`, which in a non-streaming call
/// would have been an HTTP 529: the request was already accepted, so the failure
/// has to arrive in band.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamedError {
    /// The error type, where it is one this crate knows.
    pub kind: Option<ErrorType>,
    /// The `error.type` string exactly as sent.
    pub raw_kind: String,
    /// The human-readable message.
    pub message: String,
}

impl StreamedError {
    /// One `error` frame, decoded.
    ///
    /// Infallible: a frame whose `error` object is missing or malformed still
    /// ends the stream, because the server said generation failed and refusing to
    /// decode the announcement would turn a reported failure into a silent
    /// truncation.
    fn decode(frame: &Value) -> Self {
        let error = frame.get("error").filter(|error| error.is_object());
        let raw_kind = error.map(|error| optional_string(error, "type")).unwrap_or_default();
        StreamedError {
            kind: ErrorType::from_str(&raw_kind),
            raw_kind,
            message: error.map(|error| optional_string(error, "message")).unwrap_or_default(),
        }
    }
}

// ── Events ───────────────────────────────────────────────────────────────────

/// One decoded streaming event.
///
/// The six `message_*` and `content_block_*` variants are the documented event
/// flow; [`Self::Error`] is the in-band `error` event; [`Self::Unmodeled`] is
/// `ping` and whatever Anthropic adds next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// `message_start`: the message opens with empty content.
    MessageStart(MessageStart),
    /// `content_block_start`: a block begins at `index`.
    ///
    /// Announcement only. A tool call arrives here with empty input.
    ContentBlockStart {
        /// The block's position in the final `content` array.
        index: u32,
        /// The block as far as it exists.
        block: StreamedBlock,
    },
    /// `content_block_delta`: the block at `index` grows.
    ContentBlockDelta {
        /// Which block to append to.
        index: u32,
        /// What to append.
        delta: BlockDelta,
    },
    /// `content_block_stop`: the block at `index` is complete.
    ///
    /// For a tool call this is the point at which its accumulated input is whole
    /// JSON and [`crate::content::ToolInput::decode`] is meaningful.
    ContentBlockStop {
        /// Which block finished.
        index: u32,
    },
    /// `message_delta`: the stop reason and cumulative usage.
    MessageDelta(MessageDelta),
    /// `message_stop`: the stream is over. Terminal.
    ///
    /// Carries a [`Usage`], because gateways do: the NVIDIA inference endpoint
    /// repeats `input_tokens` and `output_tokens` here. Merging is a pointwise
    /// maximum, so this partial record cannot erase the fuller one that came
    /// before it.
    MessageStop {
        /// Whatever usage this frame repeated, all zero when it carried none.
        usage: Usage,
    },
    /// The `error` event, emitted when generation itself fails. Terminal.
    Error(StreamedError),
    /// A well-formed event this crate does not model, `ping` included.
    ///
    /// Never an error, by design: see the module documentation. Ignoring it is
    /// correct, and its `kind` is worth logging once.
    Unmodeled {
        /// The event's `type`.
        kind: String,
    },
}

impl StreamEvent {
    /// The event's wire `type`.
    ///
    /// The one accessor that works uniformly across variants, so logging and
    /// metrics need no match.
    pub fn kind(&self) -> &str {
        match self {
            StreamEvent::MessageStart(_) => "message_start",
            StreamEvent::ContentBlockStart { .. } => "content_block_start",
            StreamEvent::ContentBlockDelta { .. } => "content_block_delta",
            StreamEvent::ContentBlockStop { .. } => "content_block_stop",
            StreamEvent::MessageDelta(_) => "message_delta",
            StreamEvent::MessageStop { .. } => "message_stop",
            StreamEvent::Error(_) => "error",
            StreamEvent::Unmodeled { kind } => kind,
        }
    }

    /// Whether this event ends the stream.
    ///
    /// True for `message_stop` and `error`, and deliberately *false* for
    /// `message_delta`: that event carries the stop reason, but Anthropic
    /// documents one or more of them before the single `message_stop`, and a
    /// connection dropped after a `message_delta` is still a connection that
    /// dropped. Treating the stop reason as the end is the mistake
    /// [`crate::settle::Settling`] exists to make impossible.
    pub fn is_terminal(&self) -> bool {
        matches!(self, StreamEvent::MessageStop { .. } | StreamEvent::Error(_))
    }

    /// One `data:` payload, decoded.
    pub fn decode(payload: &str) -> Result<Self, FrameError> {
        let value: Value = serde_json::from_str(payload).map_err(FrameError::NotJson)?;
        Self::from_json(&value)
    }

    /// The same, for a caller who already holds the parsed frame.
    pub fn from_json(frame: &Value) -> Result<Self, FrameError> {
        if !frame.is_object() {
            return Err(FrameError::NotAnObject);
        }
        Ok(match require_str(frame, "type")? {
            "message_start" => StreamEvent::MessageStart(decode_message_start(require(frame, "message")?)?),
            "content_block_start" => StreamEvent::ContentBlockStart {
                index: require_u32(frame, "index")?,
                block: StreamedBlock::decode(require(frame, "content_block")?)?,
            },
            "content_block_delta" => StreamEvent::ContentBlockDelta {
                index: require_u32(frame, "index")?,
                delta: BlockDelta::decode(require(frame, "delta")?)?,
            },
            "content_block_stop" => StreamEvent::ContentBlockStop { index: require_u32(frame, "index")? },
            "message_delta" => {
                let delta = frame.get("delta");
                let field = |name: &str| delta.and_then(|delta| delta.get(name)).and_then(Value::as_str);
                StreamEvent::MessageDelta(MessageDelta {
                    stop_reason: field("stop_reason").and_then(StopReason::from_str),
                    stop_sequence: field("stop_sequence").map(str::to_owned),
                    refusal: RefusalDetails::decode(delta.and_then(|delta| delta.get("stop_details"))),
                    usage: decode_usage(frame)?,
                })
            }
            "message_stop" => StreamEvent::MessageStop { usage: decode_usage(frame)? },
            "error" => StreamEvent::Error(StreamedError::decode(frame)),
            other => StreamEvent::Unmodeled { kind: other.to_owned() },
        })
    }
}

fn decode_message_start(message: &Value) -> Result<MessageStart, FrameError> {
    if !message.is_object() {
        return Err(FrameError::WrongType { field: "message", expected: "an object" });
    }
    Ok(MessageStart {
        id: require_str(message, "id")?.to_owned(),
        model: optional_string(message, "model"),
        usage: decode_usage(message)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::ToolInput;
    use serde_json::json;

    /// The `message_start` frame captured from the live gateway, field for field,
    /// including the cache write it reported.
    #[test]
    fn a_captured_message_start_decodes() {
        let event = StreamEvent::decode(
            r#"{"type": "message_start", "message": {"model": "claude-opus-5",
                "id": "msg_bdrk_762xvdbzixaxf3ot2soefimzo5seq5vb7327nuan2pjk62be6rna", "type": "message",
                "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null,
                "stop_details": null, "usage": {"input_tokens": 36, "cache_creation_input_tokens": 1043,
                "cache_read_input_tokens": 0, "cache_creation": {"ephemeral_5m_input_tokens": 1043,
                "ephemeral_1h_input_tokens": 0}, "output_tokens": 1, "service_tier": "standard"}}}"#,
        )
        .unwrap();
        let StreamEvent::MessageStart(start) = &event else { panic!("expected message_start") };
        assert_eq!(start.id, "msg_bdrk_762xvdbzixaxf3ot2soefimzo5seq5vb7327nuan2pjk62be6rna");
        assert_eq!(start.model, "claude-opus-5", "the gateway names the resolved model");
        assert_eq!(start.usage.cache_creation_input_tokens, 1_043);
        assert_eq!(start.usage.cache_creation.ephemeral_5m_input_tokens, 1_043);
        assert_eq!(event.kind(), "message_start");
        assert!(!event.is_terminal());
    }

    /// A `message_start` from the same endpoint on a cache *hit*, which is the
    /// number a caller is actually trying to prove.
    #[test]
    fn a_captured_cache_hit_is_visible_from_the_first_frame() {
        let event = StreamEvent::decode(
            r#"{"type": "message_start", "message": {"model": "claude-opus-5", "id": "msg_bdrk_x", "type": "message",
                "role": "assistant", "content": [], "usage": {"input_tokens": 36,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1043,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                "output_tokens": 2, "service_tier": "standard"}}}"#,
        )
        .unwrap();
        let StreamEvent::MessageStart(start) = event else { panic!("expected message_start") };
        assert_eq!(start.usage.cache_read_input_tokens, 1_043);
        assert_eq!(start.usage.total_input_tokens(), 1_079);
    }

    /// A captured text block: opens empty, grows by `text_delta`, then stops.
    #[test]
    fn the_captured_text_block_lifecycle_decodes() {
        let start = StreamEvent::decode(
            r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}"#,
        )
        .unwrap();
        assert_eq!(
            start,
            StreamEvent::ContentBlockStart { index: 0, block: StreamedBlock::Text { text: String::new() } }
        );

        let delta = StreamEvent::decode(
            r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}"#,
        )
        .unwrap();
        assert_eq!(
            delta,
            StreamEvent::ContentBlockDelta { index: 0, delta: BlockDelta::Text { delta: "Hello".to_owned() } }
        );

        let stop = StreamEvent::decode(r#"{"type": "content_block_stop", "index": 0}"#).unwrap();
        assert_eq!(stop, StreamEvent::ContentBlockStop { index: 0 });
        assert!(!stop.is_terminal(), "a block ending is not the stream ending");
    }

    /// A captured live `tool_use` announcement and one of its input fragments.
    #[test]
    fn a_captured_tool_use_block_and_its_fragment_decode() {
        let start = StreamEvent::decode(
            r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use",
                "id": "toolu_bdrk_01SfSun6GdgxKVDToTxtf7Ee", "name": "get_weather", "input": {}}}"#,
        )
        .unwrap();
        let StreamEvent::ContentBlockStart { index: 0, block: StreamedBlock::ToolUse { id, input, .. } } = start else {
            panic!("expected an announced tool call")
        };
        assert_eq!(id, "toolu_bdrk_01SfSun6GdgxKVDToTxtf7Ee");
        assert_eq!(input, ToolInput::default());

        let fragment = StreamEvent::decode(
            r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta",
                "partial_json": "{\"locatio"}}"#,
        )
        .unwrap();
        assert_eq!(
            fragment,
            StreamEvent::ContentBlockDelta {
                index: 0,
                delta: BlockDelta::InputJson { partial_json: r#"{"locatio"#.to_owned() }
            }
        );
    }

    /// A captured thinking block start and its signature delta.
    #[test]
    fn a_captured_thinking_block_and_signature_decode() {
        let start = StreamEvent::decode(
            r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "thinking",
                "thinking": "", "signature": ""}}"#,
        )
        .unwrap();
        assert_eq!(
            start,
            StreamEvent::ContentBlockStart {
                index: 0,
                block: StreamedBlock::Thinking { thinking: String::new(), signature: String::new() }
            }
        );

        let signature = StreamEvent::decode(
            r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta",
                "signature": "CAISyUkKcAgREAEYAipAqjMsxB4VuvEH9lSBF0/z"}}"#,
        )
        .unwrap();
        assert_eq!(
            signature,
            StreamEvent::ContentBlockDelta {
                index: 0,
                delta: BlockDelta::Signature { delta: "CAISyUkKcAgREAEYAipAqjMsxB4VuvEH9lSBF0/z".to_owned() }
            }
        );
    }

    /// The captured `message_delta`: stop reason plus cumulative usage, and
    /// pointedly not the end of the stream.
    #[test]
    fn a_captured_message_delta_decodes_and_is_not_terminal() {
        let event = StreamEvent::decode(
            r#"{"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null,
                "stop_details": null}, "usage": {"input_tokens": 505, "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0, "output_tokens": 34,
                "output_tokens_details": {"thinking_tokens": 0},
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}}}"#,
        )
        .unwrap();
        let StreamEvent::MessageDelta(delta) = &event else { panic!("expected message_delta") };
        assert_eq!(delta.stop_reason, Some(StopReason::ToolUse));
        assert_eq!(delta.stop_sequence, None);
        assert_eq!(delta.usage.output_tokens, 34);
        assert!(!event.is_terminal(), "the stop reason is not the end of the stream");
    }

    /// The captured `message_stop`, which on this gateway carries a usage object
    /// the published documentation does not show.
    #[test]
    fn a_captured_message_stop_is_terminal_and_may_carry_usage() {
        let event =
            StreamEvent::decode(r#"{"type": "message_stop", "usage": {"input_tokens": 505, "output_tokens": 34}}"#)
                .unwrap();
        assert_eq!(
            event,
            StreamEvent::MessageStop { usage: Usage { input_tokens: 505, output_tokens: 34, ..Usage::default() } }
        );
        assert!(event.is_terminal());

        let documented = StreamEvent::decode(r#"{"type": "message_stop"}"#).unwrap();
        assert_eq!(documented, StreamEvent::MessageStop { usage: Usage::default() });
        assert!(documented.is_terminal());
    }

    /// An unrecognized stop reason reads as absent: the content is still fine, so
    /// failing the frame would throw away a usable answer.
    #[test]
    fn an_unrecognized_stop_reason_reads_as_absent() {
        let event =
            StreamEvent::from_json(&json!({"type": "message_delta", "delta": {"stop_reason": "some_new_reason"}}))
                .unwrap();
        let StreamEvent::MessageDelta(delta) = event else { panic!("expected message_delta") };
        assert_eq!(delta.stop_reason, None);
        assert_eq!(delta.usage, Usage::default(), "a message_delta need not carry usage");
    }

    #[test]
    fn a_stop_sequence_names_the_sequence_that_matched() {
        let event = StreamEvent::from_json(&json!({
            "type": "message_delta", "delta": {"stop_reason": "stop_sequence", "stop_sequence": "END"}
        }))
        .unwrap();
        let StreamEvent::MessageDelta(delta) = event else { panic!("expected message_delta") };
        assert_eq!(delta.stop_reason, Some(StopReason::StopSequence));
        assert_eq!(delta.stop_sequence.as_deref(), Some("END"));
    }

    /// A refusal is a completed message, so its reason arrives beside the stop
    /// reason rather than as an error.
    #[test]
    fn a_refusal_carries_its_category_beside_the_stop_reason() {
        let event = StreamEvent::from_json(&json!({
            "type": "message_delta",
            "delta": {"stop_reason": "refusal", "stop_sequence": null,
                      "stop_details": {"type": "refusal", "category": "cyber",
                                       "explanation": "This could enable exploit development."}}
        }))
        .unwrap();
        let StreamEvent::MessageDelta(delta) = event else { panic!("expected message_delta") };
        assert_eq!(delta.stop_reason, Some(StopReason::Refusal));
        let refusal = delta.refusal.unwrap();
        assert_eq!(refusal.category, Some(RefusalCategory::Cyber));
        assert_eq!(refusal.raw_category, "cyber");
        assert_eq!(refusal.explanation.as_deref(), Some("This could enable exploit development."));
    }

    /// A captured live frame's `stop_details` is `null` on an ordinary stop, and an
    /// unreadable or unknown one must not cost the caller the frame.
    #[test]
    fn stop_details_that_name_no_refusal_read_as_absent() {
        for details in [json!(null), json!("nonsense"), json!({}), json!({"type": "something_else"})] {
            let event = StreamEvent::from_json(&json!({
                "type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_details": details}
            }))
            .unwrap();
            let StreamEvent::MessageDelta(delta) = event else { panic!("expected message_delta") };
            assert_eq!(delta.refusal, None, "{details}");
        }

        // A category this crate does not know keeps its raw string.
        let event = StreamEvent::from_json(&json!({
            "type": "message_delta",
            "delta": {"stop_reason": "refusal", "stop_details": {"type": "refusal", "category": "some_new_area"}}
        }))
        .unwrap();
        let StreamEvent::MessageDelta(delta) = event else { panic!("expected message_delta") };
        let refusal = delta.refusal.unwrap();
        assert_eq!(refusal.category, None);
        assert_eq!(refusal.raw_category, "some_new_area");
        assert_eq!(refusal.explanation, None);
    }

    /// Anthropic's documented mid-stream error, parsed into the crate's error
    /// vocabulary while keeping the raw string.
    #[test]
    fn the_documented_error_event_decodes() {
        let event =
            StreamEvent::decode(r#"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#)
                .unwrap();
        assert_eq!(
            event,
            StreamEvent::Error(StreamedError {
                kind: Some(ErrorType::Overloaded),
                raw_kind: "overloaded_error".to_owned(),
                message: "Overloaded".to_owned(),
            })
        );
        assert!(event.is_terminal());
        assert_eq!(event.kind(), "error");
    }

    /// An error type outside the documented list keeps its raw string, so a newly
    /// added one stays legible instead of failing the frame. This shape was
    /// returned by the live gateway.
    #[test]
    fn an_unknown_error_type_keeps_its_raw_string() {
        let event = StreamEvent::from_json(&json!({
            "type": "error",
            "error": {"type": "key_model_access_denied", "message": "key not allowed to access model"}
        }))
        .unwrap();
        let StreamEvent::Error(error) = event else { panic!("expected an error") };
        assert_eq!(error.kind, None);
        assert_eq!(error.raw_kind, "key_model_access_denied");
        assert_eq!(error.message, "key not allowed to access model");
    }

    /// An `error` event with no error object still ends the stream: the server
    /// said generation failed, and dropping that would look like a truncation.
    #[test]
    fn an_error_event_without_an_error_object_still_ends_the_stream() {
        let event = StreamEvent::from_json(&json!({"type": "error"})).unwrap();
        let StreamEvent::Error(error) = &event else { panic!("expected an error") };
        assert_eq!(error.raw_kind, "");
        assert_eq!(error.message, "");
        assert!(event.is_terminal());
    }

    /// `ping` exists only to hold the connection open, so it decodes to the same
    /// ignorable variant as a genuinely unknown event.
    #[test]
    fn ping_and_unknown_events_are_both_ignorable() {
        for kind in ["ping", "message_kaleidoscope", "content_block_telepathy"] {
            let event = StreamEvent::from_json(&json!({"type": kind})).unwrap();
            assert_eq!(event, StreamEvent::Unmodeled { kind: kind.to_owned() });
            assert_eq!(event.kind(), kind);
            assert!(!event.is_terminal());
        }
    }

    #[test]
    fn a_broken_frame_is_an_error() {
        assert!(matches!(StreamEvent::decode("not json at all"), Err(FrameError::NotJson(_))));
        assert!(matches!(StreamEvent::decode("[1,2]"), Err(FrameError::NotAnObject)));
        assert!(matches!(StreamEvent::decode("{}"), Err(FrameError::MissingField { field: "type" })));
        assert!(matches!(
            StreamEvent::from_json(&json!({"type": "content_block_delta", "index": 0})),
            Err(FrameError::MissingField { field: "delta" })
        ));
        assert!(matches!(
            StreamEvent::from_json(&json!({"type": "content_block_stop", "index": -1})),
            Err(FrameError::WrongType { field: "index", .. })
        ));
        assert!(matches!(
            StreamEvent::from_json(&json!({"type": "message_start"})),
            Err(FrameError::MissingField { field: "message" })
        ));
        assert!(matches!(
            StreamEvent::from_json(&json!({"type": "message_start", "message": {}})),
            Err(FrameError::MissingField { field: "id" })
        ));
        assert!(matches!(
            StreamEvent::from_json(&json!({"type": "message_start", "message": "oops"})),
            Err(FrameError::WrongType { field: "message", .. })
        ));
    }

    /// A `usage` object that will not deserialize fails the frame, because
    /// reporting no cost would hide the numbers this crate exists to report.
    #[test]
    fn an_unusable_usage_object_fails_the_frame() {
        let broken = StreamEvent::from_json(&json!({
            "type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": "not a number"}
        }));
        assert!(matches!(broken, Err(FrameError::UndecodableUsage(_))));

        let null = StreamEvent::from_json(&json!({"type": "message_stop", "usage": null})).unwrap();
        assert_eq!(null, StreamEvent::MessageStop { usage: Usage::default() }, "null usage is no usage");
    }
}
