//! A non-streamed response body, decoded.
//!
//! The same message a stream settles into, delivered in one piece. So this shares
//! the vocabulary of [`crate::content`] and [`crate::values`] rather than growing
//! a parallel one, and [`Response::text`], [`Response::thinking`] and
//! [`Response::tool_calls`] read exactly like their [`crate::settle::Settled`]
//! counterparts. A caller can switch a request between streaming and not without
//! rewriting the code that reads the answer.
//!
//! # What is not decoded
//!
//! Fields that echo the request — `role`, which is always `assistant`, and the
//! `type` discriminator — are dropped. Reading back a constant tells the caller
//! nothing it did not already know, and a field this crate ignores today cannot
//! break tomorrow.

use serde_json::Value;

use crate::content::StreamedBlock;
use crate::frame::{FrameError, decode_usage, require, require_str};
use crate::settle::ToolCall;
use crate::stream::RefusalDetails;
use crate::usage::Usage;
use crate::values::StopReason;

/// One `POST /v1/messages` response body.
///
/// Every field is final: unlike a stream, there is no state in which this is
/// half-built, which is why it needs no counterpart to
/// [`crate::settle::Settling`]. Getting one *is* the proof the call completed —
/// an HTTP error carries an error body instead, which is [`ApiError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    /// The message's identifier, worth logging for Anthropic support.
    pub id: String,
    /// The model that answered, as the API named it.
    pub model: String,
    /// The content blocks, in order.
    ///
    /// May be empty: a `max_tokens: 0` pre-warm request returns no content at
    /// all, with `stop_reason: "max_tokens"` and a fully populated usage block.
    pub blocks: Vec<StreamedBlock>,
    /// Why generation stopped. `None` when the API named a reason this crate does
    /// not know, so a newly added reason cannot cost the caller the answer.
    pub stop_reason: Option<StopReason>,
    /// Which stop sequence matched, on [`StopReason::StopSequence`].
    pub stop_sequence: Option<String>,
    /// Why the model refused, on [`StopReason::Refusal`]. `None` otherwise.
    pub refusal: Option<RefusalDetails>,
    /// What the message cost, cache counts included.
    pub usage: Usage,
}

impl Response {
    /// One response body, decoded.
    pub fn decode(body: &str) -> Result<Self, FrameError> {
        let value: Value = serde_json::from_str(body).map_err(FrameError::NotJson)?;
        Self::from_json(&value)
    }

    /// The same, for a caller who already holds the parsed body.
    pub fn from_json(body: &Value) -> Result<Self, FrameError> {
        if !body.is_object() {
            return Err(FrameError::NotAnObject);
        }
        let blocks = match body.get("content") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(blocks)) => blocks.iter().map(StreamedBlock::decode).collect::<Result<_, _>>()?,
            Some(_) => return Err(FrameError::WrongType { field: "content", expected: "an array" }),
        };
        Ok(Response {
            id: require_str(body, "id")?.to_owned(),
            model: require_str(body, "model")?.to_owned(),
            blocks,
            stop_reason: body.get("stop_reason").and_then(Value::as_str).and_then(StopReason::from_str),
            stop_sequence: body.get("stop_sequence").and_then(Value::as_str).map(str::to_owned),
            refusal: RefusalDetails::decode(body.get("stop_details")),
            usage: decode_usage(body)?,
        })
    }

    /// The whole answer text: every text block, in order, concatenated.
    ///
    /// Thinking is excluded — see [`Self::thinking`].
    pub fn text(&self) -> String {
        joined(self.blocks.iter().filter_map(StreamedBlock::text))
    }

    /// The reasoning text, in order. Empty under `display: "omitted"`.
    pub fn thinking(&self) -> String {
        joined(self.blocks.iter().filter_map(|block| match block {
            StreamedBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
            _ => None,
        }))
    }

    /// The tool calls the model made, in block order.
    ///
    /// Inputs arrive whole here rather than as fragments, but they are still
    /// [`crate::content::ToolInput`]: the bytes are what gets replayed on the next turn, and
    /// re-serializing a parsed value can reorder keys and cost the prompt cache.
    pub fn tool_calls(&self) -> impl Iterator<Item = ToolCall<'_>> {
        self.blocks.iter().filter_map(|block| match block {
            StreamedBlock::ToolUse { id, name, input } => Some(ToolCall { id, name, input }),
            _ => None,
        })
    }
}

/// An error body the API returns instead of a message.
///
/// Sent with a non-2xx HTTP status. `kind` is the parsed vocabulary and
/// `raw_kind` the string it came from, so a type Anthropic adds later stays
/// legible; [`crate::values::ErrorType::from_status`] gives the status code's
/// documented type where no body arrived at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    /// The error type, where it is one this crate knows.
    pub kind: Option<crate::values::ErrorType>,
    /// The `error.type` string exactly as sent.
    pub raw_kind: String,
    /// The human-readable message.
    pub message: String,
}

impl ApiError {
    /// One error body, decoded.
    ///
    /// Fails only when the body is not JSON, is not an object, or has no `error`
    /// object. A missing `type` or `message` *inside* that object is not fatal: an
    /// error the caller can name and log beats a decode failure that discards
    /// what the server said.
    pub fn decode(body: &str) -> Result<Self, FrameError> {
        let value: Value = serde_json::from_str(body).map_err(FrameError::NotJson)?;
        Self::from_json(&value)
    }

    /// The same, for a caller who already holds the parsed body.
    pub fn from_json(body: &Value) -> Result<Self, FrameError> {
        if !body.is_object() {
            return Err(FrameError::NotAnObject);
        }
        let error = require(body, "error")?;
        if !error.is_object() {
            return Err(FrameError::WrongType { field: "error", expected: "an object" });
        }
        let raw_kind = crate::frame::optional_string(error, "type");
        Ok(ApiError {
            kind: crate::values::ErrorType::from_str(&raw_kind),
            raw_kind,
            message: crate::frame::optional_string(error, "message"),
        })
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.raw_kind)
    }
}

impl std::error::Error for ApiError {}

fn joined<'a>(pieces: impl Iterator<Item = &'a str> + Clone) -> String {
    let mut joined = String::with_capacity(pieces.clone().map(str::len).sum());
    for piece in pieces {
        joined.push_str(piece);
    }
    joined
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A response captured live whose usage names the tier that served it, and
    /// whose `stop_details` is null because nothing was refused.
    #[test]
    fn a_captured_response_reports_its_tier_and_no_refusal() {
        let response = Response::decode(
            r#"{"model": "aws/anthropic/bedrock-claude-opus-5",
                "id": "msg_bdrk_xks345u4fdiv5rwkn7a47smi4r3rntfka5cxui7s5vvm55twxwqa", "type": "message",
                "role": "assistant", "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn", "stop_sequence": null, "stop_details": null,
                "usage": {"input_tokens": 16, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                "output_tokens": 4, "output_tokens_details": {"thinking_tokens": 0},
                "service_tier": "standard"}}"#,
        )
        .unwrap();
        assert_eq!(response.text(), "ok");
        assert_eq!(response.refusal, None, "an ordinary stop refuses nothing");
        assert_eq!(response.usage.service_tier, Some(crate::values::ServedTier::Standard));
    }

    /// A refusal is a finished message with a reason, not an error body.
    #[test]
    fn a_refusal_response_names_the_category_that_fired() {
        let response = Response::from_json(&json!({
            "id": "msg_r", "model": "claude-opus-5", "content": [], "stop_reason": "refusal",
            "stop_details": {"type": "refusal", "category": "frontier_llm",
                             "explanation": "This may assist competing model development."},
            "usage": {"input_tokens": 90, "output_tokens": 0}
        }))
        .unwrap();
        assert_eq!(response.stop_reason, Some(StopReason::Refusal));
        let refusal = response.refusal.unwrap();
        assert_eq!(refusal.category, Some(crate::values::RefusalCategory::FrontierLlm));
        assert_eq!(refusal.explanation.as_deref(), Some("This may assist competing model development."));
    }

    /// A response captured from the live gateway, on a cache hit.
    #[test]
    fn a_captured_response_decodes() {
        let response = Response::decode(
            r#"{"model": "aws/anthropic/bedrock-claude-opus-5",
                "id": "msg_bdrk_uvqh23yuvwqi3puuuh4rxmvca2sxet23o7m3jywjxaggmpaxe4zq", "type": "message",
                "role": "assistant", "stop_reason": "end_turn", "stop_sequence": null, "stop_details": null,
                "content": [{"type": "text", "text": "GCD(1071, 462) = 21."}],
                "usage": {"input_tokens": 36, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1043,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                "output_tokens": 45, "output_tokens_details": {"thinking_tokens": 0},
                "service_tier": "standard"}}"#,
        )
        .unwrap();

        assert_eq!(response.id, "msg_bdrk_uvqh23yuvwqi3puuuh4rxmvca2sxet23o7m3jywjxaggmpaxe4zq");
        assert_eq!(response.model, "aws/anthropic/bedrock-claude-opus-5");
        assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(response.text(), "GCD(1071, 462) = 21.");
        assert_eq!(response.thinking(), "");
        assert_eq!(response.tool_calls().count(), 0);
        assert_eq!(response.usage.cache_read_input_tokens, 1_043);
        assert_eq!(response.usage.total_input_tokens(), 1_079);
        assert!(response.usage.cache_hit_rate().unwrap() > 0.96);
    }

    /// A tool-use response: the input arrives whole, and is still held as bytes so
    /// replaying it cannot reorder keys and cost the cache.
    #[test]
    fn a_tool_use_response_keeps_its_input_as_bytes() {
        let response = Response::from_json(&json!({
            "id": "msg_1", "model": "claude-opus-5", "type": "message", "role": "assistant",
            "stop_reason": "tool_use", "stop_sequence": null,
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"location": "San Francisco"}}
            ],
            "usage": {"input_tokens": 505, "output_tokens": 34}
        }))
        .unwrap();

        assert_eq!(response.stop_reason, Some(StopReason::ToolUse));
        assert_eq!(response.text(), "Let me check.");
        let calls: Vec<_> = response.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "toolu_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input.decode().unwrap(), json!({"location": "San Francisco"}));
    }

    /// Thinking and answer stay apart, exactly as in a settled stream.
    #[test]
    fn thinking_and_answer_blocks_stay_apart() {
        let response = Response::from_json(&json!({
            "id": "msg_t", "model": "claude-opus-5",
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "Euclid's argument works.", "signature": "EqQBCgIYAhIM"},
                {"type": "text", "text": "There are infinitely many primes."}
            ],
            "usage": {"input_tokens": 20, "output_tokens": 90, "output_tokens_details": {"thinking_tokens": 60}}
        }))
        .unwrap();
        assert_eq!(response.text(), "There are infinitely many primes.");
        assert_eq!(response.thinking(), "Euclid's argument works.");
        assert_eq!(response.usage.output_tokens_details.thinking_tokens, 60);
    }

    /// The documented `max_tokens: 0` pre-warm response: no content, a stop
    /// reason, and a full usage block. The empty content is the point.
    #[test]
    fn a_prewarm_response_has_no_content_but_real_usage() {
        let response = Response::from_json(&json!({
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL", "type": "message", "role": "assistant", "content": [],
            "model": "claude-opus-5", "stop_reason": "max_tokens", "stop_sequence": null,
            "usage": {"input_tokens": 8, "cache_creation_input_tokens": 5120, "cache_read_input_tokens": 0,
                      "cache_creation": {"ephemeral_5m_input_tokens": 5120, "ephemeral_1h_input_tokens": 0},
                      "output_tokens": 0, "service_tier": "standard"}
        }))
        .unwrap();
        assert!(response.blocks.is_empty());
        assert_eq!(response.stop_reason, Some(StopReason::MaxTokens));
        assert_eq!(response.text(), "");
        assert_eq!(response.usage.cache_creation_input_tokens, 5_120, "the write is the whole purpose");
        assert!(response.usage.cache_creation_is_consistent());
        assert_eq!(response.usage.output_tokens, 0);
    }

    /// The documented 1-hour cache usage block, with its TTL split.
    #[test]
    fn the_documented_mixed_ttl_usage_decodes_and_adds_up() {
        let response = Response::from_json(&json!({
            "id": "msg_m", "model": "claude-opus-5", "content": [], "stop_reason": "end_turn",
            "usage": {"input_tokens": 2048, "cache_read_input_tokens": 1800, "cache_creation_input_tokens": 248,
                      "output_tokens": 503,
                      "cache_creation": {"ephemeral_5m_input_tokens": 148, "ephemeral_1h_input_tokens": 100}}
        }))
        .unwrap();
        let usage = response.usage;
        assert_eq!(usage.cache_creation.ephemeral_5m_input_tokens, 148);
        assert_eq!(usage.cache_creation.ephemeral_1h_input_tokens, 100);
        assert!(usage.cache_creation_is_consistent(), "148 + 100 = 248");
        assert_eq!(usage.total_input_tokens(), 4_096);
    }

    #[test]
    fn a_matched_stop_sequence_is_reported() {
        let response = Response::from_json(&json!({
            "id": "msg_s", "model": "claude-opus-5", "content": [{"type": "text", "text": "up to "}],
            "stop_reason": "stop_sequence", "stop_sequence": "END"
        }))
        .unwrap();
        assert_eq!(response.stop_reason, Some(StopReason::StopSequence));
        assert_eq!(response.stop_sequence.as_deref(), Some("END"));
    }

    /// An unrecognized stop reason reads as absent rather than costing the caller
    /// a perfectly good answer.
    #[test]
    fn an_unrecognized_stop_reason_reads_as_absent() {
        let response = Response::from_json(&json!({
            "id": "msg_u", "model": "claude-opus-5", "stop_reason": "some_new_reason",
            "content": [{"type": "text", "text": "still useful"}]
        }))
        .unwrap();
        assert_eq!(response.stop_reason, None);
        assert_eq!(response.text(), "still useful");
    }

    /// A server-tool block a caller did not ask for does not fail the response.
    #[test]
    fn an_unmodeled_block_kind_does_not_fail_the_response() {
        let response = Response::from_json(&json!({
            "id": "msg_w", "model": "claude-opus-5", "stop_reason": "end_turn",
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
                {"type": "text", "text": "Here is the weather."}
            ]
        }))
        .unwrap();
        assert_eq!(response.blocks.len(), 2);
        assert_eq!(response.blocks[0].kind(), "server_tool_use");
        assert_eq!(response.text(), "Here is the weather.");
    }

    #[test]
    fn a_malformed_response_is_an_error() {
        assert!(matches!(Response::decode("not json"), Err(FrameError::NotJson(_))));
        assert!(matches!(Response::decode("[]"), Err(FrameError::NotAnObject)));
        assert!(matches!(Response::decode("{}"), Err(FrameError::MissingField { field: "id" })));
        assert!(matches!(Response::from_json(&json!({"id": "m"})), Err(FrameError::MissingField { field: "model" })));
        assert!(matches!(
            Response::from_json(&json!({"id": "m", "model": "x", "content": "text"})),
            Err(FrameError::WrongType { field: "content", .. })
        ));
        assert!(matches!(
            Response::from_json(&json!({"id": "m", "model": "x", "content": [{"type": "tool_use", "id": "t"}]})),
            Err(FrameError::MissingField { field: "name" })
        ));
        assert!(matches!(
            Response::from_json(&json!({"id": "m", "model": "x", "usage": {"output_tokens": "many"}})),
            Err(FrameError::UndecodableUsage(_))
        ));
    }

    /// An absent `content` array is the empty content, matching the pre-warm case.
    #[test]
    fn an_absent_content_array_is_no_content() {
        let response = Response::from_json(&json!({"id": "m", "model": "x"})).unwrap();
        assert!(response.blocks.is_empty());
        assert_eq!(response.usage, Usage::default());
    }

    /// The error body captured from the live gateway, whose type is outside the
    /// documented list and therefore keeps its raw string.
    #[test]
    fn a_captured_error_body_decodes() {
        let error = ApiError::decode(
            r#"{"error": {"message": "key not allowed to access model", "type": "key_model_access_denied",
                "param": "model", "code": "403"}}"#,
        )
        .unwrap();
        assert_eq!(error.kind, None);
        assert_eq!(error.raw_kind, "key_model_access_denied");
        assert_eq!(error.message, "key not allowed to access model");
        assert_eq!(error.to_string(), "key not allowed to access model (key_model_access_denied)");
    }

    /// Anthropic's documented error body, parsed into the crate's vocabulary.
    #[test]
    fn a_documented_error_body_parses_into_the_error_vocabulary() {
        let error =
            ApiError::decode(r#"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#)
                .unwrap();
        assert_eq!(error.kind, Some(crate::values::ErrorType::Overloaded));
        assert_eq!(error.message, "Overloaded");
    }

    /// A body with no `error` object is not an error body; a malformed one inside
    /// it still yields a loggable error.
    #[test]
    fn error_decoding_needs_an_error_object_but_tolerates_a_thin_one() {
        assert!(matches!(ApiError::decode("{}"), Err(FrameError::MissingField { field: "error" })));
        assert!(matches!(
            ApiError::from_json(&json!({"error": "just a string"})),
            Err(FrameError::WrongType { field: "error", .. })
        ));
        let thin = ApiError::from_json(&json!({"error": {}})).unwrap();
        assert_eq!(thin.kind, None);
        assert_eq!(thin.raw_kind, "");
        assert_eq!(thin.message, "");
    }
}
