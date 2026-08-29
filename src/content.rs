//! The content blocks a message is made of, and the deltas that grow them.
//!
//! An Anthropic message is an *indexed array of content blocks*. A stream
//! announces each block with `content_block_start` at some `index`, grows it with
//! `content_block_delta` events, and closes it with `content_block_stop`. The
//! index is the block's position in the final `content` array, which is what
//! makes reassembly exact rather than a guess about arrival order.
//!
//! This module holds the block and delta types and the fold that joins them.
//! [`crate::stream`] wraps them in events; [`crate::settle`] keeps them in an
//! index-ordered map.
//!
//! # Every delta appends
//!
//! No delta replaces anything. That is why accumulation is a fold — the same
//! frame applied twice is visible as doubled text, so [`crate::settle`] never
//! applies one twice, and why a block is complete exactly when its
//! `content_block_stop` has arrived.
//!
//! # Unrecognized kinds are not failures
//!
//! Server tools introduce block kinds (`server_tool_use`,
//! `web_search_tool_result`) and delta kinds (`citations_delta`) that a caller
//! using only its own tools never sees. Anthropic adds more over time. So both
//! have an `Unmodeled` variant, for the same reason
//! [`crate::stream::StreamEvent::Unmodeled`] exists: a well-formed thing this
//! crate does not know is not a broken frame.

use serde_json::Value;

use crate::frame::{FrameError, optional_string, require_str};

// ── Tool input ───────────────────────────────────────────────────────────────

/// A tool call's `input`, exactly as the model emitted it.
///
/// Anthropic streams tool input as `input_json_delta` fragments of *partial
/// JSON*: no single fragment need parse on its own, and only the concatenation
/// after `content_block_stop` is a whole object. So this holds bytes, and
/// [`Self::decode`] is a separate step that can fail without destroying the call
/// it belongs to — the caller can still answer the model with a tool error
/// instead of dropping the turn.
///
/// Keeping the bytes also protects the prompt cache. Anthropic warns that
/// re-serializing a parsed value can reorder keys, and a reordered `tool_use`
/// block replayed on the next turn is a different prefix and therefore a cache
/// miss.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolInput(String);

impl ToolInput {
    /// The input as it arrived.
    pub fn from_wire(input: impl Into<String>) -> Self {
        Self(input.into())
    }

    /// The exact bytes, for replay.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The input as a JSON value.
    ///
    /// Three outcomes, kept apart:
    ///
    /// * Well-formed JSON decodes to it.
    /// * Empty or blank input decodes to the empty object. A tool taking no
    ///   arguments is announced as `"input": {}` and streamed as one empty
    ///   `partial_json` fragment; both mean the empty argument set.
    /// * Anything else is `Err`, carrying `serde_json`'s byte offset. A stream
    ///   cut off mid-call leaves partial JSON here, and this is where that shows
    ///   up as a failure rather than as an invented value.
    pub fn decode(&self) -> Result<Value, serde_json::Error> {
        if self.0.trim().is_empty() {
            return Ok(Value::Object(serde_json::Map::new()));
        }
        serde_json::from_str(&self.0)
    }

    /// Appends one `input_json_delta` fragment.
    pub(crate) fn push_fragment(&mut self, partial_json: &str) {
        self.0.push_str(partial_json);
    }

    /// The `input` field of a `tool_use` block, however it arrived.
    ///
    /// One rule covers both directions of the API, because they carry the same
    /// value differently:
    ///
    /// * A *stream* announces every tool call with `"input": {}` and sends the
    ///   real input afterwards as `input_json_delta` fragments. The empty object
    ///   must therefore become empty bytes, or the fragments would append to a
    ///   literal `{}` and the result would not parse.
    /// * A *non-streamed response* carries the finished input in this field and
    ///   sends no fragments, so it must be kept.
    ///
    /// The rule is "an empty object is the empty input", and it is exact rather
    /// than a heuristic about which direction the block came from: a tool taking
    /// no arguments has `{}` as its whole input, and [`Self::decode`] already maps
    /// empty bytes to `{}`. The two representations denote one value, so
    /// collapsing them loses nothing.
    ///
    /// The bytes here are `serde_json`'s canonical form, with object keys sorted,
    /// because the wire order was already lost when the body was parsed. That is
    /// still cache-safe: what the prompt cache needs is that repeated turns send
    /// *identical* bytes, and a canonical form is identical to itself. See the
    /// type documentation for why the streamed path keeps the original bytes
    /// instead, where it still can.
    fn from_json(input: Option<&Value>) -> Self {
        match input {
            None | Some(Value::Null) => Self::default(),
            Some(Value::Object(fields)) if fields.is_empty() => Self::default(),
            Some(input) => Self(input.to_string()),
        }
    }
}

// ── Blocks ───────────────────────────────────────────────────────────────────

/// One content block the model produced.
///
/// The state a block is in while it streams, and the state it settles into. A
/// tool call is announced with `"input": {}` and its real input arrives
/// afterwards, so [`Self::ToolUse`] begins with an empty [`ToolInput`] rather
/// than a decoded value.
///
/// Distinct from [`crate::context::ContentBlock`], which is what a caller *sends*
/// and carries cache-breakpoint metadata. This is what came back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamedBlock {
    /// Answer text, grown by `text_delta`.
    Text {
        /// The text so far, or all of it once the block has stopped.
        text: String,
    },
    /// A thinking block, grown by `thinking_delta` and signed by
    /// `signature_delta`.
    ///
    /// Under `display: "omitted"` the block opens, receives a single
    /// `signature_delta`, and closes with no `thinking_delta` at all — so empty
    /// `thinking` beside a non-empty `signature` is normal, not a truncation.
    /// This crate observed exactly that on the live API.
    Thinking {
        /// The reasoning text, empty under `display: "omitted"`.
        thinking: String,
        /// The signature that lets Anthropic verify the block if it is replayed.
        /// Sent as one delta just before `content_block_stop`.
        signature: String,
    },
    /// A thinking block whose contents safety systems redacted. Opaque, and
    /// replayable only as these bytes.
    RedactedThinking {
        /// The opaque payload.
        data: String,
    },
    /// A call to one of the caller's tools, grown by `input_json_delta`.
    ToolUse {
        /// The identifier a matching `tool_result` must repeat.
        id: String,
        /// Which tool to run.
        name: String,
        /// The input, undecoded. See [`ToolInput`].
        input: ToolInput,
    },
    /// A block kind this crate does not model — a server tool call, its result,
    /// or anything Anthropic adds later.
    ///
    /// Never an error, for the reason given in the module documentation. Its
    /// `kind` is worth logging once.
    Unmodeled {
        /// The block's `type`.
        kind: String,
    },
}

impl StreamedBlock {
    /// The block's wire `type`.
    pub fn kind(&self) -> &str {
        match self {
            StreamedBlock::Text { .. } => "text",
            StreamedBlock::Thinking { .. } => "thinking",
            StreamedBlock::RedactedThinking { .. } => "redacted_thinking",
            StreamedBlock::ToolUse { .. } => "tool_use",
            StreamedBlock::Unmodeled { kind } => kind,
        }
    }

    /// The block's answer text, for a block that has any.
    ///
    /// Answer text only. Thinking is deliberately excluded: folding reasoning
    /// into an answer would present it as one.
    pub fn text(&self) -> Option<&str> {
        match self {
            StreamedBlock::Text { text } => Some(text),
            _ => None,
        }
    }

    /// One `content_block_start` payload, decoded.
    pub fn decode(block: &Value) -> Result<Self, FrameError> {
        if !block.is_object() {
            return Err(FrameError::WrongType { field: "content_block", expected: "an object" });
        }
        Ok(match require_str(block, "type")? {
            "text" => StreamedBlock::Text { text: optional_string(block, "text") },
            "thinking" => StreamedBlock::Thinking {
                thinking: optional_string(block, "thinking"),
                signature: optional_string(block, "signature"),
            },
            "redacted_thinking" => StreamedBlock::RedactedThinking { data: optional_string(block, "data") },
            "tool_use" => StreamedBlock::ToolUse {
                id: require_str(block, "id")?.to_owned(),
                name: require_str(block, "name")?.to_owned(),
                input: ToolInput::from_json(block.get("input")),
            },
            other => StreamedBlock::Unmodeled { kind: other.to_owned() },
        })
    }

    /// Folds one delta into this block.
    ///
    /// A delta whose kind does not match the block — a `text_delta` addressed to
    /// a tool call, say — is ignored rather than fabricating a field the block
    /// does not have. Mismatches are not observed in practice; the API pairs each
    /// delta kind with its block kind.
    pub fn apply(&mut self, delta: &BlockDelta) {
        match (self, delta) {
            (StreamedBlock::Text { text }, BlockDelta::Text { delta }) => text.push_str(delta),
            (StreamedBlock::Thinking { thinking, .. }, BlockDelta::Thinking { delta }) => thinking.push_str(delta),
            (StreamedBlock::Thinking { signature, .. }, BlockDelta::Signature { delta }) => signature.push_str(delta),
            (StreamedBlock::ToolUse { input, .. }, BlockDelta::InputJson { partial_json }) => {
                input.push_fragment(partial_json);
            }
            _ => {}
        }
    }
}

// ── Deltas ───────────────────────────────────────────────────────────────────

/// One `content_block_delta` payload.
///
/// Every variant appends; none replaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockDelta {
    /// `text_delta`: more answer text.
    Text {
        /// The text to append.
        delta: String,
    },
    /// `thinking_delta`: more reasoning. Absent under `display: "omitted"`.
    Thinking {
        /// The reasoning to append.
        delta: String,
    },
    /// `signature_delta`: the thinking block's signature, sent once just before
    /// its `content_block_stop`.
    Signature {
        /// The signature bytes to append.
        delta: String,
    },
    /// `input_json_delta`: one fragment of a tool call's input.
    ///
    /// Partial JSON. A fragment need not parse alone — see [`ToolInput`].
    InputJson {
        /// The fragment to append.
        partial_json: String,
    },
    /// A delta kind this crate does not model, such as `citations_delta`.
    Unmodeled {
        /// The delta's `type`.
        kind: String,
    },
}

impl BlockDelta {
    /// The delta's wire `type`.
    pub fn kind(&self) -> &str {
        match self {
            BlockDelta::Text { .. } => "text_delta",
            BlockDelta::Thinking { .. } => "thinking_delta",
            BlockDelta::Signature { .. } => "signature_delta",
            BlockDelta::InputJson { .. } => "input_json_delta",
            BlockDelta::Unmodeled { kind } => kind,
        }
    }

    /// One `delta` payload, decoded.
    pub fn decode(delta: &Value) -> Result<Self, FrameError> {
        if !delta.is_object() {
            return Err(FrameError::WrongType { field: "delta", expected: "an object" });
        }
        Ok(match require_str(delta, "type")? {
            "text_delta" => BlockDelta::Text { delta: require_str(delta, "text")?.to_owned() },
            "thinking_delta" => BlockDelta::Thinking { delta: require_str(delta, "thinking")?.to_owned() },
            "signature_delta" => BlockDelta::Signature { delta: require_str(delta, "signature")?.to_owned() },
            "input_json_delta" => {
                BlockDelta::InputJson { partial_json: require_str(delta, "partial_json")?.to_owned() }
            }
            other => BlockDelta::Unmodeled { kind: other.to_owned() },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The captured `input_json_delta` fragments from a live tool call:
    /// individually invalid JSON, whole only once concatenated.
    #[test]
    fn captured_input_fragments_are_partial_on_purpose() {
        let mut input = ToolInput::default();
        for fragment in ["", r#"{"locatio"#, r#"n": "San "#, "Fra", "nc", r#"isco"}"#] {
            input.push_fragment(fragment);
        }
        assert_eq!(input.as_str(), r#"{"location": "San Francisco"}"#);
        assert_eq!(input.decode().unwrap(), json!({"location": "San Francisco"}));
    }

    /// A stream cut off mid-call leaves input that does not parse, and says so
    /// rather than inventing a value.
    #[test]
    fn truncated_input_fails_only_where_it_is_decoded() {
        let truncated = ToolInput::from_wire(r#"{"location": "San Fra"#);
        assert_eq!(truncated.as_str(), r#"{"location": "San Fra"#, "the bytes survive");
        assert!(truncated.decode().unwrap_err().to_string().contains("EOF"));
    }

    #[test]
    fn absent_input_decodes_to_the_empty_object() {
        assert_eq!(ToolInput::default().decode().unwrap(), json!({}));
        assert_eq!(ToolInput::from_wire("").decode().unwrap(), json!({}));
        assert_eq!(ToolInput::from_wire("  ").decode().unwrap(), json!({}));
    }

    /// A captured live `tool_use` announcement carries no input: it arrives as
    /// `{}` and is held as empty bytes so the fragments can append to it.
    #[test]
    fn a_captured_tool_use_announcement_starts_with_no_input() {
        let block = StreamedBlock::decode(&json!({
            "type": "tool_use", "id": "toolu_bdrk_01SfSun6GdgxKVDToTxtf7Ee", "name": "get_weather", "input": {}
        }))
        .unwrap();
        let StreamedBlock::ToolUse { id, name, input } = &block else { panic!("expected a tool call") };
        assert_eq!(id, "toolu_bdrk_01SfSun6GdgxKVDToTxtf7Ee");
        assert_eq!(name, "get_weather");
        assert_eq!(*input, ToolInput::default());
        assert_eq!(block.kind(), "tool_use");
        assert_eq!(block.text(), None);
    }

    /// The same field in the other direction: a non-streamed response carries the
    /// finished input here, and it is kept rather than discarded.
    #[test]
    fn a_finished_tool_input_survives_decoding() {
        let block = StreamedBlock::decode(&json!({
            "type": "tool_use", "id": "toolu_1", "name": "get_weather",
            "input": {"location": "San Francisco", "units": "celsius"}
        }))
        .unwrap();
        let StreamedBlock::ToolUse { input, .. } = &block else { panic!("expected a tool call") };
        assert_eq!(input.decode().unwrap(), json!({"location": "San Francisco", "units": "celsius"}));
        assert_eq!(input.as_str(), r#"{"location":"San Francisco","units":"celsius"}"#, "canonical and stable");
    }

    /// An empty object and no input at all denote one value, so they decode alike.
    /// That is what lets one rule serve a stream's `{}` announcement and a
    /// response's finished input.
    #[test]
    fn an_empty_object_input_and_an_absent_one_agree() {
        for input in [json!({}), Value::Null] {
            let block = StreamedBlock::decode(&json!({
                "type": "tool_use", "id": "t", "name": "finish", "input": input
            }))
            .unwrap();
            let StreamedBlock::ToolUse { input, .. } = &block else { panic!("expected a tool call") };
            assert_eq!(*input, ToolInput::default());
            assert_eq!(input.decode().unwrap(), json!({}));
        }
        let without = StreamedBlock::decode(&json!({"type": "tool_use", "id": "t", "name": "finish"})).unwrap();
        let StreamedBlock::ToolUse { input, .. } = &without else { panic!("expected a tool call") };
        assert_eq!(*input, ToolInput::default());
    }

    /// A captured thinking block under `display: "omitted"`: it opens, takes one
    /// signature, and closes with no reasoning text.
    #[test]
    fn a_captured_thinking_block_may_carry_only_a_signature() {
        let mut block = StreamedBlock::decode(&json!({"type": "thinking", "thinking": "", "signature": ""})).unwrap();
        block.apply(&BlockDelta::Signature { delta: "CAISyUkKcAgREAEYAipAqjMsxB4V".to_owned() });
        assert_eq!(
            block,
            StreamedBlock::Thinking { thinking: String::new(), signature: "CAISyUkKcAgREAEYAipAqjMsxB4V".to_owned() },
            "empty thinking beside a real signature is what `display: omitted` looks like"
        );
        assert_eq!(block.text(), None, "thinking is not answer text");
    }

    #[test]
    fn thinking_deltas_accumulate_ahead_of_the_signature() {
        let mut block = StreamedBlock::decode(&json!({"type": "thinking", "thinking": "", "signature": ""})).unwrap();
        for piece in ["I need to find the GCD", " of 1071 and 462", "\nThe answer is 21."] {
            block.apply(&BlockDelta::Thinking { delta: piece.to_owned() });
        }
        block.apply(&BlockDelta::Signature { delta: "sig".to_owned() });
        assert_eq!(
            block,
            StreamedBlock::Thinking {
                thinking: "I need to find the GCD of 1071 and 462\nThe answer is 21.".to_owned(),
                signature: "sig".to_owned(),
            }
        );
    }

    #[test]
    fn text_deltas_accumulate() {
        let mut block = StreamedBlock::decode(&json!({"type": "text", "text": ""})).unwrap();
        for piece in ["Hel", "lo, ", "world"] {
            block.apply(&BlockDelta::Text { delta: piece.to_owned() });
        }
        assert_eq!(block.text(), Some("Hello, world"));
    }

    #[test]
    fn a_redacted_thinking_block_keeps_its_opaque_payload() {
        let block = StreamedBlock::decode(&json!({"type": "redacted_thinking", "data": "EvwBCkYIAxgCKkC"})).unwrap();
        assert_eq!(block, StreamedBlock::RedactedThinking { data: "EvwBCkYIAxgCKkC".to_owned() });
        assert_eq!(block.kind(), "redacted_thinking");
    }

    /// A server-tool block is not a broken frame; nor is a delta kind this crate
    /// does not model.
    #[test]
    fn unmodeled_kinds_decode_rather_than_fail() {
        let block =
            StreamedBlock::decode(&json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1"})).unwrap();
        assert_eq!(block, StreamedBlock::Unmodeled { kind: "web_search_tool_result".to_owned() });
        assert_eq!(block.kind(), "web_search_tool_result");

        let delta = BlockDelta::decode(&json!({"type": "citations_delta", "citation": {}})).unwrap();
        assert_eq!(delta, BlockDelta::Unmodeled { kind: "citations_delta".to_owned() });
        assert_eq!(delta.kind(), "citations_delta");
    }

    /// A delta addressed to a block of another kind is ignored rather than
    /// inventing a field the block does not have.
    #[test]
    fn a_mismatched_delta_leaves_its_block_alone() {
        let original =
            StreamedBlock::ToolUse { id: "t1".to_owned(), name: "f".to_owned(), input: ToolInput::from_wire("{}") };
        let mut block = original.clone();
        block.apply(&BlockDelta::Text { delta: "not mine".to_owned() });
        block.apply(&BlockDelta::Thinking { delta: "nor this".to_owned() });
        block.apply(&BlockDelta::Unmodeled { kind: "citations_delta".to_owned() });
        assert_eq!(block, original);
    }

    #[test]
    fn every_delta_kind_decodes_and_names_itself() {
        let cases = [
            (json!({"type": "text_delta", "text": "x"}), "text_delta"),
            (json!({"type": "thinking_delta", "thinking": "x"}), "thinking_delta"),
            (json!({"type": "signature_delta", "signature": "x"}), "signature_delta"),
            (json!({"type": "input_json_delta", "partial_json": "x"}), "input_json_delta"),
        ];
        for (frame, kind) in cases {
            assert_eq!(BlockDelta::decode(&frame).unwrap().kind(), kind);
        }
    }

    #[test]
    fn a_malformed_block_or_delta_is_an_error() {
        assert!(matches!(StreamedBlock::decode(&json!([])), Err(FrameError::WrongType { field: "content_block", .. })));
        assert!(matches!(StreamedBlock::decode(&json!({})), Err(FrameError::MissingField { field: "type" })));
        assert!(matches!(
            StreamedBlock::decode(&json!({"type": "tool_use", "id": "t"})),
            Err(FrameError::MissingField { field: "name" })
        ));
        assert!(matches!(BlockDelta::decode(&json!("nope")), Err(FrameError::WrongType { field: "delta", .. })));
        assert!(matches!(
            BlockDelta::decode(&json!({"type": "text_delta", "text": 7})),
            Err(FrameError::WrongType { field: "text", .. })
        ));
        assert!(matches!(
            BlockDelta::decode(&json!({"type": "input_json_delta"})),
            Err(FrameError::MissingField { field: "partial_json" })
        ));
    }
}
