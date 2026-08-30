//! Accumulating a stream, and the boundary where it becomes a finished message.
//!
//! # Why "settled" is a type and not a flag
//!
//! A streamed message is only trustworthy once a terminal event arrives. A
//! connection can drop mid-answer, and the text collected so far looks exactly
//! like a complete answer — same field, same characters, no error anywhere.
//! Anything that reports "finished" with a boolean invites the caller to read the
//! half-finished case as the finished one.
//!
//! So the two states are two types. [`Settling`] accumulates and has no method
//! that yields a message. [`Settled`] is a finished message and has no method
//! that accepts more events. The only bridge is [`Settling::settle`], which
//! consumes the accumulator: a truncated stream yields [`SettleError::Truncated`],
//! and there is no other way to obtain a `Settled`. A caller who forgets to check
//! gets a compile error, not a plausible answer.
//!
//! The Anthropic protocol makes this sharper than it sounds. `message_delta`
//! carries the `stop_reason`, so a stream can look complete — "the model said
//! `end_turn`" — one frame before it actually is. Only `message_stop` (or an
//! `error`) ends it. Believing a `stop_reason` is believing a flag, and this
//! module makes that unwritable.
//!
//! # Cost of accumulation
//!
//! Text and tool input arrive as many small fragments. Each is appended to the
//! `String` of its block, which amortizes to linear in the total bytes:
//! `String::push_str` doubles capacity, so *n* fragments cost O(*n*) copying
//! rather than the O(*n*²) of rebuilding a joined string per fragment. Blocks
//! live in a `BTreeMap` keyed by their wire `index`, so a repeated index updates
//! in place instead of duplicating, and iteration is already in the order the
//! final `content` array uses.

use std::collections::BTreeMap;

use crate::content::{StreamedBlock, ToolInput};
use crate::frame::FrameError;
use crate::stream::{MessageDelta, MessageStart, RefusalDetails, StreamEvent, StreamedError};
use crate::usage::Usage;
use crate::values::StopReason;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Why a stream did not produce a finished message.
#[derive(Debug)]
pub enum SettleError {
    /// The stream ended without a terminal event.
    ///
    /// The connection dropped, the reader stopped early, or the server hung up.
    /// Whatever content had accumulated is *not* returned: handing back a partial
    /// answer typed as a whole one is the mistake this module exists to prevent.
    /// The counts are here so the failure can be logged usefully.
    Truncated {
        /// How many events had been consumed.
        events: usize,
        /// How many characters of answer text had accumulated.
        text_len: usize,
        /// Whether a `message_delta` had already named a stop reason.
        ///
        /// `true` is the interesting case, and the reason this field exists: the
        /// model finished its turn and the stream still broke before
        /// `message_stop`. A caller that trusted the stop reason would have
        /// reported success.
        had_stop_reason: bool,
    },
    /// A frame could not be decoded. Carries the frame error unchanged.
    Frame(FrameError),
}

impl std::fmt::Display for SettleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SettleError::Truncated { events, text_len, had_stop_reason } => write!(
                f,
                "stream ended without a terminal event after {events} event(s) and {text_len} character(s) of text \
                 (stop reason {})",
                if *had_stop_reason { "had already arrived" } else { "never arrived" }
            ),
            SettleError::Frame(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for SettleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SettleError::Frame(error) => Some(error),
            SettleError::Truncated { .. } => None,
        }
    }
}

impl From<FrameError> for SettleError {
    fn from(error: FrameError) -> Self {
        SettleError::Frame(error)
    }
}

// ── Settling ─────────────────────────────────────────────────────────────────

/// A stream still being read.
///
/// Deliberately offers no way to read a finished message out of itself. Feed it
/// events with [`Self::consume`]; when the stream is exhausted call
/// [`Self::settle`], which either produces a [`Settled`] or fails.
///
/// An accumulator is not a message, and the compiler enforces it rather than the
/// documentation asking politely. There is no `blocks` field to reach for:
///
/// ```compile_fail
/// let settling = anthropic::settle::Settling::new();
/// let _ = settling.blocks;
/// ```
///
/// And a `Settled` cannot be assembled by hand to bypass the check, because
/// [`Settling::settle`] is its only constructor:
///
/// ```compile_fail
/// let _ = anthropic::settle::Settled {
///     outcome: anthropic::settle::Outcome::Stopped { reason: None, stop_sequence: None },
///     id: String::new(),
///     model: String::new(),
///     blocks: Vec::new(),
///     usage: anthropic::usage::Usage::default(),
///     events: 0,
/// };
/// ```
#[derive(Debug, Default)]
pub struct Settling {
    blocks: BTreeMap<u32, StreamedBlock>,
    start: Option<MessageStart>,
    ending: Option<MessageDelta>,
    terminal: Option<Terminal>,
    usage: Usage,
    events: usize,
}

/// The terminal event, kept only so [`Settling::settle`] can turn it into an
/// [`Outcome`].
#[derive(Debug)]
enum Terminal {
    Stop,
    Error(StreamedError),
}

impl Settling {
    /// A stream with nothing in it yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// One raw `data:` payload, decoded and folded in.
    ///
    /// The convenience path: equivalent to [`StreamEvent::decode`] followed by
    /// [`Self::consume`], returning the decoded event so a caller can also
    /// forward deltas to a display as they arrive.
    ///
    /// A frame that fails to decode is not folded in and is not counted, so the
    /// accumulator is exactly as it was before the bad frame.
    pub fn consume_payload(&mut self, payload: &str) -> Result<StreamEvent, SettleError> {
        let event = StreamEvent::decode(payload)?;
        self.consume(event.clone());
        Ok(event)
    }

    /// One decoded event, folded in.
    ///
    /// Infallible on purpose. Every failure a frame can have already happened
    /// during decoding, and an unmodeled event is not a failure — so nothing here
    /// can go wrong, and the signature says so.
    ///
    /// Usage from every frame that carries it is merged by pointwise maximum, so
    /// a later partial record cannot erase an earlier fuller one. A second
    /// terminal event is ignored: the first one ended the message, and a
    /// duplicate must not overwrite a success with a later spurious failure or
    /// the reverse.
    pub fn consume(&mut self, event: StreamEvent) {
        self.events += 1;
        match event {
            StreamEvent::MessageStart(start) => {
                self.usage.merge_cumulative(&start.usage);
                if self.start.is_none() {
                    self.start = Some(start);
                }
            }
            // `start` announces the block; a repeat at the same index replaces the
            // announcement rather than adding a block, because the index *is* the
            // block's identity.
            StreamEvent::ContentBlockStart { index, block } => {
                self.blocks.insert(index, block);
            }
            // A delta for an index never announced is dropped: there is no block
            // to grow, and inventing one would guess at its kind.
            StreamEvent::ContentBlockDelta { index, delta } => {
                if let Some(block) = self.blocks.get_mut(&index) {
                    block.apply(&delta);
                }
            }
            // Nothing to record. A block is complete when its content is, and the
            // content is already here.
            StreamEvent::ContentBlockStop { .. } => {}
            StreamEvent::MessageDelta(delta) => {
                self.usage.merge_cumulative(&delta.usage);
                self.ending = Some(delta);
            }
            StreamEvent::MessageStop { usage } => {
                self.usage.merge_cumulative(&usage);
                self.finish(Terminal::Stop);
            }
            StreamEvent::Error(error) => self.finish(Terminal::Error(error)),
            StreamEvent::Unmodeled { .. } => {}
        }
    }

    fn finish(&mut self, terminal: Terminal) {
        if self.terminal.is_none() {
            self.terminal = Some(terminal);
        }
    }

    /// The answer text so far, for a live display.
    ///
    /// Named for what it is. Reading it does not settle the stream and does not
    /// claim the answer is complete — that is what [`Settled`] is for. Thinking is
    /// excluded; see [`Self::thinking_so_far`].
    pub fn text_so_far(&self) -> String {
        joined(self.blocks.values().filter_map(StreamedBlock::text))
    }

    /// The reasoning so far, for a live display. Empty under
    /// `display: "omitted"`, where the API sends no reasoning at all.
    pub fn thinking_so_far(&self) -> String {
        joined(self.blocks.values().filter_map(|block| match block {
            StreamedBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
            _ => None,
        }))
    }

    /// The usage reported so far, cache counts included.
    ///
    /// Complete from `message_start` onwards as far as caching goes: Anthropic
    /// reports the cache counts in the opening frame, because caching happens on
    /// input. Output tokens keep rising until the end.
    pub fn usage_so_far(&self) -> &Usage {
        &self.usage
    }

    /// How many events have been consumed.
    pub fn event_count(&self) -> usize {
        self.events
    }

    /// Whether a terminal event has arrived, so [`Self::settle`] will succeed.
    ///
    /// For deciding when to stop reading, not for deciding to trust the content: a
    /// `true` here still gives you nothing without calling `settle`.
    pub fn is_terminated(&self) -> bool {
        self.terminal.is_some()
    }

    /// The finished message, or a failure explaining why there is none.
    ///
    /// Consumes the accumulator, which is what makes the two states exclusive:
    /// after settling there is no `Settling` left to append to, and without
    /// settling there is no [`Settled`] at all.
    pub fn settle(self) -> Result<Settled, SettleError> {
        let terminal = match self.terminal {
            Some(terminal) => terminal,
            None => {
                return Err(SettleError::Truncated {
                    events: self.events,
                    text_len: self.text_so_far().len(),
                    had_stop_reason: self.ending.as_ref().is_some_and(|end| end.stop_reason.is_some()),
                });
            }
        };
        let (id, model) = self.start.map(|start| (start.id, start.model)).unwrap_or_default();
        let outcome = match terminal {
            Terminal::Error(error) => Outcome::Errored { error },
            Terminal::Stop => {
                let (reason, stop_sequence, refusal) =
                    self.ending.map(|end| (end.stop_reason, end.stop_sequence, end.refusal)).unwrap_or_default();
                Outcome::Stopped { reason, stop_sequence, refusal }
            }
        };
        Ok(Settled {
            outcome,
            id,
            model,
            blocks: self.blocks.into_values().collect(),
            usage: self.usage,
            events: self.events,
        })
    }
}

/// Concatenates string pieces with one allocation.
fn joined<'a>(pieces: impl Iterator<Item = &'a str> + Clone) -> String {
    let mut joined = String::with_capacity(pieces.clone().map(str::len).sum());
    for piece in pieces {
        joined.push_str(piece);
    }
    joined
}

// ── Settled ──────────────────────────────────────────────────────────────────

/// How a stream ended.
///
/// Not a status string: each ending carries exactly the data that ending has, so
/// there is no error field to read on a normal stop and no stop reason to check
/// on a failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The stream reached `message_stop`.
    ///
    /// This is not the same as "the model succeeded". [`StopReason::MaxTokens`]
    /// means the answer was cut short, [`StopReason::Refusal`] means it declined,
    /// and [`StopReason::ToolUse`] means it is waiting for a tool result — all of
    /// them arrive this way, because in each case the protocol completed. What the
    /// model *did* is the reason's business, which is why it is here rather than
    /// spread across variants.
    Stopped {
        /// Why generation stopped. `None` when no `message_delta` named a reason,
        /// or when it named one this crate does not know.
        reason: Option<StopReason>,
        /// Which stop sequence matched, on [`StopReason::StopSequence`].
        stop_sequence: Option<String>,
        /// Why the model refused, on [`StopReason::Refusal`]. Here rather than in
        /// its own outcome for the reason the whole variant exists: a refusal is a
        /// message the server finished sending, so the protocol completed.
        refusal: Option<RefusalDetails>,
    },
    /// The stream delivered an `error` event instead of finishing.
    Errored {
        /// What the API reported.
        error: StreamedError,
    },
}

/// A stream that reached a terminal event.
///
/// Obtainable only from [`Settling::settle`], so holding one is proof the stream
/// finished. Every field is final.
///
/// `#[non_exhaustive]` is what makes "only from `settle`" true rather than merely
/// intended: it forbids the struct literal outside this crate, so no caller can
/// fabricate a finished message from an unfinished stream. Reading every field
/// still works, and so does matching with `..`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Settled {
    /// How the stream ended, with that ending's own data.
    pub outcome: Outcome,
    /// The message's identifier, worth logging for Anthropic support. Empty when
    /// the stream errored before `message_start`.
    pub id: String,
    /// The model that answered, as the API named it.
    pub model: String,
    /// The content blocks, in the order of the final `content` array.
    pub blocks: Vec<StreamedBlock>,
    /// What the message cost, merged across every frame that reported it.
    ///
    /// Never optional: a stream that reached `message_start` reported real
    /// numbers, and one that errored first reports zeros, which is the honest
    /// count for work that never happened.
    pub usage: Usage,
    /// How many events the stream delivered.
    pub events: usize,
}

impl Settled {
    /// Why generation stopped, on a stream that reached `message_stop`.
    ///
    /// `None` on an `error` outcome, and also when the API named a reason this
    /// crate does not know.
    pub fn stop_reason(&self) -> Option<StopReason> {
        match &self.outcome {
            Outcome::Stopped { reason, .. } => *reason,
            Outcome::Errored { .. } => None,
        }
    }

    /// Why the model refused, on a stream that reached `message_stop` carrying
    /// [`StopReason::Refusal`]. `None` otherwise.
    pub fn refusal(&self) -> Option<&RefusalDetails> {
        match &self.outcome {
            Outcome::Stopped { refusal, .. } => refusal.as_ref(),
            Outcome::Errored { .. } => None,
        }
    }

    /// The error, on the failing outcome. `None` on a normal stop.
    pub fn error(&self) -> Option<&StreamedError> {
        match &self.outcome {
            Outcome::Errored { error } => Some(error),
            Outcome::Stopped { .. } => None,
        }
    }

    /// The whole answer text: every text block, in order, concatenated.
    ///
    /// Thinking is excluded — see [`Self::thinking`] — because presenting
    /// reasoning as an answer is exactly the confusion the block kinds prevent.
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

    /// Every place the answer drew on, in block order.
    ///
    /// Empty unless the request enabled citations on a document or search result.
    /// Flattened across blocks, because a caller rendering footnotes wants the
    /// list; the per-block association is still in
    /// [`crate::content::StreamedBlock::Text`] for one that wants to interleave.
    pub fn citations(&self) -> impl Iterator<Item = &crate::document::Citation> {
        self.blocks.iter().flat_map(|block| match block {
            crate::content::StreamedBlock::Text { citations, .. } => citations.as_slice(),
            _ => &[],
        })
    }

    /// The tool calls the model made, in block order.
    ///
    /// The list a tool-running loop iterates. Inputs are still undecoded — see
    /// [`ToolInput::decode`], which fails per call rather than per message.
    pub fn tool_calls(&self) -> impl Iterator<Item = ToolCall<'_>> {
        self.blocks.iter().filter_map(|block| match block {
            StreamedBlock::ToolUse { id, name, input } => Some(ToolCall { id, name, input }),
            _ => None,
        })
    }
}

/// One tool call, borrowed from a [`Settled`].
///
/// A borrowed view rather than a copy: a tool loop reads the name, answers with
/// the id, and decodes the input, none of which needs ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolCall<'a> {
    /// The identifier the matching `tool_result` must repeat.
    pub id: &'a str,
    /// Which tool to run.
    pub name: &'a str,
    /// The input, undecoded.
    pub input: &'a ToolInput,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A verbatim replay of a live stream: the `message_start` usage, a text
    /// block, and a `message_delta`/`message_stop` pair carrying the counts.
    const CAPTURED_TEXT_STREAM: [&str; 6] = [
        r#"{"type": "message_start", "message": {"model": "claude-opus-5", "id": "msg_bdrk_762xvdb",
            "type": "message", "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null,
            "stop_details": null, "usage": {"input_tokens": 36, "cache_creation_input_tokens": 1043,
            "cache_read_input_tokens": 0, "cache_creation": {"ephemeral_5m_input_tokens": 1043,
            "ephemeral_1h_input_tokens": 0}, "output_tokens": 1, "service_tier": "standard"}}}"#,
        r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}"#,
        r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "GCD(1071, 462)"}}"#,
        r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " = 21"}}"#,
        r#"{"type": "content_block_stop", "index": 0}"#,
        r#"{"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null,
            "stop_details": null}, "usage": {"input_tokens": 36, "cache_creation_input_tokens": 1043,
            "cache_read_input_tokens": 0, "output_tokens": 45, "output_tokens_details": {"thinking_tokens": 0},
            "cache_creation": {"ephemeral_5m_input_tokens": 1043, "ephemeral_1h_input_tokens": 0}}}"#,
    ];

    const CAPTURED_MESSAGE_STOP: &str =
        r#"{"type": "message_stop", "usage": {"input_tokens": 36, "output_tokens": 45}}"#;

    fn settle_all(payloads: &[&str]) -> Result<Settled, SettleError> {
        let mut settling = Settling::new();
        for payload in payloads {
            settling.consume_payload(payload).unwrap();
        }
        settling.settle()
    }

    /// The happy path, end to end, from captured frames.
    #[test]
    fn a_captured_text_stream_settles() {
        let mut payloads = CAPTURED_TEXT_STREAM.to_vec();
        payloads.push(CAPTURED_MESSAGE_STOP);
        let settled = settle_all(&payloads).unwrap();

        assert_eq!(
            settled.outcome,
            Outcome::Stopped { reason: Some(StopReason::EndTurn), stop_sequence: None, refusal: None }
        );
        assert_eq!(settled.text(), "GCD(1071, 462) = 21");
        assert_eq!(settled.thinking(), "", "no thinking was requested");
        assert_eq!(settled.id, "msg_bdrk_762xvdb");
        assert_eq!(settled.model, "claude-opus-5");
        assert_eq!(settled.events, 7);
        assert_eq!(settled.error(), None);
        assert_eq!(settled.tool_calls().count(), 0);

        // The `message_stop` carried only two counters; the cache numbers from
        // earlier frames survive it.
        assert_eq!(settled.usage.cache_creation_input_tokens, 1_043);
        assert_eq!(settled.usage.cache_creation.ephemeral_5m_input_tokens, 1_043);
        assert_eq!(settled.usage.output_tokens, 45);
        assert_eq!(settled.usage.total_input_tokens(), 1_079);
    }

    /// The point of the whole module: without `message_stop` there is no
    /// `Settled`, and the text that did arrive is not handed over — even though
    /// `message_delta` already said `end_turn`.
    #[test]
    fn a_stream_truncated_after_its_stop_reason_does_not_settle() {
        let mut settling = Settling::new();
        for payload in CAPTURED_TEXT_STREAM {
            settling.consume_payload(payload).unwrap();
        }
        assert_eq!(settling.text_so_far(), "GCD(1071, 462) = 21", "readable as in-progress");
        assert!(!settling.is_terminated(), "a stop reason is not a terminal event");

        let error = settling.settle().unwrap_err();
        let SettleError::Truncated { events, text_len, had_stop_reason } = error else { panic!("expected truncation") };
        assert_eq!(events, 6);
        assert_eq!(text_len, 19);
        assert!(had_stop_reason, "the model finished its turn and the stream still broke");
        assert!(error.to_string().contains("had already arrived"));
    }

    /// The degenerate truncation behaves the same way.
    #[test]
    fn an_empty_stream_does_not_settle() {
        let error = Settling::new().settle().unwrap_err();
        assert!(matches!(error, SettleError::Truncated { events: 0, text_len: 0, had_stop_reason: false }));
        assert!(error.to_string().contains("never arrived"));
    }

    /// A captured live tool-use stream, fragment for fragment. The input is whole
    /// only after `content_block_stop`.
    #[test]
    fn a_captured_tool_use_stream_settles_into_one_call() {
        let mut payloads = vec![
            r#"{"type": "message_start", "message": {"model": "claude-opus-5", "id": "msg_bdrk_rcgr6ir",
                "type": "message", "role": "assistant", "content": [], "stop_reason": null,
                "usage": {"input_tokens": 505, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                "output_tokens": 16, "service_tier": "standard"}}}"#
                .to_owned(),
            r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use",
                "id": "toolu_bdrk_01SfSun6GdgxKVDToTxtf7Ee", "name": "get_weather", "input": {}}}"#
                .to_owned(),
        ];
        for fragment in ["", r#"{\"locatio"#, r#"n\": \"San "#, "Fra", "nc", r#"isco\"}"#] {
            payloads.push(format!(
                r#"{{"type": "content_block_delta", "index": 0,
                    "delta": {{"type": "input_json_delta", "partial_json": "{fragment}"}}}}"#
            ));
        }
        payloads.push(r#"{"type": "content_block_stop", "index": 0}"#.to_owned());
        payloads.push(
            r#"{"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null,
                "stop_details": null}, "usage": {"input_tokens": 505, "output_tokens": 34,
                "output_tokens_details": {"thinking_tokens": 0},
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}}}"#
                .to_owned(),
        );
        payloads.push(r#"{"type": "message_stop", "usage": {"input_tokens": 505, "output_tokens": 34}}"#.to_owned());

        let borrowed: Vec<&str> = payloads.iter().map(String::as_str).collect();
        let settled = settle_all(&borrowed).unwrap();

        assert_eq!(settled.stop_reason(), Some(StopReason::ToolUse));
        assert_eq!(settled.text(), "", "the model called a tool instead of answering");
        let calls: Vec<_> = settled.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "toolu_bdrk_01SfSun6GdgxKVDToTxtf7Ee");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input.as_str(), r#"{"location": "San Francisco"}"#);
        assert_eq!(calls[0].input.decode().unwrap(), json!({"location": "San Francisco"}));
        assert_eq!(settled.usage.output_tokens, 34);
    }

    /// Text and a tool call in one message, the shape a tool-using turn takes:
    /// commentary at index 0, the call at index 1.
    #[test]
    fn text_and_a_tool_call_settle_together_in_index_order() {
        let settled = settle_all(&[
            r#"{"type": "message_start", "message": {"id": "msg_1", "model": "claude-opus-5", "content": []}}"#,
            r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}"#,
            r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta",
                "text": "Okay, let's check the weather for San Francisco, CA:"}}"#,
            r#"{"type": "content_block_stop", "index": 0}"#,
            r#"{"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use",
                "id": "toolu_01T1x1fJ34qAmk2tNTrN7Up6", "name": "get_weather", "input": {}}}"#,
            r#"{"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta",
                "partial_json": "{\"location\":"}}"#,
            r#"{"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta",
                "partial_json": " \"San Francisco, CA\"}"}}"#,
            r#"{"type": "content_block_stop", "index": 1}"#,
            r#"{"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null},
                "usage": {"output_tokens": 89}}"#,
            r#"{"type": "message_stop"}"#,
        ])
        .unwrap();

        assert_eq!(settled.blocks.len(), 2);
        assert_eq!(settled.text(), "Okay, let's check the weather for San Francisco, CA:");
        assert_eq!(settled.stop_reason(), Some(StopReason::ToolUse));
        let call = settled.tool_calls().next().unwrap();
        assert_eq!(call.input.decode().unwrap(), json!({"location": "San Francisco, CA"}));
        assert_eq!(settled.usage.output_tokens, 89);
    }

    /// A captured thinking stream: reasoning at index 0 with its signature, the
    /// answer at index 1. Thinking never leaks into the answer.
    #[test]
    fn a_captured_thinking_stream_keeps_reasoning_out_of_the_answer() {
        let settled = settle_all(&[
            r#"{"type": "message_start", "message": {"id": "msg_01", "model": "claude-opus-5", "content": []}}"#,
            r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "thinking",
                "thinking": "", "signature": ""}}"#,
            r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta",
                "thinking": "I need to find the GCD of 1071 and 462."}}"#,
            r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta",
                "thinking": "\n462 = 3 x 147 + 21"}}"#,
            r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta",
                "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3h"}}"#,
            r#"{"type": "content_block_stop", "index": 0}"#,
            r#"{"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}"#,
            r#"{"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta",
                "text": "The greatest common divisor of 1071 and 462 is **21**."}}"#,
            r#"{"type": "content_block_stop", "index": 1}"#,
            r#"{"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                "usage": {"output_tokens": 120, "output_tokens_details": {"thinking_tokens": 64}}}"#,
            r#"{"type": "message_stop"}"#,
        ])
        .unwrap();

        assert_eq!(settled.text(), "The greatest common divisor of 1071 and 462 is **21**.");
        assert_eq!(settled.thinking(), "I need to find the GCD of 1071 and 462.\n462 = 3 x 147 + 21");
        assert_eq!(
            settled.blocks[0],
            StreamedBlock::Thinking {
                thinking: "I need to find the GCD of 1071 and 462.\n462 = 3 x 147 + 21".to_owned(),
                signature: "EqQBCgIYAhIM1gbcDa9GJwZA2b3h".to_owned(),
            },
            "the signature is kept so the block can be replayed"
        );
        assert_eq!(settled.usage.output_tokens_details.thinking_tokens, 64);
    }

    /// A captured thinking block under `display: "omitted"`: signature only, no
    /// reasoning text. Empty thinking here is normal, not a truncation.
    #[test]
    fn a_captured_omitted_display_thinking_block_settles_with_only_its_signature() {
        let settled = settle_all(&[
            r#"{"type": "message_start", "message": {"id": "msg_o", "model": "claude-opus-5", "content": []}}"#,
            r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "thinking",
                "thinking": "", "signature": ""}}"#,
            r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta",
                "signature": "CAISyUkKcAgREAEYAipAqjMsxB4VuvEH9lSBF0/z"}}"#,
            r#"{"type": "content_block_stop", "index": 0}"#,
            r#"{"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}"#,
            r#"{"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta",
                "text": "There are infinitely many primes."}}"#,
            r#"{"type": "content_block_stop", "index": 1}"#,
            r#"{"type": "message_delta", "delta": {"stop_reason": "end_turn"}}"#,
            r#"{"type": "message_stop"}"#,
        ])
        .unwrap();

        assert_eq!(settled.thinking(), "", "`display: omitted` sends no thinking_delta");
        assert_eq!(settled.text(), "There are infinitely many primes.");
        assert!(matches!(&settled.blocks[0], StreamedBlock::Thinking { signature, .. } if !signature.is_empty()));
    }

    /// A mid-stream `error` ends the stream as its own outcome, keeping whatever
    /// arrived. Anthropic documents `overloaded_error` arriving exactly this way.
    #[test]
    fn a_mid_stream_error_settles_as_errored() {
        let settled = settle_all(&[
            r#"{"type": "message_start", "message": {"id": "msg_e", "model": "claude-opus-5", "content": [],
                "usage": {"input_tokens": 500, "cache_read_input_tokens": 400, "output_tokens": 1}}}"#,
            r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}"#,
            r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "parti"}}"#,
            r#"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#,
        ])
        .unwrap();

        assert_eq!(settled.stop_reason(), None, "an error names no stop reason");
        let error = settled.error().unwrap();
        assert_eq!(error.kind, Some(crate::values::ErrorType::Overloaded));
        assert_eq!(error.message, "Overloaded");
        assert_eq!(settled.text(), "parti", "what arrived is kept; the outcome says it failed");
        assert_eq!(settled.usage.cache_read_input_tokens, 400, "the request was still billed");
    }

    /// An event type this crate has never seen passes through the middle of a
    /// stream without disturbing it. This is the server-release case.
    #[test]
    fn an_unknown_event_interleaved_changes_nothing() {
        let mut settling = Settling::new();
        settling
            .consume_payload(
                r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}"#,
            )
            .unwrap();
        settling
            .consume_payload(
                r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "be"}}"#,
            )
            .unwrap();
        assert_eq!(
            settling.consume_payload(r#"{"type": "ping"}"#).unwrap(),
            StreamEvent::Unmodeled { kind: "ping".to_owned() }
        );
        assert_eq!(
            settling.consume_payload(r#"{"type": "message_prophecy", "oracle": true}"#).unwrap(),
            StreamEvent::Unmodeled { kind: "message_prophecy".to_owned() }
        );
        settling
            .consume_payload(
                r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "fore"}}"#,
            )
            .unwrap();
        settling.consume_payload(r#"{"type": "message_stop"}"#).unwrap();

        let settled = settling.settle().unwrap();
        assert_eq!(settled.text(), "before", "the unknown events contributed nothing and broke nothing");
        assert_eq!(settled.events, 6, "they were still counted");
    }

    /// A truncated stream that had already produced a whole tool call still does
    /// not settle. Finished structure is no substitute for a terminal event.
    #[test]
    fn a_truncated_stream_with_a_finished_tool_call_still_does_not_settle() {
        let mut settling = Settling::new();
        settling
            .consume_payload(
                r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use",
                    "id": "t1", "name": "f", "input": {}}}"#,
            )
            .unwrap();
        settling
            .consume_payload(
                r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta",
                    "partial_json": "{}"}}"#,
            )
            .unwrap();
        settling.consume_payload(r#"{"type": "content_block_stop", "index": 0}"#).unwrap();
        assert!(matches!(settling.settle(), Err(SettleError::Truncated { events: 3, .. })));
    }

    /// The first terminal event wins, so a duplicate cannot rewrite history.
    #[test]
    fn a_second_terminal_event_is_ignored() {
        let settled = settle_all(&[
            r#"{"type": "message_delta", "delta": {"stop_reason": "end_turn"}}"#,
            r#"{"type": "message_stop"}"#,
            r#"{"type": "error", "error": {"type": "api_error", "message": "too late"}}"#,
        ])
        .unwrap();
        assert_eq!(settled.stop_reason(), Some(StopReason::EndTurn));
        assert_eq!(settled.error(), None);
    }

    /// Blocks settle in wire-index order however the frames arrive.
    #[test]
    fn blocks_settle_in_index_order() {
        let mut settling = Settling::new();
        for (index, text) in [(2u32, "third"), (0, "first "), (1, "second ")] {
            settling
                .consume_payload(&format!(
                    r#"{{"type": "content_block_start", "index": {index}, "content_block":
                        {{"type": "text", "text": "{text}"}}}}"#
                ))
                .unwrap();
        }
        settling.consume_payload(r#"{"type": "message_stop"}"#).unwrap();
        assert_eq!(settling.settle().unwrap().text(), "first second third");
    }

    /// A delta for an index that was never announced is dropped rather than
    /// guessing at the block's kind.
    #[test]
    fn a_delta_for_an_unannounced_index_is_dropped() {
        let settled = settle_all(&[
            r#"{"type": "content_block_delta", "index": 7, "delta": {"type": "text_delta", "text": "orphan"}}"#,
            r#"{"type": "message_stop"}"#,
        ])
        .unwrap();
        assert!(settled.blocks.is_empty());
        assert_eq!(settled.text(), "");
        assert_eq!(settled.events, 2, "the orphan was counted, not applied");
    }

    /// A broken frame surfaces as a frame error and leaves the accumulator as it
    /// was.
    #[test]
    fn a_broken_frame_does_not_disturb_the_accumulator() {
        let mut settling = Settling::new();
        settling
            .consume_payload(
                r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "kept"}}"#,
            )
            .unwrap();
        assert!(matches!(settling.consume_payload("{ not json"), Err(SettleError::Frame(_))));
        assert!(matches!(
            settling.consume_payload(r#"{"type": "content_block_stop"}"#),
            Err(SettleError::Frame(FrameError::MissingField { field: "index" }))
        ));
        assert_eq!(settling.text_so_far(), "kept");
        assert_eq!(settling.event_count(), 1, "a frame that never decoded was never an event");
    }

    /// Usage is readable while the stream runs, and the cache counts are final
    /// from the opening frame because caching happens on input.
    #[test]
    fn usage_is_readable_mid_stream_and_its_cache_counts_are_already_final() {
        let mut settling = Settling::new();
        settling.consume_payload(CAPTURED_TEXT_STREAM[0]).unwrap();
        assert_eq!(settling.usage_so_far().cache_creation_input_tokens, 1_043);
        assert_eq!(settling.usage_so_far().output_tokens, 1, "generation has barely begun");

        for payload in &CAPTURED_TEXT_STREAM[1..] {
            settling.consume_payload(payload).unwrap();
        }
        assert_eq!(settling.usage_so_far().output_tokens, 45, "output rises");
        assert_eq!(settling.usage_so_far().cache_creation_input_tokens, 1_043, "cache counts do not");
    }

    /// A stream that stops without ever naming a reason still settles: the
    /// protocol completed, and the missing reason is reported as missing.
    #[test]
    fn a_stop_without_a_reason_settles_with_none() {
        let settled = settle_all(&[r#"{"type": "message_stop"}"#]).unwrap();
        assert_eq!(settled.outcome, Outcome::Stopped { reason: None, stop_sequence: None, refusal: None });
        assert_eq!(settled.id, "", "no message_start arrived to name it");
        assert_eq!(settled.usage, Usage::default());
    }

    /// `max_tokens` is a normal stop, not an error: the answer is cut short and
    /// the caller can still read it, which is what the reason is for.
    #[test]
    fn a_truncated_answer_settles_as_a_stop_with_its_reason() {
        let settled = settle_all(&[
            r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}"#,
            r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta",
                "text": "as far as I got"}}"#,
            r#"{"type": "message_delta", "delta": {"stop_reason": "max_tokens"},
                "usage": {"output_tokens": 1024}}"#,
            r#"{"type": "message_stop"}"#,
        ])
        .unwrap();
        assert_eq!(settled.stop_reason(), Some(StopReason::MaxTokens));
        assert_eq!(settled.error(), None, "running out of room is not an error");
        assert_eq!(settled.text(), "as far as I got");
        assert_eq!(settled.usage.output_tokens, 1_024);
    }

    #[test]
    fn a_matched_stop_sequence_is_reported_with_its_text() {
        let settled = settle_all(&[
            r#"{"type": "message_delta", "delta": {"stop_reason": "stop_sequence", "stop_sequence": "END"}}"#,
            r#"{"type": "message_stop"}"#,
        ])
        .unwrap();
        assert_eq!(
            settled.outcome,
            Outcome::Stopped {
                reason: Some(StopReason::StopSequence),
                stop_sequence: Some("END".to_owned()),
                refusal: None
            }
        );
    }

    /// A repeated `content_block_start` at the same index replaces the
    /// announcement rather than adding a block: the index is the identity.
    #[test]
    fn a_repeated_index_does_not_duplicate_a_block() {
        let settled = settle_all(&[
            r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "first"}}"#,
            r#"{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "again"}}"#,
            r#"{"type": "message_stop"}"#,
        ])
        .unwrap();
        assert_eq!(settled.blocks.len(), 1);
        assert_eq!(settled.text(), "again");
    }
}
