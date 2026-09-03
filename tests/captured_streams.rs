//! Decoding real captured traffic.
//!
//! Every file in `tests/captured/` is a verbatim body from a live Messages
//! endpoint, saved from `aws/anthropic/bedrock-claude-opus-5` via
//! the NVIDIA inference gateway. Nothing here is invented: the frame order, the
//! field spellings, the fields the documentation does not mention, and the empty
//! thinking block that `display: "omitted"` produces are all as they arrived. All
//! but `citations.sse` were saved on 2026-08-29; that one on 2026-08-30.
//! `fable-5-1-binding.sse` and `fable-5-1-thinking-dropped.json` are first-party
//! Fable 5.1 responses captured on 2026-09-03. The first reports a clean check;
//! the second reports a signed thinking block dropped after its system prefix
//! changed.
//!
//! Two files were trimmed, and only by dropping repetition: long runs of
//! `thinking_delta` and `text_delta` frames were cut to the first few per block,
//! and signature values to their first 40 characters. No frame kind, field, or
//! ordering was changed, so what is exercised is still the wire.
//!
//! The unit tests in `src/` cover each frame kind in isolation. These cover what
//! only a whole body can: that a stream of hundreds of frames settles into one
//! message, and that the failure modes are the ones the types promise.

use anthropic::content::StreamedBlock;
use anthropic::document::Citation;
use anthropic::frame::data_payload;
use anthropic::response::Response;
use anthropic::settle::{Outcome, SettleError, Settling};
use anthropic::stream::StreamEvent;
use anthropic::values::StopReason;

/// Feeds every `data:` line of a captured body into an accumulator.
fn accumulate(body: &str) -> Settling {
    let mut settling = Settling::new();
    for line in body.lines() {
        if let Some(payload) = data_payload(line) {
            settling.consume_payload(payload).expect("every captured frame must decode");
        }
    }
    settling
}

const TOOL_USE: &str = include_str!("captured/tool-use.sse");
const THINKING_SUMMARIZED: &str = include_str!("captured/thinking-summarized.sse");
const THINKING_OMITTED: &str = include_str!("captured/thinking-omitted.sse");
const CACHE_WRITE: &str = include_str!("captured/cache-write.sse");
const CACHE_READ: &str = include_str!("captured/cache-read.sse");
const CITATIONS: &str = include_str!("captured/citations.sse");
const FABLE_BINDING: &str = include_str!("captured/fable-5-1-binding.sse");
const FABLE_DROPPED: &str = include_str!("captured/fable-5-1-thinking-dropped.json");
const RESPONSE: &str = include_str!("captured/response.json");

#[test]
fn a_first_party_binding_stream_reports_that_no_input_was_dropped() {
    let settled = accumulate(FABLE_BINDING).settle().unwrap();
    assert_eq!(settled.model, "claude-fable-5-1");
    assert_eq!(settled.stop_reason(), Some(StopReason::MaxTokens));
    assert_eq!(settled.input_transformations, Some(Vec::new()));
}

#[test]
fn a_first_party_response_reports_the_exact_thinking_block_it_dropped() {
    let response = Response::decode(FABLE_DROPPED).unwrap();
    let transformations = response.input_transformations.unwrap();
    assert_eq!(transformations.len(), 1);
    assert_eq!(transformations[0].path(), Some("messages.1.content.0"));
    assert_eq!(transformations[0].reason(), Some(anthropic::ThinkingDropReason::PrefixBindingMismatch));
}

/// A cited answer, settled from the real stream: a plain-text document with
/// citations enabled, and the model grounding two of its three text blocks.
///
/// Note the frame order. The `citations_delta` for a block arrives *before* the
/// `text_delta`s it grounds, which is why a citation is appended to its block
/// rather than attached to text already present — accumulating in arrival order is
/// what makes that a non-issue.
#[test]
fn a_captured_cited_stream_grounds_its_text_in_the_document() {
    let settled = accumulate(CITATIONS).settle().unwrap();

    assert_eq!(settled.stop_reason(), Some(StopReason::EndTurn));
    assert!(settled.text().contains("At night the sky is black."));

    // Four text blocks: a cited claim, an uncited aside, a second cited claim, and
    // its uncited tail. Only the grounded ones carry citations, and the API
    // announces exactly those with a `citations` array.
    let cited: Vec<_> = settled
        .blocks
        .iter()
        .filter_map(|block| match block {
            StreamedBlock::Text { citations, .. } if !citations.is_empty() => Some(citations),
            _ => None,
        })
        .collect();
    assert_eq!(settled.blocks.len(), 4, "the model split its answer into four text blocks");
    assert_eq!(cited.len(), 2, "two of the four are grounded");

    let Citation::CharLocation { cited_text, document_title, start_char_index, end_char_index, document_index } =
        &cited[0][0]
    else {
        panic!("a plain-text document is cited by character range")
    };
    assert_eq!(cited_text, "At night it is black. ");
    assert_eq!(document_title.as_deref(), Some("Field guide"));
    assert_eq!(*document_index, 0);
    assert_eq!((*start_char_index, *end_char_index), (32, 54), "the exact characters in the document");

    // The uncited block is a text block like any other; it simply cites nothing.
    let uncited = settled
        .blocks
        .iter()
        .filter(|block| matches!(block, StreamedBlock::Text { citations, .. } if citations.is_empty()))
        .count();
    assert_eq!(uncited, 2, "the aside and the closing tail cite nothing");

    // The second citation grounds the daytime claim, in the same document.
    assert_eq!(cited[1][0].cited_text(), Some("The sky is blue during the day. "));
    assert_eq!(cited[1][0].kind(), "char_location");
}

/// A forced tool call, settled from the real stream. The input arrived in six
/// fragments, none of which is valid JSON on its own.
#[test]
fn a_captured_tool_use_stream_settles_into_one_call() {
    let settled = accumulate(TOOL_USE).settle().unwrap();

    assert_eq!(settled.stop_reason(), Some(StopReason::ToolUse));
    assert_eq!(settled.model, "claude-opus-5");
    assert!(settled.id.starts_with("msg_bdrk_"));
    assert_eq!(settled.text(), "", "the model called a tool instead of answering");

    let calls: Vec<_> = settled.tool_calls().collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert!(calls[0].id.starts_with("toolu_bdrk_"));
    assert_eq!(calls[0].input.as_str(), r#"{"location": "San Francisco"}"#);
    assert_eq!(
        calls[0].input.decode().unwrap(),
        serde_json::json!({"location": "San Francisco"}),
        "the fragments reassembled into whole JSON"
    );
    assert_eq!(settled.usage.output_tokens, 34);
}

/// Reasoning under `display: "summarized"`: thinking text plus a signature, and
/// no confusion between reasoning and answer.
///
/// This run exhausted its `max_tokens` while still thinking, which is why it is
/// worth keeping: a budget that runs out is a *normal stop*, not an error. The
/// stream completed, the reasoning is real, and the stop reason is what tells the
/// caller the answer never arrived — see [`Outcome::Stopped`].
#[test]
fn a_captured_summarized_thinking_stream_settles_even_though_the_budget_ran_out() {
    let settled = accumulate(THINKING_SUMMARIZED).settle().unwrap();

    assert_eq!(settled.stop_reason(), Some(StopReason::MaxTokens));
    assert_eq!(settled.error(), None, "running out of room is not an error");
    assert_eq!(settled.text(), "", "the budget went entirely on thinking");
    let thinking = settled.thinking();
    assert!(!thinking.is_empty(), "summarized display sends reasoning text");
    assert!(thinking.starts_with("I need"), "{thinking}");
    let StreamedBlock::Thinking { signature, .. } = &settled.blocks[0] else {
        panic!("the first block should be thinking")
    };
    assert!(!signature.is_empty(), "the signature makes the block replayable");
    assert!(settled.usage.output_tokens > 0);
}

/// Reasoning under `display: "omitted"`: the thinking block opens, takes one
/// signature, and closes with no reasoning text at all. Empty thinking beside a
/// real signature is normal here, which is exactly why the two fields are
/// separate.
#[test]
fn a_captured_omitted_thinking_stream_has_a_signed_but_empty_thinking_block() {
    let settled = accumulate(THINKING_OMITTED).settle().unwrap();

    assert_eq!(settled.thinking(), "", "omitted display sends no thinking_delta");
    let StreamedBlock::Thinking { thinking, signature } = &settled.blocks[0] else {
        panic!("the first block should be thinking")
    };
    assert!(thinking.is_empty(), "the block is empty by configuration, not by truncation");
    assert!(!signature.is_empty(), "and still signed, so it can be replayed");

    // The answer that followed it is real, and cut short by the token budget.
    assert!(settled.text().starts_with("# There Are Infinitely Many Primes"), "{}", settled.text());
    assert_eq!(settled.stop_reason(), Some(StopReason::MaxTokens));
    assert_eq!(settled.error(), None);
}

/// The pair that proves caching works, from two runs of one identical prompt.
/// The first wrote 1043 tokens; the second read the same 1043 back.
#[test]
fn a_captured_cache_write_and_read_pair_reports_its_hit_rate() {
    let write = accumulate(CACHE_WRITE).settle().unwrap();
    assert_eq!(write.usage.cache_creation_input_tokens, 1_043);
    assert_eq!(write.usage.cache_creation.ephemeral_5m_input_tokens, 1_043);
    assert_eq!(write.usage.cache_read_input_tokens, 0);
    assert!(write.usage.cache_creation_is_consistent(), "the TTL split sums to the total");
    assert_eq!(write.usage.cache_hit_rate(), Some(0.0), "a write is not yet a read");

    let read = accumulate(CACHE_READ).settle().unwrap();
    assert_eq!(read.usage.cache_read_input_tokens, 1_043, "the same prefix, read back");
    assert_eq!(read.usage.cache_creation_input_tokens, 0);
    assert_eq!(read.usage.total_input_tokens(), 1_079);
    assert!(read.usage.cache_hit_rate().unwrap() > 0.96);
}

/// This gateway's `message_stop` carries only `input_tokens` and `output_tokens`.
/// Merging by pointwise maximum means it cannot erase the cache counts that
/// arrived in `message_start` — which a last-writer-wins merge would have done.
#[test]
fn the_terminal_frame_does_not_erase_the_cache_counts() {
    let payloads: Vec<&str> = CACHE_WRITE.lines().filter_map(data_payload).collect();
    let last = StreamEvent::decode(payloads.last().unwrap()).unwrap();
    let StreamEvent::MessageStop { usage } = &last else { panic!("expected message_stop last") };
    assert_eq!(usage.cache_creation_input_tokens, 0, "the terminal frame reports no cache counts of its own");
    assert_eq!(usage.output_tokens, 45);

    let settled = accumulate(CACHE_WRITE).settle().unwrap();
    assert_eq!(settled.usage.cache_creation_input_tokens, 1_043, "kept from message_start");
    assert_eq!(settled.usage.output_tokens, 45, "and the larger output count still wins");
}

/// A non-streamed response body, on a cache hit.
#[test]
fn a_captured_response_body_decodes() {
    let response = Response::decode(RESPONSE).unwrap();
    assert_eq!(response.model, "aws/anthropic/bedrock-claude-opus-5");
    assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
    assert!(!response.text().is_empty());
    assert_eq!(response.usage.cache_read_input_tokens, 1_043);
    assert_eq!(response.usage.total_input_tokens(), 1_079);
}

/// Truncating a real stream anywhere before its `message_stop` must not settle —
/// not at the last frame, not after the `message_delta` that already reported
/// `end_turn`. This is the property the whole `settle` module exists for, checked
/// against every prefix of real traffic rather than one hand-made case.
#[test]
fn no_prefix_of_a_captured_stream_settles_before_its_terminal_frame() {
    for body in [TOOL_USE, THINKING_SUMMARIZED, THINKING_OMITTED, CACHE_WRITE, CACHE_READ, CITATIONS, FABLE_BINDING] {
        let payloads: Vec<&str> = body.lines().filter_map(data_payload).collect();
        for cut in 0..payloads.len() {
            let mut settling = Settling::new();
            for payload in &payloads[..cut] {
                settling.consume_payload(payload).unwrap();
            }
            assert!(!settling.is_terminated(), "a prefix of {cut} frame(s) claimed to be terminated");
            let error = settling.settle().unwrap_err();
            assert!(
                matches!(error, SettleError::Truncated { .. }),
                "a prefix of {cut} frame(s) settled anyway: {error}"
            );
        }
        // Only the whole stream settles.
        assert!(accumulate(body).settle().is_ok());
    }
}

/// Every prefix that reaches the `message_delta` reports that the stop reason had
/// already arrived — the exact case where trusting `stop_reason` instead of
/// `message_stop` would report a truncated answer as a whole one.
#[test]
fn a_truncation_after_the_stop_reason_says_so() {
    let payloads: Vec<&str> = TOOL_USE.lines().filter_map(data_payload).collect();
    let all_but_last = &payloads[..payloads.len() - 1];
    let mut settling = Settling::new();
    for payload in all_but_last {
        settling.consume_payload(payload).unwrap();
    }
    let error = settling.settle().unwrap_err();
    let SettleError::Truncated { had_stop_reason, events, .. } = error else { panic!("expected truncation") };
    assert!(had_stop_reason, "the model finished its turn and the stream still broke");
    assert_eq!(events, all_but_last.len());
}

/// An `error` frame spliced into real traffic ends the stream as an error while
/// keeping what had arrived. Anthropic documents `overloaded_error` arriving this
/// way mid-stream, where a non-streamed call would have returned HTTP 529.
#[test]
fn an_error_spliced_into_a_captured_stream_settles_as_errored() {
    let payloads: Vec<&str> = THINKING_OMITTED.lines().filter_map(data_payload).collect();
    let mut settling = Settling::new();
    for payload in payloads.iter().take(6) {
        settling.consume_payload(payload).unwrap();
    }
    let partial_text = settling.text_so_far();
    settling
        .consume_payload(r#"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#)
        .unwrap();

    let settled = settling.settle().unwrap();
    assert!(matches!(settled.outcome, Outcome::Errored { .. }));
    let error = settled.error().unwrap();
    assert_eq!(error.kind, Some(anthropic::values::ErrorType::Overloaded));
    assert_eq!(settled.stop_reason(), None, "an error names no stop reason");
    assert_eq!(settled.text(), partial_text, "what arrived is kept");
    assert!(!settled.blocks.is_empty(), "so are the blocks");
}

/// Unknown events interleaved through real traffic change nothing. This is the
/// server-release case: Anthropic's versioning policy permits new event types, so
/// a decoder that failed here would break on a deploy it never saw.
#[test]
fn unknown_events_interleaved_through_a_captured_stream_change_nothing() {
    let clean = accumulate(THINKING_OMITTED).settle().unwrap();

    let mut settling = Settling::new();
    let mut injected = 0;
    for payload in THINKING_OMITTED.lines().filter_map(data_payload) {
        let event = settling.consume_payload(r#"{"type": "message_astrology", "sign": "libra"}"#).unwrap();
        assert_eq!(event.kind(), "message_astrology");
        assert!(!event.is_terminal());
        injected += 1;
        settling.consume_payload(r#"{"type": "ping"}"#).unwrap();
        injected += 1;
        settling.consume_payload(payload).unwrap();
    }
    let noisy = settling.settle().unwrap();

    assert_eq!(noisy.text(), clean.text());
    assert_eq!(noisy.thinking(), clean.thinking());
    assert_eq!(noisy.blocks, clean.blocks);
    assert_eq!(noisy.usage, clean.usage);
    assert_eq!(noisy.stop_reason(), clean.stop_reason());
    assert_eq!(noisy.events, clean.events + injected, "the noise was counted and otherwise ignored");
}

/// Every frame of every captured body decodes, and none of them is an
/// `Unmodeled` event: the modeled set covers real traffic completely.
#[test]
fn every_captured_frame_decodes_into_a_modeled_event() {
    for body in [TOOL_USE, THINKING_SUMMARIZED, THINKING_OMITTED, CACHE_WRITE, CACHE_READ, CITATIONS, FABLE_BINDING] {
        for payload in body.lines().filter_map(data_payload) {
            let event = StreamEvent::decode(payload).expect(payload);
            assert!(
                !matches!(event, StreamEvent::Unmodeled { .. }),
                "captured traffic contained an unmodeled event: {}",
                event.kind()
            );
        }
    }
}

/// The `event:` lines name the same types the payloads do, so a caller may read
/// either. This crate reads the payload, which is why the redundancy is worth
/// checking once rather than trusting.
#[test]
fn the_sse_event_names_agree_with_the_payload_types() {
    for body in [TOOL_USE, THINKING_SUMMARIZED, THINKING_OMITTED, CACHE_WRITE, CACHE_READ, CITATIONS, FABLE_BINDING] {
        let mut named: Option<&str> = None;
        for line in body.lines() {
            if let Some(name) = line.strip_prefix("event: ") {
                named = Some(name.trim());
            } else if let Some(payload) = data_payload(line) {
                let event = StreamEvent::decode(payload).unwrap();
                assert_eq!(named.take(), Some(event.kind()), "the event: line disagreed with the payload");
            }
        }
    }
}
