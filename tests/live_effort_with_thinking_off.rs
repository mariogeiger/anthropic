//! Per-message effort on a model whose thinking is off, measured first-party.
//!
//! With thinking off the documentation says effort cannot change
//! mid-conversation, so the crate refuses any effort message that differs from
//! the top-level effort. The API is measured refusing a subset of those, and
//! accepting a change the final user turn inherits, which is reported rather
//! than asserted. These tests consume credentials only when a non-empty `.key`
//! exists in the crate root, and cap output at one token.

use anthropic::context::{Context, Opening};
use anthropic::request::{Model, Opus5ThinkingOffEffort, Request};
use anthropic::system::PerMessageEffort;
use anthropic::{BetaFeature, MESSAGES_PATH};
use serde_json::{Value, json};

fn read_key() -> Option<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".key");
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_owned()).filter(|s| !s.is_empty())
}

macro_rules! key_or_skip {
    () => {
        match read_key() {
            Some(key) => key,
            None => {
                eprintln!("[skip] no .key file in crate root");
                return;
            }
        }
    };
}

fn post<T: serde::Serialize>(body: &T, key: &str) -> (u16, Value) {
    let url = format!("{}{}", anthropic::API_BASE, MESSAGES_PATH);
    let payload = serde_json::to_string(body).expect("serialize body");
    let request = ureq::post(&url)
        .set(anthropic::HEADER_API_KEY, key)
        .set(anthropic::HEADER_VERSION, anthropic::VERSION)
        .set(anthropic::HEADER_BETA, BetaFeature::MidConversationOutputConfig.as_str())
        .set("content-type", "application/json");
    match request.send_string(&payload) {
        Ok(response) => {
            let status = response.status();
            let text = response.into_string().expect("read body");
            (status, serde_json::from_str(&text).unwrap_or(Value::Null))
        }
        Err(ureq::Error::Status(status, response)) => {
            let text = response.into_string().unwrap_or_default();
            (status, serde_json::from_str(&text).unwrap_or(Value::Null))
        }
        Err(ureq::Error::Transport(error)) => panic!("transport error: {error}"),
    }
}

fn assert_400(status: u16, body: &Value) -> &str {
    assert_eq!(status, 400, "expected 400, got {status}: {body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "body: {body}");
    body["error"]["message"].as_str().unwrap_or("")
}

fn effort(level: &str) -> Value {
    json!({"role": "system", "content": [], "output_config": {"effort": level}})
}

#[test]
fn live_400_opus_5_thinking_off_rejects_per_message_xhigh() {
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-opus-5", "max_tokens": 1,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "high"},
        "messages": [effort("xhigh"), {"role": "user", "content": "hi"}],
    });
    let (status, response) = post(&body, &key);
    let message = assert_400(status, &response);
    assert!(message.contains("not supported when thinking is disabled"), "{message}");
}

#[test]
fn live_400_opus_5_thinking_off_rejects_an_effort_change_at_the_final_turn() {
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-opus-5", "max_tokens": 1,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
        "messages": [
            {"role": "user", "content": "one"}, {"role": "assistant", "content": "ok"},
            effort("medium"),
            {"role": "user", "content": "two"}, {"role": "assistant", "content": "ok"},
            effort("low"),
            {"role": "user", "content": "three"},
        ],
    });
    let (status, response) = post(&body, &key);
    let message = assert_400(status, &response);
    assert!(message.starts_with("messages.5: "), "{message}");
    assert!(message.contains("effort cannot change when thinking is disabled"), "{message}");
}

#[test]
fn live_ok_opus_5_thinking_off_accepts_an_effort_message_restating_the_level() {
    let key = key_or_skip!();
    let mut context = Context::new(Opening::None);
    context.push_user_text("one");
    context.push_assistant_text("ok");
    context.push_effort(PerMessageEffort::Low);
    context.push_user_text("Reply ok");
    let model = Model::opus_5().with_thinking_off(Opus5ThinkingOffEffort::Low);
    let request = Request::new(&context, model, 1).unwrap();
    assert_eq!(request.required_beta_features().collect::<Vec<_>>(), vec![BetaFeature::MidConversationOutputConfig]);
    let (status, body) = post(&request, &key);
    assert_eq!(status, 200, "expected 200, got {status}: {body}");
}

/// The documented rule refuses this history and the crate does too; the API
/// answered 200 on 2026-09-29. Reported, not asserted, so a change in either
/// direction is visible without a false failure.
#[test]
fn live_report_opus_5_thinking_off_on_a_change_the_final_turn_inherits() {
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-opus-5", "max_tokens": 1,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
        "messages": [
            {"role": "user", "content": "one"}, {"role": "assistant", "content": "ok"},
            effort("medium"),
            {"role": "user", "content": "two"}, {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "Reply ok"},
        ],
    });
    let (status, response) = post(&body, &key);
    eprintln!("[report] an inherited effort change with thinking off: {status} {}", response["error"]["message"]);
}
