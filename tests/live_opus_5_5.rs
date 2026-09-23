//! First-party Claude Opus 5.5 request-shape checks.
//!
//! These tests consume credentials and capacity only when a non-empty `.key`
//! exists in the crate root. The positive cases use small output caps.

use anthropic::MESSAGES_PATH;
use anthropic::context::{Context, Opening, Tool};
use anthropic::request::{Model, Opus5_5Effort, OutputFormat, Request};
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

fn assert_ok(status: u16, body: &Value) {
    assert_eq!(status, 200, "expected 200, got {status}: {body}");
}

fn assert_400(status: u16, body: &Value) -> &str {
    assert_eq!(status, 400, "expected 400, got {status}: {body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "body: {body}");
    body["error"]["message"].as_str().unwrap_or("")
}

fn user_context(text: &str) -> Context {
    let mut context = Context::new(Opening::None);
    context.push_user_text(text);
    context
}

#[test]
fn live_ok_opus_5_5_low_effort() {
    let key = key_or_skip!();
    let context = user_context("Reply with the single word: ok");
    let model = Model::opus_5_5().with_effort(Opus5_5Effort::Low);
    let (status, body) = post(&Request::new(&context, model, 1).unwrap(), &key);
    assert_ok(status, &body);
    assert_eq!(body["model"], "claude-opus-5-5");
}

#[test]
fn live_ok_opus_5_5_structured_output() {
    let key = key_or_skip!();
    let mut context = Context::new(Opening::instruction("Extract the requested value."));
    context.push_user_text("The answer is ok.");
    let format = OutputFormat::json_schema(json!({
        "type": "object", "additionalProperties": false,
        "required": ["answer"], "properties": {"answer": {"type": "string"}},
    }));
    let request = Request::new(&context, Model::opus_5_5().with_effort(Opus5_5Effort::Low), 32)
        .unwrap()
        .with_output_format(format);
    let (status, body) = post(&request, &key);
    assert_ok(status, &body);
}

#[test]
fn live_ok_opus_5_5_custom_tools() {
    let key = key_or_skip!();
    let tool = Tool::new(
        "get_weather",
        json!({
            "type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"],
        }),
    );
    let mut context = Context::new(Opening::None).with_tools(vec![tool]);
    context.push_user_text("Reply ok without calling a tool.");
    let request = Request::new(&context, Model::opus_5_5().with_effort(Opus5_5Effort::Low), 1).unwrap();
    let (status, body) = post(&request, &key);
    assert_ok(status, &body);
}

#[test]
fn live_400_opus_5_5_thinking_disabled() {
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-opus-5-5", "max_tokens": 1,
        "messages": [{"role": "user", "content": "Reply ok"}],
        "thinking": {"type": "disabled"},
    });
    let (status, response) = post(&body, &key);
    let message = assert_400(status, &response);
    assert!(message.contains("thinking.type.disabled"), "message: {message}");
}

#[test]
fn live_400_opus_5_5_forced_tool_choice() {
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-opus-5-5", "max_tokens": 1,
        "messages": [{"role": "user", "content": "Reply ok"}],
        "tools": [{"name": "noop", "description": "No operation", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
    });
    let (status, response) = post(&body, &key);
    let message = assert_400(status, &response);
    assert!(message.contains("not supported"), "message: {message}");
}
