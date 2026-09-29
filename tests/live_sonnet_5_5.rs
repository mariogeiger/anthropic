//! First-party Claude Sonnet 5.5 request-shape checks.
//!
//! The `live_ok_*` cases send bodies the crate produces; the `live_400_*` cases
//! send raw bodies the crate cannot produce and read how the API refuses them.
//! They consume credentials only when a non-empty `.key` exists in the crate
//! root, and cap output at one token. First exercised on 2026-09-29.

use anthropic::ThinkingDisplayWithUpdates;
use anthropic::context::{Context, Opening};
use anthropic::request::{Model, Request, Sonnet5_5BetweenToolsEffort, Sonnet5_5Effort};
use anthropic::system::PerMessageEffort;
use anthropic::{BetaFeature, MESSAGES_PATH, PrefixMismatchBehavior};
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

fn post<T: serde::Serialize>(body: &T, key: &str, beta: &[BetaFeature]) -> (u16, Value) {
    let url = format!("{}{}", anthropic::API_BASE, MESSAGES_PATH);
    let payload = serde_json::to_string(body).expect("serialize body");
    let mut request = ureq::post(&url)
        .set(anthropic::HEADER_API_KEY, key)
        .set(anthropic::HEADER_VERSION, anthropic::VERSION)
        .set("content-type", "application/json");
    if !beta.is_empty() {
        let header = beta.iter().map(|feature| feature.as_str()).collect::<Vec<_>>().join(",");
        request = request.set(anthropic::HEADER_BETA, &header);
    }
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

fn post_request(request: &Request<'_>, key: &str) -> (u16, Value) {
    post(request, key, &request.required_beta_features().collect::<Vec<_>>())
}

fn assert_ok(status: u16, body: &Value) {
    assert_eq!(status, 200, "expected 200, got {status}: {body}");
    assert_eq!(body["model"], "claude-sonnet-5-5");
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

fn raw(fields: Value) -> Value {
    let mut body = json!({
        "model": "claude-sonnet-5-5", "max_tokens": 1,
        "messages": [{"role": "user", "content": "Reply ok"}],
    });
    body.as_object_mut().unwrap().extend(fields.as_object().unwrap().clone());
    body
}

#[test]
fn live_ok_sonnet_5_5_adaptive_max_effort() {
    let key = key_or_skip!();
    let context = user_context("Reply with the single word: ok");
    let model = Model::sonnet_5_5().with_effort(Sonnet5_5Effort::Max);
    let (status, body) = post_request(&Request::new(&context, model, 1).unwrap(), &key);
    assert_ok(status, &body);
}

#[test]
fn live_ok_sonnet_5_5_between_tools_at_each_accepted_effort() {
    let key = key_or_skip!();
    let context = user_context("Reply with the single word: ok");
    for effort in
        [Sonnet5_5BetweenToolsEffort::Low, Sonnet5_5BetweenToolsEffort::Medium, Sonnet5_5BetweenToolsEffort::High]
    {
        let model = Model::sonnet_5_5().with_thinking_between_tools(effort);
        let (status, body) = post_request(&Request::new(&context, model, 1).unwrap(), &key);
        assert_ok(status, &body);
    }
}

#[test]
fn live_ok_sonnet_5_5_updates_display_and_binding_carry_their_inferred_headers() {
    let key = key_or_skip!();
    let context = user_context("Reply with the single word: ok");
    let model = Model::sonnet_5_5().with_adaptive_thinking(ThinkingDisplayWithUpdates::Updates);
    let request = Request::new(&context, model, 1)
        .unwrap()
        .with_prefix_mismatch_behavior(PrefixMismatchBehavior::DropBlock)
        .unwrap();
    let (status, body) = post_request(&request, &key);
    assert_ok(status, &body);
    assert_eq!(body["input_transformations"], json!([]));
}

#[test]
fn live_ok_sonnet_5_5_between_tools_accepts_a_restated_effort_and_a_system_message() {
    let key = key_or_skip!();
    let mut context = user_context("one");
    context.push_assistant_text("ok");
    context.push_effort(PerMessageEffort::Low);
    context.push_user_text("Reply ok");
    context.push_system_text("Answer in one word.").unwrap();
    let model = Model::sonnet_5_5().with_thinking_between_tools(Sonnet5_5BetweenToolsEffort::Low);
    let (status, body) = post_request(&Request::new(&context, model, 1).unwrap(), &key);
    assert_ok(status, &body);
}

#[test]
fn live_400_sonnet_5_5_disabled_thinking_points_to_between_tools() {
    let key = key_or_skip!();
    let (status, response) = post(&raw(json!({"thinking": {"type": "disabled"}})), &key, &[]);
    let message = assert_400(status, &response);
    assert!(message.contains(r#""thinking": {"type": "between_tools"}"#), "{message}");
}

#[test]
fn live_400_sonnet_5_5_legacy_thinking_budget() {
    let key = key_or_skip!();
    let body = raw(json!({"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 1024}}));
    let (status, response) = post(&body, &key, &[]);
    let message = assert_400(status, &response);
    assert!(message.contains("thinking.type.enabled"), "{message}");
}

#[test]
fn live_400_sonnet_5_5_between_tools_refuses_xhigh_and_any_other_field() {
    let key = key_or_skip!();
    let xhigh = raw(json!({"thinking": {"type": "between_tools"}, "output_config": {"effort": "xhigh"}}));
    let (status, response) = post(&xhigh, &key, &[]);
    let message = assert_400(status, &response);
    assert!(message.contains("not supported when thinking is disabled"), "{message}");
    let display = raw(json!({"thinking": {"type": "between_tools", "display": "summarized"}}));
    let (status, response) = post(&display, &key, &[]);
    let message = assert_400(status, &response);
    assert!(message.contains("Extra inputs are not permitted"), "{message}");
}

#[test]
fn live_400_sonnet_5_5_between_tools_refuses_an_effort_change() {
    let key = key_or_skip!();
    let body = raw(json!({
        "thinking": {"type": "between_tools"},
        "output_config": {"effort": "low"},
        "messages": [
            {"role": "user", "content": "one"}, {"role": "assistant", "content": "ok"},
            {"role": "system", "content": [], "output_config": {"effort": "medium"}},
            {"role": "user", "content": "Reply ok"},
        ],
    }));
    let (status, response) = post(&body, &key, &[BetaFeature::MidConversationOutputConfig]);
    let message = assert_400(status, &response);
    assert!(message.contains("effort cannot change when thinking is disabled"), "{message}");
}

#[test]
fn live_400_sonnet_5_5_temperature_forced_tool_choice_and_prefill() {
    let key = key_or_skip!();
    let (status, response) = post(&raw(json!({"temperature": 0.5})), &key, &[]);
    assert!(assert_400(status, &response).contains("deprecated"));
    let forced = raw(json!({
        "tools": [{"name": "noop", "description": "No operation", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
    }));
    let (status, response) = post(&forced, &key, &[]);
    assert!(assert_400(status, &response).contains(r#"type "tool" and "any" are not supported"#));
    let prefill = raw(json!({"messages": [
        {"role": "user", "content": "Reply ok"}, {"role": "assistant", "content": "o"},
    ]}));
    let (status, response) = post(&prefill, &key, &[]);
    assert!(assert_400(status, &response).contains("prefill"));
}
