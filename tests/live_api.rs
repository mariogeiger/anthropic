//! Live integration tests against the real Messages API.
//!
//! These confront the crate's design assumptions with the actual API, using a
//! key read from a `.key` file in the crate root (git-ignored). They are
//! **gated on that file**: with no `.key`, every test prints `[skip]` and
//! passes, so `cargo test` stays green in CI and for anyone without a key.
//!
//! They cost tokens and need network. Run them deliberately:
//!   cargo test --test live_api -- --nocapture
//!
//! Two kinds of test:
//!   * `live_ok_*`  — a body the crate *produces* is accepted (HTTP 200).
//!   * `live_400_*` — a combo the crate makes *unrepresentable* really is a
//!                    400 at the API, documenting why the type system forbids it
//!                    (CLAUDE.md §2). These post raw JSON the crate can't emit.
//!
//! Every result below was observed live on 2026-05-29.

use anthropic::context::{CacheSlot, Context};
use anthropic::request::{
    CountRequest, Model, ModelId, Opus4_8Effort, Request, Sonnet4_6Effort, Temperature,
};
use anthropic::{CacheTtl, MESSAGES_PATH, ThinkingDisplay};
use serde_json::{Value, json};

// ── Harness ──────────────────────────────────────────────────────────────────

fn read_key() -> Option<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".key");
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_owned()).filter(|s| !s.is_empty())
}

/// Bind the API key or bail out of the test as a skip.
macro_rules! key_or_skip {
    () => {
        match read_key() {
            Some(k) => k,
            None => {
                eprintln!("[skip] {}: no .key file in crate root", function_name());
                return;
            }
        }
    };
}

fn function_name() -> &'static str {
    // Best-effort label; not load-bearing.
    std::thread::current().name().unwrap_or("test").to_string().leak()
}

/// POST any serializable body to `path`, returning `(status, parsed_body)`.
/// Non-2xx is captured rather than thrown, so negative tests can inspect it.
fn post<T: serde::Serialize>(path: &str, body: &T, key: &str) -> (u16, Value) {
    let url = format!("{}{}", anthropic::API_BASE, path);
    let payload = serde_json::to_string(body).expect("serialize body");
    let result = ureq::post(&url)
        .set(anthropic::HEADER_API_KEY, key)
        .set(anthropic::HEADER_VERSION, anthropic::VERSION)
        .set("content-type", "application/json")
        .send_string(&payload);
    match result {
        Ok(resp) => {
            let code = resp.status();
            let text = resp.into_string().expect("read body");
            (code, serde_json::from_str(&text).unwrap_or(Value::Null))
        }
        Err(ureq::Error::Status(code, resp)) => {
            let text = resp.into_string().unwrap_or_default();
            (code, serde_json::from_str(&text).unwrap_or(Value::Null))
        }
        Err(e) => panic!("transport error: {e}"),
    }
}

fn msg(path: &str, body: &Value, key: &str) -> (u16, Value) {
    post(path, body, key)
}

/// Assert a 200 and surface the error body otherwise.
fn assert_ok(code: u16, body: &Value) {
    assert_eq!(code, 200, "expected 200, got {code}: {body}");
}

/// Assert a 400 `invalid_request_error` and return the server message.
fn assert_400(code: u16, body: &Value) -> String {
    assert_eq!(code, 400, "expected 400, got {code}: {body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "body: {body}");
    body["error"]["message"].as_str().unwrap_or("").to_owned()
}

fn user_ctx(text: &str) -> Context {
    let mut c = Context::new();
    c.push_user_text(text);
    c
}

// ── Positive: bodies the crate produces are accepted ──────────────────────────

#[test]
fn live_ok_opus_4_8_default() {
    let key = key_or_skip!();
    let ctx = user_ctx("Reply with the single word: ok");
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, Model::opus_4_8(), 16), &key);
    assert_ok(code, &body);
    assert_eq!(body["model"], "claude-opus-4-8");
}

#[test]
fn live_ok_opus_4_8_adaptive_xhigh() {
    // `xhigh` is absent from the Models-API capability tree but IS accepted on
    // Opus — the crate is right to expose `Opus4_8Effort::Xhigh`.
    let key = key_or_skip!();
    let ctx = user_ctx("Think briefly, then reply: ok");
    let model = Model::opus_4_8()
        .with_adaptive_thinking(ThinkingDisplay::Summarized)
        .with_effort(Opus4_8Effort::Xhigh);
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, model, 64), &key);
    assert_ok(code, &body);
}

#[test]
fn live_ok_sonnet_4_6_temperature() {
    let key = key_or_skip!();
    let ctx = user_ctx("Reply with the single word: ok");
    let model = Model::sonnet_4_6().with_temperature(Temperature::new(0.3).unwrap());
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, model, 16), &key);
    assert_ok(code, &body);
    assert_eq!(body["model"], "claude-sonnet-4-6");
}

#[test]
fn live_ok_sonnet_4_6_adaptive_max_effort() {
    // `max` on Sonnet 4.6 is valid (Opus-tier-only was a stale claim) — the
    // crate correctly keeps `Sonnet4_6Effort::Max`.
    let key = key_or_skip!();
    let ctx = user_ctx("Think briefly, then reply: ok");
    let model =
        Model::sonnet_4_6().with_adaptive_thinking(ThinkingDisplay::Summarized).with_effort(Sonnet4_6Effort::Max);
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, model, 64), &key);
    assert_ok(code, &body);
}

#[test]
fn live_ok_haiku_4_5_temperature() {
    let key = key_or_skip!();
    let ctx = user_ctx("Reply with the single word: ok");
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, Model::haiku_4_5(), 16), &key);
    assert_ok(code, &body);
    assert_eq!(body["model"], "claude-haiku-4-5-20251001");
}

#[test]
fn live_ok_haiku_4_5_legacy_thinking() {
    // Haiku 4.5 accepts legacy `{type:"enabled",budget_tokens}` (adaptive 400s);
    // budget must be < max_tokens.
    let key = key_or_skip!();
    let ctx = user_ctx("Think, then reply: ok");
    let model = Model::haiku_4_5().with_thinking(1024);
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, model, 1536), &key);
    assert_ok(code, &body);
}

#[test]
fn live_ok_count_tokens() {
    let key = key_or_skip!();
    let ctx = Context::new().with_system("You are helpful.");
    let mut ctx = ctx;
    ctx.push_user_text("How many tokens is this?");
    let (code, body) = post(anthropic::COUNT_TOKENS_PATH, &CountRequest::new(&ctx, ModelId::Opus4_8), &key);
    assert_ok(code, &body);
    assert!(body["input_tokens"].as_u64().unwrap_or(0) > 0, "body: {body}");
}

#[test]
fn live_ok_prompt_cache_creation() {
    // A cached system prompt over the 4096-token Opus minimum: confirms the
    // crate's `SystemPrompt` block-array wire shape + `cache_control` actually
    // engage caching (usage reports cached tokens).
    let key = key_or_skip!();
    let big = "The quick brown fox jumps over the lazy dog. ".repeat(700); // ~7k tokens
    let ctx = Context::new()
        .with_system_cached(CacheSlot::S0, big, CacheTtl::FiveMinutes)
        .expect("anchor system cache");
    let mut ctx = ctx;
    ctx.push_user_text("Reply: ok");
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, Model::opus_4_8(), 16), &key);
    assert_ok(code, &body);
    let created = body["usage"]["cache_creation_input_tokens"].as_u64().unwrap_or(0);
    let read = body["usage"]["cache_read_input_tokens"].as_u64().unwrap_or(0);
    assert!(created + read > 0, "expected cache activity, usage: {}", body["usage"]);
}

// ── Negative: combos the crate forbids really do 400 (CLAUDE.md §2) ────────────

#[test]
fn live_400_opus_temperature() {
    // Opus rejects sampling params — why `Opus4_8` carries no temperature.
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-opus-4-8", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5,
    });
    let (code, resp) = msg(MESSAGES_PATH, &body, &key);
    assert_400(code, &resp);
}

#[test]
fn live_400_opus_legacy_thinking() {
    // Opus rejects `{type:"enabled",budget_tokens}` — why `Opus4_8Thinking` is
    // Off|Adaptive with no legacy variant.
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-opus-4-8", "max_tokens": 2048,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    });
    let (code, resp) = msg(MESSAGES_PATH, &body, &key);
    assert_400(code, &resp);
}

#[test]
fn live_400_sonnet_xhigh() {
    // `xhigh` is Opus-only — why `Sonnet4_6Effort` has no `Xhigh`.
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-sonnet-4-6", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"effort": "xhigh"},
    });
    let (code, resp) = msg(MESSAGES_PATH, &body, &key);
    let m = assert_400(code, &resp);
    assert!(m.contains("xhigh"), "unexpected message: {m}");
}

#[test]
fn live_400_haiku_adaptive_thinking() {
    // Haiku rejects adaptive thinking — why `Haiku4_5Thinking` is Off|Enabled.
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-haiku-4-5", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "adaptive"},
    });
    let (code, resp) = msg(MESSAGES_PATH, &body, &key);
    assert_400(code, &resp);
}

#[test]
fn live_400_haiku_effort() {
    // Haiku rejects `output_config.effort` — why `Haiku4_5` emits no output_config.
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-haiku-4-5", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"effort": "high"},
    });
    let (code, resp) = msg(MESSAGES_PATH, &body, &key);
    assert_400(code, &resp);
}

#[test]
fn live_400_haiku_budget_ge_max() {
    // budget_tokens must be < max_tokens. The crate CANNOT prevent this: thinking
    // lives on the model type, max_tokens on Request (CLAUDE.md §4 split). This
    // test pins the gap — a caller can build a 400 the type system won't catch.
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-haiku-4-5", "max_tokens": 1000,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 2000},
    });
    let (code, resp) = msg(MESSAGES_PATH, &body, &key);
    assert_400(code, &resp);
}

#[test]
fn live_400_ttl_1h_after_5m() {
    // Confirms the API's TTL-ordering rule and the exact flow order the crate
    // encodes in `flow_key` (tools, system, messages). The crate enforces this
    // pre-commit in `roll_cache`; here we bypass it with raw JSON to prove the
    // rule is real: 5m on system, 1h on a later message block.
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-opus-4-8", "max_tokens": 16,
        "system": [{"type": "text", "text": "sys",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"}}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi",
             "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}],
    });
    let (code, resp) = msg(MESSAGES_PATH, &body, &key);
    let m = assert_400(code, &resp);
    assert!(m.contains("ttl"), "unexpected message: {m}");
}
