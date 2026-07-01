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
//! Every result below was observed live on 2026-05-29, except the Fable 5 cases
//! (added 2026-06-18): Fable 5 is access-gated and returned 404 "not available"
//! on the test org, so those tests skipped rather than exercising 200/400. They
//! assert the documented behavior for orgs that do have access. The Sonnet 5 cases
//! (added 2026-07-01) encode the published GA behavior; Sonnet 5 is GA to all
//! customers, so they exercise 200/400 on any org with a `.key`.

use anthropic::context::{CacheSlot, Context};
use anthropic::request::{
    CountRequest, Fable5Effort, Model, ModelId, Opus4_8Effort, Request, Sonnet4_6Effort, Sonnet5Effort, Temperature,
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
/// Retries transport-level failures (e.g. connection resets) with backoff — a
/// real HTTP response, 4xx included, is the API's answer and returns at once.
fn post<T: serde::Serialize>(path: &str, body: &T, key: &str) -> (u16, Value) {
    let url = format!("{}{}", anthropic::API_BASE, path);
    let payload = serde_json::to_string(body).expect("serialize body");
    let mut last = String::new();
    for attempt in 0..5u64 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(400 * attempt));
        }
        match ureq::post(&url)
            .set(anthropic::HEADER_API_KEY, key)
            .set(anthropic::HEADER_VERSION, anthropic::VERSION)
            .set("content-type", "application/json")
            .send_string(&payload)
        {
            Ok(resp) => {
                let code = resp.status();
                let text = resp.into_string().expect("read body");
                return (code, serde_json::from_str(&text).unwrap_or(Value::Null));
            }
            // A real HTTP status (incl. 4xx/5xx) — the API answered; don't retry.
            Err(ureq::Error::Status(code, resp)) => {
                let text = resp.into_string().unwrap_or_default();
                return (code, serde_json::from_str(&text).unwrap_or(Value::Null));
            }
            // Connection reset / DNS / TLS — transient; retry.
            Err(ureq::Error::Transport(t)) => last = t.to_string(),
        }
    }
    panic!("transport error after retries: {last}");
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

/// Fable 5 is access-gated at the org level. Where it isn't provisioned the API
/// answers every Fable 5 request — valid or not — with `404 not_found_error`
/// ("not available"), which would mask the 200/400 a Fable 5 test means to
/// check. Treat that as a skip, the same way a missing `.key` is, so the suite
/// stays green on orgs without access.
fn fable5_unavailable(code: u16, body: &Value) -> bool {
    if code == 404 && body["error"]["type"] == "not_found_error" {
        eprintln!("[skip] {}: Claude Fable 5 not available on this org", function_name());
        return true;
    }
    false
}

// ── Positive: bodies the crate produces are accepted ──────────────────────────

#[test]
fn live_ok_opus_4_8_default() {
    let key = key_or_skip!();
    let ctx = user_ctx("Reply with the single word: ok");
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, Model::opus_4_8(), 16).unwrap(), &key);
    assert_ok(code, &body);
    assert_eq!(body["model"], "claude-opus-4-8");
}

#[test]
fn live_ok_opus_4_8_adaptive_xhigh() {
    // `xhigh` is absent from the Models-API capability tree but IS accepted on
    // Opus — the crate is right to expose `Opus4_8Effort::Xhigh`.
    let key = key_or_skip!();
    let ctx = user_ctx("Think briefly, then reply: ok");
    let model = Model::opus_4_8().with_adaptive_thinking(ThinkingDisplay::Summarized).with_effort(Opus4_8Effort::Xhigh);
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, model, 64).unwrap(), &key);
    assert_ok(code, &body);
}

#[test]
fn live_ok_fable_5_default() {
    // The crate's default Fable 5 body: always-on adaptive thinking (omitted
    // display), effort high, no sampling. Accepted (200) on orgs with access.
    let key = key_or_skip!();
    let ctx = user_ctx("Reply with the single word: ok");
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, Model::fable_5(), 16).unwrap(), &key);
    if fable5_unavailable(code, &body) {
        return;
    }
    assert_ok(code, &body);
    assert_eq!(body["model"], "claude-fable-5");
}

#[test]
fn live_ok_fable_5_summarized_xhigh() {
    // `xhigh` + summarized thinking — exercises `Fable5Effort::Xhigh` and the
    // visible-reasoning display.
    let key = key_or_skip!();
    let ctx = user_ctx("Think briefly, then reply: ok");
    let model = Model::fable_5().with_display(ThinkingDisplay::Summarized).with_effort(Fable5Effort::Xhigh);
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, model, 64).unwrap(), &key);
    if fable5_unavailable(code, &body) {
        return;
    }
    assert_ok(code, &body);
}

#[test]
fn live_ok_sonnet_4_6_temperature() {
    let key = key_or_skip!();
    let ctx = user_ctx("Reply with the single word: ok");
    let model = Model::sonnet_4_6().with_temperature(Temperature::new(0.3).unwrap());
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, model, 16).unwrap(), &key);
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
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, model, 64).unwrap(), &key);
    assert_ok(code, &body);
}

#[test]
fn live_ok_sonnet_5_default() {
    // The crate's default Sonnet 5 body: adaptive thinking on (omitted display),
    // effort high, no sampling.
    let key = key_or_skip!();
    let ctx = user_ctx("Reply with the single word: ok");
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, Model::sonnet_5(), 64).unwrap(), &key);
    assert_ok(code, &body);
    assert_eq!(body["model"], "claude-sonnet-5");
}

#[test]
fn live_ok_sonnet_5_thinking_off() {
    // Sonnet 5 *does* have a thinking-off state, reached via `{type:"disabled"}`
    // (contrast Fable 5, where disabled 400s). Confirms `Sonnet5Thinking::Disabled`
    // is accepted.
    let key = key_or_skip!();
    let ctx = user_ctx("Reply with the single word: ok");
    let (code, body) =
        post(MESSAGES_PATH, &Request::new(&ctx, Model::sonnet_5().with_thinking_off(), 16).unwrap(), &key);
    assert_ok(code, &body);
}

#[test]
fn live_ok_sonnet_5_xhigh() {
    // `xhigh` is accepted on Sonnet 5 — why `Sonnet5Effort` (unlike `Sonnet4_6Effort`)
    // exposes `Xhigh`.
    let key = key_or_skip!();
    let ctx = user_ctx("Think briefly, then reply: ok");
    let model = Model::sonnet_5().with_adaptive_thinking(ThinkingDisplay::Summarized).with_effort(Sonnet5Effort::Xhigh);
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, model, 64).unwrap(), &key);
    assert_ok(code, &body);
}

#[test]
fn live_ok_sonnet_5_stop_sequence() {
    // The crate's `stop_sequences` is honored on Sonnet 5: generation halts at the
    // sequence and the response reports `stop_reason: "stop_sequence"`. Thinking is
    // off so output is plain text and the sequence is hit deterministically.
    let key = key_or_skip!();
    let ctx = user_ctx("List the numbers 1 through 9 separated by single spaces, nothing else.");
    let model = Model::sonnet_5().with_thinking_off();
    let req = Request::new(&ctx, model, 64).unwrap().stop_sequences(vec!["5".into()]);
    let (code, body) = post(MESSAGES_PATH, &req, &key);
    assert_ok(code, &body);
    assert_eq!(body["stop_reason"], "stop_sequence", "body: {body}");
}

#[test]
fn live_ok_haiku_4_5_temperature() {
    let key = key_or_skip!();
    let ctx = user_ctx("Reply with the single word: ok");
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, Model::haiku_4_5(), 16).unwrap(), &key);
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
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, model, 1536).unwrap(), &key);
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
fn live_ok_count_tokens_sonnet_5_new_tokenizer() {
    // `ModelId::Sonnet5` resolves at the count endpoint, and confirms Sonnet 5 is a
    // genuinely distinct model, not a 4.6 alias: its new tokenizer yields more tokens
    // than Sonnet 4.6 for the same text (~30% more, per the launch docs).
    let key = key_or_skip!();
    let mut ctx = Context::new();
    ctx.push_user_text(
        "Tokenizers segment text into subword units; the same passage can map to \
         different token counts across model generations, which changes both cost \
         and how much text fits inside a fixed context window.",
    );
    let (c5, b5) = post(anthropic::COUNT_TOKENS_PATH, &CountRequest::new(&ctx, ModelId::Sonnet5), &key);
    assert_ok(c5, &b5);
    let (c46, b46) = post(anthropic::COUNT_TOKENS_PATH, &CountRequest::new(&ctx, ModelId::Sonnet4_6), &key);
    assert_ok(c46, &b46);
    let n5 = b5["input_tokens"].as_u64().unwrap_or(0);
    let n46 = b46["input_tokens"].as_u64().unwrap_or(0);
    assert!(n46 > 0 && n5 > n46, "expected Sonnet 5 to count more tokens than Sonnet 4.6, got s5={n5} s46={n46}");
}

#[test]
fn live_ok_prompt_cache_creation() {
    // A cached system prompt well over Opus 4.8's 1,024-token cache minimum:
    // confirms the crate's `SystemPrompt` block-array wire shape + `cache_control`
    // actually engage caching (usage reports cached tokens).
    let key = key_or_skip!();
    let big = "The quick brown fox jumps over the lazy dog. ".repeat(100); // ~1.8k tokens, over the 1,024 min
    let ctx =
        Context::new().with_system_cached(CacheSlot::S0, big, CacheTtl::FiveMinutes).expect("anchor system cache");
    let mut ctx = ctx;
    ctx.push_user_text("Reply: ok");
    let (code, body) = post(MESSAGES_PATH, &Request::new(&ctx, Model::opus_4_8(), 16).unwrap(), &key);
    assert_ok(code, &body);
    let created = body["usage"]["cache_creation_input_tokens"].as_u64().unwrap_or(0);
    let read = body["usage"]["cache_read_input_tokens"].as_u64().unwrap_or(0);
    assert!(created + read > 0, "expected cache activity, usage: {}", body["usage"]);
}

#[test]
fn live_ok_prompt_cache_read() {
    // §1's core promise plus the minimum-length boundary, end-to-end on Sonnet 5:
    //  * a prefix *over* the model's minimum is written on the 1st request and
    //    *re-read* on a 2nd identical-prefix request (cache_read_input_tokens > 0);
    //  * a prefix *below* `ModelId::min_cacheable_prefix_tokens` is a silent no-op —
    //    still marked with cache_control, yet nothing is written or read, and no 400.
    // The same `Context` is reused verbatim per case, so prefix bytes are identical
    // by construction; reads are live within the TTL.
    let key = key_or_skip!();
    let min = ModelId::Sonnet5.min_cacheable_prefix_tokens();

    // Over the minimum: caches on write, reads on the 2nd identical request.
    let big = "The quick brown fox jumps over the lazy dog. ".repeat(100); // ~1.8k tokens, over the 1,024 min
    let mut ctx =
        Context::new().with_system_cached(CacheSlot::S0, big, CacheTtl::FiveMinutes).expect("anchor system cache");
    ctx.push_user_text("Reply: ok");
    let (code1, body1) = post(MESSAGES_PATH, &Request::new(&ctx, Model::sonnet_5(), 64).unwrap(), &key);
    assert_ok(code1, &body1);
    let (code2, body2) = post(MESSAGES_PATH, &Request::new(&ctx, Model::sonnet_5(), 64).unwrap(), &key);
    assert_ok(code2, &body2);
    let read = body2["usage"]["cache_read_input_tokens"].as_u64().unwrap_or(0);
    assert!(read > 0, "expected a cache read on the 2nd identical-prefix request, usage: {}", body2["usage"]);

    // Below the minimum: cache_control is set, but the API silently declines to
    // cache — no creation, no read, no error.
    let small = "Be concise.";
    let mut ctx =
        Context::new().with_system_cached(CacheSlot::S0, small, CacheTtl::FiveMinutes).expect("anchor system cache");
    ctx.push_user_text("Reply: ok");
    // The whole-request token count is an upper bound on the cached prefix, so
    // count < min proves the prefix is genuinely below the threshold.
    let (ccode, cbody) = post(anthropic::COUNT_TOKENS_PATH, &CountRequest::new(&ctx, ModelId::Sonnet5), &key);
    assert_ok(ccode, &cbody);
    let count = cbody["input_tokens"].as_u64().unwrap_or(u64::MAX);
    assert!(count < u64::from(min), "test prefix must be below the {min}-token minimum, was {count}");
    let (code3, body3) = post(MESSAGES_PATH, &Request::new(&ctx, Model::sonnet_5(), 64).unwrap(), &key);
    assert_ok(code3, &body3);
    let created = body3["usage"]["cache_creation_input_tokens"].as_u64().unwrap_or(0);
    let read = body3["usage"]["cache_read_input_tokens"].as_u64().unwrap_or(0);
    assert_eq!(created + read, 0, "a sub-minimum prefix must not cache, usage: {}", body3["usage"]);
}

#[test]
fn live_ok_sonnet_5_roll_cache_across_turns() {
    // `roll_cache` is the crate's append-only cache evolution (§1): move a rolling
    // breakpoint to the conversation tail each turn so the growing prefix is
    // re-read, not rebuilt. No system anchor here — the breakpoint lives purely on
    // message content, so the 2nd turn's cache_read is attributable to the roll.
    let key = key_or_skip!();
    let big = "The quick brown fox jumps over the lazy dog. ".repeat(100); // ~1.8k tokens, over the 1,024 min

    // Turn 1: a large user message, breakpoint rolled to its tail.
    let mut ctx = Context::new();
    ctx.push_user_text(big);
    ctx.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).expect("roll to turn-1 tail");
    let (code1, body1) = post(MESSAGES_PATH, &Request::new(&ctx, Model::sonnet_5(), 64).unwrap(), &key);
    assert_ok(code1, &body1);

    // Turn 2: extend the conversation, then roll the breakpoint forward. The crate
    // clears the old position's cache_control and sets it on the new tail; the
    // turn-1 prefix cached above must now come back as a read.
    ctx.push_assistant_text("ok");
    ctx.push_user_text("Reply with the single word: ok");
    ctx.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).expect("roll to turn-2 tail");
    let (code2, body2) = post(MESSAGES_PATH, &Request::new(&ctx, Model::sonnet_5(), 64).unwrap(), &key);
    assert_ok(code2, &body2);
    let read = body2["usage"]["cache_read_input_tokens"].as_u64().unwrap_or(0);
    assert!(read > 0, "rolled prefix should be re-read on the 2nd turn, usage: {}", body2["usage"]);
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
fn live_400_fable_5_disabled_thinking() {
    // Fable 5 has no thinking-off state: `{type:"disabled"}` 400s — why `Fable5`
    // always emits adaptive thinking and exposes only `display` (no off variant,
    // unlike `Opus4_8Thinking::Off`).
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-fable-5", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"},
    });
    let (code, resp) = msg(MESSAGES_PATH, &body, &key);
    if fable5_unavailable(code, &resp) {
        return;
    }
    assert_400(code, &resp);
}

#[test]
fn live_400_fable_5_temperature() {
    // Fable 5 rejects sampling params (like Opus) — why `Fable5` carries no temperature.
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-fable-5", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5,
    });
    let (code, resp) = msg(MESSAGES_PATH, &body, &key);
    if fable5_unavailable(code, &resp) {
        return;
    }
    assert_400(code, &resp);
}

#[test]
fn live_400_sonnet_xhigh() {
    // `xhigh` is rejected on Sonnet 4.6 — why `Sonnet4_6Effort` has no `Xhigh`.
    // (Sonnet 5 accepts it; see `live_ok_sonnet_5_xhigh`.)
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
fn live_400_sonnet_5_temperature() {
    // Sonnet 5 rejects non-default sampling params (new for the Sonnet tier) —
    // why `Sonnet5` carries no temperature.
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-sonnet-5", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5,
    });
    let (code, resp) = msg(MESSAGES_PATH, &body, &key);
    assert_400(code, &resp);
}

#[test]
fn live_400_sonnet_5_legacy_thinking() {
    // Sonnet 5 rejects legacy `{type:"enabled",budget_tokens}` (removed, as on
    // Opus 4.8) — why `Sonnet5Thinking` is Adaptive|Disabled with no legacy variant.
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-sonnet-5", "max_tokens": 2048,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    });
    let (code, resp) = msg(MESSAGES_PATH, &body, &key);
    assert_400(code, &resp);
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
    // budget_tokens must be < max_tokens. `Request::new` now rejects this combo
    // up front (RequestError::ThinkingBudgetExceedsMaxTokens — see the offline
    // test), so the crate can't emit it; this raw request documents the
    // underlying API rule that motivates that check.
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
fn live_400_five_cache_breakpoints() {
    // The API caps `cache_control` breakpoints at 4 — which is exactly why
    // `CacheSlot` is S0..S3 and the crate cannot place a 5th. Raw JSON with 5
    // breakpoints (1 system + 4 message blocks) 400s.
    let key = key_or_skip!();
    let body = json!({
        "model": "claude-opus-4-8", "max_tokens": 16,
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "c", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "d", "cache_control": {"type": "ephemeral"}},
        ]}],
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
