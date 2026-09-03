use super::*;
use crate::ThinkingDisplay;
use crate::context::Opening;
use serde_json::Value;

fn req(m: impl Into<Model>) -> Value {
    serde_json::to_value(Request::new(&Context::new(Opening::None), m, 1024).unwrap()).unwrap()
}
fn count(id: ModelId) -> Value {
    serde_json::to_value(CountRequest::new(&Context::new(Opening::None), id)).unwrap()
}
fn approx(v: &Value, expected: f64) {
    let got = v.as_f64().expect("not a number");
    assert!((got - expected).abs() < 1e-4, "expected ~{expected}, got {got}");
}

#[test]
fn fable_5_1_default_and_constants() {
    let v = req(Model::fable_5_1());
    assert_eq!(v["model"], "claude-fable-5-1");
    assert_eq!(v["thinking"], serde_json::json!({"type": "adaptive", "display": "omitted"}));
    assert_eq!(v["output_config"]["effort"], "high");
    assert!(v.get("temperature").is_none());

    let id = ModelId::Fable5_1;
    assert_eq!(id.context_window_tokens(), 1_000_000);
    assert_eq!(id.max_output_tokens(), 128_000);
    assert_eq!(id.min_cacheable_prefix_tokens(), 512);
    assert_eq!(id.knowledge_cutoff(), YearMonth::new(2026, Month::June));
    assert_eq!(id.training_cutoff(), YearMonth::new(2026, Month::June));
    assert_eq!(id.price_per_mtok(), Pricing { input_cents_per_mtok: 1_000, output_cents_per_mtok: 5_000 });
    assert!(id.accepts_mid_conversation_system_message());
    assert!(!id.accepts_forced_tool_choice());
}

#[test]
fn fable_5_1_supports_every_documented_effort_and_summary_visibility() {
    for (effort, name) in [
        (Fable5_1Effort::Low, "low"),
        (Fable5_1Effort::Medium, "medium"),
        (Fable5_1Effort::High, "high"),
        (Fable5_1Effort::Xhigh, "xhigh"),
        (Fable5_1Effort::Max, "max"),
    ] {
        let v = req(Model::fable_5_1().with_display(ThinkingDisplay::Summarized).with_effort(effort));
        assert_eq!(v["thinking"]["display"], "summarized");
        assert_eq!(v["output_config"]["effort"], name);
    }
}

#[test]
fn fable_5_1_refuses_forced_tool_choice_before_serialization() {
    let ctx = Context::new(Opening::None);
    for choice in [ToolChoice::any(), ToolChoice::tool("read")] {
        let error = Request::new(&ctx, Model::fable_5_1(), 16).unwrap().with_tool_choice(choice).err().unwrap();
        assert!(matches!(error, RequestError::ForcedToolChoiceUnsupported { model: ModelId::Fable5_1, .. }));
    }
    assert!(Request::new(&ctx, Model::fable_5_1(), 16).unwrap().with_tool_choice(ToolChoice::auto()).is_ok());
    assert!(Request::new(&ctx, Model::fable_5_1(), 16).unwrap().with_tool_choice(ToolChoice::none()).is_ok());
    assert!(Request::new(&ctx, Model::fable_5(), 16).unwrap().with_tool_choice(ToolChoice::any()).is_ok());
}

#[test]
fn fable_5_default() {
    let v = req(Model::fable_5());
    assert_eq!(v["model"], "claude-fable-5");
    assert!(v.get("temperature").is_none(), "temperature must not be sent on Fable 5");
    // Thinking is always on — the adaptive block is always present, with the
    // default `omitted` display. There is no "off" state.
    assert_eq!(v["thinking"]["type"], "adaptive");
    assert_eq!(v["thinking"]["display"], "omitted");
    assert_eq!(v["output_config"]["effort"], "high");
}

#[test]
fn fable_5_summarized_and_xhigh() {
    let v = req(Model::fable_5().with_display(ThinkingDisplay::Summarized).with_effort(Fable5Effort::Xhigh));
    assert_eq!(v["thinking"]["type"], "adaptive");
    assert_eq!(v["thinking"]["display"], "summarized");
    assert_eq!(v["output_config"]["effort"], "xhigh");
    assert!(v.get("temperature").is_none());
}

#[test]
fn fable_5_max_effort() {
    assert_eq!(req(Model::fable_5().with_effort(Fable5Effort::Max))["output_config"]["effort"], "max");
}

#[test]
fn fable_5_model_id() {
    let m: Model = Model::fable_5().into();
    assert_eq!(m.id(), ModelId::Fable5);
    assert_eq!(m.api_id(), "claude-fable-5");
    assert_eq!(ModelId::Fable5.api_id(), "claude-fable-5");
}

#[test]
fn opus_4_8_default() {
    let v = req(Model::opus_4_8());
    assert_eq!(v["model"], "claude-opus-4-8");
    assert!(v.get("temperature").is_none(), "temperature must not be sent on Opus 4.8");
    assert!(v.get("thinking").is_none());
    assert_eq!(v["output_config"]["effort"], "high");
}

#[test]
fn opus_4_8_adaptive_thinking() {
    let v = req(Model::opus_4_8().with_adaptive_thinking(ThinkingDisplay::Summarized));
    assert_eq!(v["thinking"]["type"], "adaptive");
    assert_eq!(v["thinking"]["display"], "summarized");
    assert!(v.get("temperature").is_none());

    let v = req(Model::opus_4_8().with_adaptive_thinking(ThinkingDisplay::Omitted).with_effort(Opus4_8Effort::Xhigh));
    assert_eq!(v["thinking"]["display"], "omitted");
    assert_eq!(v["output_config"]["effort"], "xhigh");
}

#[test]
fn opus_4_8_max_effort() {
    assert_eq!(req(Model::opus_4_8().with_effort(Opus4_8Effort::Max))["output_config"]["effort"], "max");
}

#[test]
fn sonnet_5_default() {
    let v = req(Model::sonnet_5());
    assert_eq!(v["model"], "claude-sonnet-5");
    assert!(v.get("temperature").is_none(), "temperature must not be sent on Sonnet 5");
    // Adaptive thinking is on by default and emitted explicitly (omitting the
    // field would also mean on, but the body stays a complete record).
    assert_eq!(v["thinking"]["type"], "adaptive");
    assert_eq!(v["thinking"]["display"], "omitted");
    assert_eq!(v["output_config"]["effort"], "high");
}

#[test]
fn sonnet_5_adaptive_summarized_xhigh() {
    // `xhigh` is accepted on Sonnet 5 (unlike Sonnet 4.6).
    let v =
        req(Model::sonnet_5().with_adaptive_thinking(ThinkingDisplay::Summarized).with_effort(Sonnet5Effort::Xhigh));
    assert_eq!(v["thinking"]["type"], "adaptive");
    assert_eq!(v["thinking"]["display"], "summarized");
    assert_eq!(v["output_config"]["effort"], "xhigh");
    assert!(v.get("temperature").is_none());
}

#[test]
fn sonnet_5_thinking_off_is_explicit_disabled() {
    // "off" is the explicit disabled block — not an omitted field, which on
    // Sonnet 5 would leave adaptive thinking on.
    let v = req(Model::sonnet_5().with_thinking_off().with_effort(Sonnet5Effort::Max));
    assert_eq!(v["thinking"]["type"], "disabled");
    assert!(v["thinking"].get("display").is_none(), "disabled carries no display");
    assert!(v.get("temperature").is_none());
    assert_eq!(v["output_config"]["effort"], "max");
}

#[test]
fn sonnet_5_model_id() {
    let m: Model = Model::sonnet_5().with_thinking_off().into();
    assert_eq!(m.id(), ModelId::Sonnet5);
    assert_eq!(m.api_id(), "claude-sonnet-5");
    assert_eq!(ModelId::Sonnet5.api_id(), "claude-sonnet-5");
}

#[test]
fn min_cacheable_prefix_tokens() {
    assert_eq!(ModelId::Fable5.min_cacheable_prefix_tokens(), 512);
    assert_eq!(ModelId::Opus4_8.min_cacheable_prefix_tokens(), 1_024);
    assert_eq!(ModelId::Sonnet5.min_cacheable_prefix_tokens(), 1_024);
    assert_eq!(ModelId::Sonnet4_6.min_cacheable_prefix_tokens(), 1_024);
    assert_eq!(ModelId::Haiku4_5.min_cacheable_prefix_tokens(), 4_096);
    // `Model` delegates to its identity.
    let m: Model = Model::sonnet_5().into();
    assert_eq!(m.min_cacheable_prefix_tokens(), 1_024);
}

#[test]
fn model_constants() {
    assert_eq!(ModelId::Opus4_8.context_window_tokens(), 1_000_000);
    assert_eq!(ModelId::Haiku4_5.context_window_tokens(), 200_000);
    assert_eq!(ModelId::Sonnet5.max_output_tokens(), 128_000);
    assert_eq!(ModelId::Haiku4_5.max_output_tokens(), 64_000);
    assert_eq!(ModelId::Sonnet5.knowledge_cutoff(), YearMonth::new(2026, Month::January));
    assert_eq!(ModelId::Sonnet4_6.knowledge_cutoff(), YearMonth::new(2025, Month::August));
    assert_eq!(ModelId::Sonnet4_6.training_cutoff(), YearMonth::new(2026, Month::January));
    assert_eq!(ModelId::Haiku4_5.training_cutoff(), YearMonth::new(2025, Month::July));
    assert_eq!(ModelId::Opus4_8.price_per_mtok(), Pricing { input_cents_per_mtok: 500, output_cents_per_mtok: 2_500 });
    assert_eq!(ModelId::Haiku4_5.price_per_mtok(), Pricing { input_cents_per_mtok: 100, output_cents_per_mtok: 500 });
}

#[test]
fn a_year_month_is_ordered_chronologically_and_names_its_month() {
    let jan_2026 = YearMonth::new(2026, Month::January);
    assert_eq!(jan_2026.year(), 2026);
    assert_eq!(jan_2026.month(), Month::January);
    assert_eq!(jan_2026.month_number(), 1);
    assert_eq!(YearMonth::new(2025, Month::December).month_number(), 12);
    // Declaration order is calendar order, so the derived `Ord` is chronological.
    assert!(YearMonth::new(2025, Month::December) < jan_2026);
    assert!(YearMonth::new(2026, Month::February) > jan_2026);
    assert!(ModelId::Haiku4_5.knowledge_cutoff() < ModelId::Opus5.knowledge_cutoff());
    // Ordinals round-trip; anything outside 1..=12 names no month.
    for m in Month::ALL {
        assert_eq!(Month::from_number(YearMonth::new(2026, m).month_number()), Some(m));
    }
    assert_eq!(Month::from_number(0), None);
    assert_eq!(Month::from_number(13), None);
}

#[test]
fn max_tokens_must_be_in_range() {
    let ctx = Context::new(Opening::None);
    // Zero is rejected up front (the API requires >= 1).
    assert_eq!(
        Request::new(&ctx, Model::opus_4_8(), 0).err(),
        Some(RequestError::MaxTokensOutOfRange { max_tokens: 0, max_output: 128_000 }),
    );
    // Above the model's max output is rejected (Haiku 4.5 caps at 64k)...
    assert_eq!(
        Request::new(&ctx, Model::haiku_4_5(), 64_001).err(),
        Some(RequestError::MaxTokensOutOfRange { max_tokens: 64_001, max_output: 64_000 }),
    );
    // ...but 1 and exactly the max are fine.
    assert!(Request::new(&ctx, Model::opus_4_8(), 1).is_ok());
    assert!(Request::new(&ctx, Model::haiku_4_5(), 64_000).is_ok());
    assert!(Request::new(&ctx, Model::opus_4_8(), 128_000).is_ok());
}

#[test]
fn sonnet_4_6_default_uses_temperature() {
    let v = req(Model::sonnet_4_6());
    assert_eq!(v["model"], "claude-sonnet-4-6");
    approx(&v["temperature"], 1.0);
    assert!(v.get("thinking").is_none());
    assert_eq!(v["output_config"]["effort"], "high");
}

#[test]
fn sonnet_4_6_adaptive_drops_temperature() {
    let v =
        req(Model::sonnet_4_6().with_adaptive_thinking(ThinkingDisplay::Summarized).with_effort(Sonnet4_6Effort::Max));
    assert!(v.get("temperature").is_none());
    assert_eq!(v["thinking"]["type"], "adaptive");
    assert_eq!(v["thinking"]["display"], "summarized");
    assert_eq!(v["output_config"]["effort"], "max");

    let v = req(Model::sonnet_4_6().with_adaptive_thinking(ThinkingDisplay::Omitted));
    assert_eq!(v["thinking"]["display"], "omitted");
}

#[test]
fn sonnet_4_6_custom_temperature() {
    let t = Temperature::new(0.3).unwrap();
    let v = req(Model::sonnet_4_6().with_temperature(t).with_effort(Sonnet4_6Effort::Low));
    approx(&v["temperature"], 0.3);
    assert_eq!(v["output_config"]["effort"], "low");
}

#[test]
fn haiku_4_5_emits_temperature_only() {
    let v = req(Model::haiku_4_5());
    assert_eq!(v["model"], "claude-haiku-4-5");
    approx(&v["temperature"], 1.0);
    assert!(v.get("thinking").is_none());
    assert!(v.get("output_config").is_none(), "effort must not be sent on Haiku 4.5");

    approx(&req(Model::haiku_4_5().with_temperature(Temperature::new(0.5).unwrap()))["temperature"], 0.5);
}

#[test]
fn temperature_rejects_invalid() {
    assert_eq!(Temperature::new(f32::NAN), Err(TemperatureError::NotFinite));
    assert_eq!(Temperature::new(f32::INFINITY), Err(TemperatureError::NotFinite));
    assert_eq!(Temperature::new(f32::NEG_INFINITY), Err(TemperatureError::NotFinite));
    assert_eq!(Temperature::new(-0.1), Err(TemperatureError::OutOfRange(-0.1)));
    assert_eq!(Temperature::new(1.1), Err(TemperatureError::OutOfRange(1.1)));
    assert!(Temperature::new(0.0).is_ok());
    assert!(Temperature::new(1.0).is_ok());
    assert_eq!(Temperature::default().get(), 1.0);
}

#[test]
fn haiku_4_5_legacy_thinking() {
    // budget_tokens must stay below max_tokens (validated by `Request::new`).
    let ctx = Context::new(Opening::None);
    let v = serde_json::to_value(Request::new(&ctx, Model::haiku_4_5().with_thinking(1024), 1536).unwrap()).unwrap();
    assert_eq!(v["thinking"]["type"], "enabled");
    assert_eq!(v["thinking"]["budget_tokens"], 1024);
    assert!(v["thinking"].get("display").is_none(), "`display` is adaptive-only");
    approx(&v["temperature"], 1.0);

    assert!(req(Model::haiku_4_5().with_thinking(2048).with_thinking_off()).get("thinking").is_none());
}

#[test]
fn haiku_thinking_budget_must_be_below_max_tokens() {
    let ctx = Context::new(Opening::None);
    // budget_tokens >= max_tokens is refused before the API can 400.
    assert_eq!(
        Request::new(&ctx, Model::haiku_4_5().with_thinking(1024), 1024).err(),
        Some(RequestError::ThinkingBudgetExceedsMaxTokens { budget_tokens: 1024, max_tokens: 1024 }),
    );
    assert!(Request::new(&ctx, Model::haiku_4_5().with_thinking(2000), 1000).is_err());
    // budget below max is fine; models without a thinking budget never fail.
    assert!(Request::new(&ctx, Model::haiku_4_5().with_thinking(1024), 1536).is_ok());
    assert!(Request::new(&ctx, Model::haiku_4_5(), 16).is_ok());
    assert!(Request::new(&ctx, Model::opus_4_8(), 16).is_ok());
}

/// `service_tier` is a scalar the API accepts unconditionally and documents a
/// default for, so it is always emitted — the body is a complete record.
#[test]
fn the_service_tier_is_always_emitted_at_its_documented_default() {
    assert_eq!(req(Model::opus_5())["service_tier"], "auto");
    let ctx = Context::new(Opening::None);
    let v = serde_json::to_value(
        Request::new(&ctx, Model::opus_5(), 16).unwrap().with_service_tier(ServiceTier::StandardOnly),
    )
    .unwrap();
    assert_eq!(v["service_tier"], "standard_only");
    assert_eq!(
        Request::new(&ctx, Model::opus_5(), 16).unwrap().service_tier(),
        ServiceTier::Auto,
        "the reader agrees with the wire"
    );
}

/// An end user is really named or really not, so `metadata` is absent rather
/// than defaulted.
#[test]
fn an_end_user_id_appears_only_when_one_is_named() {
    let ctx = Context::new(Opening::None);
    assert!(req(Model::opus_5()).get("metadata").is_none(), "no end user, no metadata");
    let id = EndUserId::new("3f2b8c1e-0000-4a5d-9e77-1c2b3a4d5e6f").unwrap();
    let v = serde_json::to_value(Request::new(&ctx, Model::opus_5(), 16).unwrap().with_end_user_id(id)).unwrap();
    assert_eq!(v["metadata"]["user_id"], "3f2b8c1e-0000-4a5d-9e77-1c2b3a4d5e6f");
}

/// The documented 512-character bound, counted in characters as JSON Schema
/// counts it rather than in bytes.
#[test]
fn an_end_user_id_refuses_more_than_the_documented_length() {
    assert!(EndUserId::new("a".repeat(512)).is_ok());
    assert_eq!(EndUserId::new("a".repeat(513)).err(), Some(EndUserIdError::TooLong { length: 513 }));
    // 512 multi-byte characters are 512 characters, not 1,536 bytes' worth.
    assert!(EndUserId::new("é".repeat(512)).is_ok());
    assert_eq!(EndUserId::new("").unwrap().as_str(), "");
}

/// A schema rides in `output_config.format`, beside effort rather than instead
/// of it.
#[test]
fn an_output_format_joins_effort_in_the_output_config() {
    let ctx = Context::new(Opening::None);
    let schema = serde_json::json!({"type": "object", "properties": {"n": {"type": "integer"}}});
    let v = serde_json::to_value(
        Request::new(&ctx, Model::opus_5(), 16).unwrap().with_output_format(OutputFormat::json_schema(schema)),
    )
    .unwrap();
    assert_eq!(v["output_config"]["effort"], "high", "effort survives");
    assert_eq!(v["output_config"]["format"]["type"], "json_schema");
    assert_eq!(v["output_config"]["format"]["schema"]["properties"]["n"]["type"], "integer");
}

/// Haiku 4.5 takes no effort but does take a format, so `output_config`
/// appears carrying only the half that applies.
#[test]
fn a_model_without_effort_still_carries_an_output_format() {
    let ctx = Context::new(Opening::None);
    assert!(req(Model::haiku_4_5()).get("output_config").is_none(), "neither half, no object");
    let v = serde_json::to_value(
        Request::new(&ctx, Model::haiku_4_5(), 16)
            .unwrap()
            .with_output_format(OutputFormat::json_schema(serde_json::json!({"type": "object"}))),
    )
    .unwrap();
    assert!(v["output_config"].get("effort").is_none(), "effort is refused on this model");
    assert_eq!(v["output_config"]["format"]["type"], "json_schema");
}

#[test]
fn count_request_omits_sampling_and_max_tokens() {
    let v = count(ModelId::Opus4_8);
    assert_eq!(v["model"], "claude-opus-4-8");
    assert!(v["messages"].is_array());
    for f in ["max_tokens", "temperature", "thinking", "output_config", "stop_sequences", "service_tier"] {
        assert!(v.get(f).is_none(), "{f} should be omitted");
    }
}

#[test]
fn count_request_carries_system_and_tools() {
    let ctx = Context::new(Opening::instruction("sys"))
        .with_tools(vec![Tool::new("t", serde_json::json!({"type": "object"}))]);
    let v = serde_json::to_value(CountRequest::new(&ctx, ModelId::Sonnet4_6)).unwrap();
    assert_eq!(v["model"], "claude-sonnet-4-6");
    assert_eq!(v["system"], "sys");
    assert_eq!(v["tools"][0]["name"], "t");
}

#[test]
fn model_id_from_configured_model() {
    let m: Model = Model::opus_4_8().with_adaptive_thinking(ThinkingDisplay::Summarized).into();
    assert_eq!(m.id(), ModelId::Opus4_8);
    assert_eq!(m.id().api_id(), m.api_id());
}

#[test]
fn stop_sequences_roundtrip() {
    let ctx = Context::new(Opening::None);
    let v = serde_json::to_value(
        Request::new(&ctx, Model::opus_4_8(), 1024).unwrap().with_stop_sequences(vec!["STOP".into(), "END".into()]),
    )
    .unwrap();
    assert_eq!(v["stop_sequences"][0], "STOP");
    assert_eq!(v["stop_sequences"][1], "END");
    // Empty vec is skipped.
    assert!(req(Model::opus_4_8()).get("stop_sequences").is_none());
}
