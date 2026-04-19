//! Per-call request params and serialization to the `/v1/messages` body.
//!
//! Each `Model` variant carries only the parameters its underlying model accepts —
//! unrepresentable combinations cannot be constructed. Only the latest model in
//! each tier is supported: Opus 4.7, Sonnet 4.6, Haiku 4.5.

#![allow(non_camel_case_types)]

use crate::context::{Context, Message, SystemPrompt, Tool};
use crate::{ThinkingDisplay, ThinkingType};
use serde::Serialize;

// ── Model variants ───────────────────────────────────────────────────────────

/// Model identity without per-call parameters. Used where only the `model`
/// field is meaningful (e.g. `CountRequest`, which ignores sampling/thinking).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelId {
    Opus4_7,
    Sonnet4_6,
    Haiku4_5,
}

impl ModelId {
    /// The `model` field value sent on the wire.
    pub fn api_id(self) -> &'static str {
        match self {
            ModelId::Opus4_7 => "claude-opus-4-7",
            ModelId::Sonnet4_6 => "claude-sonnet-4-6",
            ModelId::Haiku4_5 => "claude-haiku-4-5",
        }
    }
}

/// A Claude model plus its per-call parameters.
pub enum Model {
    Opus4_7(Opus4_7),
    Sonnet4_6(Sonnet4_6),
    Haiku4_5(Haiku4_5),
}

impl Model {
    /// Identity without per-call parameters.
    pub fn id(&self) -> ModelId {
        match self {
            Model::Opus4_7(_) => ModelId::Opus4_7,
            Model::Sonnet4_6(_) => ModelId::Sonnet4_6,
            Model::Haiku4_5(_) => ModelId::Haiku4_5,
        }
    }

    /// The `model` field value sent on the wire.
    pub fn api_id(&self) -> &'static str {
        self.id().api_id()
    }

    /// Default params for each model. Chain `.with_*` on the returned struct,
    /// then pass to `Request::new` (which accepts `impl Into<Model>`).
    pub fn opus_4_7() -> Opus4_7 {
        Opus4_7::default()
    }
    pub fn sonnet_4_6() -> Sonnet4_6 {
        Sonnet4_6::default()
    }
    pub fn haiku_4_5() -> Haiku4_5 {
        Haiku4_5::default()
    }
}

impl From<Opus4_7> for Model {
    fn from(p: Opus4_7) -> Self {
        Model::Opus4_7(p)
    }
}
impl From<Sonnet4_6> for Model {
    fn from(p: Sonnet4_6) -> Self {
        Model::Sonnet4_6(p)
    }
}
impl From<Haiku4_5> for Model {
    fn from(p: Haiku4_5) -> Self {
        Model::Haiku4_5(p)
    }
}

// ── Opus 4.7 ─────────────────────────────────────────────────────────────────
// No sampling (temperature/top_p/top_k rejected). Adaptive thinking only;
// legacy `{type: "enabled", budget_tokens}` is removed.

pub struct Opus4_7 {
    pub thinking: Opus4_7Thinking,
    pub effort: Opus4_7Effort,
}

impl Default for Opus4_7 {
    fn default() -> Self {
        Self { thinking: Opus4_7Thinking::Off, effort: Opus4_7Effort::High }
    }
}

impl Opus4_7 {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_effort(mut self, effort: Opus4_7Effort) -> Self {
        self.effort = effort;
        self
    }

    /// Enable adaptive thinking. `display` defaults to `Omitted` on Opus 4.7
    /// (blocks stream but text is empty); pass `Summarized` for visible text.
    pub fn with_adaptive_thinking(mut self, display: ThinkingDisplay) -> Self {
        self.thinking = Opus4_7Thinking::Adaptive { display };
        self
    }

    pub fn with_thinking_off(mut self) -> Self {
        self.thinking = Opus4_7Thinking::Off;
        self
    }
}

pub enum Opus4_7Thinking {
    /// `thinking` field omitted from the request.
    Off,
    Adaptive {
        display: ThinkingDisplay,
    },
}

/// Effort levels for Opus 4.7. `Xhigh` is exclusive to Opus 4.7.
pub enum Opus4_7Effort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl Opus4_7Effort {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

// ── Sonnet 4.6 ───────────────────────────────────────────────────────────────
// Temperature OR adaptive thinking (API forces temperature=1.0 under adaptive).
// No `Xhigh` effort (Opus 4.7-only).

pub struct Sonnet4_6 {
    pub sampling: Sonnet4_6Sampling,
    pub effort: Sonnet4_6Effort,
}

impl Default for Sonnet4_6 {
    fn default() -> Self {
        Self { sampling: Sonnet4_6Sampling::Temperature(1.0), effort: Sonnet4_6Effort::High }
    }
}

impl Sonnet4_6 {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_effort(mut self, effort: Sonnet4_6Effort) -> Self {
        self.effort = effort;
        self
    }
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.sampling = Sonnet4_6Sampling::Temperature(t);
        self
    }

    /// Enable adaptive thinking. Overrides any previously-set temperature
    /// (API pins it to 1.0 internally under adaptive).
    pub fn with_adaptive_thinking(mut self) -> Self {
        self.sampling = Sonnet4_6Sampling::Adaptive;
        self
    }
}

pub enum Sonnet4_6Sampling {
    /// `Temperature(1.0)` matches the API default when `temperature` is omitted.
    Temperature(f32),
    Adaptive,
}

pub enum Sonnet4_6Effort {
    Low,
    Medium,
    High,
    Max,
}

impl Sonnet4_6Effort {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }
}

// ── Haiku 4.5 ────────────────────────────────────────────────────────────────
// Temperature only. `output_config.effort` rejected (400); no adaptive thinking.

pub struct Haiku4_5 {
    pub temperature: f32,
}

impl Default for Haiku4_5 {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

impl Haiku4_5 {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
}

// ── Request ──────────────────────────────────────────────────────────────────

/// Borrowed `Context` + per-call params. Serializes to `POST /v1/messages`.
pub struct Request<'a> {
    pub context: &'a Context,
    pub model: Model,
    pub max_tokens: u32,
    pub stop_sequences: Vec<String>,
}

impl<'a> Request<'a> {
    pub fn new(context: &'a Context, model: impl Into<Model>, max_tokens: u32) -> Self {
        Self { context, model: model.into(), max_tokens, stop_sequences: Vec::new() }
    }

    pub fn stop_sequences(mut self, seqs: Vec<String>) -> Self {
        self.stop_sequences = seqs;
        self
    }
}

// ── CountRequest ─────────────────────────────────────────────────────────────

/// Request body for `POST /v1/messages/count_tokens`. Takes only a `ModelId`:
/// the endpoint ignores sampling/thinking/effort, so exposing them here would
/// let callers set values the wire payload silently drops (violates §5).
pub struct CountRequest<'a> {
    pub context: &'a Context,
    pub model: ModelId,
}

impl<'a> CountRequest<'a> {
    pub fn new(context: &'a Context, model: ModelId) -> Self {
        Self { context, model }
    }
}

// ── Serialization ────────────────────────────────────────────────────────────
// Private wire structs: Option = real runtime absence (§3), empty vecs skipped —
// never "omit if equal to default".

#[derive(Serialize)]
struct AdaptiveThinking {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    display: Option<&'static str>,
}

#[derive(Serialize)]
struct OutputConfig {
    effort: &'static str,
}

#[derive(Serialize)]
struct RequestWire<'a> {
    model: &'static str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AdaptiveThinking>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: &'a Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a SystemPrompt>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: &'a Vec<Tool>,
    messages: &'a Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig>,
}

impl Serialize for Request<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let adaptive = |display| AdaptiveThinking { kind: ThinkingType::Adaptive.as_str(), display };
        let effort = |e: &'static str| Some(OutputConfig { effort: e });
        let (temperature, thinking, output_config) = match &self.model {
            Model::Opus4_7(p) => (
                None,
                match &p.thinking {
                    Opus4_7Thinking::Off => None,
                    Opus4_7Thinking::Adaptive { display } => Some(adaptive(Some(display.as_str()))),
                },
                effort(p.effort.as_str()),
            ),
            Model::Sonnet4_6(p) => {
                let (t, th) = match p.sampling {
                    Sonnet4_6Sampling::Temperature(t) => (Some(t), None),
                    // `display` is Opus 4.7-only
                    Sonnet4_6Sampling::Adaptive => (None, Some(adaptive(None))),
                };
                (t, th, effort(p.effort.as_str()))
            }
            Model::Haiku4_5(p) => (Some(p.temperature), None, None),
        };
        RequestWire {
            model: self.model.api_id(),
            max_tokens: self.max_tokens,
            temperature,
            thinking,
            output_config,
            stop_sequences: &self.stop_sequences,
            system: self.context.system.as_ref(),
            tools: &self.context.tools,
            messages: &self.context.messages,
        }
        .serialize(s)
    }
}

#[derive(Serialize)]
struct CountRequestWire<'a> {
    model: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a SystemPrompt>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: &'a Vec<Tool>,
    messages: &'a Vec<Message>,
}

impl Serialize for CountRequest<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        CountRequestWire {
            model: self.model.api_id(),
            system: self.context.system.as_ref(),
            tools: &self.context.tools,
            messages: &self.context.messages,
        }
        .serialize(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn req(m: impl Into<Model>) -> Value {
        serde_json::to_value(Request::new(&Context::new(), m, 1024)).unwrap()
    }
    fn count(id: ModelId) -> Value {
        serde_json::to_value(CountRequest::new(&Context::new(), id)).unwrap()
    }
    fn approx(v: &Value, expected: f64) {
        let got = v.as_f64().expect("not a number");
        assert!((got - expected).abs() < 1e-4, "expected ~{expected}, got {got}");
    }

    #[test]
    fn opus_4_7_default() {
        let v = req(Model::opus_4_7());
        assert_eq!(v["model"], "claude-opus-4-7");
        assert!(v.get("temperature").is_none(), "temperature must not be sent on Opus 4.7");
        assert!(v.get("thinking").is_none());
        assert_eq!(v["output_config"]["effort"], "high");
    }

    #[test]
    fn opus_4_7_adaptive_thinking() {
        let v = req(Model::opus_4_7().with_adaptive_thinking(ThinkingDisplay::Summarized));
        assert_eq!(v["thinking"]["type"], "adaptive");
        assert_eq!(v["thinking"]["display"], "summarized");
        assert!(v.get("temperature").is_none());

        let v =
            req(Model::opus_4_7().with_adaptive_thinking(ThinkingDisplay::Omitted).with_effort(Opus4_7Effort::Xhigh));
        assert_eq!(v["thinking"]["display"], "omitted");
        assert_eq!(v["output_config"]["effort"], "xhigh");
    }

    #[test]
    fn opus_4_7_max_effort() {
        assert_eq!(req(Model::opus_4_7().with_effort(Opus4_7Effort::Max))["output_config"]["effort"], "max");
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
    fn sonnet_4_6_adaptive_drops_temperature_and_display() {
        let v = req(Model::sonnet_4_6().with_adaptive_thinking().with_effort(Sonnet4_6Effort::Max));
        assert!(v.get("temperature").is_none());
        assert_eq!(v["thinking"]["type"], "adaptive");
        assert!(v["thinking"].get("display").is_none(), "`display` is Opus 4.7-only");
        assert_eq!(v["output_config"]["effort"], "max");
    }

    #[test]
    fn sonnet_4_6_custom_temperature() {
        let v = req(Model::sonnet_4_6().with_temperature(0.3).with_effort(Sonnet4_6Effort::Low));
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

        approx(&req(Model::haiku_4_5().with_temperature(0.5))["temperature"], 0.5);
    }

    #[test]
    fn count_request_omits_sampling_and_max_tokens() {
        let v = count(ModelId::Opus4_7);
        assert_eq!(v["model"], "claude-opus-4-7");
        assert!(v["messages"].is_array());
        for f in ["max_tokens", "temperature", "thinking", "output_config", "stop_sequences"] {
            assert!(v.get(f).is_none(), "{f} should be omitted");
        }
    }

    #[test]
    fn count_request_carries_system_and_tools() {
        let ctx =
            Context::new().with_system("sys").with_tools(vec![Tool::new("t", serde_json::json!({"type": "object"}))]);
        let v = serde_json::to_value(CountRequest::new(&ctx, ModelId::Sonnet4_6)).unwrap();
        assert_eq!(v["model"], "claude-sonnet-4-6");
        assert_eq!(v["system"], "sys");
        assert_eq!(v["tools"][0]["name"], "t");
    }

    #[test]
    fn model_id_from_configured_model() {
        let m: Model = Model::opus_4_7().with_adaptive_thinking(ThinkingDisplay::Summarized).into();
        assert_eq!(m.id(), ModelId::Opus4_7);
        assert_eq!(m.id().api_id(), m.api_id());
    }

    #[test]
    fn stop_sequences_roundtrip() {
        let ctx = Context::new();
        let v = serde_json::to_value(
            Request::new(&ctx, Model::opus_4_7(), 1024).stop_sequences(vec!["STOP".into(), "END".into()]),
        )
        .unwrap();
        assert_eq!(v["stop_sequences"][0], "STOP");
        assert_eq!(v["stop_sequences"][1], "END");
        // Empty vec is skipped.
        assert!(req(Model::opus_4_7()).get("stop_sequences").is_none());
    }
}
