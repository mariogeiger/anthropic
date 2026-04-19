//! Per-call request parameters and serialization to the `/v1/messages` body.
//!
//! # Design philosophy: model-specific parameter sets
//!
//! Each Claude model accepts a different subset of request parameters. Sending
//! a field the model doesn't accept returns 400 — for example, `temperature`
//! on Opus 4.7, or `output_config.effort` on Haiku 4.5. The [`Model`] enum
//! encodes this by making each variant carry only the parameters that
//! variant's model actually accepts. Unrepresentable combinations cannot be
//! constructed.
//!
//! Only the latest model in each tier is supported: [`Opus4_7`],
//! [`Sonnet4_6`], [`Haiku4_5`].
//!
//! Serialization always emits whatever the data represents — no
//! omit-if-default optimization. Reading a `Request` tells you exactly what
//! the model will do.

#![allow(non_camel_case_types)]

use crate::context::Context;
use crate::{ThinkingDisplay, ThinkingType};
use serde::Serialize;

// ── Model variants ───────────────────────────────────────────────────────────

/// A Claude model plus its per-call parameters. Each variant only carries
/// parameters the underlying model accepts.
pub enum Model {
    Opus4_7(Opus4_7),
    Sonnet4_6(Sonnet4_6),
    Haiku4_5(Haiku4_5),
}

impl Model {
    /// The string for the `model` field in the request body.
    pub fn api_id(&self) -> &'static str {
        match self {
            Model::Opus4_7(_) => "claude-opus-4-7",
            Model::Sonnet4_6(_) => "claude-sonnet-4-6",
            Model::Haiku4_5(_) => "claude-haiku-4-5",
        }
    }

    /// Default Opus 4.7 params. Returns the variant struct so you can chain
    /// `.with_effort(…)` / `.with_adaptive_thinking(…)` before passing to
    /// [`Request::new`] (which accepts `impl Into<Model>`).
    pub fn opus_4_7() -> Opus4_7 {
        Opus4_7::default()
    }

    /// Default Sonnet 4.6 params. See [`Model::opus_4_7`] for the chaining pattern.
    pub fn sonnet_4_6() -> Sonnet4_6 {
        Sonnet4_6::default()
    }

    /// Default Haiku 4.5 params. See [`Model::opus_4_7`] for the chaining pattern.
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

/// Claude Opus 4.7 parameters.
///
/// Adaptive thinking is the only supported thinking mode; the legacy
/// `{type: "enabled", budget_tokens: N}` form returns 400. Sampling parameters
/// (`temperature`, `top_p`, `top_k`) are rejected entirely.
pub struct Opus4_7 {
    pub thinking: Opus4_7Thinking,
    pub effort: Opus4_7Effort,
}

impl Default for Opus4_7 {
    fn default() -> Self {
        Self {
            thinking: Opus4_7Thinking::Off,
            effort: Opus4_7Effort::High,
        }
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

    /// Enable adaptive thinking. On Opus 4.7 the wire-level `display` defaults
    /// to `omitted` (thinking blocks stream but their text is empty); pass
    /// [`ThinkingDisplay::Summarized`] for visible thinking progress.
    pub fn with_adaptive_thinking(mut self, display: ThinkingDisplay) -> Self {
        self.thinking = Opus4_7Thinking::Adaptive { display };
        self
    }

    pub fn with_thinking_off(mut self) -> Self {
        self.thinking = Opus4_7Thinking::Off;
        self
    }
}

/// Thinking configuration for Opus 4.7.
pub enum Opus4_7Thinking {
    /// Thinking disabled; the `thinking` field is omitted from the request.
    Off,
    /// Adaptive thinking. On Opus 4.7 the API defaults `display` to `omitted`
    /// (thinking blocks are streamed but their text is empty); pass
    /// [`ThinkingDisplay::Summarized`] to get visible thinking text back.
    Adaptive { display: ThinkingDisplay },
}

/// Effort levels accepted by Opus 4.7. `Xhigh` is Opus 4.7-exclusive and sits
/// between `High` and `Max`.
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

/// Claude Sonnet 4.6 parameters.
///
/// Supports either temperature sampling or adaptive thinking (the API fixes
/// `temperature` to 1.0 internally under adaptive thinking). Effort supports
/// `Low`/`Medium`/`High`/`Max` — `Xhigh` is Opus 4.7-only.
pub struct Sonnet4_6 {
    pub sampling: Sonnet4_6Sampling,
    pub effort: Sonnet4_6Effort,
}

impl Default for Sonnet4_6 {
    fn default() -> Self {
        Self {
            sampling: Sonnet4_6Sampling::Temperature(1.0),
            effort: Sonnet4_6Effort::High,
        }
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

    /// Enable adaptive thinking. The API fixes `temperature` to 1.0 internally
    /// under adaptive thinking — any previously-set `with_temperature` is overridden.
    pub fn with_adaptive_thinking(mut self) -> Self {
        self.sampling = Sonnet4_6Sampling::Adaptive;
        self
    }
}

pub enum Sonnet4_6Sampling {
    /// Standard sampling. `Temperature(1.0)` matches the API default when
    /// `temperature` is omitted.
    Temperature(f32),
    /// Adaptive thinking. The API fixes temperature at 1.0 internally.
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

/// Claude Haiku 4.5 parameters.
///
/// Temperature sampling only. The `output_config.effort` parameter is not
/// supported (returns 400), and adaptive thinking is not supported either.
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

/// A complete Messages-API request: a borrowed [`Context`] + per-call params.
/// Serializes to the JSON body for `POST /v1/messages`.
pub struct Request<'a> {
    pub context: &'a Context,
    pub model: Model,
    pub max_tokens: u32,
    pub stop_sequences: Vec<String>,
}

impl<'a> Request<'a> {
    pub fn new(context: &'a Context, model: impl Into<Model>, max_tokens: u32) -> Self {
        Self {
            context,
            model: model.into(),
            max_tokens,
            stop_sequences: Vec::new(),
        }
    }

    pub fn stop_sequences(mut self, seqs: Vec<String>) -> Self {
        self.stop_sequences = seqs;
        self
    }
}

// ── CountRequest ─────────────────────────────────────────────────────────────

/// Request body for `POST /v1/messages/count_tokens`.
///
/// Unlike [`Request`], the count-tokens endpoint ignores sampling, thinking,
/// effort, `max_tokens`, and `stop_sequences` — only `model`, `system`,
/// `tools`, and `messages` matter. Per-call parameters on [`Model`] are
/// accepted (for ergonomics / symmetry with `Request`) but not serialized.
pub struct CountRequest<'a> {
    pub context: &'a Context,
    pub model: Model,
}

impl<'a> CountRequest<'a> {
    pub fn new(context: &'a Context, model: impl Into<Model>) -> Self {
        Self {
            context,
            model: model.into(),
        }
    }
}

impl Serialize for CountRequest<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;

        let ctx = self.context;

        // Always present: model, messages.
        let mut n = 2;
        if ctx.system.is_some() {
            n += 1;
        }
        if !ctx.tools.is_empty() {
            n += 1;
        }

        let mut st = s.serialize_struct("CountRequest", n)?;
        st.serialize_field("model", self.model.api_id())?;
        if let Some(sys) = &ctx.system {
            st.serialize_field("system", sys)?;
        }
        if !ctx.tools.is_empty() {
            st.serialize_field("tools", &ctx.tools)?;
        }
        st.serialize_field("messages", &ctx.messages)?;
        st.end()
    }
}

// ── Serialization ────────────────────────────────────────────────────────────

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

/// Resolved wire-level view of `Model`-specific fields.
struct ModelFields {
    temperature: Option<f32>,
    thinking: Option<AdaptiveThinking>,
    output_config: Option<OutputConfig>,
}

impl Model {
    fn wire_fields(&self) -> ModelFields {
        match self {
            Model::Opus4_7(p) => ModelFields {
                temperature: None,
                thinking: match &p.thinking {
                    Opus4_7Thinking::Off => None,
                    Opus4_7Thinking::Adaptive { display } => Some(AdaptiveThinking {
                        kind: ThinkingType::Adaptive.as_str(),
                        display: Some(display.as_str()),
                    }),
                },
                output_config: Some(OutputConfig {
                    effort: p.effort.as_str(),
                }),
            },
            Model::Sonnet4_6(p) => {
                let (temperature, thinking) = match p.sampling {
                    Sonnet4_6Sampling::Temperature(t) => (Some(t), None),
                    Sonnet4_6Sampling::Adaptive => (
                        None,
                        Some(AdaptiveThinking {
                            kind: ThinkingType::Adaptive.as_str(),
                            display: None,
                        }),
                    ),
                };
                ModelFields {
                    temperature,
                    thinking,
                    output_config: Some(OutputConfig {
                        effort: p.effort.as_str(),
                    }),
                }
            }
            Model::Haiku4_5(p) => ModelFields {
                temperature: Some(p.temperature),
                thinking: None,
                output_config: None,
            },
        }
    }
}

impl Serialize for Request<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;

        let ctx = self.context;
        let fields = self.model.wire_fields();

        // Always present: model, max_tokens, messages.
        let mut n = 3;
        if fields.temperature.is_some() {
            n += 1;
        }
        if fields.thinking.is_some() {
            n += 1;
        }
        if fields.output_config.is_some() {
            n += 1;
        }
        if !self.stop_sequences.is_empty() {
            n += 1;
        }
        if ctx.system.is_some() {
            n += 1;
        }
        if !ctx.tools.is_empty() {
            n += 1;
        }

        let mut st = s.serialize_struct("Request", n)?;
        st.serialize_field("model", self.model.api_id())?;
        st.serialize_field("max_tokens", &self.max_tokens)?;
        if let Some(t) = fields.temperature {
            st.serialize_field("temperature", &t)?;
        }
        if let Some(th) = &fields.thinking {
            st.serialize_field("thinking", th)?;
        }
        if !self.stop_sequences.is_empty() {
            st.serialize_field("stop_sequences", &self.stop_sequences)?;
        }
        if let Some(sys) = &ctx.system {
            st.serialize_field("system", sys)?;
        }
        if !ctx.tools.is_empty() {
            st.serialize_field("tools", &ctx.tools)?;
        }
        st.serialize_field("messages", &ctx.messages)?;
        if let Some(oc) = &fields.output_config {
            st.serialize_field("output_config", oc)?;
        }
        st.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;

    fn body(req: Request<'_>) -> serde_json::Value {
        serde_json::to_value(req).unwrap()
    }

    fn approx(v: &serde_json::Value, expected: f64) {
        let got = v.as_f64().expect("not a number");
        assert!(
            (got - expected).abs() < 1e-4,
            "expected ~{expected}, got {got}"
        );
    }

    // ── Opus 4.7 ────────────────────────────────────────────────────────────

    #[test]
    fn opus_4_7_default_omits_temperature_and_thinking() {
        let ctx = Context::new();
        let v = body(Request::new(&ctx, Model::opus_4_7(), 1024));
        assert_eq!(v["model"], "claude-opus-4-7");
        assert!(
            v.get("temperature").is_none(),
            "temperature must not be sent on Opus 4.7"
        );
        assert!(v.get("thinking").is_none());
        assert_eq!(v["output_config"]["effort"], "high");
    }

    #[test]
    fn opus_4_7_adaptive_summarized_display() {
        let ctx = Context::new();
        let v = body(Request::new(
            &ctx,
            Model::opus_4_7().with_adaptive_thinking(ThinkingDisplay::Summarized),
            1024,
        ));
        assert_eq!(v["thinking"]["type"], "adaptive");
        assert_eq!(v["thinking"]["display"], "summarized");
        assert!(v.get("temperature").is_none());
    }

    #[test]
    fn opus_4_7_adaptive_omitted_display() {
        let ctx = Context::new();
        let v = body(Request::new(
            &ctx,
            Model::opus_4_7()
                .with_adaptive_thinking(ThinkingDisplay::Omitted)
                .with_effort(Opus4_7Effort::Xhigh),
            1024,
        ));
        assert_eq!(v["thinking"]["display"], "omitted");
        assert_eq!(v["output_config"]["effort"], "xhigh");
    }

    #[test]
    fn opus_4_7_max_effort() {
        let ctx = Context::new();
        let v = body(Request::new(
            &ctx,
            Model::opus_4_7().with_effort(Opus4_7Effort::Max),
            1024,
        ));
        assert_eq!(v["output_config"]["effort"], "max");
    }

    // ── Sonnet 4.6 ──────────────────────────────────────────────────────────

    #[test]
    fn sonnet_4_6_default_uses_temperature() {
        let ctx = Context::new();
        let v = body(Request::new(&ctx, Model::sonnet_4_6(), 1024));
        assert_eq!(v["model"], "claude-sonnet-4-6");
        approx(&v["temperature"], 1.0);
        assert!(v.get("thinking").is_none());
        assert_eq!(v["output_config"]["effort"], "high");
    }

    #[test]
    fn sonnet_4_6_adaptive_thinking_drops_temperature() {
        let ctx = Context::new();
        let v = body(Request::new(
            &ctx,
            Model::sonnet_4_6()
                .with_adaptive_thinking()
                .with_effort(Sonnet4_6Effort::Max),
            1024,
        ));
        assert!(v.get("temperature").is_none());
        assert_eq!(v["thinking"]["type"], "adaptive");
        // `display` is Opus 4.7-only; not emitted for Sonnet.
        assert!(v["thinking"].get("display").is_none());
        assert_eq!(v["output_config"]["effort"], "max");
    }

    #[test]
    fn sonnet_4_6_custom_temperature() {
        let ctx = Context::new();
        let v = body(Request::new(
            &ctx,
            Model::sonnet_4_6()
                .with_temperature(0.3)
                .with_effort(Sonnet4_6Effort::Low),
            1024,
        ));
        approx(&v["temperature"], 0.3);
        assert_eq!(v["output_config"]["effort"], "low");
    }

    // ── Haiku 4.5 ───────────────────────────────────────────────────────────

    #[test]
    fn haiku_4_5_default_emits_temperature_only() {
        let ctx = Context::new();
        let v = body(Request::new(&ctx, Model::haiku_4_5(), 1024));
        assert_eq!(v["model"], "claude-haiku-4-5");
        approx(&v["temperature"], 1.0);
        assert!(v.get("thinking").is_none());
        assert!(
            v.get("output_config").is_none(),
            "effort must not be sent on Haiku 4.5"
        );
    }

    #[test]
    fn haiku_4_5_custom_temperature() {
        let ctx = Context::new();
        let v = body(Request::new(
            &ctx,
            Model::haiku_4_5().with_temperature(0.5),
            1024,
        ));
        approx(&v["temperature"], 0.5);
    }

    // ── CountRequest ───────────────────────────────────────────────────────

    fn count_body(req: CountRequest<'_>) -> serde_json::Value {
        serde_json::to_value(req).unwrap()
    }

    #[test]
    fn count_request_omits_messages_params_and_max_tokens() {
        let ctx = Context::new();
        let v = count_body(CountRequest::new(&ctx, Model::opus_4_7()));
        assert_eq!(v["model"], "claude-opus-4-7");
        assert!(v["messages"].is_array());
        assert!(v.get("max_tokens").is_none());
        assert!(v.get("temperature").is_none());
        assert!(v.get("thinking").is_none());
        assert!(v.get("output_config").is_none());
        assert!(v.get("stop_sequences").is_none());
    }

    #[test]
    fn count_request_carries_system_and_tools_from_context() {
        use crate::context::Tool;
        let ctx = Context::new()
            .with_system("sys")
            .with_tools(vec![Tool::new("t", serde_json::json!({"type": "object"}))]);
        let v = count_body(CountRequest::new(&ctx, Model::sonnet_4_6()));
        assert_eq!(v["model"], "claude-sonnet-4-6");
        assert_eq!(v["system"], "sys");
        assert_eq!(v["tools"][0]["name"], "t");
    }

    // ── Request-level fields ────────────────────────────────────────────────

    #[test]
    fn stop_sequences_emitted_when_non_empty() {
        let ctx = Context::new();
        let v = body(
            Request::new(&ctx, Model::opus_4_7(), 1024)
                .stop_sequences(vec!["STOP".into(), "END".into()]),
        );
        assert_eq!(v["stop_sequences"][0], "STOP");
        assert_eq!(v["stop_sequences"][1], "END");
    }

    #[test]
    fn stop_sequences_omitted_when_empty() {
        let ctx = Context::new();
        let v = body(Request::new(&ctx, Model::opus_4_7(), 1024));
        assert!(v.get("stop_sequences").is_none());
    }
}
