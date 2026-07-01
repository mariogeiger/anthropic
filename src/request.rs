//! Per-call request params and serialization to the `/v1/messages` body.
//!
//! Each `Model` variant carries only the parameters its underlying model accepts —
//! unrepresentable combinations cannot be constructed. The latest model in each
//! tier is supported: the Fable 5 frontier tier, plus Opus 4.8, Sonnet 5, Haiku 4.5.
//! Sonnet 4.6 is kept as the prior Sonnet tier (still an active model).

#![allow(non_camel_case_types)]

use crate::context::{Context, Message, SystemPrompt, Tool};
use crate::values::api_enum;
use crate::{ThinkingDisplay, ThinkingType};
use serde::Serialize;

// ── Temperature ──────────────────────────────────────────────────────────────

/// Sampling temperature. API-accepted range is `[0.0, 1.0]` and the value must
/// be finite — constructing a `Temperature` is the only way to prove that,
/// so downstream code never has to re-check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Temperature(f32);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemperatureError {
    /// Value was NaN or infinite.
    NotFinite,
    /// Value was finite but outside the API-accepted `[0.0, 1.0]` range.
    OutOfRange(f32),
}

impl std::fmt::Display for TemperatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemperatureError::NotFinite => write!(f, "temperature must be finite"),
            TemperatureError::OutOfRange(v) => write!(f, "temperature {v} is outside [0.0, 1.0]"),
        }
    }
}

impl std::error::Error for TemperatureError {}

impl Temperature {
    pub fn new(v: f32) -> Result<Self, TemperatureError> {
        if !v.is_finite() {
            Err(TemperatureError::NotFinite)
        } else if !(0.0..=1.0).contains(&v) {
            Err(TemperatureError::OutOfRange(v))
        } else {
            Ok(Self(v))
        }
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl Default for Temperature {
    /// API default is `1.0` (per Anthropic docs).
    fn default() -> Self {
        Self(1.0)
    }
}

// ── Model variants ───────────────────────────────────────────────────────────

/// Model identity without per-call parameters. Used where only the `model`
/// field is meaningful (e.g. `CountRequest`, which ignores sampling/thinking).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelId {
    Fable5,
    Opus4_8,
    Sonnet5,
    Sonnet4_6,
    Haiku4_5,
}

impl ModelId {
    /// The `model` field value sent on the wire.
    pub fn api_id(self) -> &'static str {
        match self {
            ModelId::Fable5 => "claude-fable-5",
            ModelId::Opus4_8 => "claude-opus-4-8",
            ModelId::Sonnet5 => "claude-sonnet-5",
            ModelId::Sonnet4_6 => "claude-sonnet-4-6",
            ModelId::Haiku4_5 => "claude-haiku-4-5",
        }
    }

    /// Minimum prefix length, in tokens, that this model will cache. A cached
    /// prefix shorter than this is a *silent* no-op — the API caches nothing and
    /// returns no error (detectable only via `usage.cache_read_input_tokens`), so
    /// this is documented behavior, not a request-validity rule the type system
    /// can enforce. Values per the Anthropic prompt-caching docs (first-party API;
    /// some platforms differ — e.g. Fable 5 is 1024 on Amazon Bedrock).
    pub fn min_cacheable_prefix_tokens(self) -> u32 {
        match self {
            ModelId::Fable5 => 512,
            ModelId::Opus4_8 => 1_024,
            ModelId::Sonnet5 => 1_024,
            ModelId::Sonnet4_6 => 1_024,
            ModelId::Haiku4_5 => 4_096,
        }
    }
}

/// A Claude model plus its per-call parameters.
pub enum Model {
    Fable5(Fable5),
    Opus4_8(Opus4_8),
    Sonnet5(Sonnet5),
    Sonnet4_6(Sonnet4_6),
    Haiku4_5(Haiku4_5),
}

impl Model {
    /// Identity without per-call parameters.
    pub fn id(&self) -> ModelId {
        match self {
            Model::Fable5(_) => ModelId::Fable5,
            Model::Opus4_8(_) => ModelId::Opus4_8,
            Model::Sonnet5(_) => ModelId::Sonnet5,
            Model::Sonnet4_6(_) => ModelId::Sonnet4_6,
            Model::Haiku4_5(_) => ModelId::Haiku4_5,
        }
    }

    /// The `model` field value sent on the wire.
    pub fn api_id(&self) -> &'static str {
        self.id().api_id()
    }

    /// Minimum cacheable prefix length, in tokens
    /// (see [`ModelId::min_cacheable_prefix_tokens`]).
    pub fn min_cacheable_prefix_tokens(&self) -> u32 {
        self.id().min_cacheable_prefix_tokens()
    }

    /// Default params for each model. Chain `.with_*` on the returned struct,
    /// then pass to `Request::new` (which accepts `impl Into<Model>`).
    pub fn fable_5() -> Fable5 {
        Fable5::default()
    }
    pub fn opus_4_8() -> Opus4_8 {
        Opus4_8::default()
    }
    pub fn sonnet_5() -> Sonnet5 {
        Sonnet5::default()
    }
    pub fn sonnet_4_6() -> Sonnet4_6 {
        Sonnet4_6::default()
    }
    pub fn haiku_4_5() -> Haiku4_5 {
        Haiku4_5::default()
    }
}

impl From<Fable5> for Model {
    fn from(p: Fable5) -> Self {
        Model::Fable5(p)
    }
}
impl From<Opus4_8> for Model {
    fn from(p: Opus4_8) -> Self {
        Model::Opus4_8(p)
    }
}
impl From<Sonnet5> for Model {
    fn from(p: Sonnet5) -> Self {
        Model::Sonnet5(p)
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

// ── Fable 5 ──────────────────────────────────────────────────────────────────
// Frontier tier. No sampling (temperature/top_p/top_k rejected). Thinking is
// always on: `{type: "disabled"}` and legacy `{type: "enabled", budget_tokens}`
// both 400, so unlike Opus 4.8 there is no "off" state — the only knob is
// `display`. Depth is controlled by `output_config.effort` (low..=max, incl.
// xhigh). `display` defaults to `Omitted` (blocks stream, text empty).

pub struct Fable5 {
    pub display: ThinkingDisplay,
    pub effort: Fable5Effort,
}

impl Default for Fable5 {
    fn default() -> Self {
        Self { display: ThinkingDisplay::Omitted, effort: Fable5Effort::High }
    }
}

impl Fable5 {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_effort(mut self, effort: Fable5Effort) -> Self {
        self.effort = effort;
        self
    }

    /// Set the thinking summary visibility. Thinking can't be turned off on
    /// Fable 5; pass `Summarized` for visible reasoning text, `Omitted` (default)
    /// for empty thinking blocks.
    pub fn with_display(mut self, display: ThinkingDisplay) -> Self {
        self.display = display;
        self
    }
}

// Same range as Opus-tier (`xhigh` is Opus 4.7+/Fable; Sonnet rejects it).
api_enum! { Fable5Effort {
    Low => "low", Medium => "medium", High => "high", Xhigh => "xhigh", Max => "max",
}}

// ── Opus 4.8 ─────────────────────────────────────────────────────────────────
// No sampling (temperature/top_p/top_k rejected). Adaptive thinking only;
// legacy `{type: "enabled", budget_tokens}` is removed.

pub struct Opus4_8 {
    pub thinking: Opus4_8Thinking,
    pub effort: Opus4_8Effort,
}

impl Default for Opus4_8 {
    fn default() -> Self {
        Self { thinking: Opus4_8Thinking::Off, effort: Opus4_8Effort::High }
    }
}

impl Opus4_8 {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_effort(mut self, effort: Opus4_8Effort) -> Self {
        self.effort = effort;
        self
    }

    /// Enable adaptive thinking. `display` defaults to `Omitted` on Opus 4.8
    /// (blocks stream but text is empty); pass `Summarized` for visible text.
    pub fn with_adaptive_thinking(mut self, display: ThinkingDisplay) -> Self {
        self.thinking = Opus4_8Thinking::Adaptive { display };
        self
    }

    pub fn with_thinking_off(mut self) -> Self {
        self.thinking = Opus4_8Thinking::Off;
        self
    }
}

pub enum Opus4_8Thinking {
    /// `thinking` field omitted from the request.
    Off,
    Adaptive {
        display: ThinkingDisplay,
    },
}

// `Xhigh` is Opus-tier only (Opus 4.7 and later); Sonnet 4.6 rejects it (400).
api_enum! { Opus4_8Effort {
    Low => "low", Medium => "medium", High => "high", Xhigh => "xhigh", Max => "max",
}}

// ── Sonnet 5 ─────────────────────────────────────────────────────────────────
// Current Sonnet tier. No sampling (temperature/top_p/top_k non-default rejected,
// like Opus 4.8). Adaptive thinking is *on by default*: omitting `thinking` leaves
// it on, so "off" has to be sent explicitly as `{type: "disabled"}` — unlike Opus
// 4.8, whose off state is simply the omitted field. Legacy `{type: "enabled",
// budget_tokens}` is removed (400). Full Opus-tier effort incl. `xhigh` (Sonnet 4.6
// rejects `xhigh`). New tokenizer (~30% more tokens than Sonnet 4.6) — no wire effect.

pub struct Sonnet5 {
    pub thinking: Sonnet5Thinking,
    pub effort: Sonnet5Effort,
}

impl Default for Sonnet5 {
    /// Adaptive thinking on with `Omitted` display — the runtime default the API
    /// applies when `thinking` is absent, emitted explicitly (§5).
    fn default() -> Self {
        Self { thinking: Sonnet5Thinking::Adaptive { display: ThinkingDisplay::Omitted }, effort: Sonnet5Effort::High }
    }
}

impl Sonnet5 {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_effort(mut self, effort: Sonnet5Effort) -> Self {
        self.effort = effort;
        self
    }

    /// Set adaptive thinking's summary visibility. `display` defaults to `Omitted`
    /// (blocks stream but text is empty); pass `Summarized` for visible text.
    pub fn with_adaptive_thinking(mut self, display: ThinkingDisplay) -> Self {
        self.thinking = Sonnet5Thinking::Adaptive { display };
        self
    }

    /// Turn thinking off. Emits `{type: "disabled"}` explicitly: on Sonnet 5 an
    /// omitted `thinking` field leaves adaptive thinking on, so off must be stated.
    pub fn with_thinking_off(mut self) -> Self {
        self.thinking = Sonnet5Thinking::Disabled;
        self
    }
}

pub enum Sonnet5Thinking {
    Adaptive {
        display: ThinkingDisplay,
    },
    /// Explicit `{type: "disabled"}` — distinct from an omitted field, which on
    /// Sonnet 5 means adaptive thinking on.
    Disabled,
}

// Full Opus-tier range: `xhigh` is accepted on Sonnet 5 (Sonnet 4.6 rejects it).
api_enum! { Sonnet5Effort {
    Low => "low", Medium => "medium", High => "high", Xhigh => "xhigh", Max => "max",
}}

// ── Sonnet 4.6 ───────────────────────────────────────────────────────────────
// Temperature OR adaptive thinking (API forces temperature=1.0 under adaptive).
// No `Xhigh` effort (Opus-tier only; Sonnet rejects it).

pub struct Sonnet4_6 {
    pub sampling: Sonnet4_6Sampling,
    pub effort: Sonnet4_6Effort,
}

impl Default for Sonnet4_6 {
    fn default() -> Self {
        Self { sampling: Sonnet4_6Sampling::Temperature(Temperature::default()), effort: Sonnet4_6Effort::High }
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
    pub fn with_temperature(mut self, t: Temperature) -> Self {
        self.sampling = Sonnet4_6Sampling::Temperature(t);
        self
    }

    /// Enable adaptive thinking. Overrides any previously-set temperature
    /// (API pins it to 1.0 internally under adaptive).
    pub fn with_adaptive_thinking(mut self, display: ThinkingDisplay) -> Self {
        self.sampling = Sonnet4_6Sampling::Adaptive { display };
        self
    }
}

pub enum Sonnet4_6Sampling {
    /// `Temperature::default()` (1.0) matches the API default when `temperature` is omitted.
    Temperature(Temperature),
    Adaptive {
        display: ThinkingDisplay,
    },
}

api_enum! { Sonnet4_6Effort {
    Low => "low", Medium => "medium", High => "high", Max => "max",
}}

// ── Haiku 4.5 ────────────────────────────────────────────────────────────────
// Temperature + legacy fixed-budget thinking. `output_config.effort` rejected
// (400); adaptive thinking rejected (400).

pub struct Haiku4_5 {
    pub temperature: Temperature,
    pub thinking: Haiku4_5Thinking,
}

impl Default for Haiku4_5 {
    fn default() -> Self {
        Self { temperature: Temperature::default(), thinking: Haiku4_5Thinking::Off }
    }
}

impl Haiku4_5 {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_temperature(mut self, t: Temperature) -> Self {
        self.temperature = t;
        self
    }

    /// Enable legacy fixed-budget thinking. Haiku 4.5 accepts the legacy
    /// `{type: "enabled", budget_tokens: N}` form; adaptive thinking is rejected.
    /// `budget_tokens` must be below the request's `max_tokens` — `Request::new`
    /// enforces this and returns `RequestError` otherwise.
    pub fn with_thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking = Haiku4_5Thinking::Enabled { budget_tokens };
        self
    }

    pub fn with_thinking_off(mut self) -> Self {
        self.thinking = Haiku4_5Thinking::Off;
        self
    }
}

pub enum Haiku4_5Thinking {
    /// `thinking` field omitted from the request.
    Off,
    /// Legacy fixed-budget thinking: `{type: "enabled", budget_tokens: N}`.
    Enabled { budget_tokens: u32 },
}

// ── Request ──────────────────────────────────────────────────────────────────

/// Construction-time rejection for a cross-field invariant the §4 state/per-call
/// split can't express in the type system. Same "error before commit" approach
/// as the cache ops (§1): refuse rather than let the API answer with a 400.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestError {
    /// Legacy fixed-budget thinking requires `budget_tokens < max_tokens`; the
    /// API rejects `budget_tokens >= max_tokens` with a 400. Only reachable on
    /// Haiku 4.5 via `with_thinking` — adaptive-thinking models carry no budget.
    ThinkingBudgetExceedsMaxTokens { budget_tokens: u32, max_tokens: u32 },
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestError::ThinkingBudgetExceedsMaxTokens { budget_tokens, max_tokens } => {
                write!(f, "thinking budget_tokens ({budget_tokens}) must be less than max_tokens ({max_tokens})")
            }
        }
    }
}

impl std::error::Error for RequestError {}

/// Borrowed `Context` + per-call params. Serializes to `POST /v1/messages`.
pub struct Request<'a> {
    pub context: &'a Context,
    pub model: Model,
    pub max_tokens: u32,
    pub stop_sequences: Vec<String>,
}

impl<'a> Request<'a> {
    /// `new` is the single construction path, so the check can't be bypassed.
    /// It validates the one invariant the type system can't (the model carries
    /// `budget_tokens`, the request carries `max_tokens` — they only meet here):
    /// legacy `budget_tokens` must be below `max_tokens`.
    pub fn new(context: &'a Context, model: impl Into<Model>, max_tokens: u32) -> Result<Self, RequestError> {
        let model = model.into();
        if let Model::Haiku4_5(h) = &model
            && let Haiku4_5Thinking::Enabled { budget_tokens } = h.thinking
            && budget_tokens >= max_tokens
        {
            return Err(RequestError::ThinkingBudgetExceedsMaxTokens { budget_tokens, max_tokens });
        }
        Ok(Self { context, model, max_tokens, stop_sequences: Vec::new() })
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
struct EnabledThinking {
    #[serde(rename = "type")]
    kind: &'static str,
    budget_tokens: u32,
}

#[derive(Serialize)]
struct DisabledThinking {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ThinkingWire {
    Adaptive(AdaptiveThinking),
    Enabled(EnabledThinking),
    Disabled(DisabledThinking),
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
    thinking: Option<ThinkingWire>,
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
        let adaptive =
            |display| ThinkingWire::Adaptive(AdaptiveThinking { kind: ThinkingType::Adaptive.as_str(), display });
        let enabled = |budget_tokens| {
            ThinkingWire::Enabled(EnabledThinking { kind: ThinkingType::Enabled.as_str(), budget_tokens })
        };
        let effort = |e: &'static str| Some(OutputConfig { effort: e });
        let (temperature, thinking, output_config) = match &self.model {
            // Thinking is always on — always emit the adaptive block (§5: the
            // request is a complete record of what the model sees).
            Model::Fable5(p) => (None, Some(adaptive(Some(p.display.as_str()))), effort(p.effort.as_str())),
            Model::Opus4_8(p) => (
                None,
                match &p.thinking {
                    Opus4_8Thinking::Off => None,
                    Opus4_8Thinking::Adaptive { display } => Some(adaptive(Some(display.as_str()))),
                },
                effort(p.effort.as_str()),
            ),
            // Adaptive thinking is always emitted explicitly; "off" is the explicit
            // disabled block, not an omitted field (§5). No sampling.
            Model::Sonnet5(p) => (
                None,
                Some(match &p.thinking {
                    Sonnet5Thinking::Adaptive { display } => adaptive(Some(display.as_str())),
                    Sonnet5Thinking::Disabled => {
                        ThinkingWire::Disabled(DisabledThinking { kind: ThinkingType::Disabled.as_str() })
                    }
                }),
                effort(p.effort.as_str()),
            ),
            Model::Sonnet4_6(p) => {
                let (t, th) = match p.sampling {
                    Sonnet4_6Sampling::Temperature(t) => (Some(t.get()), None),
                    Sonnet4_6Sampling::Adaptive { display } => (None, Some(adaptive(Some(display.as_str())))),
                };
                (t, th, effort(p.effort.as_str()))
            }
            Model::Haiku4_5(p) => {
                let th = match p.thinking {
                    Haiku4_5Thinking::Off => None,
                    Haiku4_5Thinking::Enabled { budget_tokens } => Some(enabled(budget_tokens)),
                };
                (Some(p.temperature.get()), th, None)
            }
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
        serde_json::to_value(Request::new(&Context::new(), m, 1024).unwrap()).unwrap()
    }
    fn count(id: ModelId) -> Value {
        serde_json::to_value(CountRequest::new(&Context::new(), id)).unwrap()
    }
    fn approx(v: &Value, expected: f64) {
        let got = v.as_f64().expect("not a number");
        assert!((got - expected).abs() < 1e-4, "expected ~{expected}, got {got}");
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

        let v =
            req(Model::opus_4_8().with_adaptive_thinking(ThinkingDisplay::Omitted).with_effort(Opus4_8Effort::Xhigh));
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
        // field would also mean on, but §5 keeps the body a complete record).
        assert_eq!(v["thinking"]["type"], "adaptive");
        assert_eq!(v["thinking"]["display"], "omitted");
        assert_eq!(v["output_config"]["effort"], "high");
    }

    #[test]
    fn sonnet_5_adaptive_summarized_xhigh() {
        // `xhigh` is accepted on Sonnet 5 (unlike Sonnet 4.6).
        let v = req(Model::sonnet_5()
            .with_adaptive_thinking(ThinkingDisplay::Summarized)
            .with_effort(Sonnet5Effort::Xhigh));
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
    fn sonnet_4_6_default_uses_temperature() {
        let v = req(Model::sonnet_4_6());
        assert_eq!(v["model"], "claude-sonnet-4-6");
        approx(&v["temperature"], 1.0);
        assert!(v.get("thinking").is_none());
        assert_eq!(v["output_config"]["effort"], "high");
    }

    #[test]
    fn sonnet_4_6_adaptive_drops_temperature() {
        let v = req(Model::sonnet_4_6()
            .with_adaptive_thinking(ThinkingDisplay::Summarized)
            .with_effort(Sonnet4_6Effort::Max));
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
        let ctx = Context::new();
        let v =
            serde_json::to_value(Request::new(&ctx, Model::haiku_4_5().with_thinking(1024), 1536).unwrap()).unwrap();
        assert_eq!(v["thinking"]["type"], "enabled");
        assert_eq!(v["thinking"]["budget_tokens"], 1024);
        assert!(v["thinking"].get("display").is_none(), "`display` is adaptive-only");
        approx(&v["temperature"], 1.0);

        assert!(req(Model::haiku_4_5().with_thinking(2048).with_thinking_off()).get("thinking").is_none());
    }

    #[test]
    fn haiku_thinking_budget_must_be_below_max_tokens() {
        let ctx = Context::new();
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

    #[test]
    fn count_request_omits_sampling_and_max_tokens() {
        let v = count(ModelId::Opus4_8);
        assert_eq!(v["model"], "claude-opus-4-8");
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
        let m: Model = Model::opus_4_8().with_adaptive_thinking(ThinkingDisplay::Summarized).into();
        assert_eq!(m.id(), ModelId::Opus4_8);
        assert_eq!(m.id().api_id(), m.api_id());
    }

    #[test]
    fn stop_sequences_roundtrip() {
        let ctx = Context::new();
        let v = serde_json::to_value(
            Request::new(&ctx, Model::opus_4_8(), 1024).unwrap().stop_sequences(vec!["STOP".into(), "END".into()]),
        )
        .unwrap();
        assert_eq!(v["stop_sequences"][0], "STOP");
        assert_eq!(v["stop_sequences"][1], "END");
        // Empty vec is skipped.
        assert!(req(Model::opus_4_8()).get("stop_sequences").is_none());
    }
}
