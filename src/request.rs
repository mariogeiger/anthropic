//! Per-call parameters, and the bodies they serialize into.
//!
//! Conversation state lives in [`Context`] and is stable across turns; what
//! changes per call is here. That split is what lets one conversation be sent to
//! different models or with different sampling settings without touching the
//! type that upholds the cache invariants.
//!
//! [`Request`] is the `/v1/messages` body and [`CountRequest`] its
//! token-counting sibling, which takes only a [`ModelId`] because the endpoint
//! ignores sampling and thinking — exposing them there would let a caller set
//! values the payload silently drops.
//!
//! Serialization is explicit: what a request value says is what the model sees.
//! There is no omit-if-default, and scalar parameters with a documented
//! server-side default are emitted anyway, so a change to the provider's
//! defaults cannot quietly change behavior. The two exceptions are transport
//! rather than content — `stream`, which the model never sees, and
//! `tool_choice`, whose absence must stay byte-identical to a request that never
//! mentioned it or the message cache key moves.

use crate::context::{Context, Message, SystemPrompt, Tool};
use crate::tool_choice::ToolChoice;
use crate::values::{OutputFormatType, ServiceTier, ThinkingDisplay, ThinkingType};
use serde::Serialize;

// The model types were part of this module before they outgrew it. Named
// explicitly rather than glob-imported so `anthropic::request::Model` keeps
// meaning what it always did without shadowing this module's own wire structs.
// The canonical home is [`crate::model`].
pub use crate::model::{
    Fable5, Fable5Effort, Haiku4_5, Haiku4_5Thinking, Model, ModelId, Month, Opus4_8, Opus4_8Effort, Opus4_8Thinking,
    Opus5, Opus5Effort, Opus5Thinking, Opus5ThinkingOffEffort, Pricing, Sonnet4_6, Sonnet4_6Effort, Sonnet4_6Sampling,
    Sonnet5, Sonnet5Effort, Sonnet5Thinking, Temperature, TemperatureError, YearMonth,
};

// ── Request ──────────────────────────────────────────────────────────────────

/// An opaque identifier for the end user a request is made on behalf of.
///
/// Anthropic uses it to detect abuse, and documents that it must carry no
/// identifying information — a UUID or a hash, never a name, email, or phone
/// number. A newtype rather than a bare `String` so the constructor is the place
/// that warning lives, and so `metadata` cannot be confused for anything else.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EndUserId(String);

impl EndUserId {
    /// An opaque identifier, at most 512 characters.
    ///
    /// Pass a UUID or a hash. Anything identifying — a name, an email address, a
    /// phone number — must not go here; the API accepts it and Anthropic asks that
    /// you do not send it, which is a rule a type cannot enforce and documentation
    /// must therefore state.
    pub fn new(id: impl Into<String>) -> Result<Self, EndUserIdError> {
        let id = id.into();
        // The bound is on characters rather than bytes: the API documents
        // `maxLength: 512`, which JSON Schema counts in code points.
        let length = id.chars().count();
        if length > Self::MAX_CHARS {
            return Err(EndUserIdError::TooLong { length });
        }
        Ok(Self(id))
    }

    /// The documented maximum length, in characters.
    pub const MAX_CHARS: usize = 512;

    /// The identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Why an end-user identifier was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndUserIdError {
    /// Longer than the documented 512 characters, which the API rejects.
    TooLong {
        /// How many characters were given.
        length: usize,
    },
}

impl std::fmt::Display for EndUserIdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EndUserIdError::TooLong { length } => {
                write!(f, "end-user id is {length} characters, over the documented maximum of {}", EndUserId::MAX_CHARS)
            }
        }
    }
}

impl std::error::Error for EndUserIdError {}

/// A JSON schema the answer must satisfy.
///
/// The model's answer arrives as ordinary text in a text block; what this changes
/// is that the text is guaranteed to parse against the schema. So it is a per-call
/// parameter and not a content type: nothing about the conversation changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputFormat {
    /// The schema the answer conforms to.
    pub schema: serde_json::Value,
}

impl OutputFormat {
    /// An answer conforming to this JSON schema.
    pub fn json_schema(schema: serde_json::Value) -> Self {
        Self { schema }
    }
}

/// Wire shape: the enum in the `type` field, as every closed vocabulary is.
#[derive(Serialize)]
struct OutputFormatWire<'a> {
    #[serde(rename = "type")]
    kind: OutputFormatType,
    schema: &'a serde_json::Value,
}

impl Serialize for OutputFormat {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        OutputFormatWire { kind: OutputFormatType::JsonSchema, schema: &self.schema }.serialize(s)
    }
}

/// Construction-time rejection for a cross-field invariant the state/per-call
/// split cannot express in the type system. Same "error before commit" approach
/// as the cache ops: refuse rather than let the API answer with a 400.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestError {
    /// Legacy fixed-budget thinking requires `budget_tokens < max_tokens`; the
    /// API rejects `budget_tokens >= max_tokens` with a 400. Only reachable on
    /// Haiku 4.5 via `with_thinking` — adaptive-thinking models carry no budget.
    ThinkingBudgetExceedsMaxTokens {
        /// The budget asked for.
        budget_tokens: u32,
        /// The output budget it had to stay below.
        max_tokens: u32,
    },
    /// `max_tokens` must fall within `1..=ModelId::max_output_tokens()`. The API
    /// rejects 0 and any value above the model's synchronous max output with a 400.
    MaxTokensOutOfRange {
        /// The value asked for.
        max_tokens: u32,
        /// The model's maximum synchronous output.
        max_output: u32,
    },
    /// A mid-conversation system message must end `messages` or be followed by an
    /// assistant turn; the API answers `role 'system' must precede an 'assistant'
    /// message or end the array`.
    ///
    /// Checked here rather than at
    /// [`crate::context::Context::push_system`] because it is a property of the
    /// *finished* history: an append that was legal becomes illegal when a user
    /// turn is appended after it, so only the request knows.
    SystemMessageNotFollowedByAssistant {
        /// Index in `messages` of the offending system message.
        at: usize,
    },
    /// Every declared tool was deferred. The API answers `At least one tool must
    /// have defer_loading=false`, because a model served an empty schema has
    /// nothing to search with.
    ///
    /// Checked here rather than on [`crate::context::Tool`] because it is a
    /// relation across the whole list, which no single tool's type can carry.
    EveryToolDeferred {
        /// How many tools were declared, all of them deferred.
        tools: usize,
    },
    /// The conversation holds a mid-conversation system message and this model
    /// does not accept one. The documentation states the feature is available on
    /// Fable 5, Mythos 5, Opus 4.8 and Opus 5, and "not available on Claude
    /// Sonnet 5; use the top-level `system` field instead".
    ///
    /// # Why this is refused here and not made unrepresentable
    ///
    /// This crate's rule is that a refused combination should not compile, and a
    /// type per model is how it holds that rule. The rule is followed here as far
    /// as it reaches, and this is where it stops.
    ///
    /// A model-specific *parameter* lives on the model's own type, so omitting it
    /// there makes the bad request unwritable with no runtime check at all. This
    /// combination is not of that shape. It pairs the model, which is a per-call
    /// parameter, with the conversation's message list, which is
    /// [`crate::context::Context`] — and those are deliberately different types
    /// so that *one* conversation can be sent to *several* models. Making the
    /// pairing unrepresentable would mean parameterizing `Context` by model,
    /// which destroys exactly that property: a conversation carrying an
    /// instruction added mid-session could then never be counted against a Sonnet
    /// 5 tokenizer, or replayed on a cheaper model, even though both are
    /// legitimate. The type would buy one 400 at the cost of the crate's central
    /// separation.
    ///
    /// It also cannot be caught at append time. `Context::push_system` does not
    /// know which model the conversation will be sent to, and by design never
    /// will. The request is the first place both facts are present, which makes
    /// it the only place the check can live — the same reason
    /// [`Self::MaxTokensOutOfRange`] and
    /// [`Self::SystemMessageNotFollowedByAssistant`] are here.
    ///
    /// So the refusal is typed, names the model, and names the index to remove.
    MidConversationSystemMessageUnsupported {
        /// The model that does not accept one.
        model: ModelId,
        /// Index in `messages` of the first system message, the entry to remove.
        at: usize,
    },
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestError::ThinkingBudgetExceedsMaxTokens { budget_tokens, max_tokens } => {
                write!(f, "thinking budget_tokens ({budget_tokens}) must be less than max_tokens ({max_tokens})")
            }
            RequestError::MaxTokensOutOfRange { max_tokens, max_output } => {
                write!(f, "max_tokens ({max_tokens}) must be in 1..={max_output} for this model")
            }
            RequestError::SystemMessageNotFollowedByAssistant { at } => {
                write!(f, "the system message at index {at} must end the conversation or precede an assistant turn")
            }
            RequestError::EveryToolDeferred { tools } => {
                write!(f, "all {tools} declared tools are deferred; at least one must not be")
            }
            RequestError::MidConversationSystemMessageUnsupported { model, at } => write!(
                f,
                "{} does not accept a mid-conversation system message; \
                 remove the one at index {at} and use the top-level system prompt",
                model.api_id()
            ),
        }
    }
}

impl std::error::Error for RequestError {}

/// Borrowed [`Context`] + per-call params. Serializes to `POST /v1/messages`.
///
/// Private fields with readers, not public fields, and for one reason:
/// `max_tokens` and `model` are bound by a cross-field invariant that no single
/// type can carry. `max_tokens` must lie in `1..=` *this model's* maximum output,
/// and Haiku 4.5's legacy thinking budget must stay below it. [`Request::new`]
/// checks both, so it must be the only way in — a public field would let a caller
/// assign around the check and get the 400 the check exists to prevent. The
/// builders below only move between states that stay valid.
pub struct Request<'a> {
    context: &'a Context,
    model: Model,
    max_tokens: u32,
    stop_sequences: Vec<String>,
    stream: bool,
    tool_choice: Option<ToolChoice>,
    service_tier: ServiceTier,
    end_user_id: Option<EndUserId>,
    output_format: Option<OutputFormat>,
}

impl<'a> Request<'a> {
    /// `new` is the only construction path, which is what makes the checks
    /// unbypassable rather than merely conventional. It validates the invariants
    /// no single type can express: `max_tokens` must fall within `1..=` the
    /// model's max output, and legacy `budget_tokens` (Haiku 4.5 only) must be
    /// below `max_tokens`.
    pub fn new(context: &'a Context, model: impl Into<Model>, max_tokens: u32) -> Result<Self, RequestError> {
        let model = model.into();
        let max_output = model.id().max_output_tokens();
        if max_tokens == 0 || max_tokens > max_output {
            return Err(RequestError::MaxTokensOutOfRange { max_tokens, max_output });
        }
        // A system message is legal only at the end or immediately before an
        // assistant turn, and that is decidable only now that the history is
        // final — see `RequestError::SystemMessageNotFollowedByAssistant`.
        if let Some(at) = context.misplaced_system_message() {
            return Err(RequestError::SystemMessageNotFollowedByAssistant { at });
        }
        // Availability is per model, and the model is only known here — see
        // `RequestError::MidConversationSystemMessageUnsupported`.
        if !model.id().accepts_mid_conversation_system_message()
            && let Some(at) = context.first_system_message()
        {
            return Err(RequestError::MidConversationSystemMessageUnsupported { model: model.id(), at });
        }
        let tools = context.tools();
        if !tools.is_empty() && tools.iter().all(|tool| tool.defer_loading) {
            return Err(RequestError::EveryToolDeferred { tools: tools.len() });
        }
        if let Model::Haiku4_5(h) = &model
            && let Haiku4_5Thinking::Enabled { budget_tokens } = h.thinking
            && budget_tokens >= max_tokens
        {
            return Err(RequestError::ThinkingBudgetExceedsMaxTokens { budget_tokens, max_tokens });
        }
        Ok(Self {
            context,
            model,
            max_tokens,
            stop_sequences: Vec::new(),
            stream: false,
            tool_choice: None,
            service_tier: ServiceTier::Auto,
            end_user_id: None,
            output_format: None,
        })
    }

    /// The conversation state this call is made against.
    pub fn context(&self) -> &'a Context {
        self.context
    }

    /// Which model answers, with its per-call parameters.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The output-token budget, already checked against the model's maximum.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Ends generation when any of these sequences appears, reported back as
    /// [`crate::values::StopReason::StopSequence`].
    pub fn with_stop_sequences(mut self, seqs: Vec<String>) -> Self {
        self.stop_sequences = seqs;
        self
    }

    /// The sequences whose appearance ends generation.
    pub fn stop_sequences(&self) -> &[String] {
        &self.stop_sequences
    }

    /// Asks for the response as Server-Sent Events.
    ///
    /// A per-call decision, not a property of the conversation: the same
    /// [`Context`] may be streamed on one turn and not the next, and the request
    /// body is otherwise identical. Decode the events with
    /// [`crate::stream::StreamEvent`] and accumulate them with
    /// [`crate::settle::Settling`]; decode the single body with
    /// [`crate::response::Response`].
    pub fn streamed(mut self) -> Self {
        self.stream = true;
        self
    }

    /// Whether Server-Sent Events were asked for.
    pub fn is_streamed(&self) -> bool {
        self.stream
    }

    /// Constrains which tool the model may or must call.
    ///
    /// Invalidates the message cache on change; see [`ToolChoice`].
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Whether, and which, tool the model must call. `None` leaves the field off
    /// the wire, which the API reads as `auto`.
    pub fn tool_choice(&self) -> Option<&ToolChoice> {
        self.tool_choice.as_ref()
    }

    /// Which capacity may serve the request.
    ///
    /// A plain field with the documented default, always emitted: the API accepts
    /// it unconditionally and documents `auto`, so there is no runtime absence to
    /// represent. Read [`crate::usage::Usage::service_tier`] to learn which tier
    /// actually served it, which need not be the one asked for.
    pub fn with_service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = tier;
        self
    }

    /// Which capacity may serve the request.
    pub fn service_tier(&self) -> ServiceTier {
        self.service_tier
    }

    /// Names the end user this request is made on behalf of, for abuse detection.
    ///
    /// A real runtime distinction rather than a defaulted value — a request is
    /// either made on someone's behalf or it is not — so it is an `Option` and the
    /// absent case sends no `metadata` at all.
    pub fn with_end_user_id(mut self, id: EndUserId) -> Self {
        self.end_user_id = Some(id);
        self
    }

    /// The end user this request names, if any.
    pub fn end_user_id(&self) -> Option<&EndUserId> {
        self.end_user_id.as_ref()
    }

    /// Requires the answer to conform to a JSON schema.
    ///
    /// The answer still arrives as text in a text block; what changes is that the
    /// text parses against the schema.
    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    /// The schema the answer must satisfy, if any.
    pub fn output_format(&self) -> Option<&OutputFormat> {
        self.output_format.as_ref()
    }
}

// ── CountRequest ─────────────────────────────────────────────────────────────

/// Request body for `POST /v1/messages/count_tokens`. Takes only a [`ModelId`]:
/// the endpoint ignores sampling/thinking/effort, so exposing them here would
/// let callers set values the wire payload silently drops, which explicit
/// serialization forbids.
///
/// Private fields with readers, matching [`Request`]: this endpoint's body is
/// fixed at construction and there is nothing to reassign afterwards, so a
/// settable field would only offer a way to build a body no constructor would.
pub struct CountRequest<'a> {
    context: &'a Context,
    model: ModelId,
}

impl<'a> CountRequest<'a> {
    /// A token-count request for this conversation and model.
    pub fn new(context: &'a Context, model: ModelId) -> Self {
        Self { context, model }
    }

    /// The conversation state to count.
    pub fn context(&self) -> &'a Context {
        self.context
    }

    /// Which model's tokenizer counts it.
    pub fn model(&self) -> ModelId {
        self.model
    }
}

// ── Serialization ────────────────────────────────────────────────────────────
// Private wire structs: Option = real runtime absence, empty vecs skipped —
// never "omit if equal to default".

#[derive(Serialize)]
struct AdaptiveThinking {
    #[serde(rename = "type")]
    kind: ThinkingType,
    #[serde(skip_serializing_if = "Option::is_none")]
    display: Option<ThinkingDisplay>,
}

#[derive(Serialize)]
struct EnabledThinking {
    #[serde(rename = "type")]
    kind: ThinkingType,
    budget_tokens: u32,
}

#[derive(Serialize)]
struct DisabledThinking {
    #[serde(rename = "type")]
    kind: ThinkingType,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ThinkingWire {
    Adaptive(AdaptiveThinking),
    Enabled(EnabledThinking),
    Disabled(DisabledThinking),
}

#[derive(Serialize)]
struct OutputConfig<'a> {
    // A `&'static str` rather than a per-model effort enum, because there is no
    // one such enum: each model accepts its own effort set, and this struct
    // is private, so no caller can write the field at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'a OutputFormat>,
}

/// The `metadata` object, whose only documented field is the end-user id. A
/// wrapper rather than a flattened field because the API nests it, and a request
/// that names no end user sends no `metadata` at all.
#[derive(Serialize)]
struct Metadata<'a> {
    user_id: &'a EndUserId,
}

#[derive(Serialize)]
struct RequestWire<'a> {
    model: &'static str,
    max_tokens: u32,
    // Emitted only when streaming was asked for: the API reads an absent field
    // as `false`, and this is a transport choice the model never sees, so it is
    // outside the rule that the body is a complete record of what the model sees.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
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
    output_config: Option<OutputConfig<'a>>,
    // Always emitted with its documented default: a scalar the API accepts
    // unconditionally is a complete record of what was asked for.
    service_tier: ServiceTier,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Metadata<'a>>,
    // Absent means `auto`, the documented default. Omitting rather than
    // defaulting keeps the message-cache key identical to a request that never
    // mentioned the field — see `ToolChoice`.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a ToolChoice>,
}

impl Serialize for Request<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let adaptive = |display| ThinkingWire::Adaptive(AdaptiveThinking { kind: ThinkingType::Adaptive, display });
        let enabled =
            |budget_tokens| ThinkingWire::Enabled(EnabledThinking { kind: ThinkingType::Enabled, budget_tokens });
        let effort = |e: &'static str| Some(e);
        let (temperature, thinking, output_config) = match &self.model {
            // Thinking is always on — always emit the adaptive block (the
            // request is a complete record of what the model sees).
            Model::Fable5(p) => (None, Some(adaptive(Some(p.display))), effort(p.effort.as_str())),
            // Adaptive thinking is always emitted explicitly; "off" is the explicit
            // disabled block, not an omitted field. No sampling: `temperature`
            // is rejected as deprecated on this model.
            Model::Opus5(p) => match &p.thinking {
                Opus5Thinking::Adaptive { display, effort: e } => {
                    (None, Some(adaptive(Some(*display))), effort(e.as_str()))
                }
                Opus5Thinking::Disabled { effort: e } => (
                    None,
                    Some(ThinkingWire::Disabled(DisabledThinking { kind: ThinkingType::Disabled })),
                    effort(e.as_str()),
                ),
            },
            Model::Opus4_8(p) => (
                None,
                match &p.thinking {
                    Opus4_8Thinking::Off => None,
                    Opus4_8Thinking::Adaptive { display } => Some(adaptive(Some(*display))),
                },
                effort(p.effort.as_str()),
            ),
            // Adaptive thinking is always emitted explicitly; "off" is the explicit
            // disabled block, not an omitted field. No sampling.
            Model::Sonnet5(p) => (
                None,
                Some(match &p.thinking {
                    Sonnet5Thinking::Adaptive { display } => adaptive(Some(*display)),
                    Sonnet5Thinking::Disabled => {
                        ThinkingWire::Disabled(DisabledThinking { kind: ThinkingType::Disabled })
                    }
                }),
                effort(p.effort.as_str()),
            ),
            Model::Sonnet4_6(p) => {
                let (t, th) = match p.sampling {
                    Sonnet4_6Sampling::Temperature(t) => (Some(t.get()), None),
                    Sonnet4_6Sampling::Adaptive { display } => (None, Some(adaptive(Some(display)))),
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
        // `output_config` carries effort and format independently: Haiku 4.5 takes
        // no effort but does take a format, so the object appears whenever either
        // half is present and is omitted only when neither is.
        let format = self.output_format.as_ref();
        let output_config =
            (output_config.is_some() || format.is_some()).then_some(OutputConfig { effort: output_config, format });
        RequestWire {
            model: self.model.api_id(),
            max_tokens: self.max_tokens,
            stream: self.stream,
            temperature,
            thinking,
            output_config,
            service_tier: self.service_tier,
            metadata: self.end_user_id.as_ref().map(|user_id| Metadata { user_id }),
            stop_sequences: &self.stop_sequences,
            system: self.context.system.as_ref(),
            tools: &self.context.tools,
            messages: &self.context.messages,
            tool_choice: self.tool_choice.as_ref(),
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
    use crate::ThinkingDisplay;
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
        // field would also mean on, but the body stays a complete record).
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
    fn model_constants() {
        assert_eq!(ModelId::Opus4_8.context_window_tokens(), 1_000_000);
        assert_eq!(ModelId::Haiku4_5.context_window_tokens(), 200_000);
        assert_eq!(ModelId::Sonnet5.max_output_tokens(), 128_000);
        assert_eq!(ModelId::Haiku4_5.max_output_tokens(), 64_000);
        assert_eq!(ModelId::Sonnet5.knowledge_cutoff(), YearMonth::new(2026, Month::January));
        assert_eq!(ModelId::Sonnet4_6.knowledge_cutoff(), YearMonth::new(2025, Month::August));
        assert_eq!(ModelId::Sonnet4_6.training_cutoff(), YearMonth::new(2026, Month::January));
        assert_eq!(ModelId::Haiku4_5.training_cutoff(), YearMonth::new(2025, Month::July));
        assert_eq!(
            ModelId::Opus4_8.price_per_mtok(),
            Pricing { input_cents_per_mtok: 500, output_cents_per_mtok: 2_500 }
        );
        assert_eq!(
            ModelId::Haiku4_5.price_per_mtok(),
            Pricing { input_cents_per_mtok: 100, output_cents_per_mtok: 500 }
        );
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
        let ctx = Context::new();
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

    /// `service_tier` is a scalar the API accepts unconditionally and documents a
    /// default for, so it is always emitted — the body is a complete record.
    #[test]
    fn the_service_tier_is_always_emitted_at_its_documented_default() {
        assert_eq!(req(Model::opus_5())["service_tier"], "auto");
        let ctx = Context::new();
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
        let ctx = Context::new();
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
        let ctx = Context::new();
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
        let ctx = Context::new();
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
            Request::new(&ctx, Model::opus_4_8(), 1024).unwrap().with_stop_sequences(vec!["STOP".into(), "END".into()]),
        )
        .unwrap();
        assert_eq!(v["stop_sequences"][0], "STOP");
        assert_eq!(v["stop_sequences"][1], "END");
        // Empty vec is skipped.
        assert!(req(Model::opus_4_8()).get("stop_sequences").is_none());
    }
}
