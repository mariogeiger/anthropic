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
    Fable5, Fable5_1, Fable5_1Effort, Fable5Effort, Haiku4_5, Haiku4_5Thinking, Model, ModelId, Month, Opus4_8,
    Opus4_8Effort, Opus4_8Thinking, Opus5, Opus5Effort, Opus5Thinking, Opus5ThinkingOffEffort, Pricing, Sonnet4_6,
    Sonnet4_6Effort, Sonnet4_6Sampling, Sonnet5, Sonnet5Effort, Sonnet5Thinking, Temperature, TemperatureError,
    YearMonth,
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
    /// This model requires tool choice to remain `auto` or `none`, so forcing
    /// `any` or one named tool is rejected before serialization.
    ForcedToolChoiceUnsupported {
        /// The model that must choose its own call.
        model: ModelId,
        /// The rejected tool-choice wire name.
        choice: &'static str,
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
            RequestError::ForcedToolChoiceUnsupported { model, choice } => {
                write!(f, "{} does not accept forced tool_choice {choice}; use auto or none", model.api_id())
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
    ///
    /// # Errors
    ///
    /// Returns [`RequestError::ForcedToolChoiceUnsupported`] when Fable 5.1 is
    /// asked for `any` or one named tool. It supports only `auto` and `none`.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Result<Self, RequestError> {
        if !self.model.id().accepts_forced_tool_choice()
            && matches!(choice, ToolChoice::Any { .. } | ToolChoice::Tool { .. })
        {
            return Err(RequestError::ForcedToolChoiceUnsupported { model: self.model.id(), choice: choice.as_str() });
        }
        self.tool_choice = Some(choice);
        Ok(self)
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
            Model::Fable5_1(p) => (None, Some(adaptive(Some(p.display))), effort(p.effort.as_str())),
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
#[path = "request/tests.rs"]
mod tests;
