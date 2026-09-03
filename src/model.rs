//! One type per model, carrying only the parameters that model accepts.
//!
//! The API returns 400 for a parameter a model does not take, and the subsets
//! differ per model in ways no shared struct can express honestly. So each model
//! is its own type: a parameter Sonnet 4.6 rejects does not exist on `Sonnet4_6`,
//! and the compiler refuses the request the API would have refused.
//!
//! The differences are not cosmetic. Thinking is always on for Fable 5.1 and
//! Fable 5 and has no off state; Fable 5.1 additionally refuses forced tool
//! choice. On Opus 4.8, "off" is an *omitted* `thinking` field; on Opus 5 and
//! Sonnet 5 an omitted field means thinking stays *on*, so off must be stated
//! explicitly. Opus 5 goes further and makes the accepted effort range depend on
//! whether thinking is on, which is why its effort lives inside
//! [`Opus5Thinking`] rather than beside it: the refused combination is
//! unwritable rather than rejected at runtime.
//!
//! Mutually exclusive settings are sum types, never two optional fields a caller
//! must keep in sync — [`Sonnet4_6Sampling`] is temperature *or* adaptive
//! thinking, because the API pins temperature to 1.0 under adaptive thinking.
//!
//! Alongside the parameters, [`ModelId`] carries the documented per-model
//! constants: context window, maximum output, cacheable-prefix minimum,
//! knowledge and training cutoffs, and list pricing.

#![allow(non_camel_case_types)]

use crate::ThinkingDisplay;
use crate::values::api_enum;

// ── Temperature ──────────────────────────────────────────────────────────────

/// Sampling temperature. API-accepted range is `[0.0, 1.0]` and the value must
/// be finite — constructing a `Temperature` is the only way to prove that,
/// so downstream code never has to re-check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Temperature(f32);

/// Why a temperature was refused.
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
    /// A temperature, if the value is finite and within `[0.0, 1.0]`.
    pub fn new(v: f32) -> Result<Self, TemperatureError> {
        if !v.is_finite() {
            Err(TemperatureError::NotFinite)
        } else if !(0.0..=1.0).contains(&v) {
            Err(TemperatureError::OutOfRange(v))
        } else {
            Ok(Self(v))
        }
    }

    /// The value.
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
    /// `claude-fable-5-1`, the current frontier tier.
    Fable5_1,
    /// `claude-fable-5`, the prior frontier tier.
    Fable5,
    /// `claude-opus-5`, the current Opus tier.
    Opus5,
    /// `claude-opus-4-8`, the prior Opus tier.
    Opus4_8,
    /// `claude-sonnet-5`, the current Sonnet tier.
    Sonnet5,
    /// `claude-sonnet-4-6`, the prior Sonnet tier.
    Sonnet4_6,
    /// `claude-haiku-4-5`, the small tier.
    Haiku4_5,
}

/// Standard list price per million tokens (MTok), in US cents (e.g. 500 = $5.00).
///
/// Cache reads are explicit because Fable 5.1's $0.25 rate is 2.5% of its base
/// input price, while the other modeled models use the usual 10% rate. Batch,
/// cache-write, promotional, and per-platform rates are not represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pricing {
    /// Price of a million uncached input tokens, in US cents.
    pub input_cents_per_mtok: u32,
    /// Price of a million input tokens read from prompt cache, in US cents.
    pub cache_read_input_cents_per_mtok: u32,
    /// Price of a million output tokens, in US cents.
    pub output_cents_per_mtok: u32,
}

/// A calendar month, used for documented model cutoff dates (no day granularity).
///
/// Readers plus one checked constructor, not public fields, because "the month, 1
/// through 12" is a claim a `pub month: u8` cannot keep. Ordering is derived and
/// therefore correct: the fields are declared most-significant first, so the
/// lexicographic order Rust derives *is* chronological order, and comparing two
/// cutoffs needs no helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct YearMonth {
    year: u16,
    month: Month,
}

impl YearMonth {
    /// A calendar month.
    pub fn new(year: u16, month: Month) -> Self {
        Self { year, month }
    }

    /// The year.
    pub fn year(self) -> u16 {
        self.year
    }

    /// The month.
    pub fn month(self) -> Month {
        self.month
    }

    /// The month as its ordinal, 1 for January through 12 for December.
    pub fn month_number(self) -> u8 {
        self.month as u8 + 1
    }
}

/// A month of the year.
///
/// Twelve variants rather than a range-checked integer, so there is no invalid
/// month to construct and no validation to run. Ordering is declaration order,
/// which is calendar order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Month {
    /// January.
    January,
    /// February.
    February,
    /// March.
    March,
    /// April.
    April,
    /// May.
    May,
    /// June.
    June,
    /// July.
    July,
    /// August.
    August,
    /// September.
    September,
    /// October.
    October,
    /// November.
    November,
    /// December.
    December,
}

impl Month {
    /// Every month, in calendar order.
    pub const ALL: [Month; 12] = [
        Month::January,
        Month::February,
        Month::March,
        Month::April,
        Month::May,
        Month::June,
        Month::July,
        Month::August,
        Month::September,
        Month::October,
        Month::November,
        Month::December,
    ];

    /// The month an ordinal names, 1 for January through 12 for December, or
    /// `None` for an ordinal that names no month.
    pub fn from_number(n: u8) -> Option<Self> {
        Self::ALL.get(usize::from(n.checked_sub(1)?)).copied()
    }
}

impl ModelId {
    /// The `model` field value sent on the wire.
    pub fn api_id(self) -> &'static str {
        match self {
            ModelId::Fable5_1 => "claude-fable-5-1",
            ModelId::Fable5 => "claude-fable-5",
            ModelId::Opus5 => "claude-opus-5",
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
            ModelId::Fable5_1 | ModelId::Fable5 => 512,
            ModelId::Opus5 => 512,
            ModelId::Opus4_8 => 1_024,
            ModelId::Sonnet5 => 1_024,
            ModelId::Sonnet4_6 => 1_024,
            ModelId::Haiku4_5 => 4_096,
        }
    }

    /// Total context-window size in tokens. Input and output share this budget.
    pub fn context_window_tokens(self) -> u32 {
        match self {
            ModelId::Fable5_1 | ModelId::Fable5 => 1_000_000,
            ModelId::Opus5 => 1_000_000,
            ModelId::Opus4_8 => 1_000_000,
            ModelId::Sonnet5 => 1_000_000,
            ModelId::Sonnet4_6 => 1_000_000,
            ModelId::Haiku4_5 => 200_000,
        }
    }

    /// Whether this model accepts a `{"role": "system"}` entry inside `messages`.
    ///
    /// A closed list, not a tier rule: the feature is documented as available on
    /// Fable 5.1, Fable 5, Mythos 5.1, Mythos 5, Opus 4.8 and Opus 5, and
    /// *not* on Sonnet 5, where the documentation says to use the top-level
    /// `system` field instead. Every model the list omits is therefore `false`
    /// here, so adding a model states its answer rather than inheriting a guess.
    ///
    /// Mid-conversation *tool* changes carry the same availability, and need no
    /// second predicate: a [`crate::system::SystemBlock::ToolAddition`] or
    /// `ToolRemoval` can only ride inside a system message, so refusing the
    /// message refuses the tool change with it.
    ///
    /// [`crate::request::Request::new`] refuses a conversation holding one on a
    /// model that answers `false`; see
    /// [`crate::request::RequestError::MidConversationSystemMessageUnsupported`]
    /// for why that is a runtime refusal rather than a type error.
    pub fn accepts_mid_conversation_system_message(self) -> bool {
        match self {
            ModelId::Fable5_1 | ModelId::Fable5 | ModelId::Opus5 | ModelId::Opus4_8 => true,
            ModelId::Sonnet5 | ModelId::Sonnet4_6 | ModelId::Haiku4_5 => false,
        }
    }

    /// Whether this model accepts an effort-only system message.
    ///
    /// The beta is a closed model list: Fable 5.1, Mythos 5.1, and Opus 5.
    /// Mythos is not modeled by this crate, so exactly two variants answer true.
    pub fn accepts_per_message_effort(self) -> bool {
        matches!(self, ModelId::Fable5_1 | ModelId::Opus5)
    }

    /// Whether this model accepts `tool_choice` values that force a call.
    ///
    /// Fable 5.1 rejects both `any` and a named `tool`: always-on thinking must
    /// be allowed to run before the model chooses a call. Every earlier model
    /// carried here accepts the full tool-choice vocabulary.
    pub fn accepts_forced_tool_choice(self) -> bool {
        !matches!(self, ModelId::Fable5_1)
    }

    /// Maximum output tokens in a single synchronous Messages API response.
    /// `Request::new` rejects `max_tokens` outside `1..=max_output_tokens()`.
    /// (The Message Batches API permits more on some models via a beta header,
    /// which this crate does not model.)
    pub fn max_output_tokens(self) -> u32 {
        match self {
            ModelId::Fable5_1 | ModelId::Fable5 => 128_000,
            ModelId::Opus5 => 128_000,
            ModelId::Opus4_8 => 128_000,
            ModelId::Sonnet5 => 128_000,
            ModelId::Sonnet4_6 => 128_000,
            ModelId::Haiku4_5 => 64_000,
        }
    }

    /// Reliable knowledge cutoff: the date through which the model's knowledge is
    /// most extensive and reliable (per the Anthropic models docs).
    pub fn knowledge_cutoff(self) -> YearMonth {
        let (year, month) = match self {
            ModelId::Fable5_1 => (2026, Month::June),
            ModelId::Fable5 => (2026, Month::January),
            ModelId::Opus5 => (2026, Month::May),
            ModelId::Opus4_8 => (2026, Month::January),
            ModelId::Sonnet5 => (2026, Month::January),
            ModelId::Sonnet4_6 => (2025, Month::August),
            ModelId::Haiku4_5 => (2025, Month::February),
        };
        YearMonth::new(year, month)
    }

    /// Training data cutoff: the broader end of the training-data date range.
    pub fn training_cutoff(self) -> YearMonth {
        let (year, month) = match self {
            ModelId::Fable5_1 => (2026, Month::June),
            ModelId::Fable5 => (2026, Month::January),
            ModelId::Opus5 => (2026, Month::May),
            ModelId::Opus4_8 => (2026, Month::January),
            ModelId::Sonnet5 => (2026, Month::January),
            ModelId::Sonnet4_6 => (2026, Month::January),
            ModelId::Haiku4_5 => (2025, Month::July),
        };
        YearMonth::new(year, month)
    }

    /// Standard list price per MTok (see [`Pricing`] for caveats).
    pub fn price_per_mtok(self) -> Pricing {
        let (input, cache_read_input, output) = match self {
            ModelId::Fable5_1 => (1_000, 25, 5_000),
            ModelId::Fable5 => (1_000, 100, 5_000),
            ModelId::Opus5 | ModelId::Opus4_8 => (500, 50, 2_500),
            // Sonnet 5 standard price; intro $2/$10 through 2026-08-31 not represented.
            ModelId::Sonnet5 | ModelId::Sonnet4_6 => (300, 30, 1_500),
            ModelId::Haiku4_5 => (100, 10, 500),
        };
        Pricing {
            input_cents_per_mtok: input,
            cache_read_input_cents_per_mtok: cache_read_input,
            output_cents_per_mtok: output,
        }
    }
}

/// A Claude model plus its per-call parameters.
pub enum Model {
    /// Fable 5.1 and its parameters.
    Fable5_1(Fable5_1),
    /// Fable 5 and its parameters.
    Fable5(Fable5),
    /// Opus 5 and its parameters.
    Opus5(Opus5),
    /// Opus 4.8 and its parameters.
    Opus4_8(Opus4_8),
    /// Sonnet 5 and its parameters.
    Sonnet5(Sonnet5),
    /// Sonnet 4.6 and its parameters.
    Sonnet4_6(Sonnet4_6),
    /// Haiku 4.5 and its parameters.
    Haiku4_5(Haiku4_5),
}

impl Model {
    /// Identity without per-call parameters.
    pub fn id(&self) -> ModelId {
        match self {
            Model::Fable5_1(_) => ModelId::Fable5_1,
            Model::Fable5(_) => ModelId::Fable5,
            Model::Opus5(_) => ModelId::Opus5,
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
    /// Fable 5.1 with its default parameters.
    pub fn fable_5_1() -> Fable5_1 {
        Fable5_1::default()
    }
    /// Fable 5 with its default parameters.
    pub fn fable_5() -> Fable5 {
        Fable5::default()
    }
    /// Opus 5 with its default parameters.
    pub fn opus_5() -> Opus5 {
        Opus5::default()
    }
    /// Opus 4.8 with its default parameters.
    pub fn opus_4_8() -> Opus4_8 {
        Opus4_8::default()
    }
    /// Sonnet 5 with its default parameters.
    pub fn sonnet_5() -> Sonnet5 {
        Sonnet5::default()
    }
    /// Sonnet 4.6 with its default parameters.
    pub fn sonnet_4_6() -> Sonnet4_6 {
        Sonnet4_6::default()
    }
    /// Haiku 4.5 with its default parameters.
    pub fn haiku_4_5() -> Haiku4_5 {
        Haiku4_5::default()
    }
}

impl From<Fable5_1> for Model {
    fn from(p: Fable5_1) -> Self {
        Model::Fable5_1(p)
    }
}
impl From<Fable5> for Model {
    fn from(p: Fable5) -> Self {
        Model::Fable5(p)
    }
}
impl From<Opus5> for Model {
    fn from(p: Opus5) -> Self {
        Model::Opus5(p)
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

// ── Fable 5.1 ────────────────────────────────────────────────────────────────

mod fable;
pub use fable::{Fable5_1, Fable5_1Effort, FableThinkingDisplay};

// ── Fable 5 ──────────────────────────────────────────────────────────────────
// Frontier tier. No sampling (temperature/top_p/top_k rejected). Thinking is
// always on: `{type: "disabled"}` and legacy `{type: "enabled", budget_tokens}`
// both 400, so unlike Opus 4.8 there is no "off" state — the only knob is
// `display`. Depth is controlled by `output_config.effort` (low..=max, incl.
// xhigh). `display` defaults to `Omitted` (blocks stream, text empty).

/// Fable 5's per-call parameters.
///
/// Thinking is always on: both `{type: "disabled"}` and the legacy fixed-budget
/// form are refused, so unlike Opus 4.8 there is no off state to express. Depth is
/// [`Fable5Effort`]; visibility is `display`.
pub struct Fable5 {
    /// Whether reasoning text is sent back. `Omitted` by default.
    pub display: FableThinkingDisplay,
    /// How much thinking to spend.
    pub effort: Fable5Effort,
}

impl Default for Fable5 {
    fn default() -> Self {
        Self { display: FableThinkingDisplay::Omitted, effort: Fable5Effort::High }
    }
}

impl Fable5 {
    /// The default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets how much thinking to spend.
    pub fn with_effort(mut self, effort: Fable5Effort) -> Self {
        self.effort = effort;
        self
    }

    /// Set the thinking summary visibility. Thinking can't be turned off on
    /// Fable 5; pass `Summarized` for visible reasoning text, `Omitted` (default)
    /// for empty thinking blocks.
    pub fn with_display(mut self, display: FableThinkingDisplay) -> Self {
        self.display = display;
        self
    }
}

api_enum! {
    /// How much thinking Fable 5 spends, as `output_config.effort`. The full
    /// Opus-tier range: thinking cannot be turned off on this model, so effort is
    /// the only depth control it has.
    Fable5Effort {
        /// The least thinking.
        Low => "low",
        /// Between `low` and `high`.
        Medium => "medium",
        /// The documented default.
        High => "high",
        /// Above `high`. Opus-tier and Fable only; Sonnet 4.6 rejects it.
        Xhigh => "xhigh",
        /// The most thinking.
        Max => "max",
    }
}

// ── Opus 5 ───────────────────────────────────────────────────────────────────
// Current Opus tier. No sampling: `temperature` is rejected outright as
// deprecated for this model, so — unlike Sonnet 4.6, where temperature and
// adaptive thinking are alternatives — there is no sampling knob to model at
// all. Adaptive thinking is *on by default*: omitting `thinking` leaves it on,
// so "off" must be stated as `{type: "disabled"}`, exactly as on Sonnet 5 and
// unlike Opus 4.8, whose off state is the omitted field.
//
// Effort belongs to the thinking state rather than beside it, because the two
// are not independent: with thinking on the full Opus-tier range applies,
// `xhigh` and `max` included; with it off the API accepts only `high` and
// below. Carrying effort on each variant makes the refused pair unwritable.

/// Opus 5's per-call parameters.
///
/// No sampling knob at all: `temperature` is refused as deprecated on this model.
/// Effort lives inside [`Opus5Thinking`] rather than beside it, because the two
/// are not independent — the accepted effort range narrows once thinking is off,
/// and carrying effort per variant makes the refused pair unwritable.
pub struct Opus5 {
    /// Whether the model thinks, and how much.
    pub thinking: Opus5Thinking,
}

impl Default for Opus5 {
    /// Adaptive thinking on with `Omitted` display and the documented default
    /// effort, `high` — the runtime default the API applies when `thinking` is
    /// absent, emitted explicitly.
    fn default() -> Self {
        Self { thinking: Opus5Thinking::Adaptive { display: ThinkingDisplay::Omitted, effort: Opus5Effort::High } }
    }
}

impl Opus5 {
    /// The default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the effort thinking is spent at. Only meaningful with thinking on;
    /// with it off, effort is chosen through [`Opus5::with_thinking_off`],
    /// whose narrower range is the one the API accepts in that state.
    pub fn with_effort(mut self, effort: Opus5Effort) -> Self {
        let display = match self.thinking {
            Opus5Thinking::Adaptive { display, .. } => display,
            Opus5Thinking::Disabled { .. } => ThinkingDisplay::Omitted,
        };
        self.thinking = Opus5Thinking::Adaptive { display, effort };
        self
    }

    /// Set adaptive thinking's summary visibility, turning thinking on where it
    /// was off. `display` defaults to `Omitted` (blocks stream but text is
    /// empty); pass `Summarized` for visible text.
    pub fn with_adaptive_thinking(mut self, display: ThinkingDisplay) -> Self {
        let effort = match self.thinking {
            Opus5Thinking::Adaptive { effort, .. } => effort,
            Opus5Thinking::Disabled { .. } => Opus5Effort::High,
        };
        self.thinking = Opus5Thinking::Adaptive { display, effort };
        self
    }

    /// Turn thinking off at `effort`. Emits `{type: "disabled"}` explicitly: on
    /// Opus 5 an omitted `thinking` field leaves adaptive thinking on, so off
    /// must be stated. The effort range narrows to `high` and below, which is
    /// what [`Opus5ThinkingOffEffort`] carries.
    pub fn with_thinking_off(mut self, effort: Opus5ThinkingOffEffort) -> Self {
        self.thinking = Opus5Thinking::Disabled { effort };
        self
    }
}

/// Whether Opus 5 thinks, and at what effort.
pub enum Opus5Thinking {
    /// Adaptive thinking on. The state an omitted `thinking` field would also
    /// produce, emitted explicitly.
    Adaptive {
        /// Whether reasoning text is sent back.
        display: ThinkingDisplay,
        /// How much thinking to spend, over the full range.
        effort: Opus5Effort,
    },
    /// Explicit `{type: "disabled"}` — distinct from an omitted field, which on
    /// Opus 5 means adaptive thinking on. Carries its own effort, because the
    /// API accepts a narrower range with thinking off (`high` and below) than
    /// with it on: `xhigh` and `max` are refused as
    /// "not supported when thinking is disabled on this model".
    Disabled {
        /// How much effort to spend, over the narrower range this state accepts.
        effort: Opus5ThinkingOffEffort,
    },
}

api_enum! {
    /// How much thinking Opus 5 spends with thinking *on*. Reachable only in that
    /// state; see [`Opus5ThinkingOffEffort`] for the other.
    Opus5Effort {
        /// The least thinking.
        Low => "low",
        /// Between `low` and `high`.
        Medium => "medium",
        /// The documented default.
        High => "high",
        /// Above `high`. Opus-tier and Fable only; Sonnet 4.6 rejects it.
        Xhigh => "xhigh",
        /// The most thinking.
        Max => "max",
    }
}

api_enum! {
    /// How much effort Opus 5 spends with thinking *off*.
    ///
    /// `xhigh` and `max` exist on this model but not in this state — the API
    /// refuses them as unsupported with thinking disabled — so they are absent
    /// from the type rather than rejected at runtime.
    Opus5ThinkingOffEffort {
        /// The least thinking.
        Low => "low",
        /// Between `low` and `high`.
        Medium => "medium",
        /// The documented default.
        High => "high",
    }
}

// ── Opus 4.8 ─────────────────────────────────────────────────────────────────
// No sampling (temperature/top_p/top_k rejected). Adaptive thinking only;
// legacy `{type: "enabled", budget_tokens}` is removed.

/// Opus 4.8's per-call parameters.
///
/// No sampling: `temperature`, `top_p` and `top_k` are all refused. Adaptive
/// thinking only; the legacy fixed-budget form is gone. Unlike Opus 5, its off
/// state *is* the omitted field.
pub struct Opus4_8 {
    /// Whether the model thinks.
    pub thinking: Opus4_8Thinking,
    /// How much thinking to spend.
    pub effort: Opus4_8Effort,
}

impl Default for Opus4_8 {
    fn default() -> Self {
        Self { thinking: Opus4_8Thinking::Off, effort: Opus4_8Effort::High }
    }
}

impl Opus4_8 {
    /// The default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets how much thinking to spend.
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

    /// Turns thinking off by omitting the `thinking` field, which on Opus 4.8 is
    /// what off means.
    pub fn with_thinking_off(mut self) -> Self {
        self.thinking = Opus4_8Thinking::Off;
        self
    }
}

/// Whether Opus 4.8 thinks.
pub enum Opus4_8Thinking {
    /// `thinking` field omitted from the request.
    Off,
    /// Adaptive thinking on.
    Adaptive {
        /// Whether reasoning text is sent back.
        display: ThinkingDisplay,
    },
}

api_enum! {
    /// How much thinking Opus 4.8 spends, as `output_config.effort`.
    Opus4_8Effort {
        /// The least thinking.
        Low => "low",
        /// Between `low` and `high`.
        Medium => "medium",
        /// The documented default.
        High => "high",
        /// Above `high`. Opus-tier and Fable only; Sonnet 4.6 rejects it.
        Xhigh => "xhigh",
        /// The most thinking.
        Max => "max",
    }
}

// ── Sonnet 5 ─────────────────────────────────────────────────────────────────
// Current Sonnet tier. No sampling (temperature/top_p/top_k non-default rejected,
// like Opus 4.8). Adaptive thinking is *on by default*: omitting `thinking` leaves
// it on, so "off" has to be sent explicitly as `{type: "disabled"}` — unlike Opus
// 4.8, whose off state is simply the omitted field. Legacy `{type: "enabled",
// budget_tokens}` is removed (400). Full Opus-tier effort incl. `xhigh` (Sonnet 4.6
// rejects `xhigh`). New tokenizer (~30% more tokens than Sonnet 4.6) — no wire effect.

/// Sonnet 5's per-call parameters.
///
/// No sampling. Adaptive thinking is on by default, so off has to be stated
/// explicitly — an omitted field would leave it on. Accepts `xhigh`, which Sonnet
/// 4.6 does not.
pub struct Sonnet5 {
    /// Whether the model thinks.
    pub thinking: Sonnet5Thinking,
    /// How much thinking to spend.
    pub effort: Sonnet5Effort,
}

impl Default for Sonnet5 {
    /// Adaptive thinking on with `Omitted` display — the runtime default the API
    /// applies when `thinking` is absent, emitted explicitly.
    fn default() -> Self {
        Self { thinking: Sonnet5Thinking::Adaptive { display: ThinkingDisplay::Omitted }, effort: Sonnet5Effort::High }
    }
}

impl Sonnet5 {
    /// The default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets how much thinking to spend.
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

/// Whether Sonnet 5 thinks.
pub enum Sonnet5Thinking {
    /// Adaptive thinking on.
    Adaptive {
        /// Whether reasoning text is sent back.
        display: ThinkingDisplay,
    },
    /// Explicit `{type: "disabled"}` — distinct from an omitted field, which on
    /// Sonnet 5 means adaptive thinking on.
    Disabled,
}

api_enum! {
    /// How much thinking Sonnet 5 spends. The full Opus-tier range: `xhigh` is
    /// accepted here, unlike on Sonnet 4.6.
    Sonnet5Effort {
        /// The least thinking.
        Low => "low",
        /// Between `low` and `high`.
        Medium => "medium",
        /// The documented default.
        High => "high",
        /// Above `high`. Opus-tier and Fable only; Sonnet 4.6 rejects it.
        Xhigh => "xhigh",
        /// The most thinking.
        Max => "max",
    }
}

// ── Sonnet 4.6 ───────────────────────────────────────────────────────────────
// Temperature OR adaptive thinking (API forces temperature=1.0 under adaptive).
// No `Xhigh` effort (Opus-tier only; Sonnet rejects it).

/// Sonnet 4.6's per-call parameters.
///
/// Temperature *or* adaptive thinking, never both: the API pins temperature to
/// 1.0 under adaptive thinking, so the two are one sum type rather than two
/// fields a caller must keep in sync.
pub struct Sonnet4_6 {
    /// Temperature or adaptive thinking.
    pub sampling: Sonnet4_6Sampling,
    /// How much thinking to spend.
    pub effort: Sonnet4_6Effort,
}

impl Default for Sonnet4_6 {
    fn default() -> Self {
        Self { sampling: Sonnet4_6Sampling::Temperature(Temperature::default()), effort: Sonnet4_6Effort::High }
    }
}

impl Sonnet4_6 {
    /// The default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets how much thinking to spend.
    pub fn with_effort(mut self, effort: Sonnet4_6Effort) -> Self {
        self.effort = effort;
        self
    }

    /// Samples at this temperature, turning adaptive thinking off.
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

/// Sonnet 4.6's two mutually exclusive sampling modes.
pub enum Sonnet4_6Sampling {
    /// `Temperature::default()` (1.0) matches the API default when `temperature` is omitted.
    Temperature(Temperature),
    /// Adaptive thinking, under which the API pins temperature to 1.0.
    Adaptive {
        /// Whether reasoning text is sent back.
        display: ThinkingDisplay,
    },
}

api_enum! {
    /// How much thinking Sonnet 4.6 spends. No `xhigh`: that level is Opus-tier
    /// and Sonnet 4.6 rejects it, so it is absent from the type.
    Sonnet4_6Effort {
        /// The least thinking.
        Low => "low",
        /// Between `low` and `high`.
        Medium => "medium",
        /// The documented default.
        High => "high",
        /// The most thinking.
        Max => "max",
    }
}

// ── Haiku 4.5 ────────────────────────────────────────────────────────────────
// Temperature + legacy fixed-budget thinking. `output_config.effort` rejected
// (400); adaptive thinking rejected (400).

/// Haiku 4.5's per-call parameters.
///
/// Temperature plus the legacy fixed-budget thinking form. `output_config.effort`
/// and adaptive thinking are both refused, so neither exists here.
pub struct Haiku4_5 {
    /// The sampling temperature.
    pub temperature: Temperature,
    /// Whether the model thinks, and on what budget.
    pub thinking: Haiku4_5Thinking,
}

impl Default for Haiku4_5 {
    fn default() -> Self {
        Self { temperature: Temperature::default(), thinking: Haiku4_5Thinking::Off }
    }
}

impl Haiku4_5 {
    /// The default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Samples at this temperature.
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

    /// Turns thinking off by omitting the `thinking` field.
    pub fn with_thinking_off(mut self) -> Self {
        self.thinking = Haiku4_5Thinking::Off;
        self
    }
}

/// Whether Haiku 4.5 thinks, and on what budget.
pub enum Haiku4_5Thinking {
    /// `thinking` field omitted from the request.
    Off,
    /// Legacy fixed-budget thinking: `{type: "enabled", budget_tokens: N}`.
    Enabled {
        /// The thinking budget, which [`crate::request::Request::new`] checks stays below
        /// `max_tokens`.
        budget_tokens: u32,
    },
}
