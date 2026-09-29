//! Claude Opus 5 parameters, whose accepted effort depends on its thinking state.
//!
//! No sampling: `temperature` is rejected outright as deprecated. Adaptive
//! thinking is on by default, so off must be stated as `{type: "disabled"}`.
//! Effort belongs to the thinking state rather than beside it, because with
//! thinking off the API accepts only `high` and below.

use crate::ThinkingDisplay;
use crate::values::api_enum;

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
///
/// The effort lives inside the thinking state rather than beside it, because the
/// accepted range depends on whether thinking is on: the refused combination is
/// unwritable rather than rejected at runtime.
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
