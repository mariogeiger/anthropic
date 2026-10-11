//! Claude Haiku 5.5 parameters, whose accepted effort depends on its thinking state.
//!
//! No sampling: `temperature`, `top_p` and `top_k` are refused as deprecated.
//! Adaptive thinking is on by default, so off must be stated as
//! `{type: "disabled"}`. Effort belongs to the thinking state rather than beside
//! it, because with thinking off the API accepts only `high` and below.

use crate::ThinkingDisplayWithUpdates;
use crate::values::api_enum;

/// Haiku 5.5's per-call parameters.
///
/// No sampling knob at all, and no legacy fixed-budget thinking: both are
/// refused. Effort lives inside [`Haiku5_5Thinking`] rather than beside it,
/// because the accepted range narrows once thinking is off, and carrying effort
/// per variant makes the refused pair unwritable.
///
/// ```compile_fail,E0599
/// let _ = anthropic::request::Model::haiku_5_5().with_temperature(anthropic::request::Temperature::default());
/// ```
pub struct Haiku5_5 {
    /// Whether the model thinks, and how much.
    pub thinking: Haiku5_5Thinking,
}

impl Default for Haiku5_5 {
    /// Adaptive thinking on with `Omitted` display and the documented default
    /// effort, `medium` — the runtime default the API applies when `thinking` is
    /// absent, emitted explicitly.
    fn default() -> Self {
        Self {
            thinking: Haiku5_5Thinking::Adaptive {
                display: ThinkingDisplayWithUpdates::Omitted,
                effort: Haiku5_5Effort::Medium,
            },
        }
    }
}

impl Haiku5_5 {
    /// The default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Thinks adaptively at `effort`, keeping the display where thinking was
    /// already adaptive and defaulting it to `Omitted` otherwise.
    pub fn with_effort(mut self, effort: Haiku5_5Effort) -> Self {
        let display = match self.thinking {
            Haiku5_5Thinking::Adaptive { display, .. } => display,
            Haiku5_5Thinking::Disabled { .. } => ThinkingDisplayWithUpdates::Omitted,
        };
        self.thinking = Haiku5_5Thinking::Adaptive { display, effort };
        self
    }

    /// Thinks adaptively with this display, keeping the effort where thinking
    /// was already adaptive and defaulting it to `medium` otherwise.
    pub fn with_adaptive_thinking(mut self, display: ThinkingDisplayWithUpdates) -> Self {
        let effort = match self.thinking {
            Haiku5_5Thinking::Adaptive { effort, .. } => effort,
            Haiku5_5Thinking::Disabled { .. } => Haiku5_5Effort::Medium,
        };
        self.thinking = Haiku5_5Thinking::Adaptive { display, effort };
        self
    }

    /// Turns thinking off at `effort`. Emits `{type: "disabled"}` explicitly: an
    /// omitted `thinking` field leaves adaptive thinking on. The effort range
    /// narrows to `high` and below, which is what [`Haiku5_5ThinkingOffEffort`]
    /// carries.
    pub fn with_thinking_off(mut self, effort: Haiku5_5ThinkingOffEffort) -> Self {
        self.thinking = Haiku5_5Thinking::Disabled { effort };
        self
    }
}

/// Whether Haiku 5.5 thinks, and at what effort.
///
/// Effort lives inside each state because the accepted range depends on it, and
/// display lives only in the adaptive one because `disabled` takes no other
/// field.
pub enum Haiku5_5Thinking {
    /// Adaptive thinking, the state an omitted `thinking` field also produces.
    Adaptive {
        /// Which thinking text the provider returns.
        display: ThinkingDisplayWithUpdates,
        /// How much thinking to spend, over the full range.
        effort: Haiku5_5Effort,
    },
    /// Explicit `{type: "disabled"}`, distinct from an omitted field, which means
    /// adaptive thinking on. Effort cannot change mid-conversation in this state,
    /// which [`crate::request::Request::new`] checks.
    Disabled {
        /// How much effort to spend, over the narrower range this state accepts.
        effort: Haiku5_5ThinkingOffEffort,
    },
}

api_enum! {
    /// How much thinking Haiku 5.5 spends with thinking *on*.
    Haiku5_5Effort {
        /// The least thinking.
        Low => "low",
        /// The documented default.
        Medium => "medium",
        /// Between `medium` and `xhigh`.
        High => "high",
        /// Above `high`.
        Xhigh => "xhigh",
        /// The most thinking.
        Max => "max",
    }
}

api_enum! {
    /// How much effort Haiku 5.5 spends with thinking *off*.
    ///
    /// `xhigh` and `max` exist on this model but not in this state: the API
    /// answers `effort 'xhigh' is not supported when thinking is disabled on this
    /// model`, so they are absent from the type.
    ///
    /// ```compile_fail,E0599
    /// let _ = anthropic::request::Haiku5_5ThinkingOffEffort::Xhigh;
    /// ```
    Haiku5_5ThinkingOffEffort {
        /// The least effort.
        Low => "low",
        /// The documented default.
        Medium => "medium",
        /// The most effort this state accepts.
        High => "high",
    }
}
