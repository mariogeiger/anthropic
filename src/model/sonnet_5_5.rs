//! Claude Sonnet 5.5 parameters: adaptive thinking, or none before responding.
//!
//! No sampling: `temperature`, `top_p` and `top_k` are refused as deprecated.
//! Adaptive thinking is on by default. Its off state is not `disabled`, which
//! this model refuses, but `between_tools`: the model does not think before
//! responding and still writes short progress updates between tool calls,
//! returned as thinking blocks. Effort lives inside the thinking state, because
//! `between_tools` accepts only `high` and below and takes no other field.

use crate::values::api_enum;

api_enum! {
    /// What Claude Sonnet 5.5 returns from adaptive thinking.
    ///
    /// `Updates` requires the `thinking-display-updates-2026-08-18` beta
    /// header, which [`crate::request::Request::required_beta_features`]
    /// infers from the selected value.
    Sonnet5_5ThinkingDisplay {
        /// Provider-safe reasoning summaries are returned.
        Summarized => "summarized",
        /// No reasoning text is returned. The documented default.
        Omitted => "omitted",
        /// Only short progress updates between tool calls are returned; private
        /// reasoning remains hidden.
        Updates => "updates",
    }
}

api_enum! {
    /// How much work Claude Sonnet 5.5 spends with adaptive thinking.
    Sonnet5_5Effort {
        /// The least work.
        Low => "low",
        /// Between `low` and `high`.
        Medium => "medium",
        /// The documented default.
        High => "high",
        /// Above `high`.
        Xhigh => "xhigh",
        /// The most work.
        Max => "max",
    }
}

api_enum! {
    /// How much work Claude Sonnet 5.5 spends without up-front thinking.
    ///
    /// `xhigh` and `max` exist on this model but not in this state: the API
    /// answers `effort 'xhigh' is not supported when thinking is disabled on this
    /// model`, so they are absent from the type.
    ///
    /// ```compile_fail
    /// let _ = anthropic::request::Sonnet5_5BetweenToolsEffort::Xhigh;
    /// ```
    Sonnet5_5BetweenToolsEffort {
        /// The least work.
        Low => "low",
        /// Between `low` and `high`.
        Medium => "medium",
        /// The documented default.
        High => "high",
    }
}

/// Claude Sonnet 5.5's per-call parameters.
///
/// Thinking off is [`Sonnet5_5Thinking::BetweenTools`], reached through
/// [`Sonnet5_5::with_thinking_between_tools`]; there is no `disabled` form to
/// write, because the API refuses it. Forced tool choice is refused too, and is
/// checked when [`crate::request::Request::with_tool_choice`] combines the model
/// and the choice.
///
/// ```compile_fail
/// let _ = anthropic::request::Model::sonnet_5_5().with_thinking_off();
/// ```
pub struct Sonnet5_5 {
    /// Whether the model thinks before responding, and how much work it spends.
    pub thinking: Sonnet5_5Thinking,
}

impl Default for Sonnet5_5 {
    /// Adaptive thinking with `Omitted` display at the documented default
    /// effort, `high`: what the API applies when `thinking` is absent, emitted
    /// explicitly.
    fn default() -> Self {
        Self {
            thinking: Sonnet5_5Thinking::Adaptive {
                display: Sonnet5_5ThinkingDisplay::Omitted,
                effort: Sonnet5_5Effort::High,
            },
        }
    }
}

impl Sonnet5_5 {
    /// The documented defaults: adaptive thinking, no returned thinking text,
    /// and high effort.
    pub fn new() -> Self {
        Self::default()
    }

    /// Thinks adaptively at `effort`, keeping the display where thinking was
    /// already adaptive and defaulting it to `Omitted` otherwise.
    pub fn with_effort(mut self, effort: Sonnet5_5Effort) -> Self {
        let display = match self.thinking {
            Sonnet5_5Thinking::Adaptive { display, .. } => display,
            Sonnet5_5Thinking::BetweenTools { .. } => Sonnet5_5ThinkingDisplay::Omitted,
        };
        self.thinking = Sonnet5_5Thinking::Adaptive { display, effort };
        self
    }

    /// Thinks adaptively with this display, keeping the effort where thinking
    /// was already adaptive and defaulting it to `high` otherwise.
    pub fn with_adaptive_thinking(mut self, display: Sonnet5_5ThinkingDisplay) -> Self {
        let effort = match self.thinking {
            Sonnet5_5Thinking::Adaptive { effort, .. } => effort,
            Sonnet5_5Thinking::BetweenTools { .. } => Sonnet5_5Effort::High,
        };
        self.thinking = Sonnet5_5Thinking::Adaptive { display, effort };
        self
    }

    /// Turns up-front thinking off at `effort`, emitting `{type:
    /// "between_tools"}`. The model still writes progress updates between tool
    /// calls, returned as thinking blocks to pass back unchanged.
    pub fn with_thinking_between_tools(mut self, effort: Sonnet5_5BetweenToolsEffort) -> Self {
        self.thinking = Sonnet5_5Thinking::BetweenTools { effort };
        self
    }
}

/// Whether Claude Sonnet 5.5 thinks before responding, and at what effort.
///
/// Effort lives inside each state because the accepted range depends on it, and
/// display lives only in the adaptive one because `between_tools` refuses every
/// other field: `thinking.between_tools.display: Extra inputs are not
/// permitted`.
pub enum Sonnet5_5Thinking {
    /// Adaptive thinking, the state an omitted `thinking` field also produces.
    Adaptive {
        /// Which thinking text the provider returns.
        display: Sonnet5_5ThinkingDisplay,
        /// How much work to spend, over the full range.
        effort: Sonnet5_5Effort,
    },
    /// `{type: "between_tools"}`: no thinking before responding, only progress
    /// updates between tool calls. Effort cannot change mid-conversation in
    /// this state, which [`crate::request::Request::new`] checks.
    BetweenTools {
        /// How much work to spend, over the narrower range this state accepts.
        effort: Sonnet5_5BetweenToolsEffort,
    },
}
