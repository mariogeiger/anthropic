//! Claude Opus 5.5 parameters and its always-on thinking controls.

use crate::values::api_enum;

api_enum! {
    /// What Claude Opus 5.5 returns from adaptive thinking.
    ///
    /// `Updates` requires the `thinking-display-updates-2026-08-18` beta
    /// header, which [`crate::request::Request::required_beta_features`]
    /// infers from the selected value.
    Opus5_5ThinkingDisplay {
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
    /// How much thinking Claude Opus 5.5 spends.
    ///
    /// Thinking cannot be disabled on this model, so effort is its only depth,
    /// latency, and cost control.
    Opus5_5Effort {
        /// The least thinking.
        Low => "low",
        /// Moderate thinking and the documented default.
        Medium => "medium",
        /// Deep thinking for difficult work.
        High => "high",
        /// More exploration than `high`.
        Xhigh => "xhigh",
        /// The most thinking.
        Max => "max",
    }
}

/// Claude Opus 5.5's per-call parameters.
///
/// Thinking is always adaptive: both the disabled and legacy fixed-budget wire
/// forms are rejected. Forced tool choice is likewise rejected and is checked
/// when [`crate::request::Request::with_tool_choice`] combines the model and
/// choice.
///
/// ```compile_fail
/// let _ = anthropic::request::Model::opus_5_5().with_thinking_off();
/// ```
pub struct Opus5_5 {
    /// Which safe subset of thinking text the provider returns.
    pub display: Opus5_5ThinkingDisplay,
    /// How much work the model spends over the complete supported range.
    pub effort: Opus5_5Effort,
}

impl Default for Opus5_5 {
    fn default() -> Self {
        Self { display: Opus5_5ThinkingDisplay::Omitted, effort: Opus5_5Effort::Medium }
    }
}

impl Opus5_5 {
    /// The documented defaults: adaptive thinking, no returned thinking text,
    /// and medium effort.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets how much work the model spends.
    pub fn with_effort(mut self, effort: Opus5_5Effort) -> Self {
        self.effort = effort;
        self
    }

    /// Chooses whether summaries, progress updates, or no thinking text returns.
    pub fn with_display(mut self, display: Opus5_5ThinkingDisplay) -> Self {
        self.display = display;
        self
    }
}
