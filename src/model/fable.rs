//! Claude Fable 5.1 parameters and its always-on thinking controls.

use super::Fable5Effort;
use crate::ThinkingDisplayWithUpdates;

/// Fable 5.1's per-call parameters.
///
/// Thinking is always adaptive: disabling it and the legacy fixed-budget form
/// are both rejected. Unlike Fable 5, forced tool choice is also rejected; that
/// request-level relation is enforced by [`crate::request::Request::with_tool_choice`].
///
/// ```compile_fail
/// let _ = anthropic::request::Model::fable_5_1().with_thinking_off();
/// ```
pub struct Fable5_1 {
    /// Which safe subset of thinking text the provider returns.
    pub display: ThinkingDisplayWithUpdates,
    /// How much work the model spends over the complete supported range.
    pub effort: Fable5_1Effort,
}

impl Default for Fable5_1 {
    fn default() -> Self {
        Self { display: ThinkingDisplayWithUpdates::Omitted, effort: Fable5Effort::High }
    }
}

impl Fable5_1 {
    /// The documented defaults: adaptive thinking, no returned thinking text,
    /// and high effort.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets how much work the model spends.
    pub fn with_effort(mut self, effort: Fable5_1Effort) -> Self {
        self.effort = effort;
        self
    }

    /// Chooses whether summaries, progress updates, or no thinking text returns.
    pub fn with_display(mut self, display: ThinkingDisplayWithUpdates) -> Self {
        self.display = display;
        self
    }
}

/// Fable 5.1 accepts the same five effort levels as Fable 5.
pub type Fable5_1Effort = Fable5Effort;
