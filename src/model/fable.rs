//! Claude Fable parameters and their model-specific thinking display.

use super::Fable5Effort;
use crate::values::api_enum;

api_enum! {
    /// What Fable 5 and Fable 5.1 return from adaptive thinking.
    ///
    /// `Updates` is intentionally absent from [`crate::ThinkingDisplay`]: only
    /// these Fable models produce progress-update blocks, so the separate type
    /// keeps the beta value off models that cannot use it.
    FableThinkingDisplay {
        /// Provider-safe reasoning summaries are returned.
        Summarized => "summarized",
        /// No reasoning text is returned. The documented default.
        Omitted => "omitted",
        /// Only short progress updates between tool calls are returned; private
        /// reasoning remains hidden.
        Updates => "updates",
    }
}

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
    pub display: FableThinkingDisplay,
    /// How much work the model spends over the complete supported range.
    pub effort: Fable5_1Effort,
}

impl Default for Fable5_1 {
    fn default() -> Self {
        Self { display: FableThinkingDisplay::Omitted, effort: Fable5Effort::High }
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
    pub fn with_display(mut self, display: FableThinkingDisplay) -> Self {
        self.display = display;
        self
    }
}

/// Fable 5.1 accepts the same five effort levels as Fable 5.
pub type Fable5_1Effort = Fable5Effort;
