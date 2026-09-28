//! The tools a request declares: the caller's own, and the server tools Anthropic
//! runs itself.
//!
//! The `tools` array is a union on the wire. A custom tool is a schema the caller
//! answers with a `tool_result`; a server tool is a versioned `type` the API
//! executes before the model continues, returning its result inside the same
//! assistant turn. [`ToolDefinition`] is that union, so one list holds both in the
//! order the caller chose — the order is part of the cached prefix.

use serde::Serialize;
use serde_json::Value;

use crate::context::CacheControl;
use crate::web_search::WebSearchTool;

/// One tool the model may call.
///
/// Changing any of these fields invalidates the whole cache — tools sit first in
/// the `tools → system → messages` hierarchy, so a change there invalidates every
/// level. Compare [`crate::tool_choice::ToolChoice`], which costs only the message
/// cache.
#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    /// The name the model calls it by, and that its `tool_use` blocks carry.
    pub name: String,
    /// What it does, in the model's words.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Its JSON Schema. Key order matters for caching: a schema whose keys move
    /// between requests is a different prefix.
    pub input_schema: Value,
    /// Whether to withhold this tool from the served schema until a tool search
    /// returns a reference to it.
    ///
    /// A `bool` rather than an `Option`, because every tool is either deferred or
    /// not, and it is emitted only when `true`: the field is rendered into the
    /// prompt, so emitting `false` where the caller never asked for it writes a
    /// different prefix and a different cache key — the same reasoning as
    /// [`crate::tool_choice::ToolChoice`]'s parallel-use flag.
    ///
    /// The API refuses a request whose every tool is deferred (`At least one tool
    /// must have defer_loading=false`). That is a relation across the tool list,
    /// not a property of one tool, so [`crate::request::Request::new`] checks it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub defer_loading: bool,
    /// Whether the API validates the model's tool names and inputs against the
    /// schema. Emitted only when `true`, for the same prompt-identity reason.
    ///
    /// Measured: the inference gateway refuses this field with
    /// `tools.0.custom.strict: Extra inputs are not permitted`. It is in the
    /// documented stable schema, so it is here.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub strict: bool,
    /// Example inputs shown to the model beside the schema. Empty means none;
    /// there is no "no examples" distinct from "an empty list of examples".
    ///
    /// Measured: the inference gateway refuses this field with
    /// `tools.0.custom.input_examples: Extra inputs are not permitted`. It is in
    /// the documented stable schema, so it is here.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub input_examples: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

impl Tool {
    /// A tool with this name and schema, and no description yet.
    pub fn new(name: impl Into<String>, input_schema: Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            input_schema,
            defer_loading: false,
            strict: false,
            input_examples: Vec::new(),
            cache_control: None,
        }
    }
    /// Describes what the tool does.
    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }
    /// Withholds this tool from the served schema until a tool search finds it.
    ///
    /// Worth it for a large tool set: the deferred tools cost no prompt tokens
    /// until the model asks for them. At least one tool must stay undeferred, which
    /// [`crate::request::Request::new`] checks.
    pub fn deferred(mut self) -> Self {
        self.defer_loading = true;
        self
    }
    /// Has the API validate the model's tool names and inputs against the schema.
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }
    /// Shows the model example inputs beside the schema.
    pub fn input_examples(mut self, examples: Vec<Value>) -> Self {
        self.input_examples = examples;
        self
    }
}

/// One entry of the request's `tools` array.
///
/// A sum type rather than a list of custom tools beside optional server tools,
/// because the API holds them in one ordered array and the order is part of the
/// prompt-cache key: a separate field would fix server tools at one end of the
/// list and make every other order unwritable.
#[derive(Debug, Clone)]
pub enum ToolDefinition {
    /// A tool the caller runs and answers with a `tool_result`.
    Custom(Tool),
    /// Anthropic's web search, run by the API inside the model's turn.
    WebSearch(WebSearchTool),
}

impl From<Tool> for ToolDefinition {
    fn from(tool: Tool) -> Self {
        Self::Custom(tool)
    }
}

impl From<WebSearchTool> for ToolDefinition {
    fn from(tool: WebSearchTool) -> Self {
        Self::WebSearch(tool)
    }
}

impl Serialize for ToolDefinition {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            // A custom tool omits `type`, which the API reads as `custom`; writing it
            // would change the prefix of every existing conversation's cache key.
            Self::Custom(tool) => tool.serialize(s),
            Self::WebSearch(tool) => tool.serialize(s),
        }
    }
}

impl ToolDefinition {
    /// The name the model calls it by: the custom tool's own, or the server tool's
    /// fixed one.
    pub fn name(&self) -> &str {
        match self {
            Self::Custom(tool) => &tool.name,
            Self::WebSearch(_) => crate::web_search::WEB_SEARCH_NAME,
        }
    }

    /// Whether the served schema withholds it until a tool search finds it.
    ///
    /// A server tool this crate declares is never deferred, so a list holding one
    /// always satisfies the API's rule that some tool stays loaded.
    pub fn is_deferred(&self) -> bool {
        match self {
            Self::Custom(tool) => tool.defer_loading,
            Self::WebSearch(_) => false,
        }
    }

    pub(crate) fn cache_control_mut(&mut self) -> &mut Option<CacheControl> {
        match self {
            Self::Custom(tool) => &mut tool.cache_control,
            Self::WebSearch(tool) => &mut tool.cache_control,
        }
    }
}
