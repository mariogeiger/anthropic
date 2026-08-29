//! Whether, and which, tool the model must call.
//!
//! # This parameter costs message cache, and only message cache
//!
//! Anthropic documents the cache hierarchy as `tools → system → messages`, where
//! a change at one level invalidates that level and every level after it. Under
//! that rule `tool_choice` is a special case worth knowing: it invalidates the
//! *message* cache while leaving the tools and system caches valid.
//!
//! | What changes      | Tools cache | System cache | Messages cache |
//! |-------------------|-------------|--------------|----------------|
//! | Tool definitions  | invalid     | invalid      | invalid        |
//! | `tool_choice`     | **valid**   | **valid**    | invalid        |
//!
//! The asymmetry has a reason. `tool_choice` is rendered into the prompt near the
//! messages, not into the tool definitions, so the prefix covering tools and
//! system is byte-identical across the change. This is why it is a type of its
//! own carrying that fact, rather than a field on a model: a caller who
//! recomputes it per turn pays for a fresh message-cache write every turn, and
//! nothing about the code would say so.
//!
//! A conversation whose tools and system prompt are large and whose message tail
//! is short therefore loses little by changing it — and one that caches a long
//! message history loses that history. Anthropic's own troubleshooting list names
//! `tool_choice` first among things to hold constant between calls.
//!
//! # `None` versus an empty tool list
//!
//! [`ToolChoice::None`] withholds every tool while leaving the definitions in the
//! prompt, so the tools cache stays warm and a later turn can allow them again
//! without a re-write. Sending no tools at all changes the definitions, which
//! invalidates everything. They are different operations, and only one of them is
//! cheap.

use serde::Serialize;

/// How the model must treat the available tools.
///
/// Absent from a request means [`Self::Auto`], the API's documented default.
/// [`crate::request::Request`] holds an `Option` of this so that "not specified"
/// stays distinguishable from "explicitly auto" — the two are equivalent to the
/// model but not to the wire, and the crate does not invent a value the caller
/// did not send.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model decides whether to call a tool. The API's default.
    Auto {
        /// Whether to forbid more than one tool call per turn.
        ///
        /// `false`, the default, lets the model request several calls at once,
        /// which a caller can then run in parallel. `true` is for a tool whose
        /// effects must be observed before the next call is chosen.
        disable_parallel_tool_use: bool,
    },
    /// The model must call some tool, choosing which itself.
    Any {
        /// Whether to forbid more than one tool call per turn.
        disable_parallel_tool_use: bool,
    },
    /// The model must call this specific tool.
    Tool {
        /// The tool's name, matching a [`crate::context::Tool`] in the request.
        name: String,
        /// Whether to forbid more than one tool call per turn.
        disable_parallel_tool_use: bool,
    },
    /// The model may not call any tool.
    ///
    /// The definitions stay in the prompt, so the tools cache stays warm — see
    /// the module documentation. Carries no parallel-use flag: with no calls
    /// permitted there is nothing to parallelize, and the API rejects the
    /// combination, so it is absent from the type rather than refused at runtime.
    None,
}

impl ToolChoice {
    /// The model decides, with parallel calls allowed. The API's default.
    pub fn auto() -> Self {
        ToolChoice::Auto { disable_parallel_tool_use: false }
    }

    /// The model must call some tool, with parallel calls allowed.
    pub fn any() -> Self {
        ToolChoice::Any { disable_parallel_tool_use: false }
    }

    /// The model must call this tool, with parallel calls allowed.
    pub fn tool(name: impl Into<String>) -> Self {
        ToolChoice::Tool { name: name.into(), disable_parallel_tool_use: false }
    }

    /// The model may not call any tool.
    pub fn none() -> Self {
        ToolChoice::None
    }

    /// Forbids more than one tool call per turn.
    ///
    /// A no-op on [`Self::None`], which permits no calls at all.
    pub fn without_parallel_use(self) -> Self {
        match self {
            ToolChoice::Auto { .. } => ToolChoice::Auto { disable_parallel_tool_use: true },
            ToolChoice::Any { .. } => ToolChoice::Any { disable_parallel_tool_use: true },
            ToolChoice::Tool { name, .. } => ToolChoice::Tool { name, disable_parallel_tool_use: true },
            ToolChoice::None => ToolChoice::None,
        }
    }

    /// The `type` value sent on the wire.
    pub fn as_str(&self) -> &'static str {
        match self {
            ToolChoice::Auto { .. } => "auto",
            ToolChoice::Any { .. } => "any",
            ToolChoice::Tool { .. } => "tool",
            ToolChoice::None => "none",
        }
    }

    /// Whether parallel tool use is forbidden. Always `true` for [`Self::None`],
    /// which permits no calls at all.
    pub fn parallel_use_is_disabled(&self) -> bool {
        match self {
            ToolChoice::Auto { disable_parallel_tool_use }
            | ToolChoice::Any { disable_parallel_tool_use }
            | ToolChoice::Tool { disable_parallel_tool_use, .. } => *disable_parallel_tool_use,
            ToolChoice::None => true,
        }
    }
}

/// The wire shape: `type`, plus `name` and the flag where they apply.
///
/// `disable_parallel_tool_use` is emitted only when `true`. This is the one place
/// the crate omits a value that equals its default, and the reason is the module's
/// subject: the field is rendered into the prompt, so emitting `false` where the
/// caller never asked for it would write a different prompt — and a different
/// message-cache key — than omitting it.
#[derive(Serialize)]
struct ToolChoiceWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    disable_parallel_tool_use: bool,
}

impl Serialize for ToolChoice {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ToolChoiceWire {
            kind: self.as_str(),
            name: match self {
                ToolChoice::Tool { name, .. } => Some(name),
                _ => None,
            },
            disable_parallel_tool_use: match self {
                ToolChoice::None => false,
                other => other.parallel_use_is_disabled(),
            },
        }
        .serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn wire(choice: ToolChoice) -> Value {
        serde_json::to_value(choice).unwrap()
    }

    /// The four shapes the API documents, as the live gateway accepted them.
    #[test]
    fn every_choice_serializes_to_its_documented_shape() {
        assert_eq!(wire(ToolChoice::auto()), json!({"type": "auto"}));
        assert_eq!(wire(ToolChoice::any()), json!({"type": "any"}));
        assert_eq!(wire(ToolChoice::tool("get_weather")), json!({"type": "tool", "name": "get_weather"}));
        assert_eq!(wire(ToolChoice::none()), json!({"type": "none"}));
    }

    /// The flag appears only when the caller asked for it, because emitting it
    /// otherwise would write a different prompt and a different cache key.
    #[test]
    fn the_parallel_flag_is_emitted_only_when_set() {
        assert_eq!(wire(ToolChoice::auto()).get("disable_parallel_tool_use"), None);
        assert_eq!(
            wire(ToolChoice::auto().without_parallel_use()),
            json!({"type": "auto", "disable_parallel_tool_use": true})
        );
        assert_eq!(
            wire(ToolChoice::tool("get_weather").without_parallel_use()),
            json!({"type": "tool", "name": "get_weather", "disable_parallel_tool_use": true})
        );
        assert_eq!(wire(ToolChoice::any().without_parallel_use())["disable_parallel_tool_use"], true);
    }

    /// `none` permits no calls, so it carries no flag on the wire while reporting
    /// parallel use as disabled — there is nothing to parallelize.
    #[test]
    fn none_carries_no_flag_yet_permits_no_parallel_use() {
        assert!(ToolChoice::none().parallel_use_is_disabled());
        assert_eq!(wire(ToolChoice::none().without_parallel_use()), json!({"type": "none"}));
    }

    #[test]
    fn parallel_use_is_reported_per_variant() {
        assert!(!ToolChoice::auto().parallel_use_is_disabled());
        assert!(!ToolChoice::any().parallel_use_is_disabled());
        assert!(!ToolChoice::tool("t").parallel_use_is_disabled());
        assert!(ToolChoice::any().without_parallel_use().parallel_use_is_disabled());
    }

    /// Restricting parallel use keeps the chosen tool's name.
    #[test]
    fn restricting_parallel_use_preserves_the_tool_name() {
        assert_eq!(
            ToolChoice::tool("get_weather").without_parallel_use(),
            ToolChoice::Tool { name: "get_weather".to_owned(), disable_parallel_tool_use: true }
        );
    }

    #[test]
    fn each_variant_names_its_wire_type() {
        assert_eq!(ToolChoice::auto().as_str(), "auto");
        assert_eq!(ToolChoice::any().as_str(), "any");
        assert_eq!(ToolChoice::tool("t").as_str(), "tool");
        assert_eq!(ToolChoice::none().as_str(), "none");
    }
}
