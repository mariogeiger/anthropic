//! System content: the top-level prompt, and an instruction added mid-conversation.
//!
//! # Why a mid-conversation system message exists
//!
//! An instruction that arrives partway through a conversation has nowhere good to
//! go. Rewriting the top-level system prompt changes the first bytes of the
//! request, which invalidates the prompt cache for the whole conversation.
//! Wrapping the instruction in a user turn works, but the model reads it as
//! something the user said rather than as a directive.
//!
//! A `{"role": "system"}` entry inside `messages` solves both: it sits after the
//! cached prefix, so nothing before it changes, and the model reads it as an
//! instruction. That is why [`SystemBlock`] exists as its own block set rather
//! than reusing [`crate::block::ContentBlock`] — the two positions admit
//! different content, and a type that admitted the union would let the API's
//! refusal be written.
//!
//! # Why both positions exist
//!
//! The two look interchangeable and are not. Sending the same text in the
//! top-level `system` field and as a leading `{"role": "system"}` message
//! produces identical behaviour and identical `input_token` counts, which is what
//! makes the question reasonable. They are nonetheless **disjoint by position**,
//! and the documented rules say so twice.
//!
//! *By placement.* "A system message cannot be the first entry in `messages`. Use
//! the top-level `system` field for instructions that apply from the very start."
//! So the top-level field is the *only* legal home for an instruction that holds
//! from the beginning, and a system message is the only home for one that begins
//! partway through. Neither position can be spelled the other way, and the reason
//! is the prompt cache: the top-level field sits near the start of the hashed
//! prefix, so editing it re-processes the whole conversation, while a system
//! message appends after the prefix and costs nothing. Measured over a
//! 12,600-token cached prefix: editing a *trailing* system message still read the
//! whole prefix from cache, while editing the top-level `system` field read 0 and
//! rewrote all of it.
//!
//! *By model.* Mid-conversation system messages are documented as available on
//! Fable 5, Mythos 5, Opus 4.8 and Opus 5, and "not available on Claude Sonnet 5;
//! use the top-level `system` field instead". The top-level field works on every
//! model, so it is also the only *portable* position. See
//! [`crate::model::ModelId::accepts_mid_conversation_system_message`] and
//! [`crate::request::RequestError::MidConversationSystemMessageUnsupported`].
//!
//! Both rules are stated in
//! <https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages>,
//! and the documentation is the authority: a gateway may accept a shape the API
//! forbids, so a 200 does not make it legal.
//!
//! # What each position admits
//!
//! The top-level `system` field takes text and nothing else. A system *message*
//! takes text, [`SystemBlock::ToolAddition`], and [`SystemBlock::ToolRemoval`].
//! The shared part is [`crate::block::TextBlock`], which both hold; the
//! difference is what else they hold, which is why they are two types.
//!
//! # Placement
//!
//! [`crate::context::Context::push_system`] and [`crate::request::Request::new`]
//! enforce the placement rules between them; see
//! [`crate::context::SystemMessageError`] for what they are and why the check is
//! split across the two.

use serde::Serialize;

use crate::block::TextBlock;
use crate::context::CacheControl;
use crate::values::{SystemBlockType, ToolReferenceType};

/// One block of a mid-conversation system message.
///
/// Exactly the three the API accepts there. An image or a tool result in a system
/// message is a 400 (`role 'system' supports text, tool_addition, and
/// tool_removal blocks only`), so neither is a variant here.
#[derive(Debug, Clone)]
pub enum SystemBlock {
    /// The instruction itself.
    Text(TextBlock),
    /// Offers a tool the model could not call before.
    ToolAddition(ToolChangeBlock),
    /// Withdraws a tool the model could call before.
    ToolRemoval(ToolChangeBlock),
}

impl SystemBlock {
    /// An instruction.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(TextBlock::new(text))
    }

    /// Offers the referenced tool from this point in the conversation on.
    pub fn tool_addition(tool: ToolReference) -> Self {
        Self::ToolAddition(ToolChangeBlock { tool, cache_control: None })
    }

    /// Withdraws the referenced tool from this point in the conversation on.
    pub fn tool_removal(tool: ToolReference) -> Self {
        Self::ToolRemoval(ToolChangeBlock { tool, cache_control: None })
    }

    /// The block's wire `type`.
    pub fn kind(&self) -> SystemBlockType {
        match self {
            SystemBlock::Text(_) => SystemBlockType::Text,
            SystemBlock::ToolAddition(_) => SystemBlockType::ToolAddition,
            SystemBlock::ToolRemoval(_) => SystemBlockType::ToolRemoval,
        }
    }

    pub(crate) fn cache_control_mut(&mut self) -> &mut Option<CacheControl> {
        match self {
            SystemBlock::Text(block) => &mut block.cache_control,
            SystemBlock::ToolAddition(block) | SystemBlock::ToolRemoval(block) => &mut block.cache_control,
        }
    }
}

/// Wire shape: the `type` written from [`SystemBlock::kind`], and then the
/// variant's own fields flattened in beside it.
#[derive(Serialize)]
struct SystemBlockWire<'a, T: Serialize> {
    #[serde(rename = "type")]
    kind: SystemBlockType,
    #[serde(flatten)]
    body: &'a T,
}

impl Serialize for SystemBlock {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let kind = self.kind();
        match self {
            SystemBlock::Text(body) => SystemBlockWire { kind, body }.serialize(s),
            SystemBlock::ToolAddition(body) | SystemBlock::ToolRemoval(body) => {
                SystemBlockWire { kind, body }.serialize(s)
            }
        }
    }
}

/// A tool offered or withdrawn partway through a conversation.
///
/// The change applies from this position onwards, so the tool definitions in
/// `tools` stay byte-identical and the tools cache stays warm — the same reason
/// [`crate::tool_choice::ToolChoice::None`] beats sending no tools at all.
#[derive(Debug, Clone, Serialize)]
pub struct ToolChangeBlock {
    /// Which tool changes.
    pub tool: ToolReference,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

/// Wire shape: the `type` written from [`ToolReference::kind`], because `serde`'s
/// derived tag would spell the variant's name rather than the API's — and the API
/// says `tool_reference` where the variant says `Tool`.
#[derive(Serialize)]
struct ToolReferenceWire<'a> {
    #[serde(rename = "type")]
    kind: ToolReferenceType,
    #[serde(skip_serializing_if = "Option::is_none")]
    server_name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
}

impl Serialize for ToolReference {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let (server_name, name) = match self {
            ToolReference::Tool { name } => (None, Some(name.as_str())),
            ToolReference::McpTool { server_name, name } => (Some(server_name.as_str()), Some(name.as_str())),
            ToolReference::McpToolset { server_name } => (Some(server_name.as_str()), None),
        };
        ToolReferenceWire { kind: self.kind(), server_name, name }.serialize(s)
    }
}

/// Which tool a [`ToolChangeBlock`] names.
///
/// Three ways to name one, because a tool's identity depends on where it came
/// from. A tool the caller declared in `tools` is named directly. An
/// MCP-resolved tool is named by server *and* tool, because the server assigns it
/// a composed `{server}_{name}` identifier that [`Self::Tool`] deliberately does
/// not accept. A whole MCP server's toolset is named by server alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolReference {
    /// A tool the caller declared directly in `tools`.
    Tool {
        /// Its name, as declared.
        name: String,
    },
    /// One tool of an MCP server's toolset.
    McpTool {
        /// The MCP server that serves it.
        server_name: String,
        /// The tool's own name, not the composed `{server}_{name}` form.
        name: String,
    },
    /// Every tool of an MCP server's toolset at once.
    McpToolset {
        /// The MCP server whose whole toolset changes.
        server_name: String,
    },
}

impl ToolReference {
    /// A tool declared directly in `tools`.
    pub fn tool(name: impl Into<String>) -> Self {
        Self::Tool { name: name.into() }
    }

    /// One tool of an MCP server's toolset.
    pub fn mcp_tool(server_name: impl Into<String>, name: impl Into<String>) -> Self {
        Self::McpTool { server_name: server_name.into(), name: name.into() }
    }

    /// Every tool of an MCP server's toolset.
    pub fn mcp_toolset(server_name: impl Into<String>) -> Self {
        Self::McpToolset { server_name: server_name.into() }
    }

    /// The reference's wire `type`.
    pub fn kind(&self) -> ToolReferenceType {
        match self {
            ToolReference::Tool { .. } => ToolReferenceType::Tool,
            ToolReference::McpTool { .. } => ToolReferenceType::McpTool,
            ToolReference::McpToolset { .. } => ToolReferenceType::McpToolset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The wire shapes, as the documented schema states them.
    #[test]
    fn every_system_block_serializes_to_its_documented_shape() {
        assert_eq!(
            serde_json::to_value(SystemBlock::text("Answer in French.")).unwrap(),
            json!({"type": "text", "text": "Answer in French."})
        );
        assert_eq!(
            serde_json::to_value(SystemBlock::tool_removal(ToolReference::tool("get_time"))).unwrap(),
            json!({"type": "tool_removal", "tool": {"type": "tool_reference", "name": "get_time"}})
        );
        assert_eq!(
            serde_json::to_value(SystemBlock::tool_addition(ToolReference::mcp_tool("weather", "forecast"))).unwrap(),
            json!({
                "type": "tool_addition",
                "tool": {"type": "mcp_tool_reference", "server_name": "weather", "name": "forecast"}
            })
        );
        assert_eq!(
            serde_json::to_value(SystemBlock::tool_removal(ToolReference::mcp_toolset("weather"))).unwrap(),
            json!({"type": "tool_removal", "tool": {"type": "mcp_toolset_reference", "server_name": "weather"}})
        );
    }

    #[test]
    fn every_block_and_reference_names_its_wire_type() {
        assert_eq!(SystemBlock::text("x").kind().as_str(), "text");
        assert_eq!(SystemBlock::tool_addition(ToolReference::tool("t")).kind().as_str(), "tool_addition");
        assert_eq!(SystemBlock::tool_removal(ToolReference::tool("t")).kind().as_str(), "tool_removal");
        assert_eq!(ToolReference::tool("t").kind().as_str(), "tool_reference");
        assert_eq!(ToolReference::mcp_tool("s", "t").kind().as_str(), "mcp_tool_reference");
        assert_eq!(ToolReference::mcp_toolset("s").kind().as_str(), "mcp_toolset_reference");
    }
}
