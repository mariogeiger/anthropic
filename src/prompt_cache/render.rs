//! A request becomes the prompt the server caches: one byte string in server
//! order, cut at every position a cache entry can end, each cut keyed by the
//! SHA-256 digest of the bytes before it.
//!
//! The bytes are the crate's own serialization, so two requests share a prefix
//! exactly when the crate would send the same content for it. Only the digests
//! are kept: a prefix is compared, never read back, and a digest is a key of
//! fixed size where the bytes would grow with the conversation. What the server
//! renders but the body does not spell out in place — the model, the toggles
//! that move a whole level, and the thinking configuration where a
//! model renders it — enters as a header piece that is not itself a position.

use serde_json::{Map, Value, json};
use sha2::{Digest as _, Sha256};

use super::{Cut, Digest, Position};
use crate::CacheTtl;
use crate::model::ModelId;
use crate::request::Request;

/// Where a model renders its thinking configuration and top-level effort.
///
/// Documented as model-specific without a table, so each value was measured
/// first-party on 2026-09-25 by changing only effort, or only the thinking mode,
/// between two requests with a breakpoint on the tools, the system prompt and the
/// last message, and reading which of the three still hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigurationLevel {
    /// Ahead of the tools: every breakpoint misses after a change.
    BeforeTools,
    /// Between the system prompt and the messages, as the documentation's
    /// default describes.
    BeforeMessages,
    /// After every cacheable position: a change misses nothing.
    AfterPrefix,
}

fn configuration_level(model: ModelId) -> ConfigurationLevel {
    match model {
        ModelId::Opus5_5 | ModelId::Fable5_1 => ConfigurationLevel::AfterPrefix,
        ModelId::Haiku4_5 => ConfigurationLevel::BeforeMessages,
        ModelId::Fable5 | ModelId::Opus5 | ModelId::Opus4_8 | ModelId::Sonnet5 | ModelId::Sonnet4_6 => {
            ConfigurationLevel::BeforeTools
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Run {
    ToolUse,
    ToolResult,
}

struct Renderer {
    hasher: Sha256,
    length: u64,
    cuts: Vec<Cut>,
    run: Option<Run>,
}

impl Renderer {
    fn extend(&mut self, piece: &Value) {
        let bytes = piece.to_string().into_bytes();
        self.hasher.update(&bytes);
        self.length += bytes.len() as u64;
    }

    fn header(&mut self, piece: Value) {
        self.extend(&piece);
        self.run = None;
    }

    fn position(&mut self, position: Position, mut block: Value) {
        let mark = block.as_object_mut().and_then(|b| b.remove("cache_control")).map(|control| {
            control.get("ttl").and_then(Value::as_str).and_then(CacheTtl::from_str).unwrap_or(CacheTtl::FiveMinutes)
        });
        let run = match block.get("type").and_then(Value::as_str) {
            Some("tool_use") => Some(Run::ToolUse),
            Some("tool_result") => Some(Run::ToolResult),
            _ => None,
        };
        let continues = run.is_some() && run == self.run;
        let unit = match self.cuts.last() {
            Some(last) if continues => last.unit,
            Some(last) => last.unit + 1,
            None => 0,
        };
        self.extend(&block);
        let digest = Digest(self.hasher.clone().finalize().into());
        self.cuts.push(Cut { position, end: self.length, unit, mark, digest });
        self.run = run;
    }
}

/// Render `request` as the server's cache sees it: every position, in server
/// order.
pub(super) fn render(request: &Request<'_>) -> Vec<Cut> {
    let body = serde_json::to_value(request).expect("a request serializes to JSON");
    let mut renderer = Renderer { hasher: Sha256::new(), length: 0, cuts: Vec::new(), run: None };
    let level = configuration_level(request.model().id());
    let configuration = configuration(&body);
    let at = |wanted: ConfigurationLevel| if level == wanted { configuration.clone() } else { Value::Null };

    renderer.header(json!({
        "model": body["model"],
        "format": body["output_config"]["format"],
        "configuration": at(ConfigurationLevel::BeforeTools),
    }));
    for (i, tool) in array(&body["tools"]).iter().enumerate() {
        if tool.get("defer_loading") != Some(&Value::Bool(true)) {
            renderer.position(Position::Tool(i), tool.clone());
        }
    }

    renderer.header(json!({ "citations": enables_citations(&body["system"]) || enables_citations(&body["messages"]) }));
    match &body["system"] {
        Value::String(text) => renderer.position(Position::System(0), json!({ "type": "text", "text": text })),
        system => {
            for (i, block) in array(system).iter().enumerate() {
                renderer.position(Position::System(i), block.clone());
            }
        }
    }

    let messages = array(&body["messages"]);
    renderer.header(json!({
        "tool_forcing": tool_forcing(&body["tool_choice"]),
        "configuration": at(ConfigurationLevel::BeforeMessages),
    }));
    for (m, message) in messages.iter().enumerate() {
        if cleared(messages, m) {
            continue;
        }
        let mut opening = message.as_object().cloned().unwrap_or_default();
        let content = opening.remove("content");
        renderer.header(Value::Object(opening));
        match content {
            Some(Value::Array(blocks)) => {
                for (b, block) in blocks.into_iter().enumerate() {
                    renderer.position(Position::Message { message: m, block: b }, block);
                }
            }
            Some(Value::String(text)) => {
                renderer.position(Position::Message { message: m, block: 0 }, json!({ "type": "text", "text": text }))
            }
            _ => {}
        }
    }
    renderer.cuts
}

fn array(value: &Value) -> &[Value] {
    value.as_array().map(Vec::as_slice).unwrap_or_default()
}

/// The documented rendered part of the thinking configuration — its mode and
/// budget, not how its output is displayed — and the resolved effort.
fn configuration(body: &Value) -> Value {
    let thinking = &body["thinking"];
    let mut rendered = Map::new();
    for field in ["type", "budget_tokens"] {
        if let Some(value) = thinking.get(field) {
            rendered.insert(field.to_owned(), value.clone());
        }
    }
    json!({ "thinking": rendered, "effort": body["output_config"]["effort"] })
}

/// A turn-scoped system message is gone from the prompt once a user message
/// follows it.
fn cleared(messages: &[Value], m: usize) -> bool {
    messages[m]["role"] == "system"
        && messages[m].get("clear_at").is_some()
        && messages[m + 1..].iter().any(|later| later["role"] == "user")
}

/// What `tool_choice` renders ahead of the messages: nothing unless a tool call
/// is forced, and then only whether parallel calls stay allowed.
///
/// Measured first-party on 2026-09-25 on every model that accepts forcing:
/// `auto` and `none` read each other's entries, `any` and a named `tool` read
/// each other's, and `disable_parallel_tool_use` moves the messages only under
/// forcing. Which tool is forced renders after every cacheable position.
fn tool_forcing(tool_choice: &Value) -> Value {
    match tool_choice["type"].as_str() {
        Some("any" | "tool") => json!({ "parallel": tool_choice["disable_parallel_tool_use"] != true }),
        _ => Value::Null,
    }
}

fn enables_citations(value: &Value) -> bool {
    match value {
        Value::Object(fields) => {
            fields.get("citations").and_then(|c| c.get("enabled")) == Some(&Value::Bool(true))
                || fields.values().any(enables_citations)
        }
        Value::Array(items) => items.iter().any(enables_citations),
        _ => false,
    }
}
