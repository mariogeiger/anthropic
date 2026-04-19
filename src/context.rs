//! Cache-safe, append-only `Context` that serializes to a Messages-API request body.
//!
//! The type prevents by construction every way to break the prompt cache:
//!   - No `&mut` access to existing messages → past bytes are frozen
//!   - `system` and `tools` are set at construction only
//!   - The 4 cache breakpoints live in named slots (`CacheSlot::S0..S3`) with 1:1
//!     mapping to the Anthropic limit of 4 per request — impossible to exceed
//!   - `roll_cache` only moves a slot to the current tail (metadata change only,
//!     never rewrites past content), `clear_cache` only removes metadata
//!   - TTL ordering (1h before 5m) is validated at each `roll_cache` call
//!
//! `Context` holds conversation state (system + tools + messages + cache slots)
//! only. Per-call parameters (model, max_tokens, sampling) live on
//! [`crate::request::Request`], which borrows a `Context` and serializes to the
//! full `/v1/messages` body. This split lets you re-send the same stable
//! context with different models or sampling settings.
//!
//! # Design philosophy: model the runtime behavior, not HTTP field presence
//!
//! Types here describe what the model actually *sees*, not which JSON fields
//! happen to appear on the wire. Consequences:
//!
//! - Every `Option<T>` in this file corresponds to a real runtime distinction
//!   (a content block either has a cache breakpoint or it doesn't; a tool
//!   either has a description or it doesn't) — not "the HTTP field was
//!   omitted". We never use `Option` just to mirror wire-format optionality.
//! - `SystemPrompt` is a single struct with `{ text, cache_control }`. The
//!   fact that the API accepts both a bare string and a block array is a
//!   serialization detail, not a runtime one — the custom `Serialize` impl
//!   picks the right shape.
//! - `is_error: bool` on `ToolResult` (not `Option<bool>`): every tool result
//!   is either an error or a success at runtime. `false` matches the API
//!   default when the field is omitted.
//! - Defaults for `Request::sampling` / `effort` come straight from the
//!   Anthropic documentation (`temperature = 1.0`, `effort = "high"`). We
//!   don't invent defaults.

use crate::{CacheControlType, CacheTtl, ImageMediaType};
use serde::Serialize;
use serde_json::Value;

// ── Cache control ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub ttl: &'static str,
}

impl CacheControl {
    pub fn ephemeral(ttl: CacheTtl) -> Self {
        Self {
            kind: CacheControlType::Ephemeral.as_str(),
            ttl: ttl.as_str(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheSlot {
    S0,
    S1,
    S2,
    S3,
}

impl CacheSlot {
    const ALL: [CacheSlot; 4] = [CacheSlot::S0, CacheSlot::S1, CacheSlot::S2, CacheSlot::S3];

    fn idx(self) -> usize {
        match self {
            CacheSlot::S0 => 0,
            CacheSlot::S1 => 1,
            CacheSlot::S2 => 2,
            CacheSlot::S3 => 3,
        }
    }
}

// Anchor slots are written to `system` or `tools` at construction and never move.
// Rolling slots point at a specific content block in `messages`.
#[derive(Debug, Clone, Copy)]
enum SlotLocation {
    System,
    Tools,
    Message { msg: usize, block: usize },
}

#[derive(Debug, Clone, Copy)]
struct SlotState {
    location: SlotLocation,
    ttl: CacheTtl,
}

// ── Images & tool results ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 {
        media_type: &'static str,
        data: String,
    },
    Url {
        url: String,
    },
    File {
        file_id: String,
    },
}

impl ImageSource {
    pub fn base64(media_type: ImageMediaType, data: impl Into<String>) -> Self {
        ImageSource::Base64 {
            media_type: media_type.as_str(),
            data: data.into(),
        }
    }

    pub fn url(url: impl Into<String>) -> Self {
        ImageSource::Url { url: url.into() }
    }

    pub fn file(file_id: impl Into<String>) -> Self {
        ImageSource::File {
            file_id: file_id.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    Text { text: String },
    Image { source: ImageSource },
}

// ── Content blocks ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        is_error: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Thinking {
        thinking: String,
        signature: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    RedactedThinking {
        data: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text {
            text: text.into(),
            cache_control: None,
        }
    }

    pub fn image(source: ImageSource) -> Self {
        ContentBlock::Image {
            source,
            cache_control: None,
        }
    }

    pub fn tool_use(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        ContentBlock::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
            cache_control: None,
        }
    }

    pub fn tool_result(tool_use_id: impl Into<String>, content: ToolResultContent) -> Self {
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.into(),
            content,
            is_error: false,
            cache_control: None,
        }
    }

    fn cache_control_mut(&mut self) -> &mut Option<CacheControl> {
        match self {
            ContentBlock::Text { cache_control, .. }
            | ContentBlock::Image { cache_control, .. }
            | ContentBlock::ToolUse { cache_control, .. }
            | ContentBlock::ToolResult { cache_control, .. }
            | ContentBlock::Thinking { cache_control, .. }
            | ContentBlock::RedactedThinking { cache_control, .. } => cache_control,
        }
    }
}

// ── Messages, tools, system ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: &'static str,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl Tool {
    pub fn new(name: impl Into<String>, input_schema: Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            input_schema,
            cache_control: None,
        }
    }

    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }
}

/// A system prompt with optional cache breakpoint.
///
/// Runtime state is a single string plus an optional cache placement. The
/// wire format has two shapes — bare string when there's no `cache_control`,
/// one-element block array when there is — chosen at serialization time.
#[derive(Debug, Clone)]
pub(crate) struct SystemPrompt {
    pub(crate) text: String,
    pub(crate) cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
struct SystemTextBlockRef<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
    cache_control: &'a CacheControl,
}

impl Serialize for SystemPrompt {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match &self.cache_control {
            None => self.text.serialize(s),
            Some(cc) => {
                use serde::ser::SerializeSeq;
                let mut seq = s.serialize_seq(Some(1))?;
                seq.serialize_element(&SystemTextBlockRef {
                    kind: "text",
                    text: &self.text,
                    cache_control: cc,
                })?;
                seq.end()
            }
        }
    }
}

// ── Errors ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollCacheError {
    /// Slot is occupied by a `system` or `tools` anchor and cannot be moved.
    SlotOccupiedByAnchor(CacheSlot),
    /// Context has no message content blocks to attach a rolling breakpoint to.
    NoBlocksToCache,
    /// Target position already has a different TTL set. Anthropic returns 400 for this.
    ConflictingTtlAtSamePosition,
    /// The final 1h-before-5m ordering rule would be violated by this call.
    TtlOrderingViolation,
}

impl std::fmt::Display for RollCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RollCacheError::SlotOccupiedByAnchor(s) => {
                write!(f, "cache slot {s:?} is occupied by a system/tools anchor")
            }
            RollCacheError::NoBlocksToCache => {
                write!(f, "no content blocks to attach a cache breakpoint to")
            }
            RollCacheError::ConflictingTtlAtSamePosition => {
                write!(
                    f,
                    "target position already has a different TTL (API returns 400)"
                )
            }
            RollCacheError::TtlOrderingViolation => {
                write!(f, "all 1h breakpoints must come before any 5m breakpoints")
            }
        }
    }
}

impl std::error::Error for RollCacheError {}

// ── Context ──────────────────────────────────────────────────────────────────

pub struct Context {
    pub(crate) system: Option<SystemPrompt>,
    pub(crate) tools: Vec<Tool>,
    pub(crate) messages: Vec<Message>,
    slots: [Option<SlotState>; 4],
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    // ── Construction ────────────────────────────────────────────────────────

    pub fn new() -> Self {
        Self {
            system: None,
            tools: Vec::new(),
            messages: Vec::new(),
            slots: [None, None, None, None],
        }
    }

    pub fn with_system(mut self, text: impl Into<String>) -> Self {
        self.system = Some(SystemPrompt {
            text: text.into(),
            cache_control: None,
        });
        self
    }

    /// Set the system prompt and attach a cache breakpoint on it.
    ///
    /// Panics if `slot` is already occupied.
    pub fn with_system_cached(
        mut self,
        slot: CacheSlot,
        text: impl Into<String>,
        ttl: CacheTtl,
    ) -> Self {
        assert!(
            self.slots[slot.idx()].is_none(),
            "cache slot {slot:?} already in use"
        );
        self.system = Some(SystemPrompt {
            text: text.into(),
            cache_control: Some(CacheControl::ephemeral(ttl)),
        });
        self.slots[slot.idx()] = Some(SlotState {
            location: SlotLocation::System,
            ttl,
        });
        self
    }

    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    /// Set the tools and attach a cache breakpoint on the last tool.
    ///
    /// Panics if `slot` is already occupied or `tools` is empty.
    pub fn with_tools_cached(
        mut self,
        slot: CacheSlot,
        mut tools: Vec<Tool>,
        ttl: CacheTtl,
    ) -> Self {
        assert!(
            self.slots[slot.idx()].is_none(),
            "cache slot {slot:?} already in use"
        );
        assert!(
            !tools.is_empty(),
            "with_tools_cached: tools must not be empty"
        );
        tools.last_mut().unwrap().cache_control = Some(CacheControl::ephemeral(ttl));
        self.tools = tools;
        self.slots[slot.idx()] = Some(SlotState {
            location: SlotLocation::Tools,
            ttl,
        });
        self
    }

    // ── Evolution: append-only ──────────────────────────────────────────────

    pub fn push_user_text(&mut self, text: impl Into<String>) {
        self.messages.push(Message {
            role: "user",
            content: vec![ContentBlock::text(text)],
        });
    }

    pub fn push_user(&mut self, blocks: Vec<ContentBlock>) {
        self.messages.push(Message {
            role: "user",
            content: blocks,
        });
    }

    pub fn push_assistant_text(&mut self, text: impl Into<String>) {
        self.messages.push(Message {
            role: "assistant",
            content: vec![ContentBlock::text(text)],
        });
    }

    pub fn push_assistant(&mut self, blocks: Vec<ContentBlock>) {
        self.messages.push(Message {
            role: "assistant",
            content: blocks,
        });
    }

    pub fn push_tool_result(&mut self, tool_use_id: impl Into<String>, content: ToolResultContent) {
        self.messages.push(Message {
            role: "user",
            content: vec![ContentBlock::tool_result(tool_use_id, content)],
        });
    }

    // ── Cache slot operations ───────────────────────────────────────────────

    /// Move `slot` to the last block of the last message, with the given TTL.
    ///
    /// Clears the slot's previous `cache_control` (if any) — bytes of past content
    /// are never touched. Validates the 1h-before-5m ordering rule and rejects
    /// mid-evolution conflicts before mutating state.
    pub fn roll_cache(&mut self, slot: CacheSlot, ttl: CacheTtl) -> Result<(), RollCacheError> {
        let i = slot.idx();
        if let Some(state) = self.slots[i] {
            if matches!(state.location, SlotLocation::System | SlotLocation::Tools) {
                return Err(RollCacheError::SlotOccupiedByAnchor(slot));
            }
        }

        let (msg, block) = self.tail_position()?;
        let target = SlotLocation::Message { msg, block };

        // If another slot already marks the same position with a different TTL,
        // Anthropic returns 400. Catch it before we commit.
        for (j, other) in self.slots.iter().enumerate() {
            if j == i {
                continue;
            }
            if let Some(s) = other
                && same_location(s.location, target)
                && s.ttl != ttl
            {
                return Err(RollCacheError::ConflictingTtlAtSamePosition);
            }
        }

        self.validate_ordering_with_override(slot, Some((target, ttl)))?;

        // Commit: clear old, write new.
        if let Some(state) = self.slots[i].take() {
            self.write_cache_control(state.location, None);
        }
        self.write_cache_control(target, Some(CacheControl::ephemeral(ttl)));
        self.slots[i] = Some(SlotState {
            location: target,
            ttl,
        });
        Ok(())
    }

    /// Remove `slot` and clear its `cache_control` metadata. No-op if already empty.
    ///
    /// Refuses to clear an anchor slot (set via `with_system_cached` / `with_tools_cached`)
    /// — anchors are immutable for the lifetime of the Context.
    pub fn clear_cache(&mut self, slot: CacheSlot) -> Result<(), RollCacheError> {
        let i = slot.idx();
        if let Some(state) = self.slots[i] {
            if matches!(state.location, SlotLocation::System | SlotLocation::Tools) {
                return Err(RollCacheError::SlotOccupiedByAnchor(slot));
            }
            self.write_cache_control(state.location, None);
            self.slots[i] = None;
        }
        Ok(())
    }

    pub fn breakpoint_count(&self) -> u8 {
        self.slots.iter().filter(|s| s.is_some()).count() as u8
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    // ── Internals ───────────────────────────────────────────────────────────

    fn tail_position(&self) -> Result<(usize, usize), RollCacheError> {
        let msg_idx = self
            .messages
            .len()
            .checked_sub(1)
            .ok_or(RollCacheError::NoBlocksToCache)?;
        let block_idx = self.messages[msg_idx]
            .content
            .len()
            .checked_sub(1)
            .ok_or(RollCacheError::NoBlocksToCache)?;
        Ok((msg_idx, block_idx))
    }

    fn write_cache_control(&mut self, loc: SlotLocation, cc: Option<CacheControl>) {
        match loc {
            SlotLocation::System => {
                if let Some(sp) = &mut self.system {
                    sp.cache_control = cc;
                }
            }
            SlotLocation::Tools => {
                if let Some(t) = self.tools.last_mut() {
                    t.cache_control = cc;
                }
            }
            SlotLocation::Message { msg, block } => {
                if let Some(b) = self
                    .messages
                    .get_mut(msg)
                    .and_then(|m| m.content.get_mut(block))
                {
                    *b.cache_control_mut() = cc;
                }
            }
        }
    }

    /// Verify the final request order is `[1h…1h, 5m…5m]`. `override_slot` lets us
    /// simulate the effect of a pending `roll_cache` before committing.
    fn validate_ordering_with_override(
        &self,
        override_slot: CacheSlot,
        new_state: Option<(SlotLocation, CacheTtl)>,
    ) -> Result<(), RollCacheError> {
        let mut placements: Vec<(usize, CacheTtl)> = Vec::new();
        for slot in CacheSlot::ALL {
            let state = if slot == override_slot {
                new_state.map(|(loc, ttl)| SlotState { location: loc, ttl })
            } else {
                self.slots[slot.idx()]
            };
            if let Some(s) = state {
                placements.push((flow_index(s.location), s.ttl));
            }
        }
        placements.sort_by_key(|&(pos, _)| pos);

        let mut seen_5m = false;
        for (_, ttl) in placements {
            match ttl {
                CacheTtl::FiveMinutes => seen_5m = true,
                CacheTtl::OneHour => {
                    if seen_5m {
                        return Err(RollCacheError::TtlOrderingViolation);
                    }
                }
            }
        }
        Ok(())
    }
}

fn same_location(a: SlotLocation, b: SlotLocation) -> bool {
    match (a, b) {
        (SlotLocation::System, SlotLocation::System) => true,
        (SlotLocation::Tools, SlotLocation::Tools) => true,
        (
            SlotLocation::Message { msg: m1, block: b1 },
            SlotLocation::Message { msg: m2, block: b2 },
        ) => m1 == m2 && b1 == b2,
        _ => false,
    }
}

/// A position key that reflects the request-flow order `tools → system → messages`,
/// used only to sort breakpoints for TTL ordering validation.
fn flow_index(loc: SlotLocation) -> usize {
    const TOOLS: usize = 0;
    const SYSTEM: usize = usize::MAX / 4;
    const MSG_BASE: usize = usize::MAX / 2;
    match loc {
        SlotLocation::Tools => TOOLS,
        SlotLocation::System => SYSTEM,
        SlotLocation::Message { msg, block } => MSG_BASE + msg * 1024 + block,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{Model, Request};

    #[test]
    fn empty_request_serializes() {
        let ctx = Context::new();
        let v = serde_json::to_value(Request::new(&ctx, Model::opus_4_7(), 1024)).unwrap();
        assert_eq!(v["model"], "claude-opus-4-7");
        assert_eq!(v["max_tokens"], 1024);
        assert!(v["messages"].is_array());
    }

    #[test]
    fn roll_cache_on_empty_errors() {
        let mut ctx = Context::new();
        let err = ctx
            .roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes)
            .unwrap_err();
        assert_eq!(err, RollCacheError::NoBlocksToCache);
    }

    fn req(ctx: &Context) -> serde_json::Value {
        serde_json::to_value(Request::new(ctx, Model::opus_4_7(), 1024)).unwrap()
    }

    #[test]
    fn roll_cache_tail_places_on_last_block() {
        let mut ctx = Context::new();
        ctx.push_user_text("hi");
        ctx.roll_cache(CacheSlot::S3, CacheTtl::FiveMinutes)
            .unwrap();
        let v = req(&ctx);
        assert_eq!(v["messages"][0]["content"][0]["cache_control"]["ttl"], "5m");
    }

    #[test]
    fn roll_cache_moves_clears_old_position() {
        let mut ctx = Context::new();
        ctx.push_user_text("one");
        ctx.roll_cache(CacheSlot::S3, CacheTtl::FiveMinutes)
            .unwrap();
        ctx.push_assistant_text("two");
        ctx.push_user_text("three");
        ctx.roll_cache(CacheSlot::S3, CacheTtl::FiveMinutes)
            .unwrap();
        let v = req(&ctx);
        assert!(
            v["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(v["messages"][2]["content"][0]["cache_control"]["ttl"], "5m");
    }

    #[test]
    fn anchors_cannot_be_rolled() {
        let mut ctx = Context::new().with_system_cached(CacheSlot::S0, "sys", CacheTtl::OneHour);
        ctx.push_user_text("hi");
        let err = ctx
            .roll_cache(CacheSlot::S0, CacheTtl::OneHour)
            .unwrap_err();
        assert_eq!(err, RollCacheError::SlotOccupiedByAnchor(CacheSlot::S0));
    }

    #[test]
    fn ttl_ordering_1h_after_5m_rejected() {
        let mut ctx = Context::new();
        ctx.push_user_text("one");
        ctx.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes)
            .unwrap();
        ctx.push_user_text("two");
        let err = ctx
            .roll_cache(CacheSlot::S1, CacheTtl::OneHour)
            .unwrap_err();
        assert_eq!(err, RollCacheError::TtlOrderingViolation);
    }

    #[test]
    fn ttl_ordering_1h_system_then_5m_tail_ok() {
        let mut ctx = Context::new().with_system_cached(CacheSlot::S0, "sys", CacheTtl::OneHour);
        ctx.push_user_text("hi");
        ctx.roll_cache(CacheSlot::S3, CacheTtl::FiveMinutes)
            .unwrap();
        assert_eq!(ctx.breakpoint_count(), 2);
    }

    #[test]
    fn clear_cache_removes_metadata() {
        let mut ctx = Context::new();
        ctx.push_user_text("hi");
        ctx.roll_cache(CacheSlot::S3, CacheTtl::FiveMinutes)
            .unwrap();
        ctx.clear_cache(CacheSlot::S3).unwrap();
        let v = req(&ctx);
        assert!(
            v["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(ctx.breakpoint_count(), 0);
    }

    #[test]
    fn system_without_cache_serializes_as_plain_string() {
        let ctx = Context::new().with_system("you are helpful");
        let v = req(&ctx);
        assert_eq!(v["system"], "you are helpful");
    }

    #[test]
    fn system_with_cache_serializes_as_block_array() {
        let ctx = Context::new().with_system_cached(CacheSlot::S0, "sys", CacheTtl::OneHour);
        let v = req(&ctx);
        assert_eq!(v["system"][0]["type"], "text");
        assert_eq!(v["system"][0]["text"], "sys");
        assert_eq!(v["system"][0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn tool_result_is_error_emitted_as_bool() {
        let mut ctx = Context::new();
        ctx.push_user(vec![ContentBlock::tool_result(
            "tu_1",
            ToolResultContent::Text("oops".into()),
        )]);
        let v = req(&ctx);
        assert_eq!(v["messages"][0]["content"][0]["is_error"], false);
    }

    #[test]
    fn tools_cached_marks_last_tool() {
        let tools = vec![
            Tool::new("one", serde_json::json!({"type": "object"})),
            Tool::new("two", serde_json::json!({"type": "object"})),
        ];
        let ctx = Context::new().with_tools_cached(CacheSlot::S1, tools, CacheTtl::OneHour);
        let v = req(&ctx);
        assert!(v["tools"][0].get("cache_control").is_none());
        assert_eq!(v["tools"][1]["cache_control"]["ttl"], "1h");
    }
}
