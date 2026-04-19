//! Cache-safe, append-only conversation state.
//!
//! Invariants enforced by construction: past bytes are frozen (no `&mut` to old
//! messages); `system`/`tools` set at construction only; 4 cache breakpoints live
//! in named slots (`CacheSlot::S0..S3`) — impossible to exceed the API limit;
//! `roll_cache` only moves slot metadata, never rewrites content; TTL ordering
//! (1h before 5m) validated before every commit.
//!
//! Types model what the model *sees*, not wire-format field presence: every
//! `Option` represents a real runtime distinction. `SystemPrompt` is one struct
//! with two wire shapes (bare string vs one-element array); the serializer picks.
//!
//! `cache_control` is not reachable from outside the crate: `CacheControl` has
//! no public constructor and no public fields, and the `cache_control` slot on
//! every content block and `Tool` is crate-private. The only way to attach a
//! breakpoint is through `CacheSlot` via `with_system_cached`,
//! `with_tools_cached`, or `roll_cache`, which keeps slot bookkeeping
//! consistent with content.

use crate::{CacheControlType, CacheTtl, ImageMediaType};
use serde::Serialize;
use serde_json::Value;

// ── Cache control ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) ttl: &'static str,
}

impl CacheControl {
    pub(crate) fn ephemeral(ttl: CacheTtl) -> Self {
        Self { kind: CacheControlType::Ephemeral.as_str(), ttl: ttl.as_str() }
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
        self as usize
    }
}

// Anchor slots (System/Tools) are set at construction and immutable.
// Rolling slots point into `messages`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    Base64 { media_type: &'static str, data: String },
    Url { url: String },
    File { file_id: String },
}

impl ImageSource {
    pub fn base64(media_type: ImageMediaType, data: impl Into<String>) -> Self {
        ImageSource::Base64 { media_type: media_type.as_str(), data: data.into() }
    }
    pub fn url(url: impl Into<String>) -> Self {
        ImageSource::Url { url: url.into() }
    }
    pub fn file(file_id: impl Into<String>) -> Self {
        ImageSource::File { file_id: file_id.into() }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultItem>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultItem {
    Text { text: String },
    Image { source: ImageSource },
}

// ── Content blocks ───────────────────────────────────────────────────────────
// Each variant wraps a `*Block` struct whose `cache_control` is crate-private:
// callers can read the content fields but cannot set, swap, or clone a
// `CacheControl` into place. The only path is through a `CacheSlot`.

#[derive(Debug, Clone, Serialize)]
pub struct TextBlock {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageBlock {
    pub source: ImageSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    pub input: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    pub content: ToolResultContent,
    pub is_error: bool, // runtime bool, not Option<bool>: every result is error-or-not
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ThinkingBlock {
    pub thinking: String,
    pub signature: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RedactedThinkingBlock {
    pub data: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text(TextBlock),
    Image(ImageBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Thinking(ThinkingBlock),
    RedactedThinking(RedactedThinkingBlock),
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(TextBlock { text: text.into(), cache_control: None })
    }
    pub fn image(source: ImageSource) -> Self {
        Self::Image(ImageBlock { source, cache_control: None })
    }
    pub fn tool_use(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        Self::ToolUse(ToolUseBlock { id: id.into(), name: name.into(), input, cache_control: None })
    }
    pub fn tool_result(tool_use_id: impl Into<String>, content: ToolResultContent) -> Self {
        Self::ToolResult(ToolResultBlock {
            tool_use_id: tool_use_id.into(),
            content,
            is_error: false,
            cache_control: None,
        })
    }
    pub fn tool_result_err(tool_use_id: impl Into<String>, content: ToolResultContent) -> Self {
        Self::ToolResult(ToolResultBlock {
            tool_use_id: tool_use_id.into(),
            content,
            is_error: true,
            cache_control: None,
        })
    }

    fn cache_control_mut(&mut self) -> &mut Option<CacheControl> {
        match self {
            Self::Text(b) => &mut b.cache_control,
            Self::Image(b) => &mut b.cache_control,
            Self::ToolUse(b) => &mut b.cache_control,
            Self::ToolResult(b) => &mut b.cache_control,
            Self::Thinking(b) => &mut b.cache_control,
            Self::RedactedThinking(b) => &mut b.cache_control,
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
    pub(crate) cache_control: Option<CacheControl>,
}

impl Tool {
    pub fn new(name: impl Into<String>, input_schema: Value) -> Self {
        Self { name: name.into(), description: None, input_schema, cache_control: None }
    }
    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }
}

// Wire shape: bare string when no cache_control, one-element block array when set.
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
            Some(cc) => [SystemTextBlockRef { kind: "text", text: &self.text, cache_control: cc }].serialize(s),
        }
    }
}

// ── Errors ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollCacheError {
    /// Slot is occupied by a `system`/`tools` anchor and cannot be moved.
    SlotOccupiedByAnchor(CacheSlot),
    /// Context has no message content to attach a rolling breakpoint to.
    NoBlocksToCache,
    /// Another slot already points at this position with a different TTL.
    /// Committing would overwrite the other slot's `cache_control` and desync
    /// slot bookkeeping from the content. The API never sees this case (a
    /// block carries one `cache_control`); it is an internal invariant.
    ConflictingTtlAtSamePosition,
    /// Would violate the 1h-before-5m ordering rule.
    TtlOrderingViolation,
}

impl std::fmt::Display for RollCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RollCacheError::SlotOccupiedByAnchor(s) => {
                write!(f, "cache slot {s:?} is occupied by a system/tools anchor")
            }
            RollCacheError::NoBlocksToCache => write!(f, "no content blocks to attach a cache breakpoint to"),
            RollCacheError::ConflictingTtlAtSamePosition => {
                write!(
                    f,
                    "another slot already points at this position with a different TTL (would corrupt slot bookkeeping)"
                )
            }
            RollCacheError::TtlOrderingViolation => write!(f, "all 1h breakpoints must come before any 5m breakpoints"),
        }
    }
}

impl std::error::Error for RollCacheError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnchorError {
    /// A cache slot already holds a breakpoint. Anchors never overwrite.
    SlotAlreadyInUse(CacheSlot),
    /// `with_tools_cached` was called with an empty tool list.
    NoToolsToCache,
}

impl std::fmt::Display for AnchorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnchorError::SlotAlreadyInUse(s) => write!(f, "cache slot {s:?} is already in use"),
            AnchorError::NoToolsToCache => write!(f, "no tools to attach a cache breakpoint to"),
        }
    }
}

impl std::error::Error for AnchorError {}

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
    pub fn new() -> Self {
        Self { system: None, tools: Vec::new(), messages: Vec::new(), slots: [None; 4] }
    }

    pub fn with_system(mut self, text: impl Into<String>) -> Self {
        self.system = Some(SystemPrompt { text: text.into(), cache_control: None });
        self
    }

    /// Set the system prompt with a cache breakpoint.
    pub fn with_system_cached(
        mut self,
        slot: CacheSlot,
        text: impl Into<String>,
        ttl: CacheTtl,
    ) -> Result<Self, AnchorError> {
        self.system = Some(SystemPrompt { text: text.into(), cache_control: None });
        self.place_anchor(slot, SlotLocation::System, ttl)?;
        Ok(self)
    }

    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    /// Attach a cache breakpoint on the last tool.
    pub fn with_tools_cached(mut self, slot: CacheSlot, tools: Vec<Tool>, ttl: CacheTtl) -> Result<Self, AnchorError> {
        if tools.is_empty() {
            return Err(AnchorError::NoToolsToCache);
        }
        self.tools = tools;
        self.place_anchor(slot, SlotLocation::Tools, ttl)?;
        Ok(self)
    }

    // ── Append-only evolution ───────────────────────────────────────────────

    fn push(&mut self, role: &'static str, content: Vec<ContentBlock>) {
        self.messages.push(Message { role, content });
    }
    pub fn push_user(&mut self, blocks: Vec<ContentBlock>) {
        self.push("user", blocks);
    }
    pub fn push_assistant(&mut self, blocks: Vec<ContentBlock>) {
        self.push("assistant", blocks);
    }
    pub fn push_user_text(&mut self, text: impl Into<String>) {
        self.push("user", vec![ContentBlock::text(text)]);
    }
    pub fn push_assistant_text(&mut self, text: impl Into<String>) {
        self.push("assistant", vec![ContentBlock::text(text)]);
    }
    pub fn push_tool_result(&mut self, tool_use_id: impl Into<String>, content: ToolResultContent) {
        self.push("user", vec![ContentBlock::tool_result(tool_use_id, content)]);
    }

    // ── Cache slot ops ──────────────────────────────────────────────────────

    /// Move `slot` to the last block of the last message with the given TTL.
    /// Clears any previous placement (metadata only — content never touched).
    /// Validates TTL ordering and mid-evolution conflicts before mutating.
    pub fn roll_cache(&mut self, slot: CacheSlot, ttl: CacheTtl) -> Result<(), RollCacheError> {
        let i = slot.idx();
        if let Some(s) = self.slots[i]
            && matches!(s.location, SlotLocation::System | SlotLocation::Tools)
        {
            return Err(RollCacheError::SlotOccupiedByAnchor(slot));
        }

        let (msg, block) = self.tail_position()?;
        let target = SlotLocation::Message { msg, block };

        // Another slot at the same position with a different TTL would
        // overwrite its cache_control on commit — refuse before mutating.
        for (j, other) in self.slots.iter().enumerate() {
            if j != i
                && let Some(s) = other
                && s.location == target
                && s.ttl != ttl
            {
                return Err(RollCacheError::ConflictingTtlAtSamePosition);
            }
        }

        self.validate_ordering_with_override(slot, Some((target, ttl)))?;

        // Commit: clear old position's metadata, write new.
        if let Some(state) = self.slots[i].take() {
            self.write_cache_control(state.location, None);
        }
        self.write_cache_control(target, Some(CacheControl::ephemeral(ttl)));
        self.slots[i] = Some(SlotState { location: target, ttl });
        Ok(())
    }

    /// Remove `slot` and clear its `cache_control`. No-op if empty.
    /// Refuses anchor slots (immutable for the Context's lifetime).
    pub fn clear_cache(&mut self, slot: CacheSlot) -> Result<(), RollCacheError> {
        let i = slot.idx();
        let Some(state) = self.slots[i] else { return Ok(()) };
        if matches!(state.location, SlotLocation::System | SlotLocation::Tools) {
            return Err(RollCacheError::SlotOccupiedByAnchor(slot));
        }
        self.write_cache_control(state.location, None);
        self.slots[i] = None;
        Ok(())
    }

    pub fn breakpoint_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    // ── Internals ───────────────────────────────────────────────────────────

    fn tail_position(&self) -> Result<(usize, usize), RollCacheError> {
        let m = self.messages.len().checked_sub(1).ok_or(RollCacheError::NoBlocksToCache)?;
        let b = self.messages[m].content.len().checked_sub(1).ok_or(RollCacheError::NoBlocksToCache)?;
        Ok((m, b))
    }

    /// Record an anchor (System/Tools) in `slot` and stamp its `cache_control`.
    /// Caller must have already installed `self.system` / `self.tools` with
    /// `cache_control: None`. Refuses to overwrite an occupied slot so anchors
    /// never clobber an existing breakpoint.
    fn place_anchor(&mut self, slot: CacheSlot, location: SlotLocation, ttl: CacheTtl) -> Result<(), AnchorError> {
        debug_assert!(matches!(location, SlotLocation::System | SlotLocation::Tools));
        if self.slots[slot.idx()].is_some() {
            return Err(AnchorError::SlotAlreadyInUse(slot));
        }
        self.write_cache_control(location, Some(CacheControl::ephemeral(ttl)));
        self.slots[slot.idx()] = Some(SlotState { location, ttl });
        Ok(())
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
                if let Some(b) = self.messages.get_mut(msg).and_then(|m| m.content.get_mut(block)) {
                    *b.cache_control_mut() = cc;
                }
            }
        }
    }

    /// Verify final wire order is `[1h…, 5m…]`. `new_state` simulates a pending
    /// `roll_cache` before committing.
    fn validate_ordering_with_override(
        &self,
        override_slot: CacheSlot,
        new_state: Option<(SlotLocation, CacheTtl)>,
    ) -> Result<(), RollCacheError> {
        let mut placements: Vec<(FlowKey, CacheTtl)> = CacheSlot::ALL
            .iter()
            .filter_map(|&slot| {
                let s = if slot == override_slot {
                    new_state.map(|(location, ttl)| SlotState { location, ttl })
                } else {
                    self.slots[slot.idx()]
                };
                s.map(|s| (flow_key(s.location), s.ttl))
            })
            .collect();
        placements.sort_by_key(|&(pos, _)| pos);

        let mut seen_5m = false;
        for (_, ttl) in placements {
            match ttl {
                CacheTtl::FiveMinutes => seen_5m = true,
                CacheTtl::OneHour if seen_5m => return Err(RollCacheError::TtlOrderingViolation),
                CacheTtl::OneHour => {}
            }
        }
        Ok(())
    }
}

// Sort key for TTL-ordering: request-flow order tools→system→messages, then
// (msg, block) within messages. Tuple ordering gives the same result as the
// old hand-rolled sentinels, without the `usize::MAX/N` magic.
type FlowKey = (u8, usize, usize);

fn flow_key(loc: SlotLocation) -> FlowKey {
    match loc {
        SlotLocation::Tools => (0, 0, 0),
        SlotLocation::System => (1, 0, 0),
        SlotLocation::Message { msg, block } => (2, msg, block),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{Model, Request};

    fn req(ctx: &Context) -> serde_json::Value {
        serde_json::to_value(Request::new(ctx, Model::opus_4_7(), 1024)).unwrap()
    }

    #[test]
    fn empty_request_serializes() {
        let v = req(&Context::new());
        assert_eq!(v["model"], "claude-opus-4-7");
        assert_eq!(v["max_tokens"], 1024);
        assert!(v["messages"].is_array());
    }

    #[test]
    fn roll_cache_on_empty_errors() {
        let mut ctx = Context::new();
        assert_eq!(ctx.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap_err(), RollCacheError::NoBlocksToCache,);
    }

    #[test]
    fn roll_cache_tail_and_move() {
        let mut ctx = Context::new();
        ctx.push_user_text("one");
        ctx.roll_cache(CacheSlot::S3, CacheTtl::FiveMinutes).unwrap();
        assert_eq!(req(&ctx)["messages"][0]["content"][0]["cache_control"]["ttl"], "5m");

        // Rolling to a new tail clears the old position's cache_control.
        ctx.push_assistant_text("two");
        ctx.push_user_text("three");
        ctx.roll_cache(CacheSlot::S3, CacheTtl::FiveMinutes).unwrap();
        let v = req(&ctx);
        assert!(v["messages"][0]["content"][0].get("cache_control").is_none());
        assert_eq!(v["messages"][2]["content"][0]["cache_control"]["ttl"], "5m");
    }

    #[test]
    fn anchors_cannot_be_rolled() {
        let mut ctx = Context::new().with_system_cached(CacheSlot::S0, "sys", CacheTtl::OneHour).unwrap();
        ctx.push_user_text("hi");
        assert_eq!(
            ctx.roll_cache(CacheSlot::S0, CacheTtl::OneHour).unwrap_err(),
            RollCacheError::SlotOccupiedByAnchor(CacheSlot::S0),
        );
    }

    #[test]
    fn ttl_ordering_enforced() {
        let mut ctx = Context::new();
        ctx.push_user_text("one");
        ctx.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
        ctx.push_user_text("two");
        // 1h after 5m rejected.
        assert_eq!(ctx.roll_cache(CacheSlot::S1, CacheTtl::OneHour).unwrap_err(), RollCacheError::TtlOrderingViolation,);

        // 1h system anchor then 5m tail is fine.
        let mut ctx = Context::new().with_system_cached(CacheSlot::S0, "sys", CacheTtl::OneHour).unwrap();
        ctx.push_user_text("hi");
        ctx.roll_cache(CacheSlot::S3, CacheTtl::FiveMinutes).unwrap();
        assert_eq!(ctx.breakpoint_count(), 2);
    }

    #[test]
    fn conflicting_ttl_at_same_position_rejected() {
        let mut ctx = Context::new();
        ctx.push_user_text("one");
        ctx.roll_cache(CacheSlot::S0, CacheTtl::OneHour).unwrap();
        // S1 targets the same tail block with a different TTL — committing would
        // overwrite S0's cache_control and desync slot bookkeeping.
        assert_eq!(
            ctx.roll_cache(CacheSlot::S1, CacheTtl::FiveMinutes).unwrap_err(),
            RollCacheError::ConflictingTtlAtSamePosition,
        );
        // Same position with matching TTL is fine (idempotent co-location).
        ctx.roll_cache(CacheSlot::S1, CacheTtl::OneHour).unwrap();
        assert_eq!(ctx.breakpoint_count(), 2);
    }

    #[test]
    fn clear_cache_removes_metadata() {
        let mut ctx = Context::new();
        ctx.push_user_text("hi");
        ctx.roll_cache(CacheSlot::S3, CacheTtl::FiveMinutes).unwrap();
        ctx.clear_cache(CacheSlot::S3).unwrap();
        assert!(req(&ctx)["messages"][0]["content"][0].get("cache_control").is_none());
        assert_eq!(ctx.breakpoint_count(), 0);
    }

    #[test]
    fn system_wire_shape_switches_on_cache() {
        // Plain string when no cache_control.
        assert_eq!(req(&Context::new().with_system("you are helpful"))["system"], "you are helpful");

        // One-element block array when cached.
        let v = req(&Context::new().with_system_cached(CacheSlot::S0, "sys", CacheTtl::OneHour).unwrap());
        assert_eq!(v["system"][0]["type"], "text");
        assert_eq!(v["system"][0]["text"], "sys");
        assert_eq!(v["system"][0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn tool_result_is_error_emitted_as_bool() {
        let mut ctx = Context::new();
        ctx.push_user(vec![
            ContentBlock::tool_result("tu_1", ToolResultContent::Text("ok".into())),
            ContentBlock::tool_result_err("tu_2", ToolResultContent::Text("oops".into())),
        ]);
        let v = req(&ctx);
        assert_eq!(v["messages"][0]["content"][0]["is_error"], false);
        assert_eq!(v["messages"][0]["content"][1]["is_error"], true);
    }

    #[test]
    fn tools_cached_marks_last_tool() {
        let tools = vec![
            Tool::new("one", serde_json::json!({"type": "object"})),
            Tool::new("two", serde_json::json!({"type": "object"})),
        ];
        let v = req(&Context::new().with_tools_cached(CacheSlot::S1, tools, CacheTtl::OneHour).unwrap());
        assert!(v["tools"][0].get("cache_control").is_none());
        assert_eq!(v["tools"][1]["cache_control"]["ttl"], "1h");
    }
}
