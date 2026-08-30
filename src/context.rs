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
//!
//! Every wire value drawn from a closed API vocabulary is the matching enum from
//! [`crate::values`], never the string it serializes to. A `&'static str` field is
//! writable with any string; an enum field is writable only with a value the API
//! accepts.
//!
//! [`Message`] goes one step further: the role is not a field at all. The three
//! roles do not admit the same content — a system message takes text and tool
//! changes only — so the role is the *variant*, and [`Message::role`] derives the
//! wire value from it. A role beside a free content list cannot be written:
//!
//! ```compile_fail
//! use anthropic::context::{ContentBlock, Message};
//!
//! // There is no `role` field to set, so this does not compile.
//! let _ = Message { role: anthropic::Role::System, content: Vec::<ContentBlock>::new() };
//! ```
//!
//! ```compile_fail
//! use anthropic::context::{ContentBlock, Message};
//!
//! // Nor can a system message hold an ordinary content block: its variant takes
//! // `SystemBlock`s, which have no image or tool-result form.
//! let _ = Message::System(vec![ContentBlock::text("no")]);
//! ```

use crate::block::{ContentBlock, TextBlock, ToolResultContent};
use crate::system::SystemBlock;
use crate::{CacheControlType, CacheTtl, Role};
use serde::Serialize;
use serde_json::Value;

// ── Cache control ────────────────────────────────────────────────────────────

/// A cache breakpoint's wire form.
///
/// Has no public constructor and no public fields, and every `cache_control` slot
/// that holds one is crate-private. So the only way to place a breakpoint is
/// through a [`CacheSlot`], which keeps slot bookkeeping consistent with content.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub(crate) kind: CacheControlType,
    pub(crate) ttl: CacheTtl,
}

impl CacheControl {
    pub(crate) fn ephemeral(ttl: CacheTtl) -> Self {
        Self { kind: CacheControlType::Ephemeral, ttl }
    }
}

/// One of the four cache breakpoints a request may carry.
///
/// A fixed, named set that mirrors the API's limit one-to-one, so asking for more
/// breakpoints than the API accepts is not a runtime error but an unwritable
/// program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheSlot {
    /// The first slot.
    S0,
    /// The second slot.
    S1,
    /// The third slot.
    S2,
    /// The fourth slot.
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

// ── Messages, tools, system ──────────────────────────────────────────────────

/// One entry of the conversation.
///
/// An enum rather than a `role` field beside a `content` field, because the three
/// roles do not admit the same content: a system message takes text and tool
/// changes only, and the API answers `role 'system' supports text,
/// tool_addition, and tool_removal blocks only` for anything else. A struct with
/// both fields public would let that rejection be written; here the variant
/// carries exactly the blocks its role admits, and [`Self::role`] derives the wire
/// value.
///
/// The blocks themselves stay public, because any sequence of them is a legal
/// `content` array and pattern matching over replayed history is worth keeping.
/// What is *not* public is `cache_control` on the blocks inside — that carries a
/// cross-message invariant, so it stays crate-private and reachable only through a
/// [`CacheSlot`].
#[derive(Debug, Clone)]
pub enum Message {
    /// The caller's turn. Tool results go here too.
    User(
        /// The turn's content blocks.
        Vec<ContentBlock>,
    ),
    /// The model's own turn, replayed back into the conversation.
    Assistant(
        /// The turn's content blocks.
        Vec<ContentBlock>,
    ),
    /// An instruction added partway through, after the cached prefix. See
    /// [`crate::system`].
    System(
        /// The instruction's blocks. Never empty: the API answers `system content
        /// must contain at least one block`, and [`Context::push_system`] refuses
        /// an empty one before it commits.
        Vec<SystemBlock>,
    ),
}

/// The wire shape both content kinds share: a role beside its blocks.
#[derive(Serialize)]
struct MessageWire<'a, B> {
    role: Role,
    content: &'a Vec<B>,
}

impl Serialize for Message {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // The role is written from `role()` rather than stored, so it cannot
        // disagree with the variant that decides what content is legal.
        match self {
            Message::User(content) | Message::Assistant(content) => {
                MessageWire { role: self.role(), content }.serialize(s)
            }
            Message::System(content) => MessageWire { role: self.role(), content }.serialize(s),
        }
    }
}

impl Message {
    /// Whose entry this is.
    pub fn role(&self) -> Role {
        match self {
            Message::User(_) => Role::User,
            Message::Assistant(_) => Role::Assistant,
            Message::System(_) => Role::System,
        }
    }

    /// The entry's blocks, for the two roles that carry [`ContentBlock`]s.
    /// `None` on a system message, whose blocks are [`SystemBlock`]s — a
    /// different set, which is the whole reason this type is an enum.
    pub fn content(&self) -> Option<&[ContentBlock]> {
        match self {
            Message::User(content) | Message::Assistant(content) => Some(content),
            Message::System(_) => None,
        }
    }

    /// A system message's blocks. `None` on the other two roles.
    pub fn system_content(&self) -> Option<&[SystemBlock]> {
        match self {
            Message::System(content) => Some(content),
            Message::User(_) | Message::Assistant(_) => None,
        }
    }

    fn cache_control_at(&mut self, block: usize) -> Option<&mut Option<CacheControl>> {
        match self {
            Message::User(content) | Message::Assistant(content) => {
                content.get_mut(block).map(ContentBlock::cache_control_mut)
            }
            Message::System(content) => content.get_mut(block).map(SystemBlock::cache_control_mut),
        }
    }

    fn block_count(&self) -> usize {
        match self {
            Message::User(content) | Message::Assistant(content) => content.len(),
            Message::System(content) => content.len(),
        }
    }
}

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
    /// not a property of one tool, so [`Context::with_tools`] checks it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub defer_loading: bool,
    /// Whether the API validates the model's tool names and inputs against the
    /// schema. Emitted only when `true`, for the same prompt-identity reason.
    ///
    /// Measured: the NVIDIA inference gateway refuses this field with
    /// `tools.0.custom.strict: Extra inputs are not permitted`. It is in the
    /// documented stable schema, so it is here.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub strict: bool,
    /// Example inputs shown to the model beside the schema. Empty means none;
    /// there is no "no examples" distinct from "an empty list of examples".
    ///
    /// Measured: the NVIDIA inference gateway refuses this field with
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
    /// [`Context::with_tools`] checks.
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

// Wire shape: bare string when no cache_control, one-element block array when set.
#[derive(Debug, Clone)]
pub(crate) struct SystemPrompt {
    pub(crate) text: String,
    pub(crate) cache_control: Option<CacheControl>,
}

impl Serialize for SystemPrompt {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match &self.cache_control {
            // The bare string, which is the shape a prompt with no breakpoint takes.
            None => self.text.serialize(s),
            // A one-element array of text blocks, which is the only shape that can
            // carry a breakpoint. `TextBlock` already serializes as one.
            Some(_) => [TextBlock { text: self.text.clone(), cache_control: self.cache_control }].serialize(s),
        }
    }
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// Why a rolling cache breakpoint could not be placed or moved.
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

/// Why a `system` or `tools` cache anchor could not be placed.
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

/// Why a mid-conversation system message could not be appended.
///
/// Each variant is a placement rule the API enforces with a 400, refused here
/// before the request is built. The rule about what must *follow* the message is
/// not here, because no append can decide it — see
/// [`crate::request::RequestError::SystemMessageNotFollowedByAssistant`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemMessageError {
    /// No blocks. The API answers `system content must contain at least one
    /// block`; a system message says something or it is not one.
    Empty,
    /// It would be the first entry, with no user turn for it to follow. A
    /// conversation that opens with an instruction has a system *prompt*, which is
    /// [`Context::with_system`] and is cached better anyway.
    First,
    /// It would follow an assistant turn. The API accepts one only after a user
    /// turn, or after an assistant turn ending in a server tool result — a block
    /// this crate cannot yet build, so no assistant tail it produces qualifies.
    AfterAssistant,
}

impl std::fmt::Display for SystemMessageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SystemMessageError::Empty => write!(f, "a system message must carry at least one block"),
            SystemMessageError::First => {
                write!(f, "a system message cannot be first; use `with_system` for a system prompt")
            }
            SystemMessageError::AfterAssistant => {
                write!(f, "a system message must follow a user turn, not an assistant turn")
            }
        }
    }
}

impl std::error::Error for SystemMessageError {}

// ── Context ──────────────────────────────────────────────────────────────────

/// The conversation, and the cache invariants that hold over it.
///
/// Append-only by construction: there is no `&mut` path to a committed message,
/// because rewriting history silently invalidates the prompt cache. System prompt
/// and tools are set once, at construction. Breakpoints live in four named
/// [`CacheSlot`]s and are moved by metadata-only operations that validate TTL
/// ordering *before* they commit.
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
    /// An empty conversation with no system prompt, tools, or breakpoints.
    pub fn new() -> Self {
        Self { system: None, tools: Vec::new(), messages: Vec::new(), slots: [None; 4] }
    }

    /// Sets the system prompt, uncached.
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

    /// Sets the tools, uncached.
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

    /// Appends a user turn.
    pub fn push_user(&mut self, blocks: Vec<ContentBlock>) {
        self.messages.push(Message::User(blocks));
    }
    /// Appends an assistant turn — the model's own reply, replayed.
    pub fn push_assistant(&mut self, blocks: Vec<ContentBlock>) {
        self.messages.push(Message::Assistant(blocks));
    }
    /// Appends a user turn of one text block.
    pub fn push_user_text(&mut self, text: impl Into<String>) {
        self.push_user(vec![ContentBlock::text(text)]);
    }
    /// Appends an assistant turn of one text block.
    pub fn push_assistant_text(&mut self, text: impl Into<String>) {
        self.push_assistant(vec![ContentBlock::text(text)]);
    }
    /// Appends a tool result as a user turn, which is where the API expects one.
    pub fn push_tool_result(&mut self, tool_use_id: impl Into<String>, content: ToolResultContent) {
        self.push_user(vec![ContentBlock::tool_result(tool_use_id, content)]);
    }

    /// Appends a mid-conversation system message: an instruction, or a tool
    /// offered or withdrawn, from this point in the conversation onwards.
    ///
    /// Cache-safe by construction, which is the reason to prefer it over rewriting
    /// the system prompt: it appends, so every byte before it is unchanged and the
    /// cached prefix still matches. See [`crate::system`].
    ///
    /// The API enforces placement rules with a 400, and the two that can be
    /// decided from the history so far are checked here — see
    /// [`SystemMessageError`]. The remaining one is "must end the array or precede
    /// an assistant turn", which no append can decide because a later append can
    /// break it; [`crate::request::Request::new`] checks that one, where the
    /// history is final.
    pub fn push_system(&mut self, blocks: Vec<SystemBlock>) -> Result<(), SystemMessageError> {
        if blocks.is_empty() {
            return Err(SystemMessageError::Empty);
        }
        match self.messages.last() {
            None => return Err(SystemMessageError::First),
            // An assistant turn ending in a server tool use is also legal, but
            // this crate cannot yet build a server-tool block, so no assistant
            // tail it can produce satisfies the rule.
            Some(Message::Assistant(_)) => return Err(SystemMessageError::AfterAssistant),
            Some(Message::User(_) | Message::System(_)) => {}
        }
        self.messages.push(Message::System(blocks));
        Ok(())
    }

    /// Appends a mid-conversation system message of one instruction.
    pub fn push_system_text(&mut self, text: impl Into<String>) -> Result<(), SystemMessageError> {
        self.push_system(vec![SystemBlock::text(text)])
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

    /// How many of the four slots hold a breakpoint.
    pub fn breakpoint_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// The tools the model may call.
    ///
    /// Readable because a request-level invariant depends on the whole list —
    /// see [`crate::request::RequestError::EveryToolDeferred`] — and because a
    /// caller replaying a conversation wants to see what it offered.
    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// How many entries the conversation holds.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Where a system message sits that neither ends the conversation nor
    /// precedes an assistant turn, if any.
    ///
    /// The half of the placement rules an append cannot decide, because appending
    /// a user turn after a legal system message makes it illegal. Read by
    /// [`crate::request::Request::new`], where the history is final.
    pub(crate) fn misplaced_system_message(&self) -> Option<usize> {
        self.messages.iter().enumerate().find_map(|(at, message)| {
            let misplaced = matches!(message, Message::System(_))
                && !matches!(self.messages.get(at + 1), None | Some(Message::Assistant(_)));
            misplaced.then_some(at)
        })
    }

    // ── Internals ───────────────────────────────────────────────────────────

    fn tail_position(&self) -> Result<(usize, usize), RollCacheError> {
        let m = self.messages.len().checked_sub(1).ok_or(RollCacheError::NoBlocksToCache)?;
        let b = self.messages[m].block_count().checked_sub(1).ok_or(RollCacheError::NoBlocksToCache)?;
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
                if let Some(slot) = self.messages.get_mut(msg).and_then(|m| m.cache_control_at(block)) {
                    *slot = cc;
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
        serde_json::to_value(Request::new(ctx, Model::opus_4_8(), 1024).unwrap()).unwrap()
    }

    #[test]
    fn empty_request_serializes() {
        let v = req(&Context::new());
        assert_eq!(v["model"], "claude-opus-4-8");
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
    fn roles_reach_the_wire_as_their_strings() {
        let mut ctx = Context::new();
        ctx.push_user_text("one");
        ctx.push_assistant_text("two");
        ctx.push_tool_result("tu_1", ToolResultContent::Text("three".into()));
        let v = req(&ctx);
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][1]["role"], "assistant");
        // A tool result is a user turn, which is where the API expects one.
        assert_eq!(v["messages"][2]["role"], "user");
        // The role is derived from the variant, and is the same value the wire
        // string names.
        assert_eq!(ctx.messages[0].role(), Role::User);
        assert_eq!(ctx.messages[1].role(), Role::Assistant);
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
    }

    /// A mid-conversation system message: appended after the prefix, so nothing
    /// before it moves and the cache still matches.
    #[test]
    fn a_system_message_reaches_the_wire_after_the_turn_it_follows() {
        let mut ctx = Context::new().with_system("you are helpful");
        ctx.push_user_text("name a fruit");
        ctx.push_system_text("Answer in French.").unwrap();
        let v = req(&ctx);
        assert_eq!(v["system"], "you are helpful", "the cached prompt is untouched");
        assert_eq!(v["messages"][1]["role"], "system");
        assert_eq!(v["messages"][1]["content"][0], serde_json::json!({"type": "text", "text": "Answer in French."}));
        assert_eq!(ctx.messages[1].role(), Role::System);
        assert_eq!(ctx.messages[1].system_content().unwrap().len(), 1);
        assert!(ctx.messages[1].content().is_none(), "a system message holds SystemBlocks, not ContentBlocks");
    }

    /// A tool withdrawn mid-conversation, which leaves `tools` byte-identical and
    /// so leaves the tools cache warm.
    #[test]
    fn a_tool_change_rides_in_a_system_message_without_touching_the_tool_definitions() {
        use crate::system::{SystemBlock, ToolReference};
        let tools = vec![Tool::new("get_time", serde_json::json!({"type": "object"}))];
        let mut ctx = Context::new().with_tools(tools);
        ctx.push_user_text("what time is it");
        ctx.push_system(vec![
            SystemBlock::text("The clock is offline."),
            SystemBlock::tool_removal(ToolReference::tool("get_time")),
        ])
        .unwrap();
        let v = req(&ctx);
        assert_eq!(v["tools"][0]["name"], "get_time", "the definition stays, so its cache stays");
        assert_eq!(v["messages"][1]["content"][1]["type"], "tool_removal");
        assert_eq!(v["messages"][1]["content"][1]["tool"]["name"], "get_time");
    }

    /// The placement rules the API enforces with a 400, refused before the request
    /// is built. Each is stated by the server; see `SystemMessageError`.
    #[test]
    fn a_system_message_refuses_the_placements_the_api_rejects() {
        let mut empty = Context::new();
        empty.push_user_text("hi");
        assert_eq!(empty.push_system(vec![]), Err(SystemMessageError::Empty));

        let mut first = Context::new();
        assert_eq!(first.push_system_text("Answer in French.").err(), Some(SystemMessageError::First));

        let mut after_assistant = Context::new();
        after_assistant.push_user_text("hi");
        after_assistant.push_assistant_text("hello");
        assert_eq!(after_assistant.push_system_text("x").err(), Some(SystemMessageError::AfterAssistant));

        // Two in a row is accepted, which the live API confirms.
        let mut adjacent = Context::new();
        adjacent.push_user_text("hi");
        adjacent.push_system_text("first").unwrap();
        adjacent.push_system_text("second").unwrap();
        assert_eq!(adjacent.message_count(), 3);
    }

    /// The other half of the rules: what must *follow* a system message is a
    /// property of the finished history, so appending a user turn after one turns
    /// a legal conversation into an illegal request.
    #[test]
    fn a_system_message_followed_by_a_user_turn_is_caught_at_the_request() {
        let mut ctx = Context::new();
        ctx.push_user_text("name a fruit");
        ctx.push_system_text("Answer in French.").unwrap();
        assert_eq!(ctx.misplaced_system_message(), None, "ending the array is legal");

        ctx.push_user_text("and a color");
        assert_eq!(ctx.misplaced_system_message(), Some(1), "a user turn after it is not");
        assert_eq!(
            crate::request::Request::new(&ctx, Model::opus_5(), 1024).err(),
            Some(crate::request::RequestError::SystemMessageNotFollowedByAssistant { at: 1 }),
        );

        // Preceding an assistant turn is legal, so the same history plus a reply is
        // a request again.
        ctx.push_assistant_text("Une pomme.");
        assert_eq!(ctx.misplaced_system_message(), Some(1), "the user turn at index 2 is still in the way");
    }

    /// A cache breakpoint lands on a system message's inner block, which is where
    /// the API accepts one: `cache_control on mid_conv_system is not supported;
    /// set it on an inner content block instead`.
    #[test]
    fn a_breakpoint_lands_on_a_system_messages_inner_block() {
        let mut ctx = Context::new();
        ctx.push_user_text("hi");
        ctx.push_system_text("Answer in French.").unwrap();
        ctx.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
        let v = req(&ctx);
        assert_eq!(v["messages"][1]["content"][0]["cache_control"]["ttl"], "5m");
        assert!(v["messages"][1].get("cache_control").is_none(), "never on the message itself");
    }

    /// The tool flags are rendered into the prompt, so each appears only when the
    /// caller asked for it: emitting `false` writes a different prefix and a
    /// different cache key.
    #[test]
    fn tool_flags_appear_only_when_set() {
        let plain = Tool::new("one", serde_json::json!({"type": "object"}));
        let v = req(&Context::new().with_tools(vec![plain]));
        for field in ["defer_loading", "strict", "input_examples"] {
            assert!(v["tools"][0].get(field).is_none(), "{field} should be absent when unset");
        }

        let configured = Tool::new("two", serde_json::json!({"type": "object"}))
            .strict()
            .input_examples(vec![serde_json::json!({"city": "Paris"})]);
        let v = req(&Context::new().with_tools(vec![configured]));
        assert_eq!(v["tools"][0]["strict"], true);
        assert_eq!(v["tools"][0]["input_examples"][0]["city"], "Paris");
    }

    /// A deferred tool costs no prompt tokens until a tool search finds it, but the
    /// API refuses a request in which every tool is deferred — a relation across
    /// the list, so it is checked where the request is built.
    #[test]
    fn deferring_every_tool_is_refused_at_the_request() {
        let deferred = || Tool::new("t", serde_json::json!({"type": "object"})).deferred();
        let ctx = Context::new().with_tools(vec![deferred()]);
        assert_eq!(
            crate::request::Request::new(&ctx, Model::opus_4_8(), 16).err(),
            Some(crate::request::RequestError::EveryToolDeferred { tools: 1 }),
        );

        // One undeferred tool is enough, and only the deferred one says so.
        let ctx =
            Context::new().with_tools(vec![Tool::new("eager", serde_json::json!({"type": "object"})), deferred()]);
        let v = req(&ctx);
        assert!(v["tools"][0].get("defer_loading").is_none());
        assert_eq!(v["tools"][1]["defer_loading"], true);
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
