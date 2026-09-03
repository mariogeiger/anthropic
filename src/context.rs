//! Cache-safe, append-only conversation state.
//!
//! Invariants enforced by construction: past bytes are frozen (no `&mut` to old
//! messages); the [`Opening`] is an argument to [`Context::new`] and `tools` are
//! set at construction, so the wire order `tools → system → messages` is the
//! order the type holds them in rather than one rebuilt at serialization; 4 cache
//! breakpoints live in named slots (`CacheSlot::S0..S3`) — impossible to exceed
//! the API limit; `roll_cache` only moves slot metadata, never rewrites content;
//! TTL ordering (1h before 5m) validated before every commit.
//!
//! Types model what the model *sees*, not wire-format field presence: every
//! `Option` represents a real runtime distinction. `SystemPrompt` is one struct
//! with two wire shapes (bare string vs one-element array); the serializer picks.
//!
//! `cache_control` is not reachable from outside the crate: `CacheControl` has
//! no public constructor and no public fields, and the `cache_control` slot on
//! every content block and `Tool` is crate-private. The only way to attach a
//! breakpoint is through `CacheSlot` via `Opening::CachedInstruction`,
//! `with_tools_cached`, or `roll_cache`, which keeps slot bookkeeping
//! consistent with content.
//!
//! Every wire value drawn from a closed API vocabulary is the matching enum from
//! [`crate::values`], never the string it serializes to. A `&'static str` field is
//! writable with any string; an enum field is writable only with a value the API
//! accepts.
//!
//! [`Message`] goes one step further: the role is not a field at all. The three
//! roles do not admit the same content — a system message is one of the three
//! exact shapes in [`SystemMessage`] — so the role is the *variant*, and
//! [`Message::role`] derives the wire value from it. A role beside a free content
//! list cannot be written:
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
//! use anthropic::system::SystemMessage;
//!
//! // Nor can persistent system content hold an ordinary content block.
//! let _ = Message::System(SystemMessage::Persistent(vec![ContentBlock::text("no")]));
//! ```
//!
//! The placement rules are therefore checked in one place each, and the check
//! cannot be walked around: `messages` is private and there is no `&mut` path to
//! it, so [`Context::push_system`] is the only way a system message enters a
//! conversation and [`crate::request::Request::new`] the only way one leaves.
//!
//! ```compile_fail
//! use anthropic::context::{Context, Opening};
//! use anthropic::system::SystemMessage;
//!
//! // The field is private, so a leading system message cannot be installed
//! // behind `push_system`'s back.
//! let mut ctx = Context::new(Opening::None);
//! ctx.messages.push(anthropic::context::Message::System(
//!     SystemMessage::Persistent(Vec::new()),
//! ));
//! ```
//!
//! The opening is the same story one level up. It is an argument to
//! [`Context::new`], not a field and not a builder step, so a conversation always
//! has its opening decided before it can hold a message and no later call can
//! replace it:
//!
//! ```compile_fail
//! use anthropic::context::{Context, Opening};
//!
//! // There is no `with_system` to install an opening after the fact.
//! let ctx = Context::new(Opening::None).with_system("too late");
//! ```
//!
//! ```compile_fail
//! use anthropic::context::{Context, Opening};
//!
//! // Nor is the opening an assignable field.
//! let mut ctx = Context::new(Opening::None);
//! ctx.system = Some(String::from("too late"));
//! ```

use crate::block::{ContentBlock, TextBlock, ToolResultContent};
use crate::system::{PerMessageEffort, SystemBlock, SystemClearAt, SystemMessage};
use crate::{BetaFeature, CacheControlType, CacheTtl, Role};
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
/// roles do not admit the same content: a system message is persistent content,
/// an effort-only change, or turn-scoped text. A struct with both fields public
/// would let rejected combinations be written; here the variant carries exactly
/// the shape its role admits, and [`Self::role`] derives the wire value.
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
    /// An instruction, effort change, or turn-scoped reminder added partway
    /// through the conversation. See [`crate::system`].
    System(
        /// The exact system-message shape. The variants keep turn-scoped text,
        /// effort-only configuration, and persistent content disjoint.
        SystemMessage,
    ),
}

/// The wire shape user, assistant, and persistent system content share: a role
/// beside its blocks.
#[derive(Serialize)]
struct MessageWire<'a, B> {
    role: Role,
    content: &'a [B],
}

#[derive(Serialize)]
struct PerMessageOutputConfig {
    effort: PerMessageEffort,
}

#[derive(Serialize)]
struct EffortSystemMessageWire {
    role: Role,
    content: [(); 0],
    output_config: PerMessageOutputConfig,
}

#[derive(Serialize)]
struct TurnScopedSystemMessageWire<'a> {
    role: Role,
    clear_at: SystemClearAt,
    content: &'a [TextBlock],
}

impl Serialize for Message {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // The role is written from `role()` rather than stored, so it cannot
        // disagree with the variant that decides what content is legal.
        match self {
            Message::User(content) | Message::Assistant(content) => {
                MessageWire { role: self.role(), content }.serialize(s)
            }
            Message::System(SystemMessage::Persistent(content)) => {
                MessageWire { role: self.role(), content }.serialize(s)
            }
            Message::System(SystemMessage::Effort(effort)) => EffortSystemMessageWire {
                role: self.role(),
                content: [],
                output_config: PerMessageOutputConfig { effort: *effort },
            }
            .serialize(s),
            Message::System(SystemMessage::TurnScoped(content)) => {
                TurnScopedSystemMessageWire { role: self.role(), clear_at: SystemClearAt::NextUserMessage, content }
                    .serialize(s)
            }
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
    /// `None` on every system shape, whose own disjoint content is available
    /// through [`Self::system_message`].
    pub fn content(&self) -> Option<&[ContentBlock]> {
        match self {
            Message::User(content) | Message::Assistant(content) => Some(content),
            Message::System(_) => None,
        }
    }

    /// A persistent system message's blocks. `None` on every other shape.
    pub fn system_content(&self) -> Option<&[SystemBlock]> {
        match self {
            Message::System(content) => content.persistent_content(),
            Message::User(_) | Message::Assistant(_) => None,
        }
    }

    /// The exact system-message shape, or `None` for user and assistant turns.
    pub fn system_message(&self) -> Option<&SystemMessage> {
        match self {
            Message::System(message) => Some(message),
            Message::User(_) | Message::Assistant(_) => None,
        }
    }

    fn cache_control_at(&mut self, block: usize) -> Option<&mut Option<CacheControl>> {
        match self {
            Message::User(content) | Message::Assistant(content) => {
                content.get_mut(block).map(ContentBlock::cache_control_mut)
            }
            Message::System(content) => content.cache_control_at(block),
        }
    }

    fn block_count(&self) -> usize {
        match self {
            Message::User(content) | Message::Assistant(content) => content.len(),
            Message::System(content) => content.block_count(),
        }
    }

    fn carries_system_content(&self) -> bool {
        matches!(self, Message::System(message) if message.carries_content())
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

/// How a conversation opens, before any message.
///
/// # Why the opening is not a message
///
/// The API asks this question twice and answers it the same way both times. The
/// top-level `system` field is not an entry in `messages`, and a
/// `{"role": "system"}` entry "cannot be the first entry in `messages`" — "use the
/// top-level `system` field for instructions that apply from the very start"
/// (<https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages>).
/// So the position before the first message is reachable *only* through this
/// field, and an instruction that applies from the start has exactly one home.
///
/// The reason is the prompt cache. The hashed prefix runs `tools`, then `system`,
/// then `messages`, so the opening is the part every later turn is measured
/// against. A message is appended and costs nothing; the opening is rewritten and
/// costs the whole conversation. They are different operations on different parts
/// of the request, which is why they are different types here rather than one
/// list with a special first element.
///
/// # Three openings, because `system` is optional
///
/// The API documents `system` as optional, so "no opening at all" is a real state
/// and not the absence of a decision — a conversation may simply start with
/// messages. Those are the three variants, and the enum is what keeps them three:
/// a `None` plus a `bool` would admit a fourth, meaningless combination.
///
/// Deciding the opening at [`Context::new`] rather than through a builder is what
/// makes the wire order structural. `Context::new(Opening::None)` followed by `with_system` let
/// the opening be installed at any moment, including after messages had been
/// appended — the type said "these three things happen to be here" while the API
/// says "tools, then system, then messages". Taking it as an argument means the
/// opening exists before the first message can, and cannot be replaced afterwards
/// because no method takes `&mut` to it.
#[derive(Debug, Clone)]
pub enum Opening {
    /// No system prompt. The conversation starts with its first message, and the
    /// `system` field is absent from the body.
    None,
    /// An instruction that applies from the very start, with no cache breakpoint.
    Instruction(String),
    /// An instruction with a cache breakpoint at its end, in the named slot.
    ///
    /// The breakpoint is part of the opening rather than a later operation on it,
    /// because an anchor may not move once set: everything cached after it is
    /// measured from here. [`CacheSlot`] is the only way to name one, which is how
    /// the four-breakpoint limit stays unwritable rather than checked.
    CachedInstruction {
        /// The instruction itself.
        text: String,
        /// Which of the four breakpoints anchors it.
        slot: CacheSlot,
        /// How long the entry lives.
        ttl: CacheTtl,
    },
}

impl Opening {
    /// An instruction with no breakpoint.
    pub fn instruction(text: impl Into<String>) -> Self {
        Self::Instruction(text.into())
    }

    /// An instruction anchored by a cache breakpoint.
    pub fn cached_instruction(text: impl Into<String>, slot: CacheSlot, ttl: CacheTtl) -> Self {
        Self::CachedInstruction { text: text.into(), slot, ttl }
    }

    /// The instruction's text, or `None` when the conversation has no opening.
    pub fn text(&self) -> Option<&str> {
        match self {
            Opening::None => None,
            Opening::Instruction(text) | Opening::CachedInstruction { text, .. } => Some(text),
        }
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
            Some(_) => {
                [TextBlock { text: self.text.clone(), citations: Vec::new(), cache_control: self.cache_control }]
                    .serialize(s)
            }
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
    /// The last message is turn-scoped. Its text deliberately never enters a
    /// cache key, so the breakpoint belongs on the preceding user turn instead.
    TurnScopedMessageNotCacheable,
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
            RollCacheError::TurnScopedMessageNotCacheable => {
                write!(f, "a turn-scoped system message cannot carry a cache breakpoint")
            }
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

/// Why a `tools` cache anchor could not be placed.
///
/// The opening's anchor cannot fail — [`Context::new`] places it on a conversation
/// with no slots yet occupied — so only [`Context::with_tools_cached`] returns
/// this.
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
    /// No blocks in a persistent or turn-scoped message. An effort-only message
    /// has empty content by its distinct type and does not use this error.
    Empty,
    /// It would be the first entry, with no user turn for it to follow. A
    /// conversation that opens with an instruction has a system *prompt*, which is
    /// [`Opening::Instruction`] and is cached better anyway.
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
                write!(f, "a system message cannot be first; use `Opening::Instruction` for a system prompt")
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
/// The field order is the wire order — `tools`, then the opening, then `messages`
/// — because that is the order the API hashes the prefix in, and a type that
/// listed them in any other order would be reassembling the request at
/// serialization time instead of holding it.
///
/// Append-only by construction: there is no `&mut` path to a committed message,
/// because rewriting history silently invalidates the prompt cache. The
/// [`Opening`] is fixed at [`Context::new`] and tools at construction.
/// Breakpoints live in four named [`CacheSlot`]s and are moved by metadata-only
/// operations that validate TTL ordering *before* they commit.
pub struct Context {
    pub(crate) tools: Vec<Tool>,
    pub(crate) system: Option<SystemPrompt>,
    pub(crate) messages: Vec<Message>,
    slots: [Option<SlotState>; 4],
}

impl Default for Context {
    /// A conversation with no opening, which is one of the three [`Opening`]s
    /// rather than an unset field.
    fn default() -> Self {
        Self::new(Opening::None)
    }
}

impl Context {
    /// A conversation that opens as `opening` says, with no tools or messages yet.
    ///
    /// The opening is an argument rather than a builder step because it is the
    /// first thing on the wire: taking it here means no `Context` ever exists
    /// without its opening decided, so the opening cannot be installed after a
    /// message has been appended and cannot be replaced later.
    ///
    /// Infallible, which the old `with_system_cached` was not. A slot cannot
    /// already be occupied on a conversation that did not exist a moment ago, so
    /// the only way the anchor placement could fail is gone — one fewer `Result`
    /// for every caller, bought by construction rather than by ignoring an error.
    pub fn new(opening: Opening) -> Self {
        let mut context = Self { tools: Vec::new(), system: None, messages: Vec::new(), slots: [None; 4] };
        match opening {
            Opening::None => {}
            Opening::Instruction(text) => {
                context.system = Some(SystemPrompt { text, cache_control: None });
            }
            Opening::CachedInstruction { text, slot, ttl } => {
                context.system = Some(SystemPrompt { text, cache_control: None });
                context.write_cache_control(SlotLocation::System, Some(CacheControl::ephemeral(ttl)));
                context.slots[slot.idx()] = Some(SlotState { location: SlotLocation::System, ttl });
            }
        }
        context
    }

    /// The instruction this conversation opened with, if it opened with one.
    ///
    /// Readable because a caller replaying a conversation wants to see what the
    /// model was told from the start, and because the opening is the one part of
    /// the prefix that a cache miss is usually traced to.
    pub fn opening(&self) -> Option<&str> {
        self.system.as_ref().map(|prompt| prompt.text.as_str())
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
        self.place_tools_anchor(slot, ttl)?;
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
        self.push_content_system(SystemMessage::persistent(blocks))
    }

    /// Appends a mid-conversation system message of one persistent instruction.
    pub fn push_system_text(&mut self, text: impl Into<String>) -> Result<(), SystemMessageError> {
        self.push_system(vec![SystemBlock::text(text)])
    }

    /// Changes effort from the next user turn onward without rewriting the
    /// request's cached prefix.
    ///
    /// Effort-only messages carry empty content and are accepted anywhere in
    /// `messages`. Model support is checked by [`crate::request::Request::new`],
    /// where both the conversation and selected model are known.
    pub fn push_effort(&mut self, effort: PerMessageEffort) {
        self.messages.push(Message::System(SystemMessage::effort(effort)));
    }

    /// Appends text that clears after the next user message while remaining in
    /// the immutable transcript.
    ///
    /// The same placement rules as persistent content apply. The distinct block
    /// type makes tool changes and cache breakpoints impossible here.
    pub fn push_turn_scoped(&mut self, blocks: Vec<TextBlock>) -> Result<(), SystemMessageError> {
        if blocks.is_empty() {
            return Err(SystemMessageError::Empty);
        }
        self.push_content_system(SystemMessage::turn_scoped(blocks))
    }

    /// Appends one turn-scoped text reminder.
    pub fn push_turn_scoped_text(&mut self, text: impl Into<String>) -> Result<(), SystemMessageError> {
        self.push_turn_scoped(vec![TextBlock::new(text)])
    }

    fn push_content_system(&mut self, message: SystemMessage) -> Result<(), SystemMessageError> {
        let before_run = self
            .messages
            .iter()
            .rposition(|message| !matches!(message, Message::System(_)))
            .map(|at| &self.messages[at]);
        match before_run {
            None => return Err(SystemMessageError::First),
            // An assistant turn ending in a server tool use is also legal, but
            // this crate cannot yet build a server-tool block, so no assistant
            // tail it can produce satisfies the rule.
            Some(Message::Assistant(_)) => return Err(SystemMessageError::AfterAssistant),
            Some(Message::User(_)) => {}
            Some(Message::System(_)) => unreachable!("rposition skipped system messages"),
        }
        self.messages.push(Message::System(message));
        Ok(())
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

    /// Where a system message sits whose successor the API refuses, if any.
    ///
    /// The half of the placement rules an append cannot decide, because appending a
    /// user turn after a legal system message makes it illegal. Read by
    /// [`crate::request::Request::new`], where the history is final.
    ///
    /// Another system message is a legal successor, which is *measured* rather than
    /// read off the API's wording: the server says a system message "must precede
    /// an 'assistant' message or end the array", and yet accepts two in a row. What
    /// it enforces is that the *chain* of system messages ends the array or precedes
    /// an assistant turn, so that is what this checks. A live test holds the
    /// measurement in place.
    ///
    /// A chain is therefore reported at its *last* message, which is the one whose
    /// successor is wrong. That is the position a caller has to change.
    pub(crate) fn misplaced_system_message(&self) -> Option<usize> {
        let mut start = 0;
        while start < self.messages.len() {
            if !matches!(self.messages[start], Message::System(_)) {
                start += 1;
                continue;
            }
            let mut end = start + 1;
            while end < self.messages.len() && matches!(self.messages[end], Message::System(_)) {
                end += 1;
            }
            let carries_content = self.messages[start..end].iter().any(Message::carries_system_content);
            let valid_predecessor = start > 0 && matches!(self.messages[start - 1], Message::User(_));
            let valid_successor = matches!(self.messages.get(end), None | Some(Message::Assistant(_)));
            if carries_content && (!valid_predecessor || !valid_successor) {
                return Some(end - 1);
            }
            start = end;
        }
        None
    }

    /// Where the first content-bearing system message sits, if any.
    ///
    /// Effort-only messages are excluded: unlike persistent and turn-scoped
    /// instructions, their model support and placement rules are separate.
    pub(crate) fn first_content_system_message(&self) -> Option<usize> {
        self.messages.iter().position(Message::carries_system_content)
    }

    /// Where the first per-message effort change sits, if any.
    pub(crate) fn first_effort_message(&self) -> Option<usize> {
        self.messages.iter().position(|message| matches!(message, Message::System(SystemMessage::Effort(_))))
    }

    /// The first effort above `high` that reaches a user turn.
    ///
    /// A later effort message before that user replaces the pending level, so an
    /// `xhigh` immediately overwritten by `low` never applies and is accepted.
    pub(crate) fn first_effort_above_high_applied_to_user(&self) -> Option<(usize, PerMessageEffort)> {
        let mut effective = None;
        for (at, message) in self.messages.iter().enumerate() {
            match message {
                Message::System(SystemMessage::Effort(effort)) => effective = Some((at, *effort)),
                Message::User(_) => {
                    if let Some((at, effort @ (PerMessageEffort::Xhigh | PerMessageEffort::Max))) = effective {
                        return Some((at, effort));
                    }
                }
                Message::Assistant(_) | Message::System(_) => {}
            }
        }
        None
    }

    pub(crate) fn requires_beta(&self, feature: BetaFeature) -> bool {
        self.messages.iter().any(|message| matches!(message, Message::System(system) if system.requires(feature)))
    }

    // ── Internals ───────────────────────────────────────────────────────────

    fn tail_position(&self) -> Result<(usize, usize), RollCacheError> {
        let m = self.messages.len().checked_sub(1).ok_or(RollCacheError::NoBlocksToCache)?;
        if matches!(self.messages[m], Message::System(SystemMessage::TurnScoped(_))) {
            return Err(RollCacheError::TurnScopedMessageNotCacheable);
        }
        let b = self.messages[m].block_count().checked_sub(1).ok_or(RollCacheError::NoBlocksToCache)?;
        Ok((m, b))
    }

    /// Record the tools anchor in `slot` and stamp its `cache_control`.
    ///
    /// Only tools reach here now: the opening's anchor is placed by
    /// [`Context::new`], where no slot can yet be occupied. Refuses to overwrite
    /// an occupied slot so an anchor never clobbers an existing breakpoint.
    fn place_tools_anchor(&mut self, slot: CacheSlot, ttl: CacheTtl) -> Result<(), AnchorError> {
        if self.slots[slot.idx()].is_some() {
            return Err(AnchorError::SlotAlreadyInUse(slot));
        }
        self.write_cache_control(SlotLocation::Tools, Some(CacheControl::ephemeral(ttl)));
        self.slots[slot.idx()] = Some(SlotState { location: SlotLocation::Tools, ttl });
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
mod tests;
