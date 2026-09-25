//! What the server's prompt cache will do with a request, predicted exactly.
//!
//! The cache is invisible except through `usage`, and its rules are easy to
//! state loosely and hard to state exactly: a breakpoint that silently caches
//! nothing, an entry one position outside the lookback window, a toggle that
//! moves a whole level. [`PromptCache`] replays those rules on a sequence of
//! requests and says, for each, which entry it reads, which it writes, and the
//! token counts `usage` will report. It has no clock and no network: the caller
//! says when each request started and how many tokens each breakpoint's prefix
//! holds, so the same inputs always give the same prediction.
//!
//! # The rules
//!
//! * The prompt is `tools`, then `system`, then `messages`, and a cache entry is
//!   the exact prefix ending at one position — one tool, one system block, or
//!   one message block. Changing anything at or before a position changes its
//!   prefix. Some settings render as a whole level: enabling citations moves the
//!   system prompt and everything after it, and forcing a tool call moves the
//!   messages. Where the thinking configuration and effort render is per model.
//!   Whether an image is present renders nowhere, and which tool is forced and
//!   how thinking is displayed render after every cacheable position, so none of
//!   them moves anything. Deferred tools are not in the prompt.
//! * An entry exists only where a breakpoint placed it: a prefix another entry
//!   merely contains is not found. Every breakpoint whose prefix holds at least
//!   [`ModelId::min_cacheable_prefix_tokens`] places its entry; below that it is
//!   silently skipped.
//! * A read looks back from each breakpoint, last first, over at most
//!   [`LOOKBACK_POSITIONS`] positions counting the breakpoint itself, where a run
//!   of consecutive `tool_use` blocks counts once and so does a run of
//!   `tool_result` blocks. The first live entry found is read, which is the
//!   longest one reachable.
//! * An entry lives for its TTL from the *start* of the last request that wrote
//!   it or read through it. Reading an entry refreshes it and what it was built
//!   on: the entries its writer read or wrote before it, and theirs in turn. An
//!   entry that only shares its prefix, written by a request that missed it, is
//!   not refreshed, and writing never refreshes.
//! * `cache_read_input_tokens` is the entry's size; `cache_creation_input_tokens`
//!   is the last breakpoint's size less that, billed once however many
//!   breakpoints it spans, at the 1-hour rate up to the last 1-hour breakpoint;
//!   `input_tokens` is the rest. Breakpoints at or before the entry read place
//!   their entries free, living as long as the entry read.
//!
//! Where this departs from the documentation it follows first-party
//! measurements of 2026-09-25: the lookback window, the free entries at earlier
//! breakpoints, what a read refreshes, and that neither `tool_choice`
//! `auto` against `none` nor the presence of an image moves anything, on every
//! model measured.
//!
//! Not modelled: automatic top-level caching, two requests in flight at once
//! (an entry is readable only once its writing response has begun), thinking
//! blocks the server strips or drops from replayed history, the seconds past a
//! TTL in which the server may still read an entry (first-party: read 310 s
//! after a 5-minute write, gone by 315 s), and entries lost before their TTL.
//! Loss is real and not a function of the requests: replaying one recorded
//! sequence of bodies first-party read a different entry at some steps each
//! time, and never more than predicted. [`PromptCache::explain`] takes the
//! usage the server reported and accounts for the loss it reveals, so the
//! prediction stays exact for everything else.
//!
//! [`ModelId::min_cacheable_prefix_tokens`]: crate::model::ModelId::min_cacheable_prefix_tokens

mod render;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Duration;

use crate::CacheTtl;
use crate::request::Request;
use crate::usage::{CacheCreation, Usage};

/// How many positions a breakpoint's lookback checks, itself included.
///
/// Measured first-party on 2026-09-25 rather than taken from the documentation,
/// which says 20: an entry 21 positions before a breakpoint is read and one 22
/// positions before is not, whether the positions are blocks of one message,
/// one block per message, or cross from the system prompt into the messages.
pub const LOOKBACK_POSITIONS: usize = 22;

/// Where a cache entry ends, as indices into the request body.
///
/// Indices rather than content because a prediction is read beside the request
/// it was made for, and an index is what locates a block there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Position {
    /// `tools[i]`.
    Tool(usize),
    /// `system[i]`; a plain-string system prompt is one block.
    System(usize),
    /// `messages[message].content[block]`; plain-string content is one block.
    Message {
        /// Index into `messages`.
        message: usize,
        /// Index into that message's `content`.
        block: usize,
    },
}

/// How many tokens the prompt holds up to each breakpoint, and in total.
///
/// Supplied rather than computed because tokenization is the server's: the crate
/// cannot count tokens, and the minimum-size rule and every `usage` figure depend
/// on the counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixTokens {
    at_breakpoints: Vec<u64>,
    total: u64,
}

impl PrefixTokens {
    /// `at_breakpoints` lists one count per breakpoint, in prompt order; `total`
    /// counts the whole prompt, which is `input_tokens + cache_read_input_tokens
    /// + cache_creation_input_tokens`.
    pub fn new(at_breakpoints: Vec<u64>, total: u64) -> Self {
        Self { at_breakpoints, total }
    }

    /// The count at each breakpoint, in prompt order.
    pub fn at_breakpoints(&self) -> &[u64] {
        &self.at_breakpoints
    }

    /// The count for the whole prompt.
    pub fn total(&self) -> u64 {
        self.total
    }
}

/// The part of [`Usage`] the cache determines.
///
/// Its own type because output and server-tool counts are the model's doing, not
/// the cache's, and an exact prediction must not claim them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PromptUsage {
    /// Tokens after the last cached prefix.
    pub input_tokens: u64,
    /// Tokens read from the entry hit.
    pub cache_read_input_tokens: u64,
    /// Tokens written, split by TTL.
    pub cache_creation: CacheCreation,
}

impl From<&Usage> for PromptUsage {
    fn from(usage: &Usage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            cache_creation: usage.cache_creation,
        }
    }
}

/// An entry a request reads or writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Touched {
    /// Where its prefix ends.
    pub position: Position,
    /// Its size in tokens.
    pub tokens: u64,
    /// How long it now lives from the request's start.
    pub ttl: CacheTtl,
}

/// What one request did to the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Served {
    /// The entry read, if any.
    pub read: Option<Touched>,
    /// The entry at every breakpoint whose prefix is large enough, in prompt
    /// order. Those after the entry read are written and billed; those at or
    /// before it hold tokens already read, and are stamped at no cost.
    pub written: Vec<Touched>,
    /// The entries the lookback reached before the one read, which the server
    /// must have lost; empty unless [`PromptCache::explain`] was told less was
    /// read than predicted.
    pub lost: Vec<Touched>,
    /// What `usage` reports for the prompt.
    pub usage: PromptUsage,
}

/// Why [`PromptCache::serve`] or [`PromptCache::explain`] refused a request,
/// leaving the cache unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServeError {
    /// The counts name a different number of breakpoints than the request has.
    BreakpointCount {
        /// Breakpoints in the request.
        request: usize,
        /// Counts supplied.
        supplied: usize,
    },
    /// A breakpoint's count is below an earlier one's, or the total below the
    /// last; a longer prefix cannot hold fewer tokens.
    Decreasing,
    /// The request started before the previous one. Expiry is replayed in start
    /// order, so an earlier start would read entries already let go.
    StartedBeforePrevious {
        /// The previous request's start.
        previous: Duration,
        /// This request's start.
        started_at: Duration,
    },
    /// The observed usage is not what the rules give for any loss of entries.
    Unexplained {
        /// The nearest the rules come: the usage with the loss that matches the
        /// observed read, or with none when no loss does.
        predicted: PromptUsage,
        /// What the server reported.
        observed: PromptUsage,
    },
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BreakpointCount { request, supplied } => {
                write!(f, "the request has {request} breakpoints but {supplied} token counts were supplied")
            }
            Self::Decreasing => write!(f, "prefix token counts must not decrease along the prompt"),
            Self::StartedBeforePrevious { previous, started_at } => {
                write!(f, "a request started at {started_at:?}, before the previous one at {previous:?}")
            }
            Self::Unexplained { predicted, observed } => {
                write!(f, "no loss of entries explains the usage {observed:?}; the rules give {predicted:?}")
            }
        }
    }
}

impl std::error::Error for ServeError {}

/// A stretch of cached computation, extending its parent's.
///
/// Storage, not keys, is what expires: a key is readable while its storage and
/// every ancestor's live, and reading through a key keeps that whole line
/// alive. Measured first-party on 2026-09-25: reading a long entry kept alive a
/// shorter one written by the same request or read to build it, but not one a
/// separate miss had merely written the same prefix for.
#[derive(Debug, Clone, Copy)]
struct Storage {
    parent: Option<u64>,
    /// Byte length of the prefix this stretch computes up to.
    end: usize,
    ttl: CacheTtl,
    expires_at: Duration,
}

#[derive(Debug, Clone, Copy)]
struct Entry {
    tokens: u64,
    storage: u64,
}

/// The server's cache entries, as a sequence of requests leaves them.
///
/// Keyed by the exact rendered prefix, so an entry is found only by a request
/// whose prompt agrees with the writer's up to that position — which is the
/// whole of the server's invalidation rule.
#[derive(Debug, Default)]
pub struct PromptCache {
    entries: HashMap<Vec<u8>, Entry>,
    storage: BTreeMap<u64, Storage>,
    next_storage: u64,
    last_start: Option<Duration>,
}

fn lifetime(ttl: CacheTtl) -> Duration {
    match ttl {
        CacheTtl::FiveMinutes => Duration::from_secs(5 * 60),
        CacheTtl::OneHour => Duration::from_secs(60 * 60),
    }
}

/// A request as the cache meets it.
struct Arrival<'a> {
    rendering: render::Rendering,
    /// Cut indices of its breakpoints, in prompt order.
    marks: Vec<usize>,
    tokens: &'a PrefixTokens,
    /// The model's smallest cacheable prefix.
    minimum: u64,
    now: Duration,
}

/// A breakpoint's entry as a request will leave it.
enum Placement {
    /// Stamped free on the line of the entry read.
    Stamped(u64),
    /// Written as new storage extending the previous placement's.
    Written(CacheTtl),
}

impl PromptCache {
    /// A cache no request has touched.
    pub fn new() -> Self {
        Self::default()
    }

    /// Serve `request`, started at `started_at` on the caller's clock, whose
    /// prefixes hold `tokens`, and return what it reads, writes and will report
    /// if the server has lost nothing.
    pub fn serve(
        &mut self,
        request: &Request<'_>,
        started_at: Duration,
        tokens: &PrefixTokens,
    ) -> Result<Served, ServeError> {
        self.serve_reading(request, started_at, tokens, None)
    }

    /// Serve `request` as [`serve`](Self::serve) does, given the usage the
    /// server `observed` for it, and attribute any shortfall in what it read to
    /// entries the server lost.
    ///
    /// Exists because the server loses entries before their TTL, and not as a
    /// function of the requests, so no prediction can match every response; what
    /// can be exact is everything else. The loss attributed is the least that
    /// explains the observation: the live entries the lookback would have
    /// reached before the one actually read. Anything else the rules cannot
    /// produce is [`ServeError::Unexplained`].
    pub fn explain(
        &mut self,
        request: &Request<'_>,
        started_at: Duration,
        tokens: &PrefixTokens,
        observed: &PromptUsage,
    ) -> Result<Served, ServeError> {
        self.serve_reading(request, started_at, tokens, Some(observed))
    }

    fn serve_reading(
        &mut self,
        request: &Request<'_>,
        started_at: Duration,
        tokens: &PrefixTokens,
        observed: Option<&PromptUsage>,
    ) -> Result<Served, ServeError> {
        if let Some(previous) = self.last_start
            && started_at < previous
        {
            return Err(ServeError::StartedBeforePrevious { previous, started_at });
        }
        let rendering = render::render(request);
        let marks: Vec<usize> = (0..rendering.cuts.len()).filter(|&i| rendering.cuts[i].mark.is_some()).collect();
        if marks.len() != tokens.at_breakpoints.len() {
            return Err(ServeError::BreakpointCount { request: marks.len(), supplied: tokens.at_breakpoints.len() });
        }
        let ascending = tokens.at_breakpoints.windows(2).all(|w| w[0] <= w[1])
            && tokens.at_breakpoints.last().is_none_or(|&last| last <= tokens.total);
        if !ascending {
            return Err(ServeError::Decreasing);
        }

        let minimum = u64::from(request.model().id().min_cacheable_prefix_tokens());
        let arrival = Arrival { rendering, marks, tokens, minimum, now: started_at };
        let rendering = &arrival.rendering;
        let reachable = self.reachable(&arrival);
        let entry_at = |cut: usize| self.entries[rendering.prefix(&rendering.cuts[cut])];
        let kept = match observed {
            None => 0,
            Some(observed) => match observed.cache_read_input_tokens {
                0 => reachable.len(),
                read => match reachable.iter().position(|&cut| entry_at(cut).tokens == read) {
                    Some(kept) => kept,
                    None => {
                        let predicted = self.plan(&arrival, reachable.first().copied(), &[]).2;
                        return Err(ServeError::Unexplained { predicted, observed: *observed });
                    }
                },
            },
        };
        let (lost, hit) = (&reachable[..kept], reachable.get(kept).copied());
        let (placements, served_read, usage) = self.plan(&arrival, hit, lost);
        if let Some(observed) = observed
            && usage != *observed
        {
            return Err(ServeError::Unexplained { predicted: usage, observed: *observed });
        }

        self.last_start = Some(started_at);
        let lost: Vec<Touched> = lost
            .iter()
            .map(|&cut| {
                let entry = entry_at(cut);
                Touched {
                    position: rendering.cuts[cut].position,
                    tokens: entry.tokens,
                    ttl: self.storage[&entry.storage].ttl,
                }
            })
            .collect();
        for &cut in &reachable[..kept] {
            self.entries.remove(rendering.prefix(&rendering.cuts[cut]));
        }
        self.expire(started_at);
        let read_storage = hit.map(|cut| self.entries[rendering.prefix(&rendering.cuts[cut])].storage);
        let mut line = read_storage;
        while let Some(id) = line {
            let storage = self.storage.get_mut(&id).expect("live storage has live ancestors");
            storage.expires_at = storage.expires_at.max(started_at + lifetime(storage.ttl));
            line = storage.parent;
        }
        let mut parent = read_storage;
        let mut written = Vec::new();
        for (mark, size, placement) in placements {
            let cut = &rendering.cuts[mark];
            let storage = match placement {
                Placement::Stamped(storage) => storage,
                Placement::Written(ttl) => {
                    let id = self.next_storage;
                    self.next_storage += 1;
                    let end = cut.end;
                    self.storage.insert(id, Storage { parent, end, ttl, expires_at: started_at + lifetime(ttl) });
                    parent = Some(id);
                    id
                }
            };
            self.entries.insert(rendering.prefix(cut).to_vec(), Entry { tokens: size, storage });
            written.push(Touched { position: cut.position, tokens: size, ttl: self.storage[&storage].ttl });
        }
        Ok(Served { read: served_read, written, lost, usage })
    }

    /// Where each large-enough breakpoint's entry goes, what is read, and the
    /// usage that follows, if `hit` is read and `lost` are gone — without
    /// changing anything, so a refused request leaves the cache as it was.
    fn plan(
        &self,
        arrival: &Arrival<'_>,
        hit: Option<usize>,
        lost: &[usize],
    ) -> (Vec<(usize, u64, Placement)>, Option<Touched>, PromptUsage) {
        let Arrival { rendering, marks, tokens, minimum, now } = arrival;
        let hit_entry = hit.map(|cut| self.entries[rendering.prefix(&rendering.cuts[cut])]);
        let read = hit.zip(hit_entry).map(|(cut, entry)| Touched {
            position: rendering.cuts[cut].position,
            tokens: entry.tokens,
            ttl: self.storage[&entry.storage].ttl,
        });
        let mut placements = Vec::new();
        let mut billed: Vec<(u64, CacheTtl)> = Vec::new();
        for (&mark, &size) in marks.iter().zip(&tokens.at_breakpoints) {
            if size < *minimum {
                continue;
            }
            let cut = &rendering.cuts[mark];
            let placement = match hit_entry {
                Some(entry) if hit.is_some_and(|hit| mark <= hit) => {
                    let prefix = rendering.prefix(cut);
                    let kept =
                        self.entries.get(prefix).filter(|e| !lost.contains(&mark) && self.alive(e.storage, *now));
                    Placement::Stamped(kept.map_or_else(|| self.covering(entry.storage, prefix.len()), |e| e.storage))
                }
                _ => {
                    let ttl = cut.mark.expect("a mark");
                    billed.push((size, ttl));
                    Placement::Written(ttl)
                }
            };
            placements.push((mark, size, placement));
        }
        let read_tokens = read.map_or(0, |r| r.tokens);
        let written_through = |ttl: Option<CacheTtl>| {
            billed.iter().rev().find(|w| ttl.is_none_or(|t| w.1 == t)).map_or(0, |w| w.0).saturating_sub(read_tokens)
        };
        let creation = written_through(None);
        let one_hour = written_through(Some(CacheTtl::OneHour));
        let usage = PromptUsage {
            input_tokens: tokens.total - read_tokens - creation,
            cache_read_input_tokens: read_tokens,
            cache_creation: CacheCreation {
                ephemeral_5m_input_tokens: creation - one_hour,
                ephemeral_1h_input_tokens: one_hour,
            },
        };
        (placements, read, usage)
    }

    /// Whether the line under `storage` is still live at `now`.
    fn alive(&self, storage: u64, now: Duration) -> bool {
        let mut line = Some(storage);
        while let Some(id) = line {
            let Some(stretch) = self.storage.get(&id) else { return false };
            if stretch.expires_at <= now {
                return false;
            }
            line = stretch.parent;
        }
        true
    }

    /// Every live entry the lookback reaches, as cut indices in the order it
    /// reaches them: from each breakpoint, last first, nearest first.
    fn reachable(&self, arrival: &Arrival<'_>) -> Vec<usize> {
        let Arrival { rendering, marks, now, .. } = arrival;
        let mut reachable = Vec::new();
        for &mark in marks.iter().rev() {
            let first_unit = rendering.cuts[mark].unit;
            let window = (0..=mark).rev().take_while(|&cut| first_unit - rendering.cuts[cut].unit < LOOKBACK_POSITIONS);
            for cut in window {
                let live = self
                    .entries
                    .get(rendering.prefix(&rendering.cuts[cut]))
                    .is_some_and(|entry| self.alive(entry.storage, *now));
                if live && !reachable.contains(&cut) {
                    reachable.push(cut);
                }
            }
        }
        reachable
    }

    /// Let go of all storage whose line has expired by `now`, and the keys into it.
    fn expire(&mut self, now: Duration) {
        let mut dead = BTreeSet::new();
        for (&id, storage) in &self.storage {
            if storage.expires_at <= now || storage.parent.is_some_and(|parent| dead.contains(&parent)) {
                dead.insert(id);
            }
        }
        self.storage.retain(|id, _| !dead.contains(id));
        self.entries.retain(|_, entry| !dead.contains(&entry.storage));
    }

    /// The stretch on `line` that computes the byte `end`: the shortest one
    /// reaching it.
    fn covering(&self, line: u64, end: usize) -> u64 {
        let mut covering = line;
        let mut next = self.storage[&line].parent;
        while let Some(id) = next {
            let storage = &self.storage[&id];
            if storage.end < end {
                break;
            }
            covering = id;
            next = storage.parent;
        }
        covering
    }
}

#[cfg(test)]
#[path = "prompt_cache/tests.rs"]
mod tests;
