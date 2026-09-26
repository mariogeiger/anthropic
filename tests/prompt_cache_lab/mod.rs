//! Scenarios that confront `anthropic::prompt_cache` with the live cache, and
//! the replay that checks a recorded run against the prediction.
//!
//! A scenario is rebuilt from its name, nonce and seed alone, so a recorded
//! trace needs to hold only what the server said: when each request started,
//! how many tokens its prefixes held, and the `usage` it billed.

#![allow(dead_code)]

use std::time::Duration;

use anthropic::CacheTtl;
use anthropic::block::{ContentBlock, ImageSource, ToolResultContent};
use anthropic::context::{CacheSlot, Context, Opening, Tool};
use anthropic::document::DocumentSource;
use anthropic::prompt_cache::{CacheKeys, PrefixTokens, PromptCache, PromptUsage};
use anthropic::request::{Model, Opus5_5, Opus5_5Effort, Opus5_5ThinkingDisplay, Request};
use anthropic::tool_choice::ToolChoice;
use anthropic::usage::Usage;
use anthropic::{ImageMediaType, Role};
use serde_json::{Value, json};

const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAKUlEQVR4nA3HMQEAAAzCMIRVGGdFIXDLlyQSGxcTBIvjU6mt62cyOzcPp2MTQTYdST8AAAAASUVORK5CYII=";

const VOCAB: [&str; 32] = [
    "river", "stone", "garden", "window", "lantern", "harbor", "meadow", "copper", "violet", "orchard", "summit",
    "canyon", "willow", "thunder", "compass", "anchor", "marble", "ember", "falcon", "glacier", "island", "jasmine",
    "kettle", "ledger", "mosaic", "nectar", "oyster", "pepper", "quartz", "saddle", "timber", "velvet",
];

/// `n` words of deterministic filler, distinct per `tag`.
pub fn words(tag: &str, n: usize) -> String {
    let offset: usize = tag.bytes().map(usize::from).sum();
    let body: Vec<&str> = (0..n).map(|i| VOCAB[(i * 7 + i / 5 + offset) % VOCAB.len()]).collect();
    format!("{tag}: {}", body.join(" "))
}

/// One turn of a scripted conversation.
#[derive(Clone)]
pub struct Turn {
    pub role: Role,
    pub blocks: Vec<ContentBlock>,
    pub mark: Option<(CacheSlot, CacheTtl)>,
}

pub fn user(text: String) -> Turn {
    Turn { role: Role::User, blocks: vec![ContentBlock::text(text)], mark: None }
}

pub fn assistant(text: String) -> Turn {
    Turn { role: Role::Assistant, blocks: vec![ContentBlock::text(text)], mark: None }
}

impl Turn {
    pub fn marked(mut self, slot: CacheSlot, ttl: CacheTtl) -> Self {
        self.mark = Some((slot, ttl));
        self
    }
}

/// A request, as data a scenario can copy and edit.
#[derive(Clone)]
pub struct Script {
    pub tools: bool,
    pub system: String,
    pub system_mark: Option<CacheTtl>,
    pub turns: Vec<Turn>,
    pub tool_choice: Option<ToolChoice>,
    pub effort: Opus5_5Effort,
}

impl Script {
    pub fn new(system: String, system_mark: Option<CacheTtl>) -> Self {
        Self { tools: false, system, system_mark, turns: Vec::new(), tool_choice: None, effort: Opus5_5Effort::Low }
    }

    pub fn with(mut self, turn: Turn) -> Self {
        self.turns.push(turn);
        self
    }

    pub fn context(&self) -> Context {
        let opening = match self.system_mark {
            Some(ttl) => Opening::cached_instruction(self.system.clone(), CacheSlot::S0, ttl),
            None => Opening::instruction(self.system.clone()),
        };
        let mut context = Context::new(opening);
        if self.tools {
            let schema = json!({"type": "object", "properties": {"key": {"type": "string"}}});
            context = context.with_tools(vec![Tool::new("lookup", schema).description("Look a key up.")]);
        }
        for turn in &self.turns {
            match turn.role {
                Role::User => context.push_user(turn.blocks.clone()),
                _ => context.push_assistant(turn.blocks.clone()),
            }
            if let Some((slot, ttl)) = turn.mark {
                context.roll_cache(slot, ttl).expect("scripted breakpoint");
            }
        }
        context
    }

    pub fn request<'a>(&self, context: &'a Context) -> Request<'a> {
        let model = Opus5_5 { effort: self.effort, display: Opus5_5ThinkingDisplay::Omitted };
        let request = Request::new(context, Model::Opus5_5(model), 1).expect("valid request");
        match &self.tool_choice {
            Some(choice) => request.with_tool_choice(choice.clone()).expect("accepted tool choice"),
            None => request,
        }
    }

    /// Where each breakpoint's prefix ends, in prompt order, as a count request
    /// that stops there and the tokens that count adds after the prefix.
    ///
    /// `count_tokens` bills a whole prompt, whose rendering closes the last turn
    /// and opens the model's reply. Measured first-party on Claude Opus 5.5 on
    /// 2026-09-25: that frame is 4 tokens after a user turn and 2 after an
    /// assistant one, a lone user message `x` adds 7, and the tool-use preamble,
    /// rendered between the system prompt and the messages, adds 70.
    pub fn prefix_counts(&self) -> Vec<(Value, u64)> {
        let context = self.context();
        let body = serde_json::to_value(self.request(&context)).expect("serializable");
        let count = |messages: Value| {
            let mut count = json!({"model": body["model"], "messages": messages});
            for field in ["system", "tools", "tool_choice"] {
                if !body[field].is_null() {
                    count[field] = body[field].clone();
                }
            }
            count
        };
        let preamble = if self.tools { 70 } else { 0 };
        let mut prefixes = Vec::new();
        if self.system_mark.is_some() {
            prefixes.push((count(json!([{"role": "user", "content": "x"}])), 7 + preamble));
        }
        for (m, turn) in self.turns.iter().enumerate() {
            if turn.mark.is_some() {
                let frame = if turn.role == Role::User { 4 } else { 2 };
                prefixes.push((count(Value::Array(body["messages"].as_array().unwrap()[..=m].to_vec())), frame));
            }
        }
        prefixes.push((count(body["messages"].clone()), 0));
        prefixes
    }
}

/// A request and when, from the scenario's start, it is sent.
pub struct Step {
    pub at: Duration,
    pub script: Script,
}

fn at(seconds: u64, script: Script) -> Step {
    Step { at: Duration::from_secs(seconds), script }
}

const FIVE: CacheTtl = CacheTtl::FiveMinutes;
const HOUR: CacheTtl = CacheTtl::OneHour;

pub const SCENARIOS: [&str; 16] = [
    "reuse",
    "minimum",
    "lookback_text",
    "lookback_tools",
    "ttl_refresh",
    "ttl_expiry",
    "refresh_earlier",
    "refresh_apart",
    "one_hour",
    "write_once",
    "earlier_marks",
    "tool_choice",
    "images",
    "citations",
    "effort",
    "random",
];

/// The requests of scenario `name`, made unique to this run by `nonce`.
pub fn scenario(name: &str, nonce: &str, seed: u64) -> Vec<Step> {
    let system = |ttl| Script::new(format!("{nonce} {}", words("system", 600)), ttl);
    let base = system(Some(FIVE)).with(user(words("question", 150)).marked(CacheSlot::S1, FIVE));
    let grown = |script: Script| {
        script.with(assistant(words("answer", 80))).with(user(words("follow-up", 120)).marked(CacheSlot::S2, FIVE))
    };
    let many = |count: usize| {
        (0..count).map(|i| ContentBlock::text(format!("Item {i}: {}", words("item", 6)))).collect::<Vec<_>>()
    };
    match name {
        "reuse" => {
            let longer = system(Some(FIVE))
                .with(user(words("question", 150)))
                .with(assistant(words("answer", 80)))
                .with(user(words("follow-up", 120)).marked(CacheSlot::S1, FIVE));
            vec![at(0, base.clone()), at(5, base), at(10, longer)]
        }
        "minimum" => {
            let short = Script::new(format!("{nonce} Be brief."), None)
                .with(user(words("question", 150)).marked(CacheSlot::S1, FIVE));
            let long = Script::new(format!("{nonce} Be brief."), None)
                .with(user(words("question", 150)))
                .with(assistant(words("answer", 60)))
                .with(user(words("follow-up", 200)).marked(CacheSlot::S1, FIVE));
            vec![at(0, short.clone()), at(5, short), at(10, long.clone()), at(15, long)]
        }
        "lookback_text" => {
            let anchor = system(None).with(user(words("question", 150)).marked(CacheSlot::S1, FIVE));
            let reach = |count: usize| {
                system(None).with(user(words("question", 150))).with(assistant(words("answer", 40))).with(Turn {
                    role: Role::User,
                    blocks: many(count),
                    mark: Some((CacheSlot::S1, FIVE)),
                })
            };
            vec![at(0, anchor), at(5, reach(21)), at(10, reach(20))]
        }
        "lookback_tools" => {
            let tooled = |script: Script| Script { tools: true, ..script };
            let anchor = tooled(system(None).with(user(words("question", 150)).marked(CacheSlot::S1, FIVE)));
            let reach = |count: usize| {
                let calls = (0..5)
                    .map(|i| ContentBlock::tool_use(format!("toolu_{i:02}"), "lookup", json!({"key": format!("k{i}")})))
                    .collect();
                let mut results: Vec<ContentBlock> = (0..5)
                    .map(|i| {
                        ContentBlock::tool_result(format!("toolu_{i:02}"), ToolResultContent::Text(format!("v{i}")))
                    })
                    .collect();
                results.extend(many(count));
                tooled(system(None).with(user(words("question", 150))))
                    .with(Turn { role: Role::Assistant, blocks: calls, mark: None })
                    .with(Turn { role: Role::User, blocks: results, mark: Some((CacheSlot::S1, FIVE)) })
            };
            vec![at(0, anchor), at(5, reach(20)), at(10, reach(19))]
        }
        "ttl_refresh" => vec![at(0, base.clone()), at(280, base.clone()), at(570, base)],
        "ttl_expiry" => vec![at(0, base.clone()), at(330, base)],
        "refresh_earlier" => {
            let unmarked = system(None).with(user(words("question", 150)).marked(CacheSlot::S1, FIVE));
            let other = system(Some(FIVE)).with(user(words("other", 150)).marked(CacheSlot::S1, FIVE));
            vec![at(0, base.clone()), at(200, unmarked), at(400, other)]
        }
        "refresh_apart" => {
            let far = system(None).with(Turn { role: Role::User, blocks: many(30), mark: Some((CacheSlot::S1, FIVE)) });
            let other = system(Some(FIVE)).with(user(words("other", 150)).marked(CacheSlot::S1, FIVE));
            vec![at(0, base.clone()), at(100, far.clone()), at(200, far), at(400, other)]
        }
        "one_hour" => {
            let script = system(Some(HOUR)).with(user(words("question", 150)).marked(CacheSlot::S1, FIVE));
            vec![at(0, script.clone()), at(330, script.clone()), at(335, script)]
        }
        "write_once" => vec![at(0, grown(base.clone())), at(5, grown(base))],
        "earlier_marks" => {
            let tail = |first: Option<(CacheSlot, CacheTtl)>| {
                system(None)
                    .with(Turn { mark: first, ..user(words("question", 250)) })
                    .with(assistant(words("answer", 80)))
                    .with(user(words("follow-up", 150)).marked(CacheSlot::S1, FIVE))
            };
            let head = system(None).with(user(words("question", 250)).marked(CacheSlot::S2, FIVE));
            vec![at(0, tail(None)), at(5, tail(Some((CacheSlot::S2, FIVE)))), at(10, head)]
        }
        "tool_choice" => {
            let script = Script { tools: true, ..base };
            let none = Script { tool_choice: Some(ToolChoice::none()), ..script.clone() };
            let serial = Script { tool_choice: Some(ToolChoice::auto().without_parallel_use()), ..script.clone() };
            vec![at(0, script.clone()), at(5, none), at(10, serial)]
        }
        "images" => {
            let pictured = base.clone().with(assistant(words("answer", 80))).with(Turn {
                role: Role::User,
                blocks: vec![
                    ContentBlock::image(ImageSource::base64(ImageMediaType::Png, PNG)),
                    ContentBlock::text(words("caption", 40)),
                ],
                mark: Some((CacheSlot::S2, FIVE)),
            });
            vec![at(0, base.clone()), at(5, pictured), at(10, base)]
        }
        "citations" => {
            let cited = base.clone().with(assistant(words("answer", 80))).with(Turn {
                role: Role::User,
                blocks: vec![
                    ContentBlock::document_cited(DocumentSource::text(words("document", 120))),
                    ContentBlock::text(words("request", 30)),
                ],
                mark: Some((CacheSlot::S2, FIVE)),
            });
            vec![at(0, base.clone()), at(5, cited), at(10, base)]
        }
        "effort" => {
            let high = Script { effort: Opus5_5Effort::High, ..base.clone() };
            vec![at(0, base), at(5, high)]
        }
        "random" => random(nonce, seed),
        other => panic!("unknown scenario {other}"),
    }
}

/// A seeded walk through the moves a real conversation makes.
fn random(nonce: &str, seed: u64) -> Vec<Step> {
    let mut state = seed | 1;
    let mut next = move |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let mut turns: Vec<(String, String)> = vec![(words("question", 150), String::new())];
    let mut script = Script { tools: true, ..Script::new(format!("{nonce} {}", words("system", 600)), Some(FIVE)) };
    let mut steps = Vec::new();
    let mut earlier: Option<usize> = None;
    for i in 0..10 {
        match next(6) {
            0 | 1 => {
                let n = turns.len();
                turns.last_mut().unwrap().1 = words(&format!("answer {n}"), 40 + next(80) as usize);
                turns.push((words(&format!("question {n}"), 40 + next(160) as usize), String::new()));
            }
            2 => earlier = if turns.len() > 1 { Some(next(turns.len() as u64 - 1) as usize) } else { None },
            3 => script.tool_choice = if script.tool_choice.is_some() { None } else { Some(ToolChoice::none()) },
            4 if turns.len() > 1 => {
                turns.pop();
                turns.last_mut().unwrap().1.clear();
            }
            _ => {}
        }
        script.turns.clear();
        for (t, (question, answer)) in turns.iter().enumerate() {
            let last = t + 1 == turns.len();
            let mark = if last {
                Some((CacheSlot::S1, FIVE))
            } else if earlier == Some(t) {
                Some((CacheSlot::S2, FIVE))
            } else {
                None
            };
            script.turns.push(Turn { mark, ..user(question.clone()) });
            if !last {
                script.turns.push(assistant(answer.clone()));
            }
        }
        steps.push(at(5 * i, script.clone()));
    }
    steps
}

/// One recorded request.
pub struct Recorded {
    pub started_at: Duration,
    pub tokens: PrefixTokens,
    pub request: Value,
    pub usage: Value,
}

pub fn encode(name: &str, nonce: &str, seed: u64, steps: &[Recorded]) -> Value {
    json!({
        "scenario": name,
        "nonce": nonce,
        "seed": seed,
        "steps": steps.iter().map(|s| json!({
            "started_at_us": s.started_at.as_micros() as u64,
            "tokens": {"at_breakpoints": s.tokens.at_breakpoints(), "total": s.tokens.total()},
            "request": s.request,
            "usage": s.usage,
        })).collect::<Vec<_>>(),
    })
}

/// What replaying one trace found.
#[derive(Default)]
pub struct Replay {
    /// Steps whose usage no loss of entries explains.
    pub unexplained: Vec<String>,
    /// Steps that read less than predicted, and the entries that makes lost.
    pub losses: Vec<String>,
}

/// Replay a trace against the prediction, attributing any shortfall to loss.
pub fn replay(trace: &Value) -> Replay {
    let name = trace["scenario"].as_str().expect("scenario");
    let nonce = trace["nonce"].as_str().expect("nonce");
    let seed = trace["seed"].as_u64().expect("seed");
    let steps = scenario(name, nonce, seed);
    let recorded = trace["steps"].as_array().expect("steps");
    assert_eq!(steps.len(), recorded.len(), "{name}: step count");
    let mut cache = PromptCache::new();
    let mut replay = Replay::default();
    for (i, (step, record)) in steps.iter().zip(recorded).enumerate() {
        let context = step.script.context();
        let request = step.script.request(&context);
        let body = serde_json::to_value(&request).expect("serializable");
        assert!(body == record["request"], "{name} step {i}: the scenario no longer builds the recorded request");
        let started_at = Duration::from_micros(record["started_at_us"].as_u64().expect("start"));
        let at_breakpoints: Vec<u64> =
            record["tokens"]["at_breakpoints"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap()).collect();
        let tokens = PrefixTokens::new(at_breakpoints, record["tokens"]["total"].as_u64().unwrap());
        let usage: Usage = serde_json::from_value(record["usage"].clone()).expect("usage decodes");
        let step = format!(
            "{name} (seed {seed}) step {i} at {started_at:?}, tokens {:?} of {}",
            tokens.at_breakpoints(),
            tokens.total()
        );
        match cache.explain(&CacheKeys::of(&request), started_at, &tokens, &PromptUsage::from(&usage)) {
            Ok(served) if served.lost.is_empty() => {}
            Ok(served) => {
                let lost: Vec<_> = served.lost.iter().map(|l| (l.position, l.tokens)).collect();
                replay.losses.push(format!("{step}: lost {lost:?}"));
            }
            Err(error) => replay.unexplained.push(format!("{step}\n  {error}")),
        }
    }
    replay
}
