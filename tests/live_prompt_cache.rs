//! The prompt-cache prediction against the first-party cache.
//!
//! Ignored by default because it spends money and about ten minutes: run it with
//! `cargo test --test live_prompt_cache -- --ignored --nocapture` and a `.key`
//! in the crate root. Every scenario runs on its own thread under a fresh nonce,
//! so no entry from an earlier run can be read. Each trace is written to
//! `tests/captured/prompt_cache/` before it is judged, so a disagreement is kept
//! as evidence and `tests/prompt_cache_traces.rs` replays it offline. Set
//! `PROMPT_CACHE_SCENARIOS` to a comma-separated list to run only those, and
//! `PROMPT_CACHE_SEED` to repeat a recorded random scenario.

mod prompt_cache_lab;

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anthropic::prompt_cache::PrefixTokens;
use anthropic::{COUNT_TOKENS_PATH, MESSAGES_PATH};
use prompt_cache_lab::{Recorded, SCENARIOS, encode, replay, scenario};
use serde_json::Value;

fn read_key() -> Option<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".key");
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_owned()).filter(|s| !s.is_empty())
}

fn post(path: &str, body: &Value, key: &str) -> Value {
    let url = format!("{}{}", anthropic::API_BASE, path);
    for attempt in 0.. {
        let response = ureq::post(&url)
            .set(anthropic::HEADER_API_KEY, key)
            .set(anthropic::HEADER_VERSION, anthropic::VERSION)
            .set("content-type", "application/json")
            .send_string(&body.to_string());
        match response {
            Ok(response) => return serde_json::from_str(&response.into_string().expect("body")).expect("JSON"),
            Err(ureq::Error::Status(429 | 529, _)) if attempt < 5 => std::thread::sleep(Duration::from_secs(2)),
            Err(ureq::Error::Status(status, response)) => {
                panic!("{path} answered {status}: {}", response.into_string().unwrap_or_default())
            }
            Err(error) => panic!("{path}: {error}"),
        }
    }
    unreachable!()
}

fn run(name: &'static str, nonce: String, seed: u64, key: &str) -> Value {
    let start = Instant::now();
    let mut recorded = Vec::new();
    for step in scenario(name, &nonce, seed) {
        if let Some(wait) = step.at.checked_sub(start.elapsed()) {
            std::thread::sleep(wait);
        }
        let mut counts: Vec<u64> = step
            .script
            .prefix_counts()
            .iter()
            .map(|(body, frame)| post(COUNT_TOKENS_PATH, body, key)["input_tokens"].as_u64().expect("count") - frame)
            .collect();
        let total = counts.pop().expect("a total");
        let context = step.script.context();
        let request = serde_json::to_value(step.script.request(&context)).expect("serializable");
        let started_at = start.elapsed();
        let response = post(MESSAGES_PATH, &request, key);
        recorded.push(Recorded {
            started_at,
            tokens: PrefixTokens::new(counts, total),
            request,
            usage: response["usage"].clone(),
        });
    }
    encode(name, &nonce, seed, &recorded)
}

#[test]
#[ignore = "spends credentials and about ten minutes; run with --ignored"]
fn the_prediction_matches_the_live_cache() {
    let key = read_key().expect("a .key in the crate root");
    let clock = SystemTime::now().duration_since(UNIX_EPOCH).expect("after the epoch").as_nanos() as u64;
    let seed = std::env::var("PROMPT_CACHE_SEED").map_or(clock, |s| s.parse().expect("a u64 seed"));
    let traces: Vec<Value> = std::thread::scope(|scope| {
        let only = std::env::var("PROMPT_CACHE_SCENARIOS").ok();
        let handles: Vec<_> = SCENARIOS
            .iter()
            .filter(|name| only.as_ref().is_none_or(|only| only.split(',').any(|o| o == **name)))
            .map(|&name| {
                let nonce = format!("run-{clock:x}-{name}");
                let key = &key;
                scope.spawn(move || run(name, nonce, seed, key))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("scenario thread")).collect()
    });
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/captured/prompt_cache");
    std::fs::create_dir_all(&directory).expect("trace directory");
    let mut unexplained = Vec::new();
    for trace in &traces {
        let name = trace["scenario"].as_str().unwrap();
        let text = serde_json::to_string_pretty(trace).unwrap() + "\n";
        std::fs::write(directory.join(format!("{name}.json")), text).expect("write trace");
        let replay = replay(trace);
        for loss in &replay.losses {
            eprintln!("{loss}");
        }
        unexplained.extend(replay.unexplained);
    }
    for step in &unexplained {
        eprintln!("{step}\n");
    }
    assert!(unexplained.is_empty(), "{} of the recorded requests no loss explains", unexplained.len());
}
