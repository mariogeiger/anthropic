//! Recorded first-party cache traces, replayed against the prediction offline.
//!
//! `tests/live_prompt_cache.rs` records them; this checks that the prediction
//! still reproduces every `usage` the server billed, token for token, once the
//! entries the server lost are accounted for. Each hand-written scenario is the
//! evidence for one rule, and a loss could stand in for a rule that failed, so
//! its trace must hold no loss; the random walk checks the whole under loss.

mod prompt_cache_lab;

#[test]
fn every_recorded_trace_replays_exactly_up_to_loss() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/captured/prompt_cache");
    let mut paths: Vec<_> = std::fs::read_dir(&directory)
        .expect("trace directory")
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no recorded traces in {}", directory.display());
    let mut failures = Vec::new();
    for path in &paths {
        let trace: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("read")).expect("JSON");
        let replay = prompt_cache_lab::replay(&trace);
        failures.extend(replay.unexplained);
        if trace["scenario"] != "random" {
            failures.extend(replay.losses);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
