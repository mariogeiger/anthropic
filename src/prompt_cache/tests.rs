use std::time::Duration;

use serde_json::json;

use super::*;
use crate::block::{ContentBlock, ImageSource};
use crate::context::{CacheSlot, Context, Opening, Tool};
use crate::request::{Model, Opus5_5, Opus5_5Effort, Opus5_5ThinkingDisplay, Sonnet5, Sonnet5Effort};
use crate::tool_choice::ToolChoice;

const T0: Duration = Duration::ZERO;

fn secs(s: u64) -> Duration {
    Duration::from_secs(s)
}

fn request(context: &Context) -> Request<'_> {
    Request::new(context, Model::opus_5_5(), 1).unwrap()
}

fn usage(input: u64, read: u64, five: u64, hour: u64) -> PromptUsage {
    PromptUsage {
        input_tokens: input,
        cache_read_input_tokens: read,
        cache_creation: CacheCreation { ephemeral_5m_input_tokens: five, ephemeral_1h_input_tokens: hour },
    }
}

fn tokens(at: &[u64], total: u64) -> PrefixTokens {
    PrefixTokens::new(at.to_vec(), total)
}

/// A system prompt cached in S0 and one user message cached in S1.
fn conversation() -> Context {
    let mut context = Context::new(Opening::cached_instruction("system", CacheSlot::S0, CacheTtl::FiveMinutes));
    context.push_user_text("first");
    context.roll_cache(CacheSlot::S1, CacheTtl::FiveMinutes).unwrap();
    context
}

#[test]
fn a_cold_request_writes_every_breakpoint_and_bills_the_write_once() {
    let context = conversation();
    let served = PromptCache::new().serve(&request(&context), T0, &tokens(&[600, 900], 904)).unwrap();
    assert_eq!(served.read, None);
    assert_eq!(
        served.written.iter().map(|w| w.position).collect::<Vec<_>>(),
        [Position::System(0), Position::Message { message: 0, block: 0 }]
    );
    assert_eq!(served.usage, usage(4, 0, 900, 0));
}

#[test]
fn a_repeated_request_reads_the_longest_entry() {
    let context = conversation();
    let mut cache = PromptCache::new();
    cache.serve(&request(&context), T0, &tokens(&[600, 900], 904)).unwrap();
    let served = cache.serve(&request(&context), secs(10), &tokens(&[600, 900], 904)).unwrap();
    assert_eq!(served.read.map(|r| r.position), Some(Position::Message { message: 0, block: 0 }));
    let stamped: Vec<_> = served.written.iter().map(|w| w.position).collect();
    assert_eq!(stamped, [Position::System(0), Position::Message { message: 0, block: 0 }]);
    assert_eq!(served.usage, usage(4, 900, 0, 0));
}

#[test]
fn a_prefix_below_the_model_minimum_is_not_written() {
    let context = conversation();
    let mut cache = PromptCache::new();
    let served = cache.serve(&request(&context), T0, &tokens(&[300, 600], 604)).unwrap();
    assert_eq!(
        served.written.iter().map(|w| w.position).collect::<Vec<_>>(),
        [Position::Message { message: 0, block: 0 }]
    );
    assert_eq!(served.usage, usage(4, 0, 600, 0));
    let served = cache.serve(&request(&context), secs(1), &tokens(&[300, 600], 604)).unwrap();
    assert_eq!(served.usage, usage(4, 600, 0, 0));
}

#[test]
fn nothing_is_written_when_the_last_breakpoint_is_below_the_minimum() {
    let context = conversation();
    let served = PromptCache::new().serve(&request(&context), T0, &tokens(&[200, 500], 504)).unwrap();
    assert!(served.written.is_empty());
    assert_eq!(served.usage, usage(504, 0, 0, 0));
}

#[test]
fn an_entry_expires_its_ttl_after_the_start_of_its_last_use() {
    let context = conversation();
    let sizes = tokens(&[600, 900], 904);
    let mut cache = PromptCache::new();
    cache.serve(&request(&context), T0, &sizes).unwrap();
    assert!(cache.serve(&request(&context), secs(299), &sizes).unwrap().read.is_some());
    assert!(cache.serve(&request(&context), secs(598), &sizes).unwrap().read.is_some());
    let served = cache.serve(&request(&context), secs(898), &sizes).unwrap();
    assert_eq!(served.read, None);
    assert_eq!(served.usage, usage(4, 0, 900, 0));
}

#[test]
fn reading_an_entry_keeps_alive_what_its_writer_built_it_on() {
    let context = conversation();
    let sizes = tokens(&[600, 900], 904);
    let mut unmarked = Context::new(Opening::instruction("system"));
    unmarked.push_user_text("first");
    unmarked.roll_cache(CacheSlot::S1, CacheTtl::FiveMinutes).unwrap();
    let mut other = Context::new(Opening::cached_instruction("system", CacheSlot::S0, CacheTtl::FiveMinutes));
    other.push_user_text("second");
    other.roll_cache(CacheSlot::S1, CacheTtl::FiveMinutes).unwrap();

    let mut cache = PromptCache::new();
    cache.serve(&request(&context), T0, &sizes).unwrap();
    cache.serve(&request(&unmarked), secs(200), &tokens(&[900], 904)).unwrap();
    let served = cache.serve(&request(&other), secs(400), &tokens(&[600, 910], 914)).unwrap();
    assert_eq!(served.read.map(|r| r.position), Some(Position::System(0)));
    assert_eq!(served.usage, usage(4, 600, 310, 0));
}

#[test]
fn an_entry_written_past_a_miss_keeps_nothing_before_it_alive() {
    let anchor = conversation();
    let mut far = Context::new(Opening::instruction("system"));
    far.push_user((0..30).map(|i| ContentBlock::text(format!("block {i}"))).collect());
    far.roll_cache(CacheSlot::S1, CacheTtl::FiveMinutes).unwrap();
    let mut cache = PromptCache::new();
    cache.serve(&request(&anchor), T0, &tokens(&[600, 900], 904)).unwrap();
    assert_eq!(cache.serve(&request(&far), secs(100), &tokens(&[1500], 1504)).unwrap().read, None);
    assert!(cache.serve(&request(&far), secs(200), &tokens(&[1500], 1504)).unwrap().read.is_some());
    let mut other = Context::new(Opening::cached_instruction("system", CacheSlot::S0, CacheTtl::FiveMinutes));
    other.push_user_text("second");
    assert_eq!(cache.serve(&request(&other), secs(400), &tokens(&[600], 610)).unwrap().read, None);
}

/// Breakpoints after `a` and after `c`, and optionally one more turn.
fn marked_twice(extended: bool) -> Context {
    let mut context = Context::new(Opening::instruction("system"));
    context.push_user_text("a");
    context.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    context.push_assistant_text("b");
    context.push_user_text("c");
    context.roll_cache(CacheSlot::S1, CacheTtl::FiveMinutes).unwrap();
    if extended {
        context.push_assistant_text("d");
    }
    context
}

fn marked_once() -> Context {
    let mut context = Context::new(Opening::instruction("system"));
    context.push_user_text("a");
    context.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    context
}

#[test]
fn breakpoints_before_the_entry_read_place_their_entries_free() {
    let mut cache = PromptCache::new();
    cache.serve(&request(&marked_twice(false)), T0, &tokens(&[400, 1000], 1004)).unwrap();
    let served = cache.serve(&request(&marked_twice(true)), secs(1), &tokens(&[700, 1000], 1102)).unwrap();
    assert_eq!(served.read.map(|r| r.position), Some(Position::Message { message: 2, block: 0 }));
    assert_eq!(served.usage, usage(102, 1000, 0, 0));
    let served = cache.serve(&request(&marked_once()), secs(2), &tokens(&[700], 704)).unwrap();
    assert_eq!(served.read.map(|r| r.position), Some(Position::Message { message: 0, block: 0 }));
    assert_eq!(served.usage, usage(4, 700, 0, 0));
}

#[test]
fn a_prefix_no_breakpoint_placed_is_not_found() {
    let mut cache = PromptCache::new();
    cache.serve(&request(&marked_twice(false)), T0, &tokens(&[400, 1000], 1004)).unwrap();
    let served = cache.serve(&request(&marked_once()), secs(1), &tokens(&[700], 704)).unwrap();
    assert_eq!(served.read, None);
}

fn with_blocks(count: usize) -> Context {
    let mut context = Context::new(Opening::instruction("system"));
    context.push_user(vec![ContentBlock::text("anchor")]);
    context.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    context.push_user((0..count).map(|i| ContentBlock::text(format!("block {i}"))).collect());
    context.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    context
}

#[test]
fn the_lookback_window_holds_twenty_two_positions_counting_the_breakpoint() {
    let mut anchor = Context::new(Opening::instruction("system"));
    anchor.push_user(vec![ContentBlock::text("anchor")]);
    anchor.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    for (added, reached) in [(21, true), (22, false)] {
        let mut cache = PromptCache::new();
        cache.serve(&request(&anchor), T0, &tokens(&[600], 604)).unwrap();
        let served = cache.serve(&request(&with_blocks(added)), secs(1), &tokens(&[900], 904)).unwrap();
        assert_eq!(served.read.is_some(), reached, "{added} blocks after the entry");
    }
}

#[test]
fn a_run_of_tool_calls_counts_as_one_position() {
    let mut anchor = Context::new(Opening::instruction("system")).with_tools(vec![Tool::new("t", json!({}))]);
    anchor.push_user_text("anchor");
    anchor.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    let grown = |extra: bool| {
        let mut grown = Context::new(Opening::instruction("system")).with_tools(vec![Tool::new("t", json!({}))]);
        grown.push_user_text("anchor");
        grown.push_assistant((0..30).map(|i| ContentBlock::tool_use(format!("id{i}"), "t", json!({}))).collect());
        grown.push_user(
            (0..30)
                .map(|i| {
                    ContentBlock::tool_result(format!("id{i}"), crate::block::ToolResultContent::Text("ok".into()))
                })
                .collect(),
        );
        grown.push_assistant((0..if extra { 20 } else { 19 }).map(|i| ContentBlock::text(format!("t{i}"))).collect());
        grown.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
        grown
    };
    for (extra, reached) in [(false, true), (true, false)] {
        let mut cache = PromptCache::new();
        cache.serve(&request(&anchor), T0, &tokens(&[600], 604)).unwrap();
        let served = cache.serve(&request(&grown(extra)), secs(1), &tokens(&[900], 902)).unwrap();
        assert_eq!(served.read.is_some(), reached);
    }
}

#[test]
fn an_earlier_breakpoint_opens_its_own_window() {
    let mut anchor = Context::new(Opening::instruction("system"));
    anchor.push_user_text("anchor");
    anchor.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    let mut grown = Context::new(Opening::instruction("system"));
    grown.push_user_text("anchor");
    grown.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    grown.push_assistant((0..40).map(|i| ContentBlock::text(format!("t{i}"))).collect());
    grown.roll_cache(CacheSlot::S1, CacheTtl::FiveMinutes).unwrap();
    let mut cache = PromptCache::new();
    cache.serve(&request(&anchor), T0, &tokens(&[600], 604)).unwrap();
    let served = cache.serve(&request(&grown), secs(1), &tokens(&[600, 1200], 1202)).unwrap();
    assert_eq!(served.read.map(|r| r.tokens), Some(600));
    assert_eq!(served.usage, usage(2, 600, 600, 0));
}

#[test]
fn one_hour_tokens_run_to_the_last_one_hour_breakpoint() {
    let mut context = Context::new(Opening::cached_instruction("system", CacheSlot::S0, CacheTtl::OneHour));
    context.push_user_text("first");
    context.roll_cache(CacheSlot::S1, CacheTtl::FiveMinutes).unwrap();
    let served = PromptCache::new().serve(&request(&context), T0, &tokens(&[600, 900], 904)).unwrap();
    assert_eq!(served.usage, usage(4, 0, 300, 600));
    let mut cache = PromptCache::new();
    cache.serve(&request(&context), T0, &tokens(&[600, 900], 904)).unwrap();
    let served = cache.serve(&request(&context), secs(1000), &tokens(&[600, 900], 904)).unwrap();
    assert_eq!(served.read.map(|r| (r.position, r.ttl)), Some((Position::System(0), CacheTtl::OneHour)));
    assert_eq!(served.usage, usage(4, 600, 300, 0));
}

#[test]
fn only_forcing_a_tool_call_moves_the_messages() {
    let context = Context::new(Opening::cached_instruction("system", CacheSlot::S0, CacheTtl::FiveMinutes))
        .with_tools(vec![Tool::new("t", json!({}))]);
    let mut context = context;
    context.push_user_text("first");
    context.roll_cache(CacheSlot::S1, CacheTtl::FiveMinutes).unwrap();
    let sonnet = || Request::new(&context, Model::Sonnet5(Model::sonnet_5()), 1).unwrap();
    let sizes = tokens(&[1100, 1400], 1404);
    let forced = tokens(&[1100, 1510], 1514);
    let read = |cache: &mut PromptCache, request: Request<'_>, at: u64, sizes: &PrefixTokens| {
        cache.serve(&request, secs(at), sizes).unwrap().read.map(|r| r.tokens)
    };

    let mut cache = PromptCache::new();
    cache.serve(&sonnet(), T0, &sizes).unwrap();
    for (at, choice) in [(1, ToolChoice::none()), (2, ToolChoice::auto().without_parallel_use())] {
        assert_eq!(read(&mut cache, sonnet().with_tool_choice(choice).unwrap(), at, &sizes), Some(1400));
    }
    assert_eq!(read(&mut cache, sonnet().with_tool_choice(ToolChoice::any()).unwrap(), 3, &forced), Some(1100));
    assert_eq!(read(&mut cache, sonnet().with_tool_choice(ToolChoice::tool("t")).unwrap(), 4, &forced), Some(1510));
    let serial = ToolChoice::tool("t").without_parallel_use();
    assert_eq!(read(&mut cache, sonnet().with_tool_choice(serial).unwrap(), 5, &forced), Some(1100));
}

#[test]
fn an_image_moves_nothing_before_it() {
    let context = conversation();
    let mut pictured = conversation();
    pictured.push_assistant_text("ok");
    pictured.push_user(vec![ContentBlock::image(ImageSource::url("https://example.com/a.png"))]);
    let mut cache = PromptCache::new();
    cache.serve(&request(&context), T0, &tokens(&[600, 900], 904)).unwrap();
    let served = cache.serve(&request(&pictured), secs(1), &tokens(&[600, 900], 1500)).unwrap();
    assert_eq!(served.read.map(|r| r.position), Some(Position::Message { message: 0, block: 0 }));
}

#[test]
fn effort_renders_where_the_model_renders_it() {
    let context = Context::new(Opening::instruction("system"))
        .with_tools_cached(CacheSlot::S0, vec![Tool::new("t", json!({}))], CacheTtl::FiveMinutes)
        .unwrap();
    let mut context = context;
    context.push_user_text("first");
    context.roll_cache(CacheSlot::S1, CacheTtl::FiveMinutes).unwrap();
    let sizes = tokens(&[1100, 2100], 2104);

    let mut cache = PromptCache::new();
    let low = Model::Opus5_5(Opus5_5 { effort: Opus5_5Effort::Low, ..Model::opus_5_5() });
    let high = Model::Opus5_5(Opus5_5 { effort: Opus5_5Effort::High, ..Model::opus_5_5() });
    cache.serve(&Request::new(&context, low, 1).unwrap(), T0, &sizes).unwrap();
    let served = cache.serve(&Request::new(&context, high, 1).unwrap(), secs(1), &sizes).unwrap();
    assert_eq!(served.read.map(|r| r.position), Some(Position::Message { message: 0, block: 0 }));

    let mut cache = PromptCache::new();
    let low = Model::Sonnet5(Sonnet5 { effort: Sonnet5Effort::Low, ..Model::sonnet_5() });
    let high = Model::Sonnet5(Sonnet5 { effort: Sonnet5Effort::High, ..Model::sonnet_5() });
    cache.serve(&Request::new(&context, low, 1).unwrap(), T0, &sizes).unwrap();
    let served = cache.serve(&Request::new(&context, high, 1).unwrap(), secs(1), &sizes).unwrap();
    assert_eq!(served.read, None);
}

#[test]
fn thinking_display_is_not_rendered() {
    let context = conversation();
    let sizes = tokens(&[600, 900], 904);
    let summarized = Model::Opus5_5(Opus5_5 { display: Opus5_5ThinkingDisplay::Summarized, ..Model::opus_5_5() });
    let omitted = Model::Opus5_5(Opus5_5 { display: Opus5_5ThinkingDisplay::Omitted, ..Model::opus_5_5() });
    let mut cache = PromptCache::new();
    cache.serve(&Request::new(&context, summarized, 1).unwrap(), T0, &sizes).unwrap();
    let served = cache.serve(&Request::new(&context, omitted, 1).unwrap(), secs(1), &sizes).unwrap();
    assert_eq!(served.usage, usage(4, 900, 0, 0));
}

#[test]
fn another_model_shares_no_entry() {
    let context = conversation();
    let sizes = tokens(&[1100, 1400], 1404);
    let mut cache = PromptCache::new();
    cache.serve(&request(&context), T0, &sizes).unwrap();
    let served = cache.serve(&Request::new(&context, Model::sonnet_5(), 1).unwrap(), secs(1), &sizes).unwrap();
    assert_eq!(served.read, None);
}

#[test]
fn a_deferred_tool_is_not_in_the_prompt() {
    let tools = vec![Tool::new("eager", json!({}))];
    let mut plain = Context::new(Opening::instruction("system")).with_tools(tools.clone());
    plain.push_user_text("first");
    plain.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    let mut deferring = Context::new(Opening::instruction("system"))
        .with_tools(vec![tools[0].clone(), Tool::new("later", json!({})).deferred()]);
    deferring.push_user_text("first");
    deferring.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    let mut cache = PromptCache::new();
    cache.serve(&request(&plain), T0, &tokens(&[700], 704)).unwrap();
    assert!(cache.serve(&request(&deferring), secs(1), &tokens(&[700], 704)).unwrap().read.is_some());
}

#[test]
fn turn_scoped_text_leaves_the_prompt_after_the_next_user_message() {
    let mut context = Context::new(Opening::instruction("system"));
    context.push_user_text("first");
    context.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    context.push_turn_scoped_text("only now").unwrap();
    context.push_assistant_text("reply");
    context.roll_cache(CacheSlot::S1, CacheTtl::FiveMinutes).unwrap();
    let mut cache = PromptCache::new();
    cache.serve(&request(&context), T0, &tokens(&[600, 700], 702)).unwrap();
    let mut later = Context::new(Opening::instruction("system"));
    later.push_user_text("first");
    later.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    later.push_turn_scoped_text("only now").unwrap();
    later.push_assistant_text("reply");
    later.roll_cache(CacheSlot::S1, CacheTtl::FiveMinutes).unwrap();
    later.push_user_text("second");
    let served = cache.serve(&request(&later), secs(1), &tokens(&[600, 690], 800)).unwrap();
    assert_eq!(served.read.map(|r| r.position), Some(Position::Message { message: 0, block: 0 }));
}

#[test]
fn inconsistent_inputs_are_refused_without_changing_the_cache() {
    let context = conversation();
    let mut cache = PromptCache::new();
    assert_eq!(
        cache.serve(&request(&context), T0, &tokens(&[600], 904)),
        Err(ServeError::BreakpointCount { request: 2, supplied: 1 })
    );
    assert_eq!(cache.serve(&request(&context), T0, &tokens(&[900, 600], 904)), Err(ServeError::Decreasing));
    assert_eq!(cache.serve(&request(&context), T0, &tokens(&[600, 900], 800)), Err(ServeError::Decreasing));
    cache.serve(&request(&context), secs(5), &tokens(&[600, 900], 904)).unwrap();
    assert_eq!(
        cache.serve(&request(&context), secs(4), &tokens(&[600, 900], 904)),
        Err(ServeError::StartedBeforePrevious { previous: secs(5), started_at: secs(4) })
    );
    assert!(cache.serve(&request(&context), secs(6), &tokens(&[600, 900], 904)).unwrap().read.is_some());
}

#[test]
fn a_shortfall_is_attributed_to_the_entries_reached_before_the_one_read() {
    let context = conversation();
    let sizes = tokens(&[600, 900], 904);
    let mut cache = PromptCache::new();
    cache.serve(&request(&context), T0, &sizes).unwrap();
    let served = cache.explain(&request(&context), secs(1), &sizes, &usage(4, 600, 300, 0)).unwrap();
    assert_eq!(served.read.map(|r| r.position), Some(Position::System(0)));
    assert_eq!(
        served.lost.iter().map(|l| l.position).collect::<Vec<_>>(),
        [Position::Message { message: 0, block: 0 }]
    );
    let served = cache.explain(&request(&context), secs(2), &sizes, &usage(4, 0, 900, 0)).unwrap();
    assert_eq!(served.lost.len(), 2);
    assert_eq!(cache.serve(&request(&context), secs(3), &sizes).unwrap().usage, usage(4, 900, 0, 0));
}

#[test]
fn a_usage_no_loss_explains_is_refused_without_changing_the_cache() {
    let context = conversation();
    let sizes = tokens(&[600, 900], 904);
    let mut cache = PromptCache::new();
    let more = cache.explain(&request(&context), T0, &sizes, &usage(4, 600, 300, 0));
    assert_eq!(more, Err(ServeError::Unexplained { predicted: usage(4, 0, 900, 0), observed: usage(4, 600, 300, 0) }));
    cache.serve(&request(&context), T0, &sizes).unwrap();
    let billed = cache.explain(&request(&context), secs(1), &sizes, &usage(4, 900, 1, 0));
    assert!(matches!(billed, Err(ServeError::Unexplained { .. })));
    assert_eq!(cache.serve(&request(&context), secs(2), &sizes).unwrap().usage, usage(4, 900, 0, 0));
}
