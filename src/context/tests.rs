use super::*;
use crate::request::{Model, Request};

fn req(ctx: &Context) -> serde_json::Value {
    serde_json::to_value(Request::new(ctx, Model::opus_4_8(), 1024).unwrap()).unwrap()
}

#[test]
fn empty_request_serializes() {
    let v = req(&Context::new(Opening::None));
    assert_eq!(v["model"], "claude-opus-4-8");
    assert_eq!(v["max_tokens"], 1024);
    assert!(v["messages"].is_array());
}

#[test]
fn roll_cache_on_empty_errors() {
    let mut ctx = Context::new(Opening::None);
    assert_eq!(ctx.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap_err(), RollCacheError::NoBlocksToCache,);
}

#[test]
fn roll_cache_tail_and_move() {
    let mut ctx = Context::new(Opening::None);
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
    let mut ctx = Context::new(Opening::cached_instruction("sys", CacheSlot::S0, CacheTtl::OneHour));
    ctx.push_user_text("hi");
    assert_eq!(
        ctx.roll_cache(CacheSlot::S0, CacheTtl::OneHour).unwrap_err(),
        RollCacheError::SlotOccupiedByAnchor(CacheSlot::S0),
    );
}

#[test]
fn ttl_ordering_enforced() {
    let mut ctx = Context::new(Opening::None);
    ctx.push_user_text("one");
    ctx.roll_cache(CacheSlot::S0, CacheTtl::FiveMinutes).unwrap();
    ctx.push_user_text("two");
    // 1h after 5m rejected.
    assert_eq!(ctx.roll_cache(CacheSlot::S1, CacheTtl::OneHour).unwrap_err(), RollCacheError::TtlOrderingViolation,);

    // 1h system anchor then 5m tail is fine.
    let mut ctx = Context::new(Opening::cached_instruction("sys", CacheSlot::S0, CacheTtl::OneHour));
    ctx.push_user_text("hi");
    ctx.roll_cache(CacheSlot::S3, CacheTtl::FiveMinutes).unwrap();
    assert_eq!(ctx.breakpoint_count(), 2);
}

#[test]
fn conflicting_ttl_at_same_position_rejected() {
    let mut ctx = Context::new(Opening::None);
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
    let mut ctx = Context::new(Opening::None);
    ctx.push_user_text("hi");
    ctx.roll_cache(CacheSlot::S3, CacheTtl::FiveMinutes).unwrap();
    ctx.clear_cache(CacheSlot::S3).unwrap();
    assert!(req(&ctx)["messages"][0]["content"][0].get("cache_control").is_none());
    assert_eq!(ctx.breakpoint_count(), 0);
}

/// The three openings the API admits, each reaching its own wire shape.
///
/// Three and not two: `system` is documented as optional, so a conversation
/// with no opening at all is a real state rather than an undecided one.
#[test]
fn every_opening_reaches_its_documented_wire_shape() {
    // No opening: the field is absent, not empty. An empty string would be a
    // different prefix and a different cache key.
    let none = req(&Context::new(Opening::None));
    assert!(none.get("system").is_none(), "no opening emits no system field");
    assert_eq!(Context::new(Opening::None).opening(), None);

    // An instruction: the bare string, which is the uncached shape.
    let instruction = req(&Context::new(Opening::instruction("you are helpful")));
    assert_eq!(instruction["system"], "you are helpful");
    assert_eq!(Context::new(Opening::instruction("you are helpful")).opening(), Some("you are helpful"));

    // A cached instruction: the one-element block array, the only shape that
    // can carry a breakpoint, and the slot is accounted for.
    let cached = Context::new(Opening::cached_instruction("sys", CacheSlot::S0, CacheTtl::OneHour));
    assert_eq!(cached.breakpoint_count(), 1, "the opening's anchor occupies its slot");
    let v = req(&cached);
    assert_eq!(v["system"][0]["type"], "text");
    assert_eq!(v["system"][0]["text"], "sys");
    assert_eq!(v["system"][0]["cache_control"]["ttl"], "1h");
}

#[test]
fn system_wire_shape_switches_on_cache() {
    // Plain string when no cache_control.
    assert_eq!(req(&Context::new(Opening::instruction("you are helpful")))["system"], "you are helpful");

    // One-element block array when cached.
    let v = req(&Context::new(Opening::cached_instruction("sys", CacheSlot::S0, CacheTtl::OneHour)));
    assert_eq!(v["system"][0]["type"], "text");
    assert_eq!(v["system"][0]["text"], "sys");
    assert_eq!(v["system"][0]["cache_control"]["ttl"], "1h");
}

#[test]
fn roles_reach_the_wire_as_their_strings() {
    let mut ctx = Context::new(Opening::None);
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
    let mut ctx = Context::new(Opening::instruction("you are helpful"));
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
    let mut ctx = Context::new(Opening::None).with_tools(tools);
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
    let mut empty = Context::new(Opening::None);
    empty.push_user_text("hi");
    assert_eq!(empty.push_system(vec![]), Err(SystemMessageError::Empty));

    let mut first = Context::new(Opening::None);
    assert_eq!(first.push_system_text("Answer in French.").err(), Some(SystemMessageError::First));

    let mut after_assistant = Context::new(Opening::None);
    after_assistant.push_user_text("hi");
    after_assistant.push_assistant_text("hello");
    assert_eq!(after_assistant.push_system_text("x").err(), Some(SystemMessageError::AfterAssistant));

    // Two in a row is accepted, which the live API confirms — so a chain of
    // them is well-placed as long as the chain itself is.
    let mut adjacent = Context::new(Opening::None);
    adjacent.push_user_text("hi");
    adjacent.push_system_text("first").unwrap();
    adjacent.push_system_text("second").unwrap();
    assert_eq!(adjacent.message_count(), 3);
    assert_eq!(adjacent.misplaced_system_message(), None, "the chain ends the array");
    adjacent.push_user_text("and now a question");
    assert_eq!(
        adjacent.misplaced_system_message(),
        Some(2),
        "the chain's last message is the one whose successor is wrong"
    );
}

/// The other half of the rules: what must *follow* a system message is a
/// property of the finished history, so appending a user turn after one turns
/// a legal conversation into an illegal request.
#[test]
fn a_system_message_followed_by_a_user_turn_is_caught_at_the_request() {
    let mut ctx = Context::new(Opening::None);
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

/// The documented placement rules, one assertion each, so a regression names
/// the rule it broke. Source: the "Limitations" section of
/// <https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages>.
#[test]
fn every_documented_placement_rule_is_enforced() {
    // "A system message cannot be the first entry in messages."
    assert_eq!(Context::new(Opening::None).push_system_text("x").err(), Some(SystemMessageError::First));

    // "must immediately follow a user turn" — and a user turn carrying
    // tool_result blocks counts, which is the agentic-loop position.
    let mut after_tool_result = Context::new(Opening::None);
    after_tool_result.push_user_text("run the tests");
    after_tool_result.push_assistant(vec![ContentBlock::tool_use("toolu_01", "run_tests", serde_json::json!({}))]);
    after_tool_result.push_tool_result("toolu_01", ToolResultContent::Text("12 passed".into()));
    assert!(after_tool_result.push_system_text("the user also asked for a changelog entry").is_ok());
    assert_eq!(after_tool_result.misplaced_system_message(), None, "ending the array is legal");

    // "It cannot sit between a tool_use block and its tool_result": the only
    // entry between them is the assistant turn holding the `tool_use`, and a
    // system message may not follow an assistant turn, so that position is
    // exactly `AfterAssistant`.
    let mut between = Context::new(Opening::None);
    between.push_user_text("run the tests");
    between.push_assistant(vec![ContentBlock::tool_use("toolu_01", "run_tests", serde_json::json!({}))]);
    assert_eq!(between.push_system_text("x").err(), Some(SystemMessageError::AfterAssistant));

    // "must precede an assistant turn or end the array", checked at the
    // request because a later append can break it.
    let mut then_user = Context::new(Opening::None);
    then_user.push_user_text("hi");
    then_user.push_system_text("be terse").unwrap();
    then_user.push_user_text("again");
    assert_eq!(then_user.misplaced_system_message(), Some(1));

    // "Consecutive system messages are accepted and treated as a single
    // system section, which follows the same placement rule as a whole."
    let mut chain = Context::new(Opening::None);
    chain.push_user_text("hi");
    chain.push_system_text("first").unwrap();
    chain.push_system_text("second").unwrap();
    assert_eq!(chain.misplaced_system_message(), None);
    chain.push_assistant_text("ok");
    assert_eq!(chain.misplaced_system_message(), None, "the section precedes an assistant turn");
}

/// Availability is per model, so the same conversation is a request on Opus 5
/// and a refusal on Sonnet 5. The refusal names the model and the index.
#[test]
fn a_system_message_is_refused_on_a_model_that_does_not_accept_one() {
    use crate::request::{Request, RequestError};

    let mut ctx = Context::new(Opening::instruction("you are helpful"));
    ctx.push_user_text("name a fruit");
    ctx.push_system_text("Answer in French.").unwrap();

    assert!(Request::new(&ctx, Model::opus_5(), 1024).is_ok(), "documented as available");
    assert!(Request::new(&ctx, Model::opus_4_8(), 1024).is_ok(), "documented as available");
    assert!(Request::new(&ctx, Model::fable_5(), 1024).is_ok(), "documented as available");

    for model in [Model::from(Model::sonnet_5()), Model::sonnet_4_6().into(), Model::haiku_4_5().into()] {
        let id = model.id();
        assert_eq!(
            Request::new(&ctx, model, 1024).err(),
            Some(RequestError::MidConversationSystemMessageUnsupported { model: id, at: 1 }),
            "{} is not on the documented availability list",
            id.api_id(),
        );
    }

    // Without a system message the same models are fine, so the refusal is
    // about the pairing and not about the model.
    let mut plain = Context::new(Opening::None);
    plain.push_user_text("name a fruit");
    assert!(Request::new(&plain, Model::sonnet_5(), 1024).is_ok());
}

/// A cache breakpoint lands on a system message's inner block, which is where
/// the API accepts one: `cache_control on mid_conv_system is not supported;
/// set it on an inner content block instead`.
#[test]
fn a_breakpoint_lands_on_a_system_messages_inner_block() {
    let mut ctx = Context::new(Opening::None);
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
    let v = req(&Context::new(Opening::None).with_tools(vec![plain]));
    for field in ["defer_loading", "strict", "input_examples"] {
        assert!(v["tools"][0].get(field).is_none(), "{field} should be absent when unset");
    }

    let configured = Tool::new("two", serde_json::json!({"type": "object"}))
        .strict()
        .input_examples(vec![serde_json::json!({"city": "Paris"})]);
    let v = req(&Context::new(Opening::None).with_tools(vec![configured]));
    assert_eq!(v["tools"][0]["strict"], true);
    assert_eq!(v["tools"][0]["input_examples"][0]["city"], "Paris");
}

/// A deferred tool costs no prompt tokens until a tool search finds it, but the
/// API refuses a request in which every tool is deferred — a relation across
/// the list, so it is checked where the request is built.
#[test]
fn deferring_every_tool_is_refused_at_the_request() {
    let deferred = || Tool::new("t", serde_json::json!({"type": "object"})).deferred();
    let ctx = Context::new(Opening::None).with_tools(vec![deferred()]);
    assert_eq!(
        crate::request::Request::new(&ctx, Model::opus_4_8(), 16).err(),
        Some(crate::request::RequestError::EveryToolDeferred { tools: 1 }),
    );

    // One undeferred tool is enough, and only the deferred one says so.
    let ctx = Context::new(Opening::None)
        .with_tools(vec![Tool::new("eager", serde_json::json!({"type": "object"})), deferred()]);
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
    let v = req(&Context::new(Opening::None).with_tools_cached(CacheSlot::S1, tools, CacheTtl::OneHour).unwrap());
    assert!(v["tools"][0].get("cache_control").is_none());
    assert_eq!(v["tools"][1]["cache_control"]["ttl"], "1h");
}
