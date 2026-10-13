//! First-party check that the API continues an assistant turn left open at the
//! end of the conversation exactly where [`ModelId::accepts_prefill`] says it
//! does.
//!
//! The crate refuses such a conversation on the other models, so the body is
//! built from a conversation ending with a user turn and the assistant turn is
//! appended to the raw JSON. It consumes credentials only when a non-empty `.key`
//! exists in the crate root, and caps output at one token. First exercised on
//! 2026-10-13.

use anthropic::MESSAGES_PATH;
use anthropic::context::{Context, Opening};
use anthropic::request::{Model, Request};
use serde_json::{Value, json};

fn read_key() -> Option<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".key");
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_owned()).filter(|s| !s.is_empty())
}

fn status_of(request: &Request<'_>, key: &str) -> u16 {
    let mut body = serde_json::to_value(request).expect("serialize body");
    body["messages"].as_array_mut().expect("messages").push(json!({
        "role": "assistant",
        "content": [{"type": "text", "text": "The colour is"}],
    }));
    let url = format!("{}{}", anthropic::API_BASE, MESSAGES_PATH);
    let mut call = ureq::post(&url)
        .set(anthropic::HEADER_API_KEY, key)
        .set(anthropic::HEADER_VERSION, anthropic::VERSION)
        .set("content-type", "application/json");
    let beta = request.required_beta_features().map(|f| f.as_str()).collect::<Vec<_>>().join(",");
    if !beta.is_empty() {
        call = call.set(anthropic::HEADER_BETA, &beta);
    }
    match call.send_string(&body.to_string()) {
        Ok(response) => response.status(),
        Err(ureq::Error::Status(status, response)) => {
            let text = response.into_string().unwrap_or_default();
            let message = serde_json::from_str::<Value>(&text).unwrap_or(Value::Null)["error"]["message"].clone();
            eprintln!("{status} {message}");
            status
        }
        Err(ureq::Error::Transport(error)) => panic!("transport error: {error}"),
    }
}

#[test]
fn live_prefill_is_accepted_exactly_where_the_model_says() {
    let Some(key) = read_key() else {
        eprintln!("[skip] no .key file in crate root");
        return;
    };
    let models: Vec<Model> = vec![
        Model::fable_5_1().into(),
        Model::fable_5().into(),
        Model::opus_5_5().into(),
        Model::opus_5().into(),
        Model::opus_4_8().into(),
        Model::haiku_5_5().into(),
        Model::sonnet_5_5().into(),
        Model::sonnet_5().into(),
        Model::sonnet_4_6().into(),
        Model::haiku_4_5().into(),
    ];
    for model in models {
        let id = model.id();
        let mut context = Context::new(Opening::None);
        context.push_user_text("Name a colour.");
        let request = Request::new(&context, model, 1).expect("request");
        let status = status_of(&request, &key);
        assert_eq!(status == 200, id.accepts_prefill(), "{} answered {status}", id.api_id());
    }
}
