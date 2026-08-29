//! The Server-Sent Events envelope, and the vocabulary of a broken frame.
//!
//! Everything in this module is about *one* frame: how to find its payload in a
//! line of an SSE body, what it means for that payload to contradict the
//! documented schema, and the handful of field accessors that report such a
//! contradiction instead of guessing. [`crate::content`] and [`crate::stream`]
//! both build on it, and neither has to invent its own error type.
//!
//! What is deliberately *not* here: any notion of an unrecognized event, block,
//! or delta. Those are not broken frames — see [`crate::stream::StreamEvent`].

use serde_json::Value;

use crate::usage::Usage;

/// Why one frame could not be decoded.
///
/// Every variant describes a frame that contradicts the documented schema. An
/// event type this crate does not model is deliberately absent from this list —
/// see [`crate::stream::StreamEvent::Unmodeled`].
#[derive(Debug)]
pub enum FrameError {
    /// The payload was not JSON at all.
    NotJson(serde_json::Error),
    /// The payload parsed, but was not a JSON object.
    NotAnObject,
    /// A field the frame cannot be interpreted without was absent.
    MissingField {
        /// Name of the absent field.
        field: &'static str,
    },
    /// A field was present with the wrong JSON type.
    WrongType {
        /// Name of the offending field.
        field: &'static str,
        /// What the schema says it should have been.
        expected: &'static str,
    },
    /// A `usage` object was present but would not deserialize into [`Usage`].
    ///
    /// Reported rather than dropped: the cache counts are this crate's reason to
    /// exist, so usage that silently read as absent would hide exactly the
    /// measurement the caller is here for.
    UndecodableUsage(serde_json::Error),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::NotJson(error) => write!(f, "streamed frame is not JSON: {error}"),
            FrameError::NotAnObject => write!(f, "streamed frame is not a JSON object"),
            FrameError::MissingField { field } => write!(f, "streamed frame has no `{field}`"),
            FrameError::WrongType { field, expected } => {
                write!(f, "streamed frame field `{field}` is not {expected}")
            }
            FrameError::UndecodableUsage(error) => write!(f, "streamed `usage` object is unusable: {error}"),
        }
    }
}

impl std::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FrameError::NotJson(error) | FrameError::UndecodableUsage(error) => Some(error),
            FrameError::NotAnObject | FrameError::MissingField { .. } | FrameError::WrongType { .. } => None,
        }
    }
}

/// The value of a `data:` field, for one line of an SSE body.
///
/// `None` for everything else a stream contains — the `event:` line that names
/// the type redundantly, comment lines, and the blank line that ends a frame —
/// so a caller can pass every line through and act on what comes back.
///
/// Per the SSE grammar one optional space after the colon is framing, not data,
/// and is removed. Anthropic sends each event as a single `data:` line; a
/// payload split across several would have to be rejoined with newlines by the
/// caller before decoding.
pub fn data_payload(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("data:")?;
    Some(rest.strip_prefix(' ').unwrap_or(rest))
}

/// A field that must be present.
pub(crate) fn require<'a>(object: &'a Value, field: &'static str) -> Result<&'a Value, FrameError> {
    object.get(field).ok_or(FrameError::MissingField { field })
}

/// A field that must be present and a string.
pub(crate) fn require_str<'a>(object: &'a Value, field: &'static str) -> Result<&'a str, FrameError> {
    require(object, field)?.as_str().ok_or(FrameError::WrongType { field, expected: "a string" })
}

/// A field that must be present and a `u32`-sized non-negative integer.
pub(crate) fn require_u32(object: &Value, field: &'static str) -> Result<u32, FrameError> {
    let wrong = || FrameError::WrongType { field, expected: "a non-negative integer" };
    let number = require(object, field)?.as_u64().ok_or_else(wrong)?;
    u32::try_from(number).map_err(|_| wrong())
}

/// A string field whose absence means the empty string.
///
/// Used where the API documents a field that opens empty and is filled by
/// deltas: `text` on a text block, `thinking` and `signature` on a thinking
/// block. Absent and `""` describe the same state, so they decode alike.
pub(crate) fn optional_string(object: &Value, field: &str) -> String {
    object.get(field).and_then(Value::as_str).unwrap_or_default().to_owned()
}

/// The `usage` object of a frame that may or may not carry one.
///
/// Absent or `null` is the all-zero [`Usage`], because the API sends different
/// subsets of it at different points of a stream and a missing counter means
/// "nothing of that kind". Present but unusable is an error: quietly reporting
/// no cost would hide the numbers this crate exists to report.
pub(crate) fn decode_usage(object: &Value) -> Result<Usage, FrameError> {
    match object.get("usage") {
        None | Some(Value::Null) => Ok(Usage::default()),
        Some(usage) => serde_json::from_value(usage.clone()).map_err(FrameError::UndecodableUsage),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn only_data_lines_carry_a_payload() {
        assert_eq!(data_payload("data: {\"type\":\"ping\"}"), Some("{\"type\":\"ping\"}"));
        assert_eq!(data_payload("data:{\"type\":\"ping\"}"), Some("{\"type\":\"ping\"}"), "the space is optional");
        assert_eq!(data_payload("data:  two spaces"), Some(" two spaces"), "only one space is framing");
        assert_eq!(data_payload("event: message_stop"), None);
        assert_eq!(data_payload(": keep-alive comment"), None);
        assert_eq!(data_payload(""), None);
    }

    #[test]
    fn required_fields_report_absence_and_wrong_types() {
        let frame = json!({"index": 3, "name": "get_weather", "negative": -1, "huge": 5_000_000_000u64});
        assert_eq!(require_str(&frame, "name").unwrap(), "get_weather");
        assert_eq!(require_u32(&frame, "index").unwrap(), 3);
        assert!(matches!(require(&frame, "absent"), Err(FrameError::MissingField { field: "absent" })));
        assert!(matches!(require_str(&frame, "index"), Err(FrameError::WrongType { field: "index", .. })));
        assert!(matches!(require_u32(&frame, "negative"), Err(FrameError::WrongType { .. })));
        assert!(matches!(require_u32(&frame, "huge"), Err(FrameError::WrongType { .. })), "wider than u32");
    }

    #[test]
    fn an_absent_string_field_is_the_empty_string() {
        let frame = json!({"text": "present", "wrong": 7});
        assert_eq!(optional_string(&frame, "text"), "present");
        assert_eq!(optional_string(&frame, "absent"), "");
        assert_eq!(optional_string(&frame, "wrong"), "", "a non-string reads as empty, not as an error");
    }

    #[test]
    fn usage_is_absent_zero_or_an_error() {
        assert_eq!(decode_usage(&json!({})).unwrap(), Usage::default());
        assert_eq!(decode_usage(&json!({"usage": null})).unwrap(), Usage::default());
        assert_eq!(decode_usage(&json!({"usage": {"output_tokens": 9}})).unwrap().output_tokens, 9);
        assert!(matches!(
            decode_usage(&json!({"usage": {"input_tokens": "lots"}})),
            Err(FrameError::UndecodableUsage(_))
        ));
    }

    /// The error type carries its cause where it has one, so a caller can log a
    /// chain rather than a single line.
    #[test]
    fn frame_errors_display_and_carry_their_cause() {
        let not_json = FrameError::NotJson(serde_json::from_str::<Value>("{oops").unwrap_err());
        assert!(not_json.to_string().starts_with("streamed frame is not JSON"));
        assert!(std::error::Error::source(&not_json).is_some());

        let missing = FrameError::MissingField { field: "index" };
        assert_eq!(missing.to_string(), "streamed frame has no `index`");
        assert!(std::error::Error::source(&missing).is_none());

        let wrong = FrameError::WrongType { field: "index", expected: "a non-negative integer" };
        assert_eq!(wrong.to_string(), "streamed frame field `index` is not a non-negative integer");
        assert_eq!(FrameError::NotAnObject.to_string(), "streamed frame is not a JSON object");
    }
}
