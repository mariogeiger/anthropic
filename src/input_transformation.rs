//! Changes the API made to replayed input before the model read it.
//!
//! Preserved thinking is bound to the model and conversation that produced it.
//! With the binding-controls beta enabled, blocks the serving model cannot read
//! or whose prefix changed are reported here instead of disappearing silently.
//! The field is absent without the beta and present as an empty array when the
//! check ran but dropped nothing, so callers receive an `Option<Vec<_>>` rather
//! than those two facts being collapsed.

use serde_json::Value;

use crate::frame::{FrameError, require_str};
use crate::values::ThinkingDropReason;

/// One item in a response's top-level `input_transformations` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputTransformation {
    /// A signed thinking or redacted-thinking block was left out before
    /// inference. The caller's original `messages` array is never mutated.
    ThinkingDropped {
        /// Dot-separated position in the submitted body, such as
        /// `messages.1.content.0`.
        path: String,
        /// The reason, where it is one this crate knows.
        reason: Option<ThinkingDropReason>,
        /// The reason string exactly as sent, so a new check stays legible.
        raw_reason: String,
    },
    /// A transformation kind added after this crate version.
    Unmodeled {
        /// Its `type` string.
        kind: String,
        /// The complete object, so no new fields are lost.
        value: Value,
    },
}

impl InputTransformation {
    /// The transformation's wire `type`.
    pub fn kind(&self) -> &str {
        match self {
            Self::ThinkingDropped { .. } => "thinking_dropped",
            Self::Unmodeled { kind, .. } => kind,
        }
    }

    /// The submitted position affected, where this is a known thinking drop.
    pub fn path(&self) -> Option<&str> {
        match self {
            Self::ThinkingDropped { path, .. } => Some(path),
            Self::Unmodeled { .. } => None,
        }
    }

    /// The known reason for a thinking drop.
    pub fn reason(&self) -> Option<ThinkingDropReason> {
        match self {
            Self::ThinkingDropped { reason, .. } => *reason,
            Self::Unmodeled { .. } => None,
        }
    }

    /// The exact reason string, including a value added after this crate version.
    pub fn raw_reason(&self) -> Option<&str> {
        match self {
            Self::ThinkingDropped { raw_reason, .. } => Some(raw_reason),
            Self::Unmodeled { .. } => None,
        }
    }

    pub(crate) fn decode(value: &Value) -> Result<Self, FrameError> {
        if !value.is_object() {
            return Err(FrameError::WrongType { field: "input_transformations[]", expected: "an object" });
        }
        let kind = require_str(value, "type")?;
        if kind == "thinking_dropped" {
            let raw_reason = require_str(value, "reason")?.to_owned();
            Ok(Self::ThinkingDropped {
                path: require_str(value, "path")?.to_owned(),
                reason: ThinkingDropReason::from_str(&raw_reason),
                raw_reason,
            })
        } else {
            Ok(Self::Unmodeled { kind: kind.to_owned(), value: value.clone() })
        }
    }
}

pub(crate) fn decode_input_transformations(
    value: Option<&Value>,
) -> Result<Option<Vec<InputTransformation>>, FrameError> {
    match value {
        None => Ok(None),
        Some(Value::Array(transformations)) => {
            transformations.iter().map(InputTransformation::decode).collect::<Result<Vec<_>, _>>().map(Some)
        }
        Some(_) => Err(FrameError::WrongType { field: "input_transformations", expected: "an array" }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn known_and_future_transformations_keep_every_distinction() {
        let known = InputTransformation::decode(&json!({
            "type": "thinking_dropped",
            "path": "messages.1.content.0",
            "reason": "prefix_binding_mismatch"
        }))
        .unwrap();
        assert_eq!(known.kind(), "thinking_dropped");
        assert_eq!(known.path(), Some("messages.1.content.0"));
        assert_eq!(known.reason(), Some(ThinkingDropReason::PrefixBindingMismatch));
        assert_eq!(known.raw_reason(), Some("prefix_binding_mismatch"));

        let future = InputTransformation::decode(&json!({"type": "future_check", "new": 7})).unwrap();
        let InputTransformation::Unmodeled { kind, value } = future else { panic!("future kind must survive") };
        assert_eq!(kind, "future_check");
        assert_eq!(value["new"], 7);
    }

    #[test]
    fn absence_and_an_empty_checked_result_stay_different() {
        assert_eq!(decode_input_transformations(None).unwrap(), None);
        assert_eq!(decode_input_transformations(Some(&json!([]))).unwrap(), Some(Vec::new()));
    }
}
