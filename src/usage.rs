//! What a request cost, and what the cache did.
//!
//! These counts are the only evidence that prompt caching worked: a breakpoint
//! below the model's minimum cacheable prefix is a *silent* no-op, and the sole
//! way to notice is that both cache fields came back zero
//! (see [`crate::request::ModelId::min_cacheable_prefix_tokens`]). So this type
//! decodes them rather than skipping past them, and
//! [`Usage::merge_cumulative`] is written so a later frame can never lose them.
//!
//! # Why the counts add up the way they do
//!
//! [`Usage::input_tokens`] is *not* the whole input. Anthropic documents it as
//! the tokens after the last cache breakpoint — the part that was neither read
//! from nor written to the cache. The whole input is the sum of the three,
//! which is [`Usage::total_input_tokens`].
//!
//! # Merging is a pointwise maximum, and that is exact
//!
//! Anthropic documents the counts on `message_delta` as *cumulative*. Each
//! field is therefore a counter that never decreases across one stream, so the
//! most complete record of a stream is the pointwise maximum of every `usage`
//! object it delivered. That operation is the join of a product lattice of
//! counters: commutative, associative, idempotent, with the all-zero `Usage` as
//! its identity. Three consequences fall out for free, and none of them needs a
//! special case:
//!
//! * Frames may be merged in any order.
//! * Merging the same frame twice changes nothing.
//! * A frame that omits a field cannot zero it. This is not hypothetical: the
//!   NVIDIA inference gateway sends a `message_stop` carrying only
//!   `input_tokens` and `output_tokens`, and a last-writer-wins merge would
//!   throw away exactly the two cache numbers this module exists to report.

use serde::Deserialize;

/// Cache writes split by time-to-live.
///
/// Anthropic documents this object's fields as summing to
/// [`Usage::cache_creation_input_tokens`]; [`Usage::cache_creation_is_consistent`]
/// checks it. The split is what separates a cheap 5-minute write from a
/// 1-hour write billed at twice the base input rate, so it is worth reading
/// even when the total is all you thought you needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct CacheCreation {
    /// Tokens written to a 5-minute cache entry, billed at 1.25× base input.
    pub ephemeral_5m_input_tokens: u64,
    /// Tokens written to a 1-hour cache entry, billed at 2× base input.
    pub ephemeral_1h_input_tokens: u64,
}

impl CacheCreation {
    /// The two TTL buckets summed.
    pub fn total(self) -> u64 {
        self.ephemeral_5m_input_tokens + self.ephemeral_1h_input_tokens
    }

    /// The pointwise maximum, as described in the module documentation.
    pub fn merge_cumulative(&mut self, other: Self) {
        self.ephemeral_5m_input_tokens = self.ephemeral_5m_input_tokens.max(other.ephemeral_5m_input_tokens);
        self.ephemeral_1h_input_tokens = self.ephemeral_1h_input_tokens.max(other.ephemeral_1h_input_tokens);
    }
}

/// Tokens billed for one request.
///
/// Every field defaults to zero, because the API sends different subsets of
/// this object at different points of a stream and an absent counter means
/// "nothing of that kind", never "unknown". Unrecognized fields are ignored, so
/// a count Anthropic adds later cannot break decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct Usage {
    /// Input tokens that were neither read from nor written to the cache —
    /// everything after the last cache breakpoint. Billed at the base input
    /// rate. See [`Self::total_input_tokens`] for the whole input.
    pub input_tokens: u64,
    /// Tokens generated, thinking included.
    pub output_tokens: u64,
    /// Input tokens served from a cache entry an earlier request wrote, billed
    /// at 0.1× the base input rate.
    ///
    /// Zero here *and* in [`Self::cache_creation_input_tokens`] means nothing
    /// was cached at all. The likeliest cause is a cached prefix shorter than
    /// the model's minimum, which the API accepts in silence.
    pub cache_read_input_tokens: u64,
    /// Input tokens written to a new cache entry, billed above the base input
    /// rate. Equals [`CacheCreation::total`].
    pub cache_creation_input_tokens: u64,
    /// The same writes split by TTL.
    pub cache_creation: CacheCreation,
    /// How much of [`Self::output_tokens`] went on thinking, where the API
    /// reports it.
    pub output_tokens_details: OutputTokensDetails,
}

/// The breakdown of generated tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct OutputTokensDetails {
    /// Tokens spent thinking, already counted in [`Usage::output_tokens`].
    pub thinking_tokens: u64,
}

impl Usage {
    /// Every input token the request was billed for, cached or not.
    ///
    /// The sum Anthropic documents: reads, plus writes, plus the uncached tail.
    /// [`Self::input_tokens`] alone understates the input whenever caching is
    /// working, which is precisely when someone wants to read it.
    pub fn total_input_tokens(&self) -> u64 {
        self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens
    }

    /// The share of input tokens that came from the cache, or `None` when there
    /// were no input tokens to divide by.
    ///
    /// The one number that answers "is my cache working". A prefix below the
    /// model's minimum reads as `Some(0.0)`, which is the honest answer: the
    /// request was billed in full.
    pub fn cache_hit_rate(&self) -> Option<f64> {
        let total = self.total_input_tokens();
        (total > 0).then(|| self.cache_read_input_tokens as f64 / total as f64)
    }

    /// Whether the TTL split agrees with the write total.
    ///
    /// Anthropic documents these as equal. Reported rather than enforced: a
    /// gateway that omits the split still gives a usable write total, and
    /// refusing the whole response over a redundant field would trade a real
    /// number for a pedantic error.
    pub fn cache_creation_is_consistent(&self) -> bool {
        self.cache_creation.total() == self.cache_creation_input_tokens
    }

    /// The pointwise maximum of two cumulative records.
    ///
    /// See the module documentation for why this is the exact merge and not an
    /// approximation of one.
    pub fn merge_cumulative(&mut self, other: &Self) {
        self.input_tokens = self.input_tokens.max(other.input_tokens);
        self.output_tokens = self.output_tokens.max(other.output_tokens);
        self.cache_read_input_tokens = self.cache_read_input_tokens.max(other.cache_read_input_tokens);
        self.cache_creation_input_tokens = self.cache_creation_input_tokens.max(other.cache_creation_input_tokens);
        self.cache_creation.merge_cumulative(other.cache_creation);
        self.output_tokens_details.thinking_tokens =
            self.output_tokens_details.thinking_tokens.max(other.output_tokens_details.thinking_tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The `message_start` usage captured from the live gateway, field for field.
    #[test]
    fn a_captured_message_start_usage_decodes() {
        let usage: Usage = serde_json::from_value(json!({
            "input_tokens": 36, "cache_creation_input_tokens": 1043, "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 1043, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 1, "service_tier": "standard"
        }))
        .unwrap();
        assert_eq!(usage.input_tokens, 36);
        assert_eq!(usage.cache_creation_input_tokens, 1_043);
        assert_eq!(usage.cache_creation.ephemeral_5m_input_tokens, 1_043);
        assert_eq!(usage.total_input_tokens(), 1_079, "36 uncached + 1043 written");
        assert!(usage.cache_creation_is_consistent());
        assert_eq!(usage.cache_hit_rate(), Some(0.0), "a write is not yet a read");
    }

    /// The same prompt sent twice: the second request read the identical 1043
    /// tokens back out of the cache. This pair is the whole point of the module.
    #[test]
    fn a_captured_cache_read_reports_its_hit_rate() {
        let usage: Usage = serde_json::from_value(json!({
            "input_tokens": 36, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1043,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 2, "service_tier": "standard"
        }))
        .unwrap();
        assert_eq!(usage.cache_read_input_tokens, 1_043);
        assert_eq!(usage.total_input_tokens(), 1_079);
        let rate = usage.cache_hit_rate().unwrap();
        assert!((rate - 1_043.0 / 1_079.0).abs() < 1e-12, "{rate}");
    }

    /// The gateway's `message_stop` carries only two counters. Merging it must
    /// not zero the cache numbers a fuller earlier frame reported.
    #[test]
    fn a_partial_later_frame_cannot_erase_the_cache_counts() {
        let mut usage: Usage = serde_json::from_value(json!({
            "input_tokens": 388, "cache_creation_input_tokens": 512, "cache_read_input_tokens": 1024,
            "cache_creation": {"ephemeral_5m_input_tokens": 12, "ephemeral_1h_input_tokens": 500},
            "output_tokens": 10
        }))
        .unwrap();
        let message_stop: Usage = serde_json::from_value(json!({"input_tokens": 388, "output_tokens": 50})).unwrap();
        usage.merge_cumulative(&message_stop);

        assert_eq!(usage.output_tokens, 50, "the newer, larger count wins");
        assert_eq!(usage.cache_read_input_tokens, 1_024, "kept");
        assert_eq!(usage.cache_creation_input_tokens, 512, "kept");
        assert_eq!(usage.cache_creation.ephemeral_1h_input_tokens, 500, "kept");
    }

    /// Commutative, associative, idempotent, with zero as the identity — so
    /// frame order and duplicate frames cannot change the result.
    #[test]
    fn merging_is_a_join_and_therefore_order_free() {
        let a: Usage = serde_json::from_value(json!({"input_tokens": 5, "output_tokens": 90})).unwrap();
        let b: Usage = serde_json::from_value(json!({"input_tokens": 40, "output_tokens": 2})).unwrap();

        let mut forward = a;
        forward.merge_cumulative(&b);
        let mut backward = b;
        backward.merge_cumulative(&a);
        assert_eq!(forward, backward, "commutative");

        let mut twice = forward;
        twice.merge_cumulative(&b);
        assert_eq!(twice, forward, "idempotent");

        let mut with_identity = forward;
        with_identity.merge_cumulative(&Usage::default());
        assert_eq!(with_identity, forward, "zero is the identity");
        assert_eq!(forward.input_tokens, 40);
        assert_eq!(forward.output_tokens, 90);
    }

    #[test]
    fn an_absent_usage_object_is_all_zero() {
        let usage: Usage = serde_json::from_value(json!({})).unwrap();
        assert_eq!(usage, Usage::default());
        assert_eq!(usage.cache_hit_rate(), None, "nothing to divide by");
        assert!(usage.cache_creation_is_consistent(), "zero equals zero");
    }

    /// A missing TTL split is reported, not fatal: the write total is still real.
    #[test]
    fn an_inconsistent_ttl_split_is_reported_rather_than_refused() {
        let usage: Usage =
            serde_json::from_value(json!({"cache_creation_input_tokens": 900, "output_tokens": 1})).unwrap();
        assert_eq!(usage.cache_creation_input_tokens, 900);
        assert!(!usage.cache_creation_is_consistent());
    }

    #[test]
    fn thinking_tokens_decode_where_the_api_reports_them() {
        let usage: Usage = serde_json::from_value(json!({
            "input_tokens": 36, "output_tokens": 45, "output_tokens_details": {"thinking_tokens": 30}
        }))
        .unwrap();
        assert_eq!(usage.output_tokens_details.thinking_tokens, 30);
        assert_eq!(usage.output_tokens, 45, "thinking is already inside the output count");
    }
}
