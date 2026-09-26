//! A request reduced to what the prompt cache keys it by.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{LOOKBACK_POSITIONS, Position, render};
use crate::CacheTtl;
use crate::request::Request;

/// The SHA-256 digest of a rendered prefix, which is how an entry is found.
///
/// A digest rather than the bytes because an entry's key is the whole prefix
/// before it, which grows with the conversation; a collision would need two
/// different prompts to agree on 256 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Digest(pub(super) [u8; 32]);

impl Serialize for Digest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let hex: String = self.0.iter().map(|byte| format!("{byte:02x}")).collect();
        serializer.serialize_str(&hex)
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let hex = String::deserialize(deserializer)?;
        let invalid = || serde::de::Error::custom("a digest is 64 lowercase hexadecimal digits");
        if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return Err(invalid());
        }
        let mut digest = [0; 32];
        for (i, byte) in digest.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|_| invalid())?;
        }
        Ok(Self(digest))
    }
}

/// A breakpoint's TTL as its wire string, which is what [`CacheTtl`] spells.
mod mark {
    use serde::{Deserialize, Deserializer, Serializer};

    use crate::CacheTtl;

    pub(super) fn serialize<S: Serializer>(mark: &Option<CacheTtl>, serializer: S) -> Result<S::Ok, S::Error> {
        match mark {
            Some(ttl) => serializer.serialize_some(ttl.as_str()),
            None => serializer.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<CacheTtl>, D::Error> {
        Option::<String>::deserialize(deserializer)?
            .map(|ttl| CacheTtl::from_str(&ttl).ok_or_else(|| serde::de::Error::custom(format!("unknown TTL {ttl:?}"))))
            .transpose()
    }
}

/// One position of the rendered prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Cut {
    pub(super) position: Position,
    /// Byte length of the rendered prefix ending here, which orders the stretches
    /// of one line of storage.
    pub(super) end: u64,
    /// Positions sharing a unit count once against the lookback window.
    pub(super) unit: usize,
    /// The breakpoint placed here, if any.
    #[serde(with = "mark")]
    pub(super) mark: Option<CacheTtl>,
    /// The key of the prefix ending here.
    pub(super) digest: Digest,
}

/// A request as the prompt cache meets it: every position a breakpoint's
/// lookback reaches, keyed by a digest of the prefix ending there, with the
/// breakpoints' TTLs and the model's smallest cacheable prefix.
///
/// Exists so that a sequence of requests can be replayed without the requests.
/// A caller records these as each request is sent, at a size that does not grow
/// with the conversation, and serves them later — under the TTLs that were
/// sent, or under others through [`with_ttls`](Self::with_ttls), which is how
/// one recorded day answers what a different caching policy would have cost.
/// Positions no lookback reaches are dropped: no entry is ever looked up there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKeys {
    pub(super) cuts: Vec<Cut>,
    pub(super) minimum: u64,
}

/// Why [`CacheKeys::with_ttls`] refused a set of TTLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtlsError {
    /// A TTL is needed for each breakpoint, no more and no fewer.
    Count {
        /// Breakpoints in the request.
        breakpoints: usize,
        /// TTLs supplied.
        supplied: usize,
    },
    /// A 1-hour breakpoint followed a 5-minute one, which the API refuses.
    OneHourAfterFiveMinutes,
}

impl std::fmt::Display for TtlsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Count { breakpoints, supplied } => {
                write!(f, "the request has {breakpoints} breakpoints but {supplied} TTLs were supplied")
            }
            Self::OneHourAfterFiveMinutes => write!(f, "all 1h breakpoints must come before any 5m breakpoints"),
        }
    }
}

impl std::error::Error for TtlsError {}

impl CacheKeys {
    /// The keys `request` is served under.
    pub fn of(request: &Request<'_>) -> Self {
        let cuts = render::render(request);
        let mut reached = vec![false; cuts.len()];
        let mut next_mark_unit = None;
        for (i, cut) in cuts.iter().enumerate().rev() {
            if cut.mark.is_some() {
                next_mark_unit = Some(cut.unit);
            }
            reached[i] = next_mark_unit.is_some_and(|unit| unit - cut.unit < LOOKBACK_POSITIONS);
        }
        let cuts = cuts.into_iter().zip(reached).filter_map(|(cut, reached)| reached.then_some(cut)).collect();
        Self { cuts, minimum: u64::from(request.model().id().min_cacheable_prefix_tokens()) }
    }

    /// Each breakpoint's TTL, in prompt order.
    pub fn ttls(&self) -> Vec<CacheTtl> {
        self.cuts.iter().filter_map(|cut| cut.mark).collect()
    }

    /// The same prompt with `ttls` on its breakpoints, one per breakpoint in
    /// prompt order.
    ///
    /// A TTL renders nowhere in the prefix, so every key is unchanged: only how
    /// long each written entry lives, and at which rate it is billed.
    pub fn with_ttls(&self, ttls: &[CacheTtl]) -> Result<Self, TtlsError> {
        let breakpoints = self.cuts.iter().filter(|cut| cut.mark.is_some()).count();
        if ttls.len() != breakpoints {
            return Err(TtlsError::Count { breakpoints, supplied: ttls.len() });
        }
        if ttls.windows(2).any(|pair| pair == [CacheTtl::FiveMinutes, CacheTtl::OneHour]) {
            return Err(TtlsError::OneHourAfterFiveMinutes);
        }
        let mut ttls = ttls.iter();
        let mut keys = self.clone();
        for cut in keys.cuts.iter_mut().filter(|cut| cut.mark.is_some()) {
            cut.mark = ttls.next().copied();
        }
        Ok(keys)
    }
}
