//! Anthropic's web search server tool: how a request offers it, and what a
//! search leaves in the model's turn.
//!
//! A server tool runs on Anthropic's side, inside one request. The model emits a
//! `server_tool_use` block with its query, the API runs the search and appends a
//! `web_search_tool_result` block, and the model continues in the same assistant
//! turn — the caller answers nothing. Both blocks then belong to the conversation:
//! the next request must replay them exactly, because each result's
//! `encrypted_content` is how the API restores what the model read, and a missing
//! or altered one is a 400.
//!
//! That is why the result types here serialize as well as decode: they are the
//! same value in both directions, not an inbound view with an outbound copy.
//!
//! Only the basic tool version, `web_search_20250305`, is modeled. The later
//! versions default to calling search from inside code execution, which needs
//! `allowed_callers` and the code execution tool's own blocks; this crate does
//! not yet model either, so those versions cannot yet be declared.

use std::num::NonZeroU32;

use serde::Serialize;
use serde_json::Value;

use crate::WebSearchErrorCode;
use crate::context::CacheControl;
use crate::frame::{FrameError, require_str};

/// The `name` the web search tool carries, and that its `server_tool_use`
/// blocks repeat. Fixed by the API rather than chosen by the caller.
pub const WEB_SEARCH_NAME: &str = "web_search";

/// The one tool version this crate declares.
const WEB_SEARCH_TYPE: &str = "web_search_20250305";

// ── Declaring the tool ───────────────────────────────────────────────────────

/// Which domains a search may return results from.
///
/// A sum type because the API takes an allow-list or a block-list and refuses a
/// request carrying both with a 400; two optional lists would make that request
/// writable.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum DomainFilter {
    /// Any domain.
    #[default]
    Any,
    /// Only these domains. Entries are bare domains with an optional path,
    /// without a scheme: `example.com` or `example.com/blog`.
    Allowed(Vec<String>),
    /// Never these domains, in the same form.
    Blocked(Vec<String>),
}

/// Where the user roughly is, so results can be localized.
///
/// The API requires at least one of the four fields, so a location is built from
/// one of them and the others are added: an empty location has no constructor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UserLocation {
    #[serde(rename = "type")]
    kind: ApproximateLocation,
    #[serde(skip_serializing_if = "Option::is_none")]
    city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timezone: Option<String>,
}

/// The only location `type` the API documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ApproximateLocation {
    Approximate,
}

impl UserLocation {
    fn empty() -> Self {
        Self { kind: ApproximateLocation::Approximate, city: None, region: None, country: None, timezone: None }
    }
    /// A location known by its city.
    pub fn city(city: impl Into<String>) -> Self {
        Self::empty().with_city(city)
    }
    /// A location known by its region or state.
    pub fn region(region: impl Into<String>) -> Self {
        Self::empty().with_region(region)
    }
    /// A location known by its ISO 3166-1 alpha-2 country code. The API refuses
    /// a code it does not support with a 400.
    pub fn country(country: impl Into<String>) -> Self {
        Self::empty().with_country(country)
    }
    /// A location known by its IANA time zone.
    pub fn timezone(timezone: impl Into<String>) -> Self {
        Self::empty().with_timezone(timezone)
    }
    /// Adds the city.
    pub fn with_city(mut self, city: impl Into<String>) -> Self {
        self.city = Some(city.into());
        self
    }
    /// Adds the region or state.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }
    /// Adds the ISO 3166-1 alpha-2 country code.
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }
    /// Adds the IANA time zone.
    pub fn with_timezone(mut self, timezone: impl Into<String>) -> Self {
        self.timezone = Some(timezone.into());
        self
    }
    /// The city, if given.
    pub fn city_name(&self) -> Option<&str> {
        self.city.as_deref()
    }
    /// The region, if given.
    pub fn region_name(&self) -> Option<&str> {
        self.region.as_deref()
    }
    /// The country code, if given.
    pub fn country_code(&self) -> Option<&str> {
        self.country.as_deref()
    }
    /// The time zone, if given.
    pub fn timezone_id(&self) -> Option<&str> {
        self.timezone.as_deref()
    }
}

/// The web search tool, offered to the model beside any custom tools.
///
/// Every field absent means the API's own behavior: no limit on searches per
/// request, every domain, no localization. Each search is billed on top of
/// tokens, which is the reason [`Self::max_uses`] exists.
#[derive(Debug, Clone, Default)]
pub struct WebSearchTool {
    /// The most searches one request may run. A further attempt returns a
    /// [`WebSearchErrorCode::MaxUsesExceeded`] result instead of searching.
    pub max_uses: Option<NonZeroU32>,
    /// Which domains results may come from.
    pub domains: DomainFilter,
    /// Where to localize results for.
    pub user_location: Option<UserLocation>,
    pub(crate) cache_control: Option<CacheControl>,
}

impl WebSearchTool {
    /// The tool with the API's defaults.
    pub fn new() -> Self {
        Self::default()
    }
    /// Caps the searches one request may run.
    pub fn max_uses(mut self, max_uses: NonZeroU32) -> Self {
        self.max_uses = Some(max_uses);
        self
    }
    /// Restricts which domains results may come from.
    pub fn domains(mut self, domains: DomainFilter) -> Self {
        self.domains = domains;
        self
    }
    /// Localizes results.
    pub fn user_location(mut self, location: UserLocation) -> Self {
        self.user_location = Some(location);
        self
    }
}

#[derive(Serialize)]
struct WebSearchToolWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    name: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_uses: Option<NonZeroU32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_domains: Option<&'a Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blocked_domains: Option<&'a Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_location: Option<&'a UserLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: &'a Option<CacheControl>,
}

impl Serialize for WebSearchTool {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let (allowed_domains, blocked_domains) = match &self.domains {
            DomainFilter::Any => (None, None),
            DomainFilter::Allowed(domains) => (Some(domains), None),
            DomainFilter::Blocked(domains) => (None, Some(domains)),
        };
        WebSearchToolWire {
            kind: WEB_SEARCH_TYPE,
            name: WEB_SEARCH_NAME,
            max_uses: self.max_uses,
            allowed_domains,
            blocked_domains,
            user_location: self.user_location.as_ref(),
            cache_control: &self.cache_control,
        }
        .serialize(s)
    }
}

// ── What a search returns ────────────────────────────────────────────────────

/// One page a search returned.
///
/// `encrypted_content` is the page as the model read it, sealed. It is opaque to
/// the caller and must be replayed unchanged, which is why it is a plain field
/// kept byte for byte rather than something this crate interprets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename = "web_search_result")]
pub struct WebSearchResult {
    /// The page's URL.
    pub url: String,
    /// The page's title, where the API gave one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The page content, encrypted, for replay on later turns.
    pub encrypted_content: String,
    /// When the site was last updated, in the API's own words
    /// (`"April 30, 2025"`), where it knows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_age: Option<String>,
}

impl WebSearchResult {
    fn decode(result: &Value) -> Result<Self, FrameError> {
        let optional = |field: &'static str| match result.get(field) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(text)) => Ok(Some(text.clone())),
            Some(_) => Err(FrameError::WrongType { field, expected: "a string" }),
        };
        Ok(Self {
            url: require_str(result, "url")?.to_owned(),
            title: optional("title")?,
            encrypted_content: require_str(result, "encrypted_content")?.to_owned(),
            page_age: optional("page_age")?,
        })
    }
}

/// What one search came to: its pages, or why it failed.
///
/// A sum type because the wire is one: `content` is an array of results on
/// success and a single error object on failure. A search that succeeded and
/// matched nothing is an empty [`Self::Results`], not an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebSearchOutcome {
    /// The pages found, in the order the API ranked them.
    Results(Vec<WebSearchResult>),
    /// The search failed; the model was told why.
    Error(WebSearchErrorCode),
}

#[derive(Serialize)]
struct WebSearchErrorWire {
    #[serde(rename = "type")]
    kind: &'static str,
    error_code: WebSearchErrorCode,
}

impl Serialize for WebSearchOutcome {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Results(results) => results.serialize(s),
            Self::Error(error_code) => {
                WebSearchErrorWire { kind: "web_search_tool_result_error", error_code: *error_code }.serialize(s)
            }
        }
    }
}

/// A result block's `content` that decodes to nothing this crate knows.
///
/// Only an error code Anthropic added after this crate was written. The block
/// holding it is then kept whole as an unmodeled block rather than failing the
/// stream, for the reason every unknown kind is.
pub(crate) struct UnknownErrorCode;

impl WebSearchOutcome {
    /// One `content` field of a `web_search_tool_result` block.
    pub(crate) fn decode(content: &Value) -> Result<Result<Self, UnknownErrorCode>, FrameError> {
        match content {
            Value::Array(results) => {
                Ok(Ok(Self::Results(results.iter().map(WebSearchResult::decode).collect::<Result<_, _>>()?)))
            }
            Value::Object(_) => {
                let code = require_str(content, "error_code")?;
                Ok(WebSearchErrorCode::from_str(code).map(Self::Error).ok_or(UnknownErrorCode))
            }
            _ => Err(FrameError::WrongType { field: "content", expected: "an array or an object" }),
        }
    }

    /// The pages found, empty for a failed search.
    pub fn results(&self) -> &[WebSearchResult] {
        match self {
            Self::Results(results) => results,
            Self::Error(_) => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_default_tool_is_its_type_and_name_alone() {
        assert_eq!(
            serde_json::to_value(WebSearchTool::new()).unwrap(),
            json!({"type": "web_search_20250305", "name": "web_search"})
        );
    }

    #[test]
    fn every_setting_takes_its_documented_field() {
        let tool = WebSearchTool::new()
            .max_uses(NonZeroU32::new(5).unwrap())
            .domains(DomainFilter::Blocked(vec!["example.com".to_owned()]))
            .user_location(UserLocation::city("San Francisco").with_country("US").with_timezone("America/Los_Angeles"));
        assert_eq!(
            serde_json::to_value(tool).unwrap(),
            json!({
                "type": "web_search_20250305", "name": "web_search", "max_uses": 5,
                "blocked_domains": ["example.com"],
                "user_location": {"type": "approximate", "city": "San Francisco", "country": "US",
                                  "timezone": "America/Los_Angeles"}
            })
        );
        let allowed = WebSearchTool::new().domains(DomainFilter::Allowed(vec!["docs.rs".to_owned()]));
        let wire = serde_json::to_value(allowed).unwrap();
        assert_eq!(wire["allowed_domains"], json!(["docs.rs"]));
        assert!(wire.get("blocked_domains").is_none(), "one list or the other, never both");
    }

    /// The documented result shape decodes and serializes back to itself, which is
    /// what replay needs.
    #[test]
    fn a_documented_result_list_round_trips() {
        let content = json!([{
            "type": "web_search_result", "url": "https://en.wikipedia.org/wiki/Claude_Shannon",
            "title": "Claude Shannon - Wikipedia", "encrypted_content": "EqgfCioIARgBIiQ3YTAwMjY1Mi1mZjM5",
            "page_age": "April 30, 2025"
        }]);
        let Ok(Ok(outcome)) = WebSearchOutcome::decode(&content) else { panic!("expected results") };
        assert_eq!(outcome.results()[0].url, "https://en.wikipedia.org/wiki/Claude_Shannon");
        assert_eq!(serde_json::to_value(&outcome).unwrap(), content);
    }

    #[test]
    fn an_error_and_an_empty_search_are_different_outcomes() {
        let error = json!({"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"});
        let Ok(Ok(outcome)) = WebSearchOutcome::decode(&error) else { panic!("expected an error outcome") };
        assert_eq!(outcome, WebSearchOutcome::Error(WebSearchErrorCode::MaxUsesExceeded));
        assert_eq!(serde_json::to_value(&outcome).unwrap(), error);

        let Ok(Ok(empty)) = WebSearchOutcome::decode(&json!([])) else { panic!("expected results") };
        assert_eq!(empty, WebSearchOutcome::Results(Vec::new()));

        let novel = json!({"type": "web_search_tool_result_error", "error_code": "solar_flare"});
        assert!(matches!(WebSearchOutcome::decode(&novel), Ok(Err(UnknownErrorCode))));
        assert!(matches!(
            WebSearchOutcome::decode(&json!([{"url": "u"}])),
            Err(FrameError::MissingField { field: "encrypted_content" })
        ));
    }
}
