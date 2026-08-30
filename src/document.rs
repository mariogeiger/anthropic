//! Source material the model may quote, and the citations it quotes it with.
//!
//! # Why a document is not just text
//!
//! Pasting a document into a text block works, and loses the one thing a document
//! is for: the model can then say *where* an answer came from. A `document` block
//! with citations enabled makes the API emit a [`Citation`] beside each text block
//! it grounds, naming the exact characters, page, or block it drew on. A caller
//! can render a footnote; a caller with pasted text can only take the model's word.
//!
//! # Why citations are opt-in
//!
//! Citations cost output tokens and change how the model writes, so
//! [`DocumentBlock::cited`] asks for them and [`DocumentBlock::new`] does not.
//! Absent is not the same as `false`: [`Citations`] is emitted only when a caller
//! decides, so a request that never mentioned citations is byte-identical to what
//! it was before this module existed, and its prompt cache still matches.
//!
//! # A document and a search result differ in what they claim
//!
//! [`DocumentBlock`] is material the caller supplies: a PDF, plain text, a URL for
//! the API to fetch. [`SearchResultBlock`] is material a *search* returned, and it
//! carries the source and title that make a citation to it meaningful. The API
//! counts them separately — `document_index` versus `search_result_index` — so they
//! are two types, and a citation into one cannot be read as a citation into the
//! other.

use serde::Serialize;

use crate::block::TextBlock;
use crate::context::CacheControl;
use crate::frame::{FrameError, optional_string, require_str};
use crate::values::{CitationType, DocumentMediaType, PlainTextMediaType};

// ── Sources ──────────────────────────────────────────────────────────────────

/// Where a document's bytes come from.
///
/// Four shapes, and the media type is fixed per shape rather than free: a base64
/// document is a PDF and a text document is `text/plain`, so the enum that carries
/// the media type has one variant and there is no wrong value to write.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    /// A PDF inline, base64-encoded.
    Base64 {
        /// Always `application/pdf`; a base64 document is a PDF.
        media_type: DocumentMediaType,
        /// The base64 payload.
        data: String,
    },
    /// Plain text inline.
    Text {
        /// Always `text/plain`.
        media_type: PlainTextMediaType,
        /// The text.
        data: String,
    },
    /// A PDF the API fetches.
    Url {
        /// Where to fetch it.
        url: String,
    },
    /// A document already uploaded through the Files API.
    File {
        /// Its identifier.
        file_id: String,
    },
}

impl DocumentSource {
    /// A PDF inline, base64-encoded.
    pub fn pdf(data: impl Into<String>) -> Self {
        Self::Base64 { media_type: DocumentMediaType::Pdf, data: data.into() }
    }
    /// Plain text inline. Character-level citations point into these bytes.
    pub fn text(data: impl Into<String>) -> Self {
        Self::Text { media_type: PlainTextMediaType::Text, data: data.into() }
    }
    /// A PDF for the API to fetch.
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url { url: url.into() }
    }
    /// A document already uploaded through the Files API.
    pub fn file(file_id: impl Into<String>) -> Self {
        Self::File { file_id: file_id.into() }
    }
}

/// Whether the model may cite a block, as the API's `citations` object.
///
/// A struct with one field rather than a bare `bool`, because that is the wire
/// shape and the crate models the wire. It appears only where a caller asked; see
/// the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Citations {
    /// Whether the model may cite this block.
    pub enabled: bool,
}

// ── Blocks ───────────────────────────────────────────────────────────────────

/// Source material the caller supplies for the model to read and quote.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentBlock {
    /// Where its bytes come from.
    pub source: DocumentSource,
    /// A title the model sees, and that citations into this document repeat.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Context about the document for the model, which is *not* citable — use it
    /// to say what the document is without offering that sentence as a quotable
    /// source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Whether the model may cite it. Absent leaves the API's default, which is
    /// not to cite.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Citations>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

impl DocumentBlock {
    /// Material the model may read but not cite.
    pub fn new(source: DocumentSource) -> Self {
        Self { source, title: None, context: None, citations: None, cache_control: None }
    }

    /// Material the model may quote, emitting a [`Citation`] where it does.
    pub fn cited(source: DocumentSource) -> Self {
        Self { citations: Some(Citations { enabled: true }), ..Self::new(source) }
    }

    /// Titles the document. Citations into it repeat this title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Describes the document to the model without making the description citable.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// Material a search returned, carrying where it came from.
///
/// The source and title are required rather than optional, which is the difference
/// from [`DocumentBlock`]: a search result whose origin is unknown cannot be cited
/// usefully, so the type does not admit one.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResultBlock {
    /// Where the result came from — a URL, a path, whatever the search names.
    pub source: String,
    /// Its title.
    pub title: String,
    /// The result's text, in blocks. A citation names a range of them by index.
    pub content: Vec<TextBlock>,
    /// Whether the model may cite it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Citations>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

impl SearchResultBlock {
    /// A result the model may read but not cite.
    pub fn new(source: impl Into<String>, title: impl Into<String>, content: Vec<TextBlock>) -> Self {
        Self { source: source.into(), title: title.into(), content, citations: None, cache_control: None }
    }

    /// A result the model may quote, emitting a [`Citation`] where it does.
    pub fn cited(source: impl Into<String>, title: impl Into<String>, content: Vec<TextBlock>) -> Self {
        Self { citations: Some(Citations { enabled: true }), ..Self::new(source, title, content) }
    }
}

// ── Citations ────────────────────────────────────────────────────────────────

/// One place a text block drew on.
///
/// Inbound only: the model emits these, a caller never sends one. Which variant
/// arrives follows from what was cited — a plain-text document cites characters, a
/// PDF cites pages, a structured source cites blocks — so the variant *is* the
/// answer to "how precisely can I point at this".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Citation {
    /// A character range of a plain-text document.
    CharLocation {
        /// The quoted text, exactly.
        cited_text: String,
        /// Which document in the request, counting `document` blocks only.
        document_index: u64,
        /// Its title, where it had one.
        document_title: Option<String>,
        /// First character quoted, zero-based.
        start_char_index: u64,
        /// One past the last character quoted.
        end_char_index: u64,
    },
    /// A page range of a PDF.
    PageLocation {
        /// The quoted text, exactly.
        cited_text: String,
        /// Which document in the request.
        document_index: u64,
        /// Its title, where it had one.
        document_title: Option<String>,
        /// First page quoted, one-based as the API numbers pages.
        start_page_number: u64,
        /// One past the last page quoted.
        end_page_number: u64,
    },
    /// A block range of a structured document.
    ContentBlockLocation {
        /// The quoted text: the whole cited range, concatenated. Never a substring
        /// of one block, because a text block is the minimal citable unit.
        cited_text: String,
        /// Which document in the request.
        document_index: u64,
        /// Its title, where it had one.
        document_title: Option<String>,
        /// First block quoted, zero-based.
        start_block_index: u64,
        /// One past the last block quoted.
        end_block_index: u64,
    },
    /// A block range of a search result.
    SearchResultLocation {
        /// The quoted text, concatenated over the range.
        cited_text: String,
        /// Which search result, counted *separately* from documents and excluding
        /// the API's own web-search results.
        search_result_index: u64,
        /// Where the result came from.
        source: String,
        /// Its title, where it had one.
        title: Option<String>,
        /// First block quoted, zero-based.
        start_block_index: u64,
        /// One past the last block quoted.
        end_block_index: u64,
    },
    /// A result of the API's own web search.
    WebSearchResultLocation {
        /// The quoted text, exactly.
        cited_text: String,
        /// The page cited.
        url: String,
        /// Its title, where it had one.
        title: Option<String>,
        /// The opaque index the API uses to refer back to the result.
        encrypted_index: String,
    },
    /// A citation kind this crate does not model.
    ///
    /// Never an error, for the reason [`crate::stream::StreamEvent::Unmodeled`]
    /// exists: a well-formed citation of a new kind is not a broken frame.
    Unmodeled {
        /// The citation's `type`.
        kind: String,
    },
}

impl Citation {
    /// The citation's wire `type`.
    pub fn kind(&self) -> &str {
        match self {
            Citation::CharLocation { .. } => CitationType::CharLocation.as_str(),
            Citation::PageLocation { .. } => CitationType::PageLocation.as_str(),
            Citation::ContentBlockLocation { .. } => CitationType::ContentBlockLocation.as_str(),
            Citation::SearchResultLocation { .. } => CitationType::SearchResultLocation.as_str(),
            Citation::WebSearchResultLocation { .. } => CitationType::WebSearchResultLocation.as_str(),
            Citation::Unmodeled { kind } => kind,
        }
    }

    /// The text the model quoted. Present on every modeled kind, because a citation
    /// that names no text cites nothing.
    pub fn cited_text(&self) -> Option<&str> {
        match self {
            Citation::CharLocation { cited_text, .. }
            | Citation::PageLocation { cited_text, .. }
            | Citation::ContentBlockLocation { cited_text, .. }
            | Citation::SearchResultLocation { cited_text, .. }
            | Citation::WebSearchResultLocation { cited_text, .. } => Some(cited_text),
            Citation::Unmodeled { .. } => None,
        }
    }

    /// One citation object, decoded.
    pub fn decode(citation: &serde_json::Value) -> Result<Self, FrameError> {
        if !citation.is_object() {
            return Err(FrameError::WrongType { field: "citation", expected: "an object" });
        }
        let text = || optional_string(citation, "cited_text");
        let index = |field| citation.get(field).and_then(serde_json::Value::as_u64).unwrap_or_default();
        let optional = |field| citation.get(field).and_then(serde_json::Value::as_str).map(str::to_owned);
        Ok(match require_str(citation, "type")? {
            "char_location" => Citation::CharLocation {
                cited_text: text(),
                document_index: index("document_index"),
                document_title: optional("document_title"),
                start_char_index: index("start_char_index"),
                end_char_index: index("end_char_index"),
            },
            "page_location" => Citation::PageLocation {
                cited_text: text(),
                document_index: index("document_index"),
                document_title: optional("document_title"),
                start_page_number: index("start_page_number"),
                end_page_number: index("end_page_number"),
            },
            "content_block_location" => Citation::ContentBlockLocation {
                cited_text: text(),
                document_index: index("document_index"),
                document_title: optional("document_title"),
                start_block_index: index("start_block_index"),
                end_block_index: index("end_block_index"),
            },
            "search_result_location" => Citation::SearchResultLocation {
                cited_text: text(),
                search_result_index: index("search_result_index"),
                source: optional_string(citation, "source"),
                title: optional("title"),
                start_block_index: index("start_block_index"),
                end_block_index: index("end_block_index"),
            },
            "web_search_result_location" => Citation::WebSearchResultLocation {
                cited_text: text(),
                url: optional_string(citation, "url"),
                title: optional("title"),
                encrypted_index: optional_string(citation, "encrypted_index"),
            },
            other => Citation::Unmodeled { kind: other.to_owned() },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn every_document_source_serializes_to_its_documented_shape() {
        let wire = |source| serde_json::to_value(DocumentBlock::new(source)).unwrap()["source"].clone();
        assert_eq!(
            wire(DocumentSource::text("The sky is blue.")),
            json!({"type": "text", "media_type": "text/plain", "data": "The sky is blue."})
        );
        assert_eq!(
            wire(DocumentSource::pdf("JVBERi0=")),
            json!({"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="})
        );
        assert_eq!(
            wire(DocumentSource::url("https://example.com/a.pdf")),
            json!({"type": "url", "url": "https://example.com/a.pdf"})
        );
        assert_eq!(wire(DocumentSource::file("file_1")), json!({"type": "file", "file_id": "file_1"}));
    }

    /// Citations are opt-in: a document that never asked for them is byte-identical
    /// to one built before this module existed, so its cache prefix still matches.
    #[test]
    fn citations_appear_only_where_a_caller_asked() {
        let plain = serde_json::to_value(DocumentBlock::new(DocumentSource::text("x"))).unwrap();
        assert!(plain.get("citations").is_none());
        assert!(plain.get("title").is_none());
        assert!(plain.get("context").is_none());

        let cited = serde_json::to_value(
            DocumentBlock::cited(DocumentSource::text("The sky is blue."))
                .with_title("Colors")
                .with_context("An excerpt from a field guide."),
        )
        .unwrap();
        assert_eq!(cited["citations"], json!({"enabled": true}));
        assert_eq!(cited["title"], "Colors");
        assert_eq!(cited["context"], "An excerpt from a field guide.");
    }

    /// A search result names its origin, which is what makes a citation to it
    /// meaningful, so both fields are required rather than optional.
    #[test]
    fn a_search_result_carries_its_origin_and_blocks() {
        let wire = serde_json::to_value(SearchResultBlock::cited(
            "https://example.com/doc",
            "Field guide",
            vec![TextBlock::new("The sky is blue."), TextBlock::new("Grass is green.")],
        ))
        .unwrap();
        assert_eq!(wire["source"], "https://example.com/doc");
        assert_eq!(wire["title"], "Field guide");
        assert_eq!(wire["content"][1]["text"], "Grass is green.");
        assert_eq!(wire["citations"], json!({"enabled": true}));
    }

    /// A citation captured live: the gateway answered a plain-text document with a
    /// character range naming the exact sentence.
    #[test]
    fn a_captured_char_location_names_the_exact_characters() {
        let citation = Citation::decode(&json!({
            "type": "char_location", "cited_text": "The sky is blue.", "document_index": 0,
            "document_title": "t", "start_char_index": 0, "end_char_index": 16
        }))
        .unwrap();
        assert_eq!(
            citation,
            Citation::CharLocation {
                cited_text: "The sky is blue.".to_owned(),
                document_index: 0,
                document_title: Some("t".to_owned()),
                start_char_index: 0,
                end_char_index: 16,
            }
        );
        assert_eq!(citation.kind(), "char_location");
        assert_eq!(citation.cited_text(), Some("The sky is blue."));
    }

    #[test]
    fn every_citation_kind_decodes_and_names_itself() {
        let cases = [
            (
                json!({"type": "page_location", "cited_text": "a", "start_page_number": 3, "end_page_number": 4}),
                "page_location",
            ),
            (
                json!({"type": "content_block_location", "cited_text": "a", "start_block_index": 1, "end_block_index": 2}),
                "content_block_location",
            ),
            (
                json!({"type": "search_result_location", "cited_text": "a", "search_result_index": 2, "source": "s"}),
                "search_result_location",
            ),
            (
                json!({"type": "web_search_result_location", "cited_text": "a", "url": "u", "encrypted_index": "e"}),
                "web_search_result_location",
            ),
        ];
        for (frame, kind) in cases {
            let citation = Citation::decode(&frame).unwrap();
            assert_eq!(citation.kind(), kind);
            assert_eq!(citation.cited_text(), Some("a"));
        }
    }

    /// A citation kind this crate does not know is not a broken frame; a citation
    /// that is not an object is.
    #[test]
    fn an_unmodeled_citation_decodes_but_a_malformed_one_does_not() {
        let citation = Citation::decode(&json!({"type": "constellation_location"})).unwrap();
        assert_eq!(citation, Citation::Unmodeled { kind: "constellation_location".to_owned() });
        assert_eq!(citation.cited_text(), None);
        assert!(matches!(Citation::decode(&json!([])), Err(FrameError::WrongType { field: "citation", .. })));
        assert!(matches!(Citation::decode(&json!({})), Err(FrameError::MissingField { field: "type" })));
    }

    /// A null title reads as absent, which is how the API spells "untitled".
    #[test]
    fn a_null_title_reads_as_absent() {
        let citation = Citation::decode(&json!({
            "type": "char_location", "cited_text": "a", "document_index": 0, "document_title": null,
            "start_char_index": 0, "end_char_index": 1
        }))
        .unwrap();
        assert!(matches!(citation, Citation::CharLocation { document_title: None, .. }));
    }
}
