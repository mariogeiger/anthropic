//! The content blocks a caller sends, and the shapes their fields take.
//!
//! The outbound counterpart of [`crate::content`], which holds what the model
//! *produces*. Two sets rather than one, because they are genuinely different: a
//! caller may send a document or a tool result, which a model never emits, and a
//! model emits server-tool blocks a caller never sends. A single union type would
//! admit both sets in both directions, and the API refuses that.
//!
//! # Cache metadata is not reachable from here
//!
//! Every block carries a `cache_control` slot and every one of them is
//! crate-private, with no public constructor for the value that fills it. A
//! breakpoint is placed only through a [`crate::context::CacheSlot`], which keeps
//! the slot bookkeeping and the content it points at from disagreeing. That is why
//! these structs have public content fields and no public `cache_control`: the
//! content is already exact, the metadata carries a cross-message invariant.
//!
//! # Whose job the `type` tag is
//!
//! [`ContentBlock`] writes the tag for the whole block, so the structs inside it
//! serialize their fields alone. The same structs appear in positions where no
//! enum writes a tag — a search result's content, the top-level system prompt —
//! and there they serialize with it. [`TextBlock`] is the one that sits in both,
//! which is why its own impl includes the tag and [`ContentBlock::Text`] overrides
//! that.

use serde::Serialize;
use serde_json::Value;

use crate::context::CacheControl;
use crate::document::{DocumentBlock, DocumentSource, SearchResultBlock};
use crate::{ImageMediaType, ImageOversize, TextBlockType};

/// Where an image's bytes come from.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Bytes inline, base64-encoded.
    Base64 {
        /// The image's media type. The enum, not the string it serializes to, so
        /// a format the API does not accept cannot be named.
        media_type: ImageMediaType,
        /// The base64 payload.
        data: String,
    },
    /// A URL the API fetches.
    Url {
        /// Where to fetch it.
        url: String,
    },
    /// A file already uploaded through the Files API.
    File {
        /// Its identifier.
        file_id: String,
    },
}

impl ImageSource {
    /// Inline base64 bytes of the given media type.
    pub fn base64(media_type: ImageMediaType, data: impl Into<String>) -> Self {
        ImageSource::Base64 { media_type, data: data.into() }
    }
    /// A URL for the API to fetch.
    pub fn url(url: impl Into<String>) -> Self {
        ImageSource::Url { url: url.into() }
    }
    /// A file already uploaded through the Files API.
    pub fn file(file_id: impl Into<String>) -> Self {
        ImageSource::File { file_id: file_id.into() }
    }
}

/// What a tool returned.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    /// Plain text, the common case.
    Text(String),
    /// Several blocks, for a result carrying images alongside text.
    Blocks(Vec<ToolResultItem>),
}

/// One block of a multi-block tool result.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultItem {
    /// Text.
    Text {
        /// The text.
        text: String,
    },
    /// An image.
    Image {
        /// Where its bytes come from.
        source: ImageSource,
    },
}

// ── Blocks ───────────────────────────────────────────────────────────────────
// Each variant wraps a `*Block` struct whose `cache_control` is crate-private:
// callers can read the content fields but cannot set, swap, or clone a
// `CacheControl` into place. The only path is through a `CacheSlot`.

/// A text block to send.
///
/// The one block type every position holds, which is why it is one struct shared
/// by [`ContentBlock::Text`], [`crate::system::SystemBlock::Text`], the top-level
/// system prompt, and a search result's content, rather than four near-copies.
///
/// It serializes *with* its own `type`, because three of those four positions
/// require one — a search result whose blocks omit it is a 400
/// (`search_result.content.0.type: Field required`). The exception is
/// [`ContentBlock`], whose enum writes the tag for the whole block, so its
/// variant does not go through this impl.
#[derive(Debug, Clone)]
pub struct TextBlock {
    /// The text.
    pub text: String,
    /// Sources grounding this assistant text when it is replayed.
    pub citations: Vec<crate::document::Citation>,
    pub(crate) cache_control: Option<CacheControl>,
}

/// Wire shape: the tag, then the text, then the breakpoint where there is one.
#[derive(Serialize)]
struct TextBlockWire<'a> {
    #[serde(rename = "type")]
    kind: TextBlockType,
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: &'a Option<CacheControl>,
}

impl Serialize for TextBlock {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        TextBlockWire { kind: TextBlockType::Text, text: &self.text, cache_control: &self.cache_control }.serialize(s)
    }
}

impl TextBlock {
    /// Text with no cache breakpoint. A breakpoint is placed only through a
    /// [`crate::context::CacheSlot`], so there is no constructor that takes one.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into(), citations: Vec::new(), cache_control: None }
    }

    /// Attaches the sources the model emitted beside this assistant text.
    pub fn with_citations(mut self, citations: Vec<crate::document::Citation>) -> Self {
        self.citations = citations;
        self
    }
}

/// An image block to send.
#[derive(Debug, Clone)]
pub struct ImageBlock {
    /// Where the image's bytes come from.
    pub source: ImageSource,
    /// What the server does with an image larger than the model accepts.
    ///
    /// A real runtime distinction, not a defaulted scalar: absent means the
    /// server's own behavior, which is to scale the image down *without saying
    /// so*, and `Some(ImageOversize::Error)` is a caller asking to be told
    /// instead. Naming a policy and inheriting one are different requests, so the
    /// absent case stays absent.
    ///
    /// Measured: the NVIDIA inference gateway refuses the `transformations` object
    /// this becomes, with `messages.0.content.0.image.transformations: Extra inputs
    /// are not permitted`. It is in the documented stable schema, so it is here.
    pub oversized: Option<ImageOversize>,
    pub(crate) cache_control: Option<CacheControl>,
}

/// Wire shape: `transformations` nests the policy the API names by condition.
#[derive(Serialize)]
struct ImageTransformations {
    oversized_image: ImageOversize,
}

#[derive(Serialize)]
struct ImageBlockWire<'a> {
    source: &'a ImageSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    transformations: Option<ImageTransformations>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: &'a Option<CacheControl>,
}

impl Serialize for ImageBlock {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        ImageBlockWire {
            source: &self.source,
            transformations: self.oversized.map(|oversized_image| ImageTransformations { oversized_image }),
            cache_control: &self.cache_control,
        }
        .serialize(s)
    }
}

/// A tool call to replay into the conversation.
///
/// Sent back on the turn after the model made it, so that the tool result has a
/// call to answer.
#[derive(Debug, Clone, Serialize)]
pub struct ToolUseBlock {
    /// The identifier the matching [`ToolResultBlock`] repeats.
    pub id: String,
    /// Which tool was called.
    pub name: String,
    /// The input it was called with.
    pub input: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

/// A tool's answer to one call.
#[derive(Debug, Clone, Serialize)]
pub struct ToolResultBlock {
    /// The [`ToolUseBlock::id`] this answers.
    pub tool_use_id: String,
    /// What the tool returned.
    pub content: ToolResultContent,
    /// Whether the tool failed. A plain `bool`, not an `Option`: every result
    /// either succeeded or did not, so there is no third state to represent.
    pub is_error: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

/// A thinking block to replay into the conversation.
///
/// Sent back with its signature so the API can verify it. Thinking blocks cannot
/// carry a breakpoint of their own, but they *are* cached as part of a prefix that
/// ends after them.
#[derive(Debug, Clone, Serialize)]
pub struct ThinkingBlock {
    /// The reasoning text, empty when it was produced under
    /// [`crate::values::ThinkingDisplay::Omitted`].
    pub thinking: String,
    /// The signature the API issued for it.
    pub signature: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

impl ThinkingBlock {
    /// Replays one complete thinking block exactly as the model returned it.
    pub fn replay(thinking: impl Into<String>, signature: impl Into<String>) -> Self {
        Self { thinking: thinking.into(), signature: signature.into(), cache_control: None }
    }
}

/// A redacted thinking block to replay into the conversation, opaque bytes and all.
#[derive(Debug, Clone, Serialize)]
pub struct RedactedThinkingBlock {
    /// The opaque payload.
    pub data: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

impl RedactedThinkingBlock {
    /// Replays one redacted thinking block without interpreting its payload.
    pub fn replay(data: impl Into<String>) -> Self {
        Self { data: data.into(), cache_control: None }
    }
}

/// One block of content to send.
///
/// The outbound counterpart of [`crate::content::StreamedBlock`]: this one carries
/// cache-breakpoint metadata and models what a caller may put *into* a prompt,
/// which is not the same set the model produces.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text.
    ///
    /// Serialized through `ContentBlockTextWire` rather than [`TextBlock`]'s own
    /// impl, because this enum already writes the `type` tag and two would be one
    /// too many.
    #[serde(serialize_with = "serialize_content_text")]
    Text(TextBlock),
    /// An image.
    Image(ImageBlock),
    /// A tool call being replayed.
    ToolUse(ToolUseBlock),
    /// A tool's answer to one.
    ToolResult(ToolResultBlock),
    /// A thinking block being replayed.
    Thinking(ThinkingBlock),
    /// A redacted thinking block being replayed.
    RedactedThinking(RedactedThinkingBlock),
    /// Source material the model may read and, with citations enabled, quote.
    Document(DocumentBlock),
    /// Material a search returned, carrying where it came from.
    SearchResult(SearchResultBlock),
}

/// A text block's fields without its tag, for the position whose enum writes one.
#[derive(Serialize)]
struct UntaggedTextBlock<'a> {
    text: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    citations: &'a Vec<crate::document::Citation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: &'a Option<CacheControl>,
}

fn serialize_content_text<S: serde::Serializer>(block: &TextBlock, s: S) -> Result<S::Ok, S::Error> {
    UntaggedTextBlock { text: &block.text, citations: &block.citations, cache_control: &block.cache_control }
        .serialize(s)
}

impl ContentBlock {
    /// A text block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(TextBlock::new(text))
    }
    /// An image block, leaving the oversize policy to the server.
    pub fn image(source: ImageSource) -> Self {
        Self::Image(ImageBlock { source, oversized: None, cache_control: None })
    }
    /// An image block that states what to do if the image is too large.
    ///
    /// [`ImageOversize::Error`] is the reason to reach for this: the default
    /// silently rescales, so the model observes dimensions the caller never chose.
    pub fn image_sized(source: ImageSource, oversized: ImageOversize) -> Self {
        Self::Image(ImageBlock { source, oversized: Some(oversized), cache_control: None })
    }
    /// A tool call being replayed.
    pub fn tool_use(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        Self::ToolUse(ToolUseBlock { id: id.into(), name: name.into(), input, cache_control: None })
    }
    /// A successful tool result.
    pub fn tool_result(tool_use_id: impl Into<String>, content: ToolResultContent) -> Self {
        Self::ToolResult(ToolResultBlock {
            tool_use_id: tool_use_id.into(),
            content,
            is_error: false,
            cache_control: None,
        })
    }
    /// Source material the model may read but not cite.
    pub fn document(source: DocumentSource) -> Self {
        Self::Document(DocumentBlock::new(source))
    }
    /// Source material the model may quote, emitting a citation where it does.
    ///
    /// Takes the source rather than a built [`DocumentBlock`] for the common case;
    /// use [`Self::Document`] directly to title or describe it first.
    pub fn document_cited(source: DocumentSource) -> Self {
        Self::Document(DocumentBlock::cited(source))
    }
    /// A search result the model may read, and quote where it was built with
    /// [`SearchResultBlock::cited`].
    pub fn search_result(block: SearchResultBlock) -> Self {
        Self::SearchResult(block)
    }
    /// A failed tool result, which the model is told about via `is_error`.
    pub fn tool_result_err(tool_use_id: impl Into<String>, content: ToolResultContent) -> Self {
        Self::ToolResult(ToolResultBlock {
            tool_use_id: tool_use_id.into(),
            content,
            is_error: true,
            cache_control: None,
        })
    }

    pub(crate) fn cache_control_mut(&mut self) -> &mut Option<CacheControl> {
        match self {
            Self::Text(b) => &mut b.cache_control,
            Self::Image(b) => &mut b.cache_control,
            Self::ToolUse(b) => &mut b.cache_control,
            Self::ToolResult(b) => &mut b.cache_control,
            Self::Thinking(b) => &mut b.cache_control,
            Self::RedactedThinking(b) => &mut b.cache_control,
            Self::Document(b) => &mut b.cache_control,
            Self::SearchResult(b) => &mut b.cache_control,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Context, Opening};
    use crate::request::{Model, Request};

    /// Blocks are only observable through a request, so that is how they are read
    /// back: the wire is the thing under test.
    fn request_of(ctx: &Context) -> serde_json::Value {
        serde_json::to_value(Request::new(ctx, Model::opus_4_8(), 1024).unwrap()).unwrap()
    }

    #[test]
    fn image_media_type_serializes_from_the_enum() {
        let mut ctx = Context::new(Opening::None);
        ctx.push_user(vec![ContentBlock::image(ImageSource::base64(ImageMediaType::Png, "aGk="))]);
        let v = request_of(&ctx);
        assert_eq!(v["messages"][0]["content"][0]["source"]["media_type"], "image/png");
        assert_eq!(v["messages"][0]["content"][0]["source"]["type"], "base64");
    }

    #[test]
    fn tool_result_is_error_emitted_as_bool() {
        let mut ctx = Context::new(Opening::None);
        ctx.push_user(vec![
            ContentBlock::tool_result("tu_1", ToolResultContent::Text("ok".into())),
            ContentBlock::tool_result_err("tu_2", ToolResultContent::Text("oops".into())),
        ]);
        let v = request_of(&ctx);
        assert_eq!(v["messages"][0]["content"][0]["is_error"], false);
        assert_eq!(v["messages"][0]["content"][1]["is_error"], true);
    }

    /// The default silently rescales an oversized image, so asking to be told
    /// instead is a different request — and absent stays absent.
    #[test]
    fn an_image_states_its_oversize_policy_only_when_asked() {
        let mut ctx = Context::new(Opening::None);
        ctx.push_user(vec![ContentBlock::image(ImageSource::base64(ImageMediaType::Png, "aGk="))]);
        assert!(
            request_of(&ctx)["messages"][0]["content"][0].get("transformations").is_none(),
            "no policy named, no policy sent"
        );

        let mut ctx = Context::new(Opening::None);
        ctx.push_user(vec![ContentBlock::image_sized(
            ImageSource::base64(ImageMediaType::Png, "aGk="),
            ImageOversize::Error,
        )]);
        let v = request_of(&ctx);
        assert_eq!(v["messages"][0]["content"][0]["transformations"]["oversized_image"], "error");
        assert_eq!(v["messages"][0]["content"][0]["type"], "image", "the tag survives the manual serializer");
        assert_eq!(v["messages"][0]["content"][0]["source"]["media_type"], "image/png");
    }
}
