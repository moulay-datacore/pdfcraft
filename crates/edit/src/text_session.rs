//! A text editing session for one paragraph (#2512). Opening it reads the page once; each draft
//! is laid out from that reading, on a copy of the document as it was opened, so no draft sees
//! another one's output; the commit writes into the document it was opened on, and refuses if
//! that document has changed since.

use super::*;

/// One paragraph being edited: the document as it was opened, and the page's lines then.
pub struct TextSession {
    base: Document,
    base_state: pdfcraft_cos::ReadState,
    page: usize,
    block: usize,
    lines: Vec<TextLine>,
    source: TextBlock,
    missing: Option<MissingSourceGlyphs>,
}

impl TextSession {
    /// Opens paragraph `block` on `page` (both 0-based) of `doc`.
    pub fn open(doc: &Document, page: usize, block: usize) -> Result<Self, EditError> {
        let lines = text_lines(doc, page)?;
        let source = group_blocks(&lines)
            .get(block)
            .cloned()
            .ok_or_else(|| EditError::Invalid(format!("page {} has no paragraph {}", page + 1, block + 1)))?;
        Ok(Self { base: doc.clone(), base_state: doc.read_state(), page, block, lines, source, missing: None })
    }

    /// Missing characters by source face, cached per character for this paragraph. Font
    /// selection overrides do not use automatic subset completion.
    pub fn missing_source_glyphs(&mut self, text: &str, style: &BlockStyle) -> Result<Vec<(String, String)>, EditError> {
        if text.len() > 262_144 {
            return Err(EditError::Invalid("the replacement text exceeds 256 KiB".into()));
        }
        if style.resource.is_some() || style.family.is_some() || style.bold.is_some() {
            return Ok(Vec::new());
        }
        if self.missing.is_none() {
            self.missing = Some(MissingSourceGlyphs::new(&self.base, self.page, &self.lines, &self.source)?);
        }
        match &mut self.missing {
            Some(coverage) => coverage.missing(text),
            None => Ok(Vec::new()),
        }
    }

    /// Adds a validated font program to the preview's resources. Source lines remain valid:
    /// registering a font changes neither content operators nor their locations.
    pub fn register_font(&mut self, bytes: &[u8], grant: pdfcraft_fonts::embedding::FontGrant) -> Result<(), EditError> {
        register_text_font(&mut self.base, self.page, bytes, grant)?;
        Ok(())
    }

    /// Fonts actually registered in this paragraph's preview, for the inline editor.
    pub fn editor_fonts(&self) -> Result<Vec<TextEditorFont>, EditError> {
        text_editor_fonts(&self.base, self.page)
    }

    /// The paragraph as it was when the session opened.
    pub fn source(&self) -> &TextBlock {
        &self.source
    }

    pub fn page(&self) -> usize {
        self.page
    }

    pub fn block(&self) -> usize {
        self.block
    }

    /// Whether `doc` is still the document the session was opened on (nothing applied since).
    pub fn is_current(&self, doc: &Document) -> bool {
        self.base_state.matches(doc)
    }

    /// Lays a draft out without writing it: the rewrite Apply runs, on a copy of the opened document.
    pub fn layout(&self, text: &str, style: &BlockStyle) -> Result<TextEditLayout, EditError> {
        layout_draft(&self.base, self.page, &self.lines, &self.source, self.block, text, style)
    }

    /// Writes the draft into `doc`, which must still be the document the session was opened on.
    pub fn commit(&self, doc: &mut Document, text: Option<&str>, style: &BlockStyle) -> Result<LineEdit, EditError> {
        self.commit_with_limit(doc, text, style, MAX_TYPE3_GLYPHS)
    }

    pub(crate) fn commit_with_limit(
        &self,
        doc: &mut Document,
        text: Option<&str>,
        style: &BlockStyle,
        max_glyphs: usize,
    ) -> Result<LineEdit, EditError> {
        if !self.is_current(doc) {
            return Err(EditError::Invalid("the paragraph changed while it was being edited".into()));
        }
        rewrite_block_from(doc, self.page, self.lines.clone(), self.block, text, style, max_glyphs)
    }
}
