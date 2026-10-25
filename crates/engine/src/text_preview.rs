//! The page beneath an inline text editor, with the source glyphs removed on a COS clone.

use std::sync::Arc;

use pdfcraft_render::{PageRenderer, RenderConfig, RenderPool, RenderRequest, RenderedPage};

use crate::{Document, guard};

pub(crate) struct PreparedFonts {
    generation: u64,
    page: usize,
    block: usize,
    session: pdfcraft_edit::text::TextSession,
    programs: Vec<(String, String, Vec<u8>)>,
    attempted: Vec<(String, String)>,
    names: Vec<String>,
}

impl PreparedFonts {
    fn prepare(&mut self, fonts: &crate::text_fonts::Catalogue, text: &str, style: &pdfcraft_edit::BlockStyle) -> Result<Vec<String>, String> {
        let missing = self.session.missing_source_glyphs(text, style).map_err(|e| e.to_string())?;
        if missing == self.attempted {
            return Ok(self.names.clone());
        }
        let covers = |source: &str, absent: &str, face: &str, bytes: &[u8]| {
            (pdfcraft_fonts::embedding::same_face_name(face, source)
                || pdfcraft_fonts::embedding::face_identity(face) == pdfcraft_fonts::embedding::face_identity(source))
                && pdfcraft_fonts::embedding::covers(bytes, absent)
        };
        {
            let outstanding: Vec<_> = missing
                .iter()
                .filter(|(source, absent)| !self.programs.iter().any(|(_, face, bytes)| covers(source, absent, face, bytes)))
                .cloned()
                .collect();
            let mut budget = (32usize << 20).saturating_sub(self.programs.iter().fold(0usize, |sum, (_, _, bytes)| sum.saturating_add(bytes.len())));
            for (name, bytes) in fonts.matching_programs(&outstanding) {
                if bytes.len() > budget {
                    continue;
                }
                self.session.register_font(&bytes, fonts.grant()).map_err(|e| e.to_string())?;
                budget -= bytes.len();
                let face = pdfcraft_fonts::embedding::document_metadata(&bytes, None)?.postscript;
                self.programs.push((name, face, bytes));
            }
            self.attempted = missing.clone();
        }
        self.names = self
            .programs
            .iter()
            .filter(|(_, face, bytes)| missing.iter().any(|(source, absent)| covers(source, absent, face, bytes)))
            .map(|(name, _, _)| name.clone())
            .collect();
        Ok(self.names.clone())
    }
}

pub struct TextEditPreview {
    page: usize,
    bytes: Arc<Vec<u8>>,
    config: RenderConfig,
}

impl TextEditPreview {
    pub fn renderer(&self) -> RenderPool {
        RenderPool::new(self.bytes.clone(), 1, self.config.clone())
    }

    pub fn render_background(&self, scale: f32) -> Result<RenderedPage, String> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err("preview scale must be finite and positive".into());
        }
        guard(|| {
            let mut renderer = PageRenderer::new(self.bytes.clone(), self.config.clone());
            let out = renderer.render(RenderRequest { page: self.page, scale, ..Default::default() });
            if let Some(e) = &out.error {
                return Err(e.clone());
            }
            Ok(out)
        })?
    }
}

impl Document {
    /// Background for retyping an added text item, without changing its saved content.
    pub fn added_text_preview(&self, page: usize, index: usize) -> Result<TextEditPreview, String> {
        guard(|| {
            if !self.added().iter().filter(|a| a.page == page).nth(index).is_some_and(|a| matches!(a.content, crate::AddedContent::Text(_))) {
                return Err("the selected added item is not text".into());
            }
            let editor = self.editor.as_ref().ok_or("the document can't be read")?;
            let mut cos = editor.cos.clone();
            pdfcraft_edit::delete_content(&mut cos, page, index).map_err(|e| e.to_string())?;
            let bytes = Arc::new(pdfcraft_cos::write_incremental(&cos, &pdfcraft_cos::SaveOptions::default()).map_err(|e| e.to_string())?);
            Ok(TextEditPreview { page, bytes, config: self.config.clone() })
        })?
    }

    /// Faces a draft would register; does not change the document or undo history.
    pub fn matching_font_completions(&self, page: usize, block: usize, text: &str, style: &pdfcraft_edit::BlockStyle) -> Result<Vec<String>, String> {
        guard(|| {
            let Some(fonts) = &self.text_fonts else {
                return Ok(Vec::new());
            };
            self.with_prepared_fonts(page, block, |prepared| prepared.prepare(fonts, text, style))
        })?
    }

    /// Measure a draft without changing bytes, dirty state, or undo history.
    pub fn text_edit_layout(
        &self,
        page: usize,
        block: usize,
        text: &str,
        style: &pdfcraft_edit::BlockStyle,
    ) -> Result<pdfcraft_edit::text::TextEditLayout, String> {
        guard(|| {
            self.with_prepared_fonts(page, block, |prepared| {
                if let Some(fonts) = &self.text_fonts {
                    prepared.prepare(fonts, text, style)?;
                }
                prepared.session.layout(text, style).map_err(|e| e.to_string())
            })
        })?
    }

    fn with_prepared_fonts<T>(&self, page: usize, block: usize, run: impl FnOnce(&mut PreparedFonts) -> Result<T, String>) -> Result<T, String> {
        let editor = self.editor.as_ref().ok_or("the document can't be read")?;
        let mut cache = self.text_font_preview.lock().map_err(|_| "the text font preview cache is unavailable")?;
        if !cache.as_ref().is_some_and(|c| c.generation == self.generation && c.page == page && c.block == block) {
            #[cfg(test)]
            record_preview_open();
            *cache = Some(PreparedFonts {
                generation: self.generation,
                page,
                block,
                session: pdfcraft_edit::text::TextSession::open(&editor.cos, page, block).map_err(|e| e.to_string())?,
                programs: Vec::new(),
                attempted: Vec::new(),
                names: Vec::new(),
            });
        }
        match cache.as_mut() {
            Some(prepared) => run(prepared),
            None => Err("the text font preview could not be prepared".into()),
        }
    }

    /// Release the bounded preview when the editor closes or a document operation invalidates it.
    pub fn clear_text_font_preview(&self) {
        if let Ok(mut cache) = self.text_font_preview.lock() {
            *cache = None;
        }
    }

    /// Identifies the bounded paragraph preview's registered font set. The desktop can
    /// refresh its font stack only after a matching full face has actually been added.
    pub fn text_font_preview_key(&self) -> Option<(u64, usize, usize, usize)> {
        self.text_font_preview
            .lock()
            .ok()?
            .as_ref()
            .filter(|c| c.generation == self.generation)
            .map(|c| (c.generation, c.page, c.block, c.programs.len()))
    }

    /// Embedded source faces for a text editor, bounded and validated as sfnt programs.
    /// This is read-only; no source font is installed or exported.
    pub fn text_editor_fonts(&self, page: usize) -> Vec<pdfcraft_edit::text::TextEditorFont> {
        if let Ok(cache) = self.text_font_preview.lock()
            && let Some(prepared) = cache.as_ref().filter(|c| c.generation == self.generation && c.page == page && !c.programs.is_empty())
        {
            return prepared.session.editor_fonts().unwrap_or_default();
        }
        self.editor.as_ref().and_then(|e| pdfcraft_edit::text::text_editor_fonts(&e.cos, page).ok()).unwrap_or_default()
    }

    /// Preparing the background never changes working bytes, dirty state or undo history.
    pub fn text_edit_preview(&self, page: usize, block: usize) -> Result<TextEditPreview, String> {
        guard(|| {
            let editor = self.editor.as_ref().ok_or("the document can't be read")?;
            let mut cos = editor.cos.clone();
            pdfcraft_edit::clear_block(&mut cos, page, block).map_err(|e| e.to_string())?;
            let bytes = Arc::new(pdfcraft_cos::write_incremental(&cos, &pdfcraft_cos::SaveOptions::default()).map_err(|e| e.to_string())?);
            Ok(TextEditPreview { page, bytes, config: self.config.clone() })
        })?
    }

    /// The page beneath an added item while it is dragged: `index` counts the page's added
    /// items, as `Edit::DeleteContent` does. Read-only, like [`Self::text_edit_preview`].
    pub fn added_content_preview(&self, page: usize, index: usize) -> Result<TextEditPreview, String> {
        guard(|| {
            let editor = self.editor.as_ref().ok_or("the document can't be read")?;
            let mut cos = editor.cos.clone();
            pdfcraft_edit::delete_content(&mut cos, page, index).map_err(|e| e.to_string())?;
            let bytes = Arc::new(pdfcraft_cos::write_incremental(&cos, &pdfcraft_cos::SaveOptions::default()).map_err(|e| e.to_string())?);
            Ok(TextEditPreview { page, bytes, config: self.config.clone() })
        })?
    }
}

impl crate::Session {
    /// Enable automatic subset completion for all current and future documents. The caller
    /// chooses trusted discovery roots; automation must confine them before scanning.
    /// This changes neither PDF bytes, dirty state nor undo history.
    pub fn set_text_font_catalogue(&mut self, fonts: Arc<crate::text_fonts::Catalogue>) {
        if self.text_fonts.as_ref().is_some_and(|current| Arc::ptr_eq(current, &fonts)) {
            return;
        }
        self.text_fonts = Some(fonts.clone());
        for doc in &mut self.docs {
            doc.text_fonts = Some(fonts.clone());
            // A refreshed discovery scope must not reuse fonts from the previous scope.
            doc.text_font_preview = Default::default();
        }
    }

    pub fn clear_text_font_catalogue(&mut self) {
        self.text_fonts = None;
        for doc in &mut self.docs {
            doc.text_fonts = None;
            doc.text_font_preview = Default::default();
        }
    }

    pub fn has_text_font_catalogue(&self) -> bool {
        self.text_fonts.is_some()
    }
}

#[cfg(test)]
thread_local! {
    static PREVIEW_OPENS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_preview_open() {
    PREVIEW_OPENS.with(|count| count.set(count.get().saturating_add(1)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_keystrokes_reuse_source_lines_and_invalidate_on_generation_or_paragraph() {
        let mut session = crate::Session::new();
        let id = session.open("synthetic.pdf", None, Arc::new(crate::tests::fixture(2)), None).unwrap();
        session.set_text_font_catalogue(Arc::new(crate::text_fonts::Catalogue::discover(&[], crate::FontGrant::InstalledEmbeddable)));
        PREVIEW_OPENS.with(|count| count.set(0));
        let style = pdfcraft_edit::BlockStyle::default();
        for text in ["Page 1", "Page 11", "1 Page", "1 Page 11"] {
            let doc = session.get(id).unwrap();
            doc.text_edit_layout(0, 0, text, &style).unwrap();
            assert!(doc.matching_font_completions(0, 0, text, &style).unwrap().is_empty());
        }
        assert_eq!(PREVIEW_OPENS.with(std::cell::Cell::get), 1, "source extraction and its snapshot are reused between keystrokes");
        assert!(!session.get(id).unwrap().dirty);
        assert!(session.get(id).unwrap().can_undo().is_none());
        session.get(id).unwrap().text_edit_layout(1, 0, "Page 2", &style).unwrap();
        assert_eq!(PREVIEW_OPENS.with(std::cell::Cell::get), 2);
        session.apply(id, crate::Edit::RotatePages { pages: vec![1], degrees: 90 }).unwrap();
        session.get(id).unwrap().text_edit_layout(1, 0, "Page 2", &style).unwrap();
        assert_eq!(PREVIEW_OPENS.with(std::cell::Cell::get), 3, "the new document generation invalidates the source cache");
        session.clear_text_font_catalogue();
        assert!(session.get(id).unwrap().text_font_preview_key().is_none());
    }
}
