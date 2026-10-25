//! The page beneath an inline text editor, with the source glyphs removed on a COS clone.

use std::sync::Arc;

use pdfcraft_render::{PageRenderer, RenderConfig, RenderPool, RenderRequest, RenderedPage};

use crate::{Document, guard};

pub(crate) struct PreparedFonts {
    generation: u64,
    page: usize,
    cos: pdfcraft_cos::Document,
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
            let editor = self.editor.as_ref().ok_or("the document can't be read")?;
            let Some(fonts) = &self.text_fonts else {
                return Ok(Vec::new());
            };
            fonts.complete(&mut editor.cos.clone(), page, block, text, style).map_err(|e| e.to_string())
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
            let editor = self.editor.as_ref().ok_or("the document can't be read")?;
            if let Some(fonts) = &self.text_fonts {
                // Reuse the completed preview resources between keystrokes. Embedding and
                // building a full CJK map must not run again for every character typed.
                let mut cache = self.text_font_preview.lock().map_err(|_| "the text font preview cache is unavailable")?;
                let mut cos = cache
                    .as_ref()
                    .filter(|c| c.generation == self.generation && c.page == page)
                    .map_or_else(|| editor.cos.clone(), |c| c.cos.clone());
                fonts.complete(&mut cos, page, block, text, style).map_err(|e| e.to_string())?;
                *cache = Some(PreparedFonts { generation: self.generation, page, cos: cos.clone() });
                drop(cache);
                pdfcraft_edit::text::text_edit_layout(&cos, page, block, text, style).map_err(|e| e.to_string())
            } else {
                pdfcraft_edit::text::text_edit_layout(&editor.cos, page, block, text, style).map_err(|e| e.to_string())
            }
        })?
    }

    /// Embedded source faces for a text editor, bounded and validated as sfnt programs.
    /// This is read-only; no source font is installed or exported.
    pub fn text_editor_fonts(&self, page: usize) -> Vec<pdfcraft_edit::text::TextEditorFont> {
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
