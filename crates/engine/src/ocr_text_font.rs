//! Session-only replacement face for text recognized over a scanned image.

use std::sync::Arc;

use pdfcraft_cos::{Dict, Document as CosDocument, Object};
use pdfcraft_edit::BlockStyle;

use crate::{Document, EditError};

/// Stored with each undo snapshot. A resource keeps its object identity, so a font chosen on
/// one page can be used on another without reading a file or changing the source page.
#[derive(Clone)]
pub(crate) enum OcrFontChoice {
    Family(pdfcraft_edit::Family, bool, bool),
    Resource { object: Object, source: Arc<CosDocument> },
}

fn page_fonts(doc: &CosDocument, page: usize) -> Result<(pdfcraft_model::Page, Dict, Dict), EditError> {
    let p = pdfcraft_model::pages(doc).into_iter().nth(page).ok_or_else(|| EditError::Invalid("the font's page no longer exists".into()))?;
    let resources = p.dict.get(b"Resources").and_then(|o| doc.dict(o)).unwrap_or_default();
    let fonts = resources.get(b"Font").and_then(|o| doc.dict(o)).unwrap_or_default();
    Ok((p, resources, fonts))
}

impl OcrFontChoice {
    fn selected(doc: &CosDocument, page: usize, style: &BlockStyle) -> Result<Option<Self>, EditError> {
        if let Some(name) = &style.resource {
            let (_, _, fonts) = page_fonts(doc, page)?;
            let object =
                fonts.get(name.as_bytes()).cloned().ok_or_else(|| EditError::Invalid(format!("the selected font resource {name} is missing")))?;
            return Ok(Some(Self::Resource { object, source: Arc::new(doc.clone()) }));
        }
        Ok(style.family.map(|(family, bold, italic)| Self::Family(family, bold, italic)))
    }

    fn style(&self, doc: &CosDocument, page: usize) -> Result<BlockStyle, EditError> {
        match self {
            Self::Family(family, bold, italic) => Ok(BlockStyle { family: Some((*family, *bold, *italic)), ..Default::default() }),
            Self::Resource { object, source } => {
                let (_, _, fonts) = page_fonts(doc, page)?;
                let existing = if Arc::ptr_eq(doc.bytes(), source.bytes()) {
                    fonts.iter().filter(|(_, font)| *font == object).find_map(|(name, _)| std::str::from_utf8(name).ok().map(str::to_owned))
                } else {
                    None
                };
                let name = existing
                    .or_else(|| (0..1024).map(|n| format!("PCOCRChoice{n}")).find(|name| fonts.get(name.as_bytes()).is_none()))
                    .ok_or_else(|| EditError::Invalid("this page has too many OCR font resources".into()))?;
                Ok(BlockStyle { resource: Some(name), ..Default::default() })
            }
        }
    }

    /// Full saves renumber objects. Follow a page resource alias into the rebased graph,
    /// rather than letting a stale font reference point at an unrelated saved object.
    pub(crate) fn rebase(&self, before: &CosDocument, after: &CosDocument) -> Self {
        match self {
            Self::Family(..) => self.clone(),
            Self::Resource { object, source } => {
                // If the font has been orphaned and a full save removed it, its immutable
                // snapshot remains the source until a later OCR edit imports it again.
                if !Arc::ptr_eq(before.bytes(), source.bytes()) {
                    return self.clone();
                }
                for (page, p) in pdfcraft_model::pages(before).into_iter().enumerate() {
                    let resources = p.dict.get(b"Resources").and_then(|o| before.dict(o)).unwrap_or_default();
                    let fonts = resources.get(b"Font").and_then(|o| before.dict(o)).unwrap_or_default();
                    if let Some((name, _)) = fonts.iter().find(|(_, font)| *font == object)
                        && let Ok((_, _, saved_fonts)) = page_fonts(after, page)
                        && let Some(object) = saved_fonts.get(name).cloned()
                    {
                        return Self::Resource { object, source: Arc::new(after.clone()) };
                    }
                }
                self.clone()
            }
        }
    }

    /// Installs only on a transaction or preview clone. Existing resource names never change.
    pub(crate) fn install(&self, doc: &mut CosDocument, page: usize) -> Result<BlockStyle, EditError> {
        let style = self.style(doc, page)?;
        if let (Self::Resource { object, source }, Some(name)) = (self, style.resource.as_ref()) {
            let (p, mut resources, mut fonts) = page_fonts(doc, page)?;
            let same_graph = Arc::ptr_eq(doc.bytes(), source.bytes());
            if !same_graph || fonts.get(name.as_bytes()) != Some(object) {
                let object = if same_graph { object.clone() } else { pdfcraft_organize::import_font_resource(doc, source, object)? };
                fonts.set(name.as_bytes().to_vec(), object);
                resources.set(b"Font".to_vec(), Object::Dict(fonts));
                doc.update_dict(p.obj, |d| d.set(b"Resources".to_vec(), Object::Dict(resources))).map_err(pdfcraft_edit::EditError::from)?;
            }
        }
        Ok(style)
    }
}

fn merge_font(style: &BlockStyle, choice: BlockStyle) -> BlockStyle {
    let mut effective = style.clone();
    if effective.family.is_none() && effective.resource.is_none() {
        effective.family = choice.family.map(|(family, bold, italic)| (family, effective.bold.unwrap_or(bold), italic));
        effective.resource = choice.resource;
    }
    effective
}

pub(crate) fn prepare_edit(
    doc: &mut CosDocument,
    page: usize,
    block: usize,
    style: &BlockStyle,
    remembered: &mut Option<OcrFontChoice>,
) -> Result<BlockStyle, EditError> {
    if !pdfcraft_edit::text_blocks(doc, page)?.get(block).is_some_and(|b| b.invisible) {
        return Ok(style.clone());
    }
    let selected = match OcrFontChoice::selected(doc, page, style) {
        Ok(selected) => selected,
        Err(error) => {
            // The OCR-only inventory may expose the remembered face under an alias that
            // does not exist on this page yet. Only that exact choice may materialize it.
            if let Some(choice) = remembered.as_ref()
                && style.resource.is_some()
                && choice.style(doc, page)?.resource == style.resource
            {
                choice.install(doc, page)?;
                OcrFontChoice::selected(doc, page, style)?
            } else {
                return Err(error);
            }
        }
    };
    if let Some(selected) = selected {
        *remembered = Some(selected);
        return Ok(style.clone());
    }
    let Some(choice) = remembered.as_ref() else { return Ok(style.clone()) };
    let effective = merge_font(style, choice.install(doc, page)?);
    // Remember the installed copy's identity, avoiding duplicate imports on later edits.
    *remembered = OcrFontChoice::selected(doc, page, &effective)?;
    Ok(effective)
}

impl Document {
    /// Last explicitly chosen replacement face for scans in this document. This contains only
    /// font selection; callers must apply it only to invisible OCR blocks. It is never saved.
    pub fn ocr_text_style(&self, page: usize) -> BlockStyle {
        crate::guard(|| self.editor.as_ref().and_then(|e| e.ocr_font.as_ref().and_then(|f| f.style(&e.cos, page).ok())).unwrap_or_default())
            .unwrap_or_default()
    }

    pub(crate) fn effective_ocr_style(&self, page: usize, block: usize, style: &BlockStyle) -> Result<BlockStyle, String> {
        let editor = self.editor.as_ref().ok_or("the document can't be read")?;
        if editor.text_blocks(page).map_err(|e| e.to_string())?.get(block).is_some_and(|b| b.invisible) {
            let choice = editor.ocr_font.as_ref().map(|f| f.style(&editor.cos, page)).transpose().map_err(|e| e.to_string())?.unwrap_or_default();
            Ok(merge_font(style, choice))
        } else {
            Ok(style.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pdfcraft_cos::{Dict, Object, SaveOptions, Stream};

    use super::*;
    use crate::{Edit, FontFamily, Session};

    /// Three generated pages: two scans with deliberately uneven word matrices, then a
    /// native paragraph. The image is generated RGB data, never a user document or asset.
    fn scan_fixture() -> Arc<Vec<u8>> {
        let mut cos = CosDocument::open(Arc::new(crate::tests::fixture(3))).unwrap();
        let mut image = Dict::new();
        image.set(b"Type", Object::name("XObject"));
        image.set(b"Subtype", Object::name("Image"));
        image.set(b"Width", Object::Int(8));
        image.set(b"Height", Object::Int(8));
        image.set(b"ColorSpace", Object::name("DeviceRGB"));
        image.set(b"BitsPerComponent", Object::Int(8));
        let image = cos.add(Object::Stream(Stream::from_raw(image, vec![240; 8 * 8 * 3])));
        for page in pdfcraft_model::pages(&cos).into_iter().take(2) {
            let mut resources = page.dict.get(b"Resources").and_then(|o| cos.dict(o)).unwrap();
            let mut xobjects = Dict::new();
            xobjects.set(b"Scan", Object::Ref(image));
            resources.set(b"XObject", Object::Dict(xobjects));
            let ops = b"q 200 0 0 300 0 0 cm /Scan Do Q /PCMark /OCR BDC BT /F1 1 Tf 3 Tr 11.03 0 0 13.29 20 150 Tm (Lorem) Tj 11.34 0 0 11.25 55 150.4 Tm (ipsum) Tj ET EMC";
            let content = cos.add(Object::Stream(Stream::from_raw(Dict::new(), ops.to_vec())));
            cos.update_dict(page.obj, |d| {
                d.set(b"Contents", Object::Ref(content));
                d.set(b"Resources", Object::Dict(resources));
            })
            .unwrap();
        }
        Arc::new(pdfcraft_cos::write_incremental(&cos, &SaveOptions::default()).unwrap())
    }

    fn edit(s: &mut Session, id: crate::DocId, page: usize, style: BlockStyle) {
        s.apply(id, Edit::EditTextBlock { page, block: 0, text: "New words".into(), style }).unwrap();
    }

    fn fonts(s: &Session, id: crate::DocId, page: usize) -> Vec<String> {
        let cos = &s.get(id).unwrap().editor.as_ref().unwrap().cos;
        pdfcraft_edit::text_lines(cos, page).unwrap().into_iter().map(|l| l.base_font).collect()
    }

    #[test]
    fn ocr_family_choice_is_document_local_and_follows_undo_redo() {
        let mut s = Session::new();
        let original = scan_fixture();
        let id = s.open("synthetic.pdf", None, original.clone(), None).unwrap();
        let other = s.open("other.pdf", None, original, None).unwrap();
        assert!(s.get(id).unwrap().text_blocks(0)[0].invisible);
        assert_eq!(s.get(id).unwrap().ocr_text_style(0), BlockStyle::default());
        let chosen = BlockStyle { family: Some((FontFamily::Times, false, false)), ..Default::default() };
        edit(&mut s, id, 0, chosen.clone());
        assert_eq!(s.get(id).unwrap().ocr_text_style(1), chosen);
        assert_eq!(s.get(other).unwrap().ocr_text_style(1), BlockStyle::default());
        let doc = s.get(id).unwrap();
        let preview = doc.text_edit_layout(1, 0, "New words", &BlockStyle::default()).unwrap();
        let explicit = doc.text_edit_layout(1, 0, "New words", &chosen).unwrap();
        assert_eq!(format!("{preview:?}"), format!("{explicit:?}"), "preview uses the remembered face too");
        let before = doc.bytes.clone();
        edit(&mut s, id, 1, BlockStyle::default());
        assert!(fonts(&s, id, 1).iter().all(|f| f == "Times-Roman"));
        s.undo(id).unwrap();
        assert_eq!(s.get(id).unwrap().bytes, before);
        s.undo(id).unwrap();
        assert_eq!(s.get(id).unwrap().ocr_text_style(1), BlockStyle::default());
        s.redo(id).unwrap();
        assert_eq!(s.get(id).unwrap().ocr_text_style(1), chosen);
        s.redo(id).unwrap();
        assert!(fonts(&s, id, 1).iter().all(|f| f == "Times-Roman"));
        s.undo(id).unwrap();
        let changed = BlockStyle { family: Some((FontFamily::Courier, false, false)), ..Default::default() };
        edit(&mut s, id, 1, changed.clone());
        assert_eq!(s.get(id).unwrap().ocr_text_style(0), changed, "a new choice replaces the old one");
        s.undo(id).unwrap();
        assert_eq!(s.get(id).unwrap().ocr_text_style(0), chosen);
        s.redo(id).unwrap();
        assert_eq!(s.get(id).unwrap().ocr_text_style(0), changed);
        edit(&mut s, id, 2, BlockStyle::default());
        assert!(fonts(&s, id, 2).iter().all(|f| f == "Helvetica"), "native paragraphs retain their source family");
        let saved = s.save_bytes(id).unwrap();
        let reopened = s.open("saved.pdf", None, saved, None).unwrap();
        assert_eq!(s.get(reopened).unwrap().ocr_text_style(0), BlockStyle::default(), "the choice is session-only");
    }

    #[test]
    fn ocr_resource_choice_reuses_a_full_face_on_another_page() {
        let mut s = Session::new();
        let id = s.open("synthetic.pdf", None, scan_fixture(), None).unwrap();
        let bytes = include_bytes!("../../../assets/fonts/JetBrainsMono-Regular.ttf");
        s.apply(id, Edit::RegisterTextFont { page: 0, bytes: Arc::new(bytes.to_vec()), grant: crate::FontGrant::OpenLicence }).unwrap();
        let resource = s.get(id).unwrap().text_editor_fonts(0).into_iter().find(|f| f.face.contains("JetBrains")).unwrap().resource;
        let chosen = BlockStyle { resource: Some(resource), ..Default::default() };
        edit(&mut s, id, 0, chosen);
        let saved = s.save_full_bytes(id).unwrap();
        s.mark_saved(id, saved, None).unwrap();
        let doc = s.get(id).unwrap();
        let expected_resource = doc.ocr_text_style(1).resource.unwrap();
        assert!(doc.ocr_text_editor_fonts(1).iter().any(|f| f.resource == expected_resource && f.face.contains("JetBrains")));
        let inherited = doc.text_edit_layout(1, 0, "New words", &BlockStyle::default()).unwrap();
        let explicit = doc.text_edit_layout(1, 0, "New words", &doc.ocr_text_style(1)).unwrap();
        assert_eq!(format!("{inherited:?}"), format!("{explicit:?}"));
        edit(&mut s, id, 1, BlockStyle::default());
        assert!(fonts(&s, id, 1).iter().all(|f| f.contains("JetBrains")), "the second scan uses the chosen full face");
        s.undo(id).unwrap();
        s.redo(id).unwrap();
        assert!(fonts(&s, id, 1).iter().all(|f| f.contains("JetBrains")));
    }

    #[test]
    fn a_chosen_ocr_font_survives_deleting_its_source_page_and_full_save() {
        let mut s = Session::new();
        let id = s.open("synthetic.pdf", None, scan_fixture(), None).unwrap();
        let bytes = include_bytes!("../../../assets/fonts/JetBrainsMono-Regular.ttf");
        s.apply(id, Edit::RegisterTextFont { page: 0, bytes: Arc::new(bytes.to_vec()), grant: crate::FontGrant::OpenLicence }).unwrap();
        let resource = s.get(id).unwrap().text_editor_fonts(0).into_iter().find(|f| f.face.contains("JetBrains")).unwrap().resource;
        edit(&mut s, id, 0, BlockStyle { resource: Some(resource), ..Default::default() });
        s.apply(id, Edit::DeletePages { pages: vec![0] }).unwrap();
        let saved = s.save_full_bytes(id).unwrap();
        s.mark_saved(id, saved.clone(), None).unwrap();
        let doc = s.get(id).unwrap();
        assert!(doc.ocr_text_style(0).resource.is_some(), "removing its page does not clear the chosen replacement face");
        assert!(doc.ocr_text_editor_fonts(0).iter().any(|f| f.face.contains("JetBrains")), "the immutable snapshot supplies the orphaned face");
        doc.text_edit_layout(0, 0, "New words", &BlockStyle::default()).unwrap();
        assert_eq!(s.get(id).unwrap().bytes, saved, "preview import does not change working bytes");
        edit(&mut s, id, 0, BlockStyle::default());
        assert!(fonts(&s, id, 0).iter().all(|f| f.contains("JetBrains")));
        s.undo(id).unwrap();
        assert_eq!(s.get(id).unwrap().bytes, saved);
        s.redo(id).unwrap();
        assert!(fonts(&s, id, 0).iter().all(|f| f.contains("JetBrains")));
    }

    #[test]
    fn remembered_ocr_fonts_do_not_enter_native_font_inventory_or_layout() {
        let mut s = Session::new();
        let id = s.open("synthetic.pdf", None, scan_fixture(), None).unwrap();
        let before = s.get(id).unwrap().text_edit_layout(2, 0, "New words", &BlockStyle::default()).unwrap();
        let bytes = include_bytes!("../../../assets/fonts/JetBrainsMono-Regular.ttf");
        s.apply(id, Edit::RegisterTextFont { page: 0, bytes: Arc::new(bytes.to_vec()), grant: crate::FontGrant::OpenLicence }).unwrap();
        let resource = s.get(id).unwrap().text_editor_fonts(0).into_iter().find(|f| f.face.contains("JetBrains")).unwrap().resource;
        edit(&mut s, id, 0, BlockStyle { resource: Some(resource), ..Default::default() });
        let doc = s.get(id).unwrap();
        let working = doc.bytes.clone();
        // Warm the OCR preview on another page, so inventory queries also exercise the
        // cached-session path before returning to native text.
        doc.text_edit_layout(1, 0, "New words", &BlockStyle::default()).unwrap();
        assert!(doc.ocr_text_editor_fonts(1).iter().any(|f| f.face.contains("JetBrains")));
        assert!(doc.text_editor_fonts(1).is_empty(), "a virtual OCR face is absent from the real page inventory");
        assert!(doc.text_editor_fonts(2).is_empty(), "native font choices contain only fonts actually registered on that page");
        assert!(
            doc.ocr_text_editor_fonts(2).iter().any(|f| f.face.contains("JetBrains")),
            "the OCR-specific API scopes virtual replacements explicitly"
        );
        let after = doc.text_edit_layout(2, 0, "New words", &BlockStyle::default()).unwrap();
        assert_eq!(format!("{before:?}"), format!("{after:?}"), "native layout never receives the remembered OCR face");
        assert_eq!(doc.bytes, working, "both inventories and previews leave the document unchanged");
        edit(&mut s, id, 2, BlockStyle::default());
        assert!(fonts(&s, id, 2).iter().all(|f| f == "Helvetica"));
    }

    #[test]
    fn an_explicit_ocr_inventory_alias_materializes_only_for_ocr_edits() {
        let mut s = Session::new();
        let id = s.open("synthetic.pdf", None, scan_fixture(), None).unwrap();
        let bytes = include_bytes!("../../../assets/fonts/JetBrainsMono-Regular.ttf");
        s.apply(id, Edit::RegisterTextFont { page: 0, bytes: Arc::new(bytes.to_vec()), grant: crate::FontGrant::OpenLicence }).unwrap();
        let resource = s.get(id).unwrap().text_editor_fonts(0).into_iter().find(|f| f.face.contains("JetBrains")).unwrap().resource;
        edit(&mut s, id, 0, BlockStyle { resource: Some(resource), ..Default::default() });
        let doc = s.get(id).unwrap();
        let virtual_resource = doc.ocr_text_editor_fonts(1).into_iter().find(|f| f.face.contains("JetBrains")).unwrap().resource;
        let style = BlockStyle { resource: Some(virtual_resource), ..Default::default() };
        let saved = doc.bytes.clone();
        assert!(s.apply(id, Edit::EditTextBlock { page: 2, block: 0, text: "New words".into(), style: style.clone() }).is_err());
        assert_eq!(s.get(id).unwrap().bytes, saved, "native edits cannot materialize an OCR-only alias");
        s.get(id).unwrap().text_edit_layout(1, 0, "New words", &style).unwrap();
        edit(&mut s, id, 1, style);
        assert!(fonts(&s, id, 1).iter().all(|f| f.contains("JetBrains")));
    }

    #[test]
    fn a_failed_or_native_font_edit_does_not_replace_the_ocr_default() {
        let mut s = Session::new();
        let id = s.open("synthetic.pdf", None, scan_fixture(), None).unwrap();
        let chosen = BlockStyle { family: Some((FontFamily::Times, false, false)), ..Default::default() };
        edit(&mut s, id, 1, chosen.clone());
        assert_eq!(s.get(id).unwrap().ocr_text_style(0), chosen);
        let invalid = BlockStyle { resource: Some("Missing".into()), ..Default::default() };
        assert!(s.apply(id, Edit::EditTextBlock { page: 0, block: 0, text: "bad".into(), style: invalid }).is_err());
        assert_eq!(s.get(id).unwrap().ocr_text_style(0), chosen);
        edit(&mut s, id, 2, BlockStyle { family: Some((FontFamily::Courier, false, false)), ..Default::default() });
        assert_eq!(s.get(id).unwrap().ocr_text_style(0), chosen);
    }
}
