//! Document font selection and contextual glyph save/reopen acceptance on synthetic PDFs.
use crate::support;
use pdfcraft_cos::{Dict, Document, Object, SaveOptions, Stream};
use pdfcraft_engine::{BlockStyle, Edit, Session};
use pdfcraft_render::{PageRenderer, RenderRequest};
use std::sync::Arc;
fn document() -> Arc<Vec<u8>> {
    let mut d = Document::new_empty();
    let pages = d.add(Object::Dict(Dict::new()));
    let contents = d.add(Object::Stream(Stream::flate(Dict::new(), b"BT /F1 18 Tf 30 160 Td (Source) Tj ET")));
    let mut font = Dict::new();
    font.set(b"Type".to_vec(), Object::name("Font"));
    font.set(b"Subtype".to_vec(), Object::name("Type1"));
    font.set(b"BaseFont".to_vec(), Object::name("Helvetica"));
    let mut fonts = Dict::new();
    fonts.set(b"F1".to_vec(), Object::Dict(font));
    let mut res = Dict::new();
    res.set(b"Font".to_vec(), Object::Dict(fonts));
    let mut p = Dict::new();
    p.set(b"Type".to_vec(), Object::name("Page"));
    p.set(b"Parent".to_vec(), Object::Ref(pages));
    p.set(b"Contents".to_vec(), Object::Ref(contents));
    p.set(b"Resources".to_vec(), Object::Dict(res));
    p.set(b"MediaBox".to_vec(), Object::Array([0, 0, 300, 240].map(Object::Int).to_vec()));
    let page = d.add(Object::Dict(p));
    d.update_dict(pages, |p| {
        p.set(b"Type".to_vec(), Object::name("Pages"));
        p.set(b"Kids".to_vec(), Object::Array(vec![Object::Ref(page)]));
        p.set(b"Count".to_vec(), Object::Int(1));
    })
    .unwrap();
    let mut root = Dict::new();
    root.set(b"Type".to_vec(), Object::name("Catalog"));
    root.set(b"Pages".to_vec(), Object::Ref(pages));
    let root = d.add(Object::Dict(root));
    d.trailer_mut().set(b"Root".to_vec(), Object::Ref(root));
    Arc::new(pdfcraft_cos::write_full(&d, &SaveOptions::default()).unwrap())
}
fn pixels(bytes: Arc<Vec<u8>>) -> Vec<u8> {
    let mut renderer = PageRenderer::new(bytes, Default::default());
    let p = renderer.render(RenderRequest { page: 0, scale: 2.0, ..Default::default() });
    assert!(p.error.is_none(), "{:?}", p.error);
    assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    p.rgba.to_vec()
}
fn exercise(font: Vec<u8>, text: &str, expected_family: &str) {
    let mut s = Session::new();
    let id = s.open("source.pdf", None, document(), None).unwrap();
    let original = pixels(s.get(id).unwrap().display.clone());
    s.apply(id, Edit::RegisterTextFont { page: 0, bytes: Arc::new(font.clone()), grant: Default::default() }).unwrap();
    let resource = s.get(id).unwrap().text_editor_fonts(0).into_iter().find(|f| f.bytes == font).unwrap().resource;
    let style = BlockStyle { resource: Some(resource), width: Some(240.0), ..Default::default() };
    s.apply(id, Edit::EditTextBlock { page: 0, block: 0, text: text.into(), style: style.clone() }).unwrap();
    let edited = s.get(id).unwrap();
    let lines = edited.text_lines(0);
    assert_eq!(lines.iter().map(|l| l.text.as_str()).collect::<String>(), text);
    assert!(lines.iter().all(|l| l.base_font == expected_family));
    assert!((lines[0].baseline()[1] - 160.0).abs() < 0.01);
    let current = pixels(edited.display.clone());
    assert_ne!(current, original);
    let saved = s.save_bytes(id).unwrap();
    let re = s.open("saved.pdf", None, saved, None).unwrap();
    assert_eq!(pixels(s.get(re).unwrap().display.clone()), current);
    assert_eq!(s.get(re).unwrap().text_blocks(0)[0].editor_text, text);
    let twice = format!("{text} {text}");
    s.apply(re, Edit::EditTextBlock { page: 0, block: 0, text: twice.clone(), style: style.clone() }).unwrap();
    assert_eq!(s.get(re).unwrap().text_blocks(0)[0].editor_text, twice);
    let again = s.save_bytes(re).unwrap();
    let next = s.open("twice.pdf", None, again, None).unwrap();
    assert_eq!(s.get(next).unwrap().text_blocks(0)[0].editor_text, twice);
    assert_eq!(pixels(s.get(next).unwrap().display.clone()), pixels(s.get(re).unwrap().display.clone()));
    s.apply(re, Edit::EditTextBlock { page: 0, block: 0, text: String::new(), style }).unwrap();
    assert!(s.get(re).unwrap().text_lines(0).is_empty(), "every contextual glyph is deleted");
    let empty = pixels(s.get(re).unwrap().display.clone());
    assert!(empty.as_chunks::<4>().0.iter().all(|p| p[..3] == [255; 3]));
    s.undo(id).unwrap();
    assert_eq!(pixels(s.get(id).unwrap().display.clone()), original);
}
#[test]
fn opentype_cff_document_font_preserves_program_codes_and_rendering_after_reopen() {
    exercise(support::font_fixture::cff_font(), "A A", "RectFont-Regular");
}
#[test]
fn contextual_arabic_marks_ligatures_and_mixed_bidi_keep_logical_text_after_reopen() {
    if let Some(path) = std::env::var_os("PDFCRAFT_TEST_ARABIC_FONT") {
        exercise(std::fs::read(path).unwrap(), "سَلَام 123", "NotoSansArabic-Regular");
        return;
    }
    let Some(font) = pdfcraft_fonts::document_arabic_font() else {
        eprintln!("set CRAFT_FONTS_DIR to test full Arabic glyph programs");
        return;
    };
    if pdfcraft_fonts::embedding::validate(font.bytes).is_err() {
        eprintln!("set PDFCRAFT_TEST_ARABIC_FONT to an openly licensed static Noto Sans Arabic face");
        return;
    }
    exercise(font.bytes.to_vec(), "سَلَام 123", "NotoSansArabic-Regular");
}

// Based on #1267 by @komo1iddin: cached completion, atomic undo and portable save/reopen.
#[test]
fn matching_installed_subset_completion_is_atomic_and_portable() {
    use pdfcraft_engine::{FontGrant, text_fonts::Catalogue};
    let dir = std::env::temp_dir().join(format!("pdfcraft-matching-synthetic-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("face.otf"), support::font_fixture::named_font("Rect Font", "RectFont", 0, true)).unwrap();
    let fonts = Arc::new(Catalogue::discover(std::slice::from_ref(&dir), FontGrant::InstalledEmbeddable));
    assert_eq!(fonts.files().count(), 1);
    let source = support::source_subset::source_subset();
    let mut s = Session::new();
    let id = s.open("source.pdf", None, source.clone(), None).unwrap();
    s.set_text_font_catalogue(fonts);
    let style = BlockStyle::default();
    assert_eq!(s.get(id).unwrap().matching_font_completions(0, 0, "AB", &style).unwrap(), vec!["Rect Font Regular"]);
    s.get(id).unwrap().text_edit_layout(0, 0, "AB", &style).unwrap();
    assert!(!s.get(id).unwrap().dirty);
    s.apply(id, Edit::EditTextBlock { page: 0, block: 0, text: "AB".into(), style }).unwrap();
    let doc = s.get(id).unwrap();
    assert_eq!(doc.installed_completions, vec!["Rect Font Regular"]);
    let lines = doc.text_lines(0);
    assert_eq!(lines.iter().map(|l| l.text.as_str()).collect::<String>(), "AB");
    assert!(lines.iter().all(|l| pdfcraft_fonts::embedding::same_face_name(&l.base_font, "RectFont-Regular")));
    let saved = s.save_bytes(id).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let cos = Document::open(saved.clone()).unwrap();
    let page = pdfcraft_model::pages(&cos).remove(0);
    let resources = cos.dict(page.dict.get(b"Resources").unwrap()).unwrap();
    let fonts = cos.dict(resources.get(b"Font").unwrap()).unwrap();
    fn contents(cos: &Document, object: &Object) -> Vec<u8> {
        let object = cos.resolve(object);
        match object.as_ref() {
            Object::Stream(s) => s.decoded().unwrap(),
            Object::Array(a) => a.iter().flat_map(|o| contents(cos, o)).collect(),
            _ => Vec::new(),
        }
    }
    let mut font = String::new();
    let mut source_text = String::new();
    let mut completed_text = String::new();
    for op in pdfcraft_content::parse(&contents(&cos, page.dict.get(b"Contents").unwrap())).ops {
        if op.op == b"Tf" {
            font = String::from_utf8(op.operands[0].as_name().unwrap().to_vec()).unwrap();
        }
        let strings: Vec<_> = if op.op == b"Tj" {
            op.operands.iter().collect()
        } else if op.op == b"TJ" {
            op.operands[0].as_array().unwrap().iter().collect()
        } else {
            Vec::new()
        };
        for object in strings {
            if let Some(text) = object.as_string() {
                let metrics = pdfcraft_fonts::pdf::Metrics::from_dict(&cos, &cos.dict(fonts.get(font.as_bytes()).unwrap()).unwrap());
                let text = metrics.decode(&text.bytes);
                if font == "F1" {
                    source_text.push_str(&text);
                } else {
                    completed_text.push_str(&text);
                }
            }
        }
    }
    assert_eq!(source_text, "A");
    assert_eq!(completed_text, "B");
    let mut reopened = Session::new();
    let rid = reopened.open("saved.pdf", None, saved, None).unwrap();
    assert_eq!(reopened.get(rid).unwrap().text_blocks(0)[0].text, "AB");
    s.undo(id).unwrap();
    assert_eq!(s.get(id).unwrap().bytes, source);
    assert!(s.get(id).unwrap().can_undo().is_none());
}

#[test]
fn installed_subset_completion_requires_opt_in_same_family_and_editable_embedding() {
    use pdfcraft_engine::{FontGrant, text_fonts::Catalogue};
    for (label, family, face, permission, opt_in) in [
        ("off", "Rect Font", "RectFont-Regular", 0, false),
        ("other", "Other", "Other-Regular", 0, true),
        ("restricted", "Rect Font", "RectFont-Regular", 2, true),
        ("preview", "Rect Font", "RectFont-Regular", 4, true),
    ] {
        let dir = std::env::temp_dir().join(format!("pdfcraft-matching-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("font.otf"), support::font_fixture::named_font(family, face, permission, true)).unwrap();
        let fonts = Arc::new(Catalogue::discover(std::slice::from_ref(&dir), FontGrant::InstalledEmbeddable));
        if permission != 0 {
            assert_eq!(fonts.files().count(), 0);
        }
        let mut s = Session::new();
        let id = s.open("source.pdf", None, support::source_subset::source_subset(), None).unwrap();
        if opt_in {
            s.set_text_font_catalogue(fonts);
        }
        assert!(s.get(id).unwrap().matching_font_completions(0, 0, "AB", &BlockStyle::default()).unwrap().is_empty(), "{label}");
        s.apply(id, Edit::EditTextBlock { page: 0, block: 0, text: "AB".into(), style: BlockStyle::default() }).unwrap();
        assert!(s.get(id).unwrap().installed_completions.is_empty(), "{label}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}

// Index size must depend on metadata, not the cumulative size of installed programs.
#[test]
fn installed_metadata_index_has_no_total_program_size_budget() {
    use pdfcraft_engine::{FontGrant, text_fonts::Catalogue};
    let dir = std::env::temp_dir().join(format!("pdfcraft-metadata-budget-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for index in 0..9 {
        let path = dir.join(format!("{index}.otf"));
        std::fs::write(&path, support::font_fixture::named_font("Rect Font", &format!("RectFont-{index}"), 0, true)).unwrap();
        // Sparse padding: over 64 MiB in total, with only a few KiB of metadata.
        std::fs::OpenOptions::new().write(true).open(path).unwrap().set_len(9 << 20).unwrap();
    }
    let catalogue = Catalogue::discover(std::slice::from_ref(&dir), FontGrant::InstalledEmbeddable);
    assert_eq!(catalogue.files().count(), 9);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn installed_metadata_reads_are_capped_and_full_validation_is_deferred() {
    use pdfcraft_engine::{FontGrant, text_fonts::Catalogue};
    let dir = std::env::temp_dir().join(format!("pdfcraft-metadata-validation-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut font = support::font_fixture::named_font("Rect Font", "RectFont-Regular", 0, true);
    let count = u16::from_be_bytes(font[4..6].try_into().unwrap()) as usize;
    let name_record = (0..count).map(|i| 12 + i * 16).find(|i| &font[*i..*i + 4] == b"name").unwrap();
    let offset = u32::from_be_bytes(font[name_record + 8..name_record + 12].try_into().unwrap()) as usize;
    let original_len = font.len();
    font.resize(original_len + (1 << 20), 0);
    let length = (font.len() - offset) as u32;
    font[name_record + 12..name_record + 16].copy_from_slice(&length.to_be_bytes());
    std::fs::write(dir.join("large-name.otf"), &font).unwrap();
    let catalogue = Catalogue::discover(std::slice::from_ref(&dir), FontGrant::InstalledEmbeddable);
    assert_eq!(catalogue.files().count(), 0, "metadata table read cap");
    std::fs::remove_file(dir.join("large-name.otf")).unwrap();
    let mut font = support::font_fixture::named_font("Rect Font", "RectFont-Regular", 0, true);
    let cmap_record = (0..count).map(|i| 12 + i * 16).find(|i| &font[*i..*i + 4] == b"cmap").unwrap();
    let offset = u32::from_be_bytes(font[cmap_record + 8..cmap_record + 12].try_into().unwrap()) as usize;
    font[offset..offset + 4].fill(0);
    std::fs::write(dir.join("bad-cmap.otf"), font).unwrap();
    let catalogue = Arc::new(Catalogue::discover(std::slice::from_ref(&dir), FontGrant::InstalledEmbeddable));
    assert_eq!(catalogue.files().count(), 1, "index reads metadata only");
    let mut session = Session::new();
    let id = session.open("source.pdf", None, support::source_subset::source_subset(), None).unwrap();
    session.set_text_font_catalogue(catalogue);
    assert!(session.get(id).unwrap().matching_font_completions(0, 0, "AB", &BlockStyle::default()).unwrap().is_empty(), "full validation on use");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn installed_metadata_catalogue_skips_changed_and_removed_programs() {
    use pdfcraft_engine::{FontGrant, text_fonts::Catalogue};
    let dir = std::env::temp_dir().join(format!("pdfcraft-metadata-changed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("font.otf");
    std::fs::write(&path, support::font_fixture::named_font("Rect Font", "RectFont-Regular", 0, true)).unwrap();
    let catalogue = Arc::new(Catalogue::discover(std::slice::from_ref(&dir), FontGrant::InstalledEmbeddable));
    assert_eq!(catalogue.files().count(), 1);
    let mut session = Session::new();
    let id = session.open("source.pdf", None, support::source_subset::source_subset(), None).unwrap();
    session.set_text_font_catalogue(catalogue);
    std::fs::write(&path, support::font_fixture::named_font("Rect Font", "RectFont-Regular", 2, true)).unwrap();
    assert!(session.get(id).unwrap().matching_font_completions(0, 0, "AB", &BlockStyle::default()).unwrap().is_empty());
    std::fs::remove_file(path).unwrap();
    assert!(session.get(id).unwrap().matching_font_completions(0, 0, "AB", &BlockStyle::default()).unwrap().is_empty());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn installed_font_lists_do_not_load_unedited_programs() {
    use pdfcraft_engine::{FontGrant, text_fonts::Catalogue};
    let dir = std::env::temp_dir().join(format!("pdfcraft-metadata-list-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("font.otf"), support::font_fixture::named_font("Rect Font", "RectFont-Regular", 0, true)).unwrap();
    let mut session = Session::new();
    let id = session.open("source.pdf", None, support::source_subset::source_subset(), None).unwrap();
    let source = session.get(id).unwrap().text_editor_fonts(0);
    let catalogue = Arc::new(Catalogue::discover(std::slice::from_ref(&dir), FontGrant::InstalledEmbeddable));
    let list: Vec<_> = catalogue.files().cloned().collect();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].family, "Rect Font");
    session.set_text_font_catalogue(catalogue);
    let untouched = session.get(id).unwrap().text_editor_fonts(0);
    assert_eq!(untouched.len(), source.len(), "opening the source-font list must not load matching programs before a paragraph needs them");
    assert!(untouched.iter().all(|f| !f.resource.starts_with("PCApprovedInstalled")));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn installed_font_preview_reuses_one_paragraph_program_until_generation_changes() {
    use pdfcraft_engine::{FontGrant, text_fonts::Catalogue};
    let dir = std::env::temp_dir().join(format!("pdfcraft-preview-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("face.otf");
    std::fs::write(&path, support::font_fixture::named_font("Rect Font", "RectFont", 0, true)).unwrap();
    let fonts = Arc::new(Catalogue::discover(std::slice::from_ref(&dir), FontGrant::InstalledEmbeddable));
    let mut session = Session::new();
    let source = support::source_subset::source_subset();
    let id = session.open("source.pdf", None, source.clone(), None).unwrap();
    session.set_text_font_catalogue(fonts);
    let style = BlockStyle::default();
    let doc = session.get(id).unwrap();
    assert!(doc.text_font_preview_key().is_none());
    let expected = doc.matching_font_completions(0, 0, "AB", &style).unwrap();
    assert_eq!(expected.len(), 1);
    let key = doc.text_font_preview_key();
    assert_eq!(key.map(|(_, _, _, count)| count), Some(1));
    std::fs::remove_file(&path).unwrap();
    for text in ["ABB", "BAA", "A B", "AB"] {
        assert_eq!(doc.matching_font_completions(0, 0, text, &style).unwrap(), expected);
        doc.text_edit_layout(0, 0, text, &style).unwrap();
        assert_eq!(doc.text_font_preview_key(), key, "keystrokes reuse a validated full face without registering it again");
    }
    assert_eq!(doc.bytes, source);
    assert!(!doc.dirty && doc.can_undo().is_none());
    session.apply(id, Edit::RotatePages { pages: vec![0], degrees: 90 }).unwrap();
    assert!(session.get(id).unwrap().text_font_preview_key().is_none(), "a changed document generation cannot use the previous paragraph preview");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn installed_collection_faces_are_permission_gated_and_save_only_selected_program() {
    use pdfcraft_engine::{FontGrant, text_fonts::Catalogue};
    let dir = std::env::temp_dir().join(format!("pdfcraft-collection-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let restricted = support::font_fixture::named_font("Rect Font", "RectFont-Regular", 2, true);
    let permitted = support::font_fixture::named_font("Rect Font", "RectFont-Regular", 0, true);
    let sibling = support::font_fixture::named_font("Sibling Marker", "SiblingOnly-Regular", 0, true);
    let collection = support::font_fixture::collection(&[restricted, permitted, sibling]);
    std::fs::write(dir.join("faces.ttc"), &collection).unwrap();
    let fonts = Arc::new(Catalogue::discover(std::slice::from_ref(&dir), FontGrant::InstalledEmbeddable));
    let files: Vec<_> = fonts.files().collect();
    assert_eq!(files.len(), 2);
    assert!(files.iter().all(|f| f.face_index != 0), "restricted collection face must never be listed");
    let selected_name = files.iter().find(|f| f.face_index == 1).unwrap().full_name.clone();
    let mut session = Session::new();
    let id = session.open("source.pdf", None, support::source_subset::source_subset(), None).unwrap();
    session.set_text_font_catalogue(fonts);
    session.apply(id, Edit::EditTextBlock { page: 0, block: 0, text: "AB".into(), style: BlockStyle::default() }).unwrap();
    let doc = session.get(id).unwrap();
    assert_eq!(doc.installed_completions, vec![selected_name]);
    assert_eq!(doc.text_blocks(0)[0].text, "AB");
    let programs = doc.text_editor_fonts(0);
    let selected = programs.iter().find(|f| f.resource.starts_with("PCFull")).unwrap();
    assert_ne!(selected.bytes, collection);
    assert_ne!(selected.bytes.get(..4), Some(b"ttcf".as_slice()));
    let sibling_marker: Vec<_> = "SiblingOnly".encode_utf16().flat_map(u16::to_be_bytes).collect();
    assert!(!selected.bytes.windows(sibling_marker.len()).any(|w| w == sibling_marker));
    let saved = session.save_bytes(id).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let mut reopened = Session::new();
    let next = reopened.open("saved.pdf", None, saved, None).unwrap();
    assert_eq!(reopened.get(next).unwrap().text_blocks(0)[0].text, "AB");
    reopened.apply(next, Edit::EditTextBlock { page: 0, block: 0, text: "ABA".into(), style: BlockStyle::default() }).unwrap();
    assert_eq!(reopened.get(next).unwrap().text_blocks(0)[0].text, "ABA");
}

#[test]
fn collection_metadata_read_budget_is_cumulative_per_file() {
    use pdfcraft_engine::{FontGrant, text_fonts::Catalogue};
    let dir = std::env::temp_dir().join(format!("pdfcraft-collection-read-cap-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut face = support::font_fixture::named_font("Rect Font", "RectFont-Regular", 0, true);
    // The synthetic fixture stores name last, so harmless padding can extend just that table.
    let count = usize::from(u16::from_be_bytes(face.get(4..6).unwrap().try_into().unwrap()));
    let record = face.get_mut(12..12 + 16 * count).unwrap().as_chunks_mut::<16>().0.iter_mut().find(|r| r.get(..4) == Some(b"name")).unwrap();
    let old_len = u32::from_be_bytes(record.get(12..16).unwrap().try_into().unwrap());
    record.get_mut(12..16).unwrap().copy_from_slice(&(old_len + (192 << 10)).to_be_bytes());
    face.resize(face.len() + (192 << 10), 0);
    std::fs::write(dir.join("large-metadata.otc"), support::font_fixture::collection(&vec![face; 8])).unwrap();
    let fonts = Catalogue::discover(std::slice::from_ref(&dir), FontGrant::InstalledEmbeddable);
    assert!((1..8).contains(&fonts.files().count()), "the one MiB read allowance belongs to the whole file");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn collection_index_caps_faces_per_file() {
    use pdfcraft_engine::{FontGrant, text_fonts::Catalogue};
    let dir = std::env::temp_dir().join(format!("pdfcraft-collection-file-face-caps-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let face = support::font_fixture::named_font("Rect Font", "RectFont-Regular", 0, true);
    std::fs::write(dir.join("too-many.ttc"), support::font_fixture::collection(&vec![face.clone(); 257])).unwrap();
    assert_eq!(Catalogue::discover(std::slice::from_ref(&dir), FontGrant::InstalledEmbeddable).files().count(), 0);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn collection_index_caps_faces_across_the_catalogue() {
    use pdfcraft_engine::{FontGrant, text_fonts::Catalogue};
    let dir = std::env::temp_dir().join(format!("pdfcraft-collection-total-face-cap-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let face = support::font_fixture::named_font("Rect Font", "RectFont-Regular", 0, true);
    let collection = support::font_fixture::collection(&vec![face; 256]);
    for index in 0..17 {
        std::fs::write(dir.join(format!("{index}.ttc")), &collection).unwrap();
    }
    assert_eq!(Catalogue::discover(std::slice::from_ref(&dir), FontGrant::InstalledEmbeddable).files().count(), 4096);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[cfg(unix)]
fn installed_catalogue_skips_a_discovery_root_replaced_by_a_symlink() {
    use pdfcraft_engine::{FontGrant, text_fonts::Catalogue};
    let dir = std::env::temp_dir().join(format!("pdfcraft-catalogue-parent-link-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let root = dir.join("fonts");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("font.otf"), support::font_fixture::named_font("Rect Font", "RectFont-Regular", 0, true)).unwrap();
    let catalogue = Arc::new(Catalogue::discover(std::slice::from_ref(&root), FontGrant::InstalledEmbeddable));
    assert_eq!(catalogue.files().count(), 1);
    let moved = dir.join("moved");
    std::fs::rename(&root, &moved).unwrap();
    std::os::unix::fs::symlink(&moved, &root).unwrap();
    let mut session = Session::new();
    let id = session.open("source.pdf", None, support::source_subset::source_subset(), None).unwrap();
    session.set_text_font_catalogue(catalogue);
    assert!(session.get(id).unwrap().matching_font_completions(0, 0, "AB", &BlockStyle::default()).unwrap().is_empty());
    std::fs::remove_dir_all(dir).unwrap();
}
