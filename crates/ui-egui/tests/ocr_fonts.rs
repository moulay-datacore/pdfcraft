//! Scanned text offers an honest source notice and explicit replacement faces.

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use pdfcraft_ui_egui::PdfCraftApp;

fn source_font_fixture_with_text(body: &str) -> Vec<u8> {
    let objs: Vec<String> = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".into(),
        "<< /Type /Pages /Kids [4 0 R] /Count 1 /MediaBox [0 0 300 300] >>".into(),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Times-BoldItalic >>".into(),
        "<< /Type /Page /Parent 2 0 R /Contents 7 0 R /Resources << /Font << /F1 3 0 R /F2 5 0 R >> >> >>".into(),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Oblique >>".into(),
        "<< /Producer (PdfCraft) >>".into(),
        format!("<< /Length {} >>\nstream\n{body}\nendstream", body.len()),
    ];
    let mut out = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{o}\nendobj\n", i + 1).as_bytes());
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
    for o in offsets {
        out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1).as_bytes());
    out
}

/// Generated pixels and OCR words only; no scanned or external document enters the test.
fn ocr_font_fixture() -> Vec<u8> {
    use pdfcraft_cos::{Dict, Document, Object, SaveOptions, Stream, write_incremental};
    let body = "q 300 0 0 300 0 0 cm /Scan Do Q /PCMark /OCR BDC BT /F1 1 Tf 3 Tr 11.03 0 0 13.29 20 220 Tm (Lorem) Tj 11.34 0 0 11.25 55 220.4 Tm (ipsum) Tj 10.69 0 0 12.03 88 219.6 Tm (dolor) Tj ET EMC";
    let mut doc = Document::open(std::sync::Arc::new(source_font_fixture_with_text(body))).unwrap();
    let page = pdfcraft_model::pages(&doc)[0].clone();
    let mut image = Dict::new();
    image.set(b"Type".to_vec(), Object::name("XObject"));
    image.set(b"Subtype".to_vec(), Object::name("Image"));
    image.set(b"Width".to_vec(), Object::Int(8));
    image.set(b"Height".to_vec(), Object::Int(8));
    image.set(b"BitsPerComponent".to_vec(), Object::Int(8));
    image.set(b"ColorSpace".to_vec(), Object::name("DeviceGray"));
    let pixels: Vec<_> = (0..64).map(|i| if i % 9 == 0 { 210 } else { 255 }).collect();
    let image = doc.add(Object::Stream(Stream::flate(image, &pixels)));
    let mut resources = doc.dict(page.dict.get(b"Resources").unwrap()).unwrap();
    let mut xobjects = Dict::new();
    xobjects.set(b"Scan".to_vec(), Object::Ref(image));
    resources.set(b"XObject".to_vec(), Object::Dict(xobjects));
    let mut font = Dict::new();
    font.set(b"Type".to_vec(), Object::name("Font"));
    font.set(b"Subtype".to_vec(), Object::name("Type1"));
    font.set(b"BaseFont".to_vec(), Object::name("Helvetica"));
    font.set(b"Encoding".to_vec(), Object::name("WinAnsiEncoding"));
    let mut fonts = Dict::new();
    fonts.set(b"F1".to_vec(), Object::Dict(font));
    resources.set(b"Font".to_vec(), Object::Dict(fonts));
    doc.update_dict(page.obj, |dict| dict.set(b"Resources".to_vec(), Object::Dict(resources))).unwrap();
    write_incremental(&doc, &SaveOptions::default()).unwrap()
}

#[test]
fn scanned_text_explains_the_font_and_offers_replacement_faces() {
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_eframe(|_cc| {
        let mut app = PdfCraftApp::new();
        app.set_option("language", "en").unwrap();
        app.open_bytes("scan.pdf", None, ocr_font_fixture()).unwrap();
        app
    });
    h.run_steps(4);
    assert!(h.state_mut().execute("edit.edit_text"));
    h.run_steps(2);
    let page = h.state().views[0].page_screen_rect(0).unwrap();
    let at = egui::pos2(page.left() + 25.0 / 300.0 * page.width(), page.top() + 75.0 / 300.0 * page.height());
    h.hover_at(at);
    h.run_steps(1);
    h.drag_at(at);
    h.run_steps(1);
    h.drop_at(at);
    h.run_steps(3);
    assert!(h.state().views[0].line_editor.is_some());
    h.get_by_label("Scanned text: the original font is part of the image");
    h.get_by_label("Replacement font");
    h.get_by_value("Helvetica").click();
    h.run_steps(2);
    assert!(h.query_by_label("Keep source fonts").is_none());
    assert!(h.query_by_label("Installed fonts… ⏵").is_none(), "installed faces require the preference");
    h.get_by_label("Times");
    h.get_by_label("Courier");
    h.get_by_label("Inter Regular");
    h.state_mut().match_installed_fonts = true;
    h.run_steps(2);
    h.get_by_label("Installed fonts… ⏵");
    if let Ok(path) = std::env::var("PDFCRAFT_OCR_FONT_SHOT") {
        h.render().unwrap().save(path).unwrap();
    }
    h.get_by_label("Inter Regular").click();
    h.run_steps(4);
    let app = h.state();
    let doc = app.session.get(app.views[0].id).unwrap();
    assert!(doc.text_lines(0).iter().all(|line| line.base_font == "Inter-Regular"));
}
