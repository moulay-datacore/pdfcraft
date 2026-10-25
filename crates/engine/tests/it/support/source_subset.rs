//! An openly licensed in-memory fixture; no private PDF is included.
use pdfcraft_cos::{Dict, Document, Object, SaveOptions, Stream};
use std::sync::Arc;

/// A valid source program with a PDF encoding restricted to A. The installed file has
/// additional Unicode mappings; neither its path nor its bytes are imported as an edit.
pub fn source_subset() -> Arc<Vec<u8>> {
    let mut d = Document::new_empty();
    let pages = d.add(Object::Dict(Dict::new()));
    let mut p = Dict::new();
    p.set(b"Type".to_vec(), Object::name("Page"));
    p.set(b"Parent".to_vec(), Object::Ref(pages));
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
    let page = pdfcraft_model::pages(&d).remove(0);
    let full = super::font_fixture::cff_font();
    let mut program_dict = Dict::new();
    program_dict.set(b"Subtype".to_vec(), Object::name("OpenType"));
    let program = d.add(Object::Stream(Stream::flate(program_dict, &full)));
    let mut descriptor = Dict::new();
    descriptor.set(b"Type".to_vec(), Object::name("FontDescriptor"));
    descriptor.set(b"FontName".to_vec(), Object::name("ABCDEF+RectFont-Regular"));
    descriptor.set(b"FontFile3".to_vec(), Object::Ref(program));
    descriptor.set(b"Flags".to_vec(), Object::Int(32));
    descriptor.set(b"Ascent".to_vec(), Object::Int(1000));
    descriptor.set(b"Descent".to_vec(), Object::Int(-250));
    let descriptor = d.add(Object::Dict(descriptor));
    let mut font = Dict::new();
    font.set(b"Type".to_vec(), Object::name("Font"));
    font.set(b"Subtype".to_vec(), Object::name("Type1"));
    font.set(b"BaseFont".to_vec(), Object::name("ABCDEF+RectFont-Regular"));
    font.set(b"FontDescriptor".to_vec(), Object::Ref(descriptor));
    font.set(b"Encoding".to_vec(), Object::name("WinAnsiEncoding"));
    font.set(b"FirstChar".to_vec(), Object::Int(65));
    font.set(b"LastChar".to_vec(), Object::Int(65));
    font.set(b"Widths".to_vec(), Object::Array(vec![Object::Int(690)]));
    let mut fonts = Dict::new();
    fonts.set(b"F1".to_vec(), Object::Dict(font));
    let mut resources = Dict::new();
    resources.set(b"Font".to_vec(), Object::Dict(fonts));
    let content = d.add(Object::Stream(Stream::flate(Dict::new(), b"BT /F1 18 Tf 30 160 Td (A) Tj ET")));
    d.update_dict(page.obj, |p| {
        p.set(b"Resources".to_vec(), Object::Dict(resources));
        p.set(b"Contents".to_vec(), Object::Ref(content));
    })
    .unwrap();
    Arc::new(pdfcraft_cos::write_full(&d, &SaveOptions::default()).unwrap())
}
