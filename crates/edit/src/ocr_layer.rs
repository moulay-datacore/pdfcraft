//! PdfCraft's own OCR text layers: the invisible text Scan & OCR ▸ Recognize text puts over a
//! scan, which a second run may replace.
//!
//! A layer is one content stream that [`stamp`](crate::stamp) added with the tag [`OCR_TAG`]
//! (`/PCMark /OCR` in the stream dictionary) and that holds nothing but what the recogniser
//! writes (`pdfcraft_ocr::text_layer`): marked content `/OCR BMC … EMC` around one text object
//! in rendering mode 3, `/PCHelv` words placed with `Tm` and shown with `Tj`, inside `q … Q`
//! (files written before #3029 have no `q … Q`). The tag alone is not proof: another writer may
//! keep, copy or forge it, and Edit text rewrites a layer's words in place under the same tag
//! (a corrected word is drawn visibly). So a stream counts only when its operators are a layer
//! and nothing else; anything else on the page, an edited layer included, is never taken away.

use pdfcraft_cos::{Document, Object};

use crate::EditError;

/// The `/PCMark` tag of an OCR text layer's content stream.
pub const OCR_TAG: &str = "OCR";

/// The largest decoded stream read as a layer: a dense page's layer is a few hundred kilobytes
/// (about 60 bytes a word).
const MAX_LAYER: usize = 8 * 1024 * 1024;

/// PdfCraft's own OCR text on a page.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OcrLayers {
    /// The content streams that hold it.
    pub streams: usize,
    /// The words in them.
    pub words: usize,
}

/// The page's `/Contents` as a list, read without changing anything (a direct stream, which
/// the format doesn't allow there, lists nothing).
fn content_list(doc: &Document, page: &pdfcraft_model::Page) -> Vec<Object> {
    match page.dict.get(b"Contents") {
        Some(Object::Ref(r)) => match doc.get(*r).as_array() {
            Some(a) => a.clone(),
            None => vec![Object::Ref(*r)],
        },
        Some(Object::Array(a)) => a.clone(),
        _ => Vec::new(),
    }
}

/// The number of words in `o` when it is one of PdfCraft's own OCR layers.
fn own_layer_words(doc: &Document, o: &Object) -> Option<usize> {
    if crate::stream_tag(doc, o).as_deref() != Some(OCR_TAG) {
        return None;
    }
    let Object::Stream(stream) = &*doc.resolve(o) else { return None };
    layer_words(&stream.decoded_within(MAX_LAYER).ok()?)
}

/// The number of words in `data` when it is an OCR text layer as PdfCraft writes it (see the
/// module documentation), else `None`.
pub(crate) fn layer_words(data: &[u8]) -> Option<usize> {
    let number = |o: &Object| o.as_f64().filter(|v| v.is_finite());
    let mut ops = pdfcraft_content::ops(data);
    let (mut words, mut depth) = (0usize, 0usize);
    let (mut opened, mut closed, mut in_text, mut invisible) = (false, false, false, false);
    for op in ops.by_ref() {
        let operands = op.operands.as_slice();
        match op.op.as_slice() {
            // Nothing comes before the marked content or after it.
            _ if closed => return None,
            b"BMC" if !opened && matches!(operands, [Object::Name(n)] if n.as_slice() == OCR_TAG.as_bytes()) => opened = true,
            _ if !opened => return None,
            b"q" if operands.is_empty() && !in_text => depth = depth.checked_add(1)?,
            b"Q" if operands.is_empty() && !in_text => depth = depth.checked_sub(1)?,
            b"BT" if operands.is_empty() && !in_text => in_text = true,
            b"ET" if operands.is_empty() && in_text => in_text = false,
            // Invisible text only: any other rendering mode draws the words.
            b"Tr" if matches!(operands, [mode] if number(mode) == Some(3.0)) => invisible = true,
            b"Tf" if matches!(operands, [Object::Name(f), size] if f.as_slice() == b"PCHelv" && number(size).is_some()) => {}
            b"Tm" if in_text && operands.len() == 6 && operands.iter().all(|n| number(n).is_some()) => {}
            b"Tj" if in_text && invisible && matches!(operands, [Object::String(_)]) => words = words.checked_add(1)?,
            b"EMC" if operands.is_empty() && depth == 0 && !in_text => closed = true,
            _ => return None,
        }
    }
    (closed && ops.skipped() == 0 && ops.ambiguous_images() == 0).then_some(words)
}

/// PdfCraft's own OCR text layers on `page` (0-based).
pub fn own_ocr_layers(doc: &Document, page: usize) -> Result<OcrLayers, EditError> {
    let p = crate::page_list(doc).into_iter().nth(page).ok_or(EditError::NoSuchPage(page))?;
    let mut found = OcrLayers::default();
    for words in content_list(doc, &p).iter().filter_map(|o| own_layer_words(doc, o)) {
        found.streams = found.streams.saturating_add(1);
        found.words = found.words.saturating_add(words);
    }
    Ok(found)
}

/// Remove PdfCraft's own OCR text layers from `page` (0-based), with the `q … Q` wrapper the
/// first one may have put around the page's content when nothing after it still needs it; the
/// rest of the page is left as it is. Returns what was removed.
pub fn remove_own_ocr_layers(doc: &mut Document, page: usize) -> Result<OcrLayers, EditError> {
    let found = own_ocr_layers(doc, page)?;
    if found.streams > 0 {
        let p = crate::page_list(doc).into_iter().nth(page).ok_or(EditError::NoSuchPage(page))?;
        crate::remove_streams(doc, &p, |doc, o| own_layer_words(doc, o).is_some())?;
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pdfcraft_cos::{Dict, SaveOptions, Stream, write_incremental};

    use super::*;

    /// The layer PdfCraft writes today (two words) and the one it wrote before #3029 (no q … Q).
    const LAYER: &str = "/OCR BMC\nq\nBT\n3 Tr\n/PCHelv 1 Tf\n40 0 0 12 72 700 Tm (Hello) Tj\n30 0 0 12 120 700 Tm (world) Tj\nET\nQ\nEMC\n";
    const OLD_LAYER: &str = "/OCR BMC\nBT\n3 Tr\n/PCHelv 1 Tf\n40 0 0 12 72 650 Tm (Bonjour) Tj\nET\nEMC\n";

    /// One page whose content is `parts` (tag, bytes), in order.
    fn page(parts: &[(&str, &str)]) -> Document {
        let objs = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 600 800] >>",
            "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        ];
        let mut out = b"%PDF-1.7\n".to_vec();
        let mut offs = Vec::new();
        for (i, o) in objs.iter().enumerate() {
            offs.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n{o}\nendobj\n", i + 1).as_bytes());
        }
        let x = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
        for o in offs {
            out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n", objs.len() + 1).as_bytes());
        let mut doc = Document::open(Arc::new(out)).unwrap();
        let list = parts
            .iter()
            .map(|(tag, bytes)| {
                let dict = if tag.is_empty() { Dict::new() } else { crate::tagged(tag) };
                Object::Ref(doc.add(Object::Stream(Stream::from_raw(dict, bytes.as_bytes().to_vec()))))
            })
            .collect();
        let p = crate::page_list(&doc).remove(0);
        doc.update_dict(p.obj, |d| d.set(b"Contents".to_vec(), Object::Array(list))).unwrap();
        doc
    }

    /// The page's decoded content streams, in order, after a save and reopen.
    fn streams(doc: &Document) -> Vec<String> {
        let doc = Document::open(Arc::new(write_incremental(doc, &SaveOptions::default()).unwrap())).unwrap();
        let p = crate::page_list(&doc).remove(0);
        content_list(&doc, &p)
            .iter()
            .map(|o| match &*doc.resolve(o) {
                Object::Stream(s) => String::from_utf8_lossy(&s.decoded().unwrap()).into_owned(),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn layers_of_both_formats_are_found_and_counted() {
        assert_eq!(layer_words(LAYER.as_bytes()), Some(2));
        assert_eq!(layer_words(OLD_LAYER.as_bytes()), Some(1));
        // A layer whose words were all deleted is still a layer.
        assert_eq!(layer_words(b"/OCR BMC\nq\nBT\n3 Tr\n/PCHelv 1 Tf\nET\nQ\nEMC\n"), Some(0));
        let doc = page(&[("", "BT /F1 9 Tf 72 720 Td (Real text) Tj ET"), (OCR_TAG, LAYER), (OCR_TAG, OLD_LAYER)]);
        assert_eq!(own_ocr_layers(&doc, 0).unwrap(), OcrLayers { streams: 2, words: 3 });
        assert_eq!(own_ocr_layers(&doc, 1), Err(EditError::NoSuchPage(1)));
    }

    /// Only a layer and nothing else counts: an edited layer (a corrected word drawn visibly),
    /// another writer's stream under a copied tag, an untagged copy of a layer and anything
    /// truncated or with more after it are the page's own content.
    #[test]
    fn edited_forged_and_untagged_layers_are_not_taken() {
        let edited = LAYER.replace("30 0 0 12 120 700 Tm (world) Tj", "0 Tr 0 0 0 rg /F1 12 Tf 1 0 0 1 120 700 Tm (World) Tj");
        let not_layers = [
            edited.as_str(),
            "BT /F1 9 Tf 3 Tr (forged) Tj ET",
            "/OCR BMC\nBT\n/PCHelv 1 Tf\n1 0 0 1 0 0 Tm (shown) Tj\nET\nEMC\n",
            "/OCR BMC\nBT\n3 Tr\n/F1 1 Tf\n1 0 0 1 0 0 Tm (other font) Tj\nET\nEMC\n",
            "/OCR BMC\nBT\n3 Tr\n/PCHelv 1 Tf\n1 0 0 1 0 0 Tm (x) Tj\nET\nEMC\n0 0 10 10 re f\n",
            "/OCR BMC\nBT\n3 Tr\n/PCHelv 1 Tf\n1 0 0 1 0 0 Tm (x) Tj\nET\n",
            "/OCR BMC\nq\nBT\n3 Tr\n/PCHelv 1 Tf\nET\nEMC\n",
            "/OCR BMC\nBT\n3 Tr\n/PCHelv 1 Tf\n1 0 0 1 0 0 Tm [(x) 50 (y)] TJ\nET\nEMC\n",
            "/OCR BMC\nBT\n3 Tr\n/PCHelv 1 Tf\n1 0 0 1 0 0 Tm (x) Tj\nET\nEMC\n) stray",
            "/Artifact BMC\nBT\n3 Tr\n/PCHelv 1 Tf\nET\nEMC\n",
            "",
        ];
        for data in not_layers {
            assert_eq!(layer_words(data.as_bytes()), None, "{data:?}");
            let mut doc = page(&[(OCR_TAG, data)]);
            let before = streams(&doc);
            assert_eq!(remove_own_ocr_layers(&mut doc, 0).unwrap(), OcrLayers::default(), "{data:?}");
            assert_eq!(streams(&doc), before);
        }
        // A layer without the tag is someone else's.
        let mut doc = page(&[("", LAYER), ("HeaderFooter", OLD_LAYER)]);
        assert_eq!(remove_own_ocr_layers(&mut doc, 0).unwrap(), OcrLayers::default());
        assert_eq!(streams(&doc).len(), 2);
    }

    /// Removing the layers leaves the rest of the page as it was, and the q/Q wrapper the first
    /// stamp put around the page's content goes too once nothing after it needs it.
    #[test]
    fn removing_the_layers_keeps_the_rest_of_the_page() {
        let original = "2 0 0 2 0 0 cm BT /F1 9 Tf (Hi) Tj ET";
        let mut doc = page(&[("", original)]);
        crate::stamp(&mut doc, 0, OCR_TAG, LAYER.as_bytes().to_vec()).unwrap();
        let stamped = streams(&doc);
        assert!(stamped.len() > 2 && stamped.iter().any(|s| s == LAYER), "{stamped:?}");
        assert_eq!(remove_own_ocr_layers(&mut doc, 0).unwrap(), OcrLayers { streams: 1, words: 2 });
        assert_eq!(streams(&doc), [original], "the content list is back to what it was");

        // Content drawn after the layer stays, and so does the wrapper it relies on.
        let mut doc = page(&[("", original)]);
        crate::stamp(&mut doc, 0, OCR_TAG, LAYER.as_bytes().to_vec()).unwrap();
        crate::stamp(&mut doc, 0, "Probe", b"q 0 0 10 10 re f Q\n".to_vec()).unwrap();
        let before = streams(&doc);
        remove_own_ocr_layers(&mut doc, 0).unwrap();
        let after = streams(&doc);
        let without: Vec<&String> = before.iter().filter(|s| s.as_str() != LAYER).collect();
        assert_eq!(after.iter().collect::<Vec<_>>(), without);
        assert!(after.iter().any(|s| s.contains("re f")) && after.iter().any(|s| s == original));

        // Old- and new-format layers beside an edited one: only the edited one stays.
        let edited = LAYER.replace("(world) Tj", "(world) Tj 0 Tr (shown) Tj");
        let mut doc = page(&[("", original), (OCR_TAG, OLD_LAYER), (OCR_TAG, &edited), (OCR_TAG, LAYER)]);
        assert_eq!(remove_own_ocr_layers(&mut doc, 0).unwrap(), OcrLayers { streams: 2, words: 3 });
        assert_eq!(streams(&doc), [original.to_string(), edited]);
        assert_eq!(own_ocr_layers(&doc, 0).unwrap(), OcrLayers::default());
    }

    /// The check reads untrusted content: every cut of a layer, and bytes that are no content at
    /// all, come back as an answer, never a panic.
    #[test]
    fn any_bytes_are_answered() {
        for data in [LAYER, OLD_LAYER] {
            for end in 0..data.len() {
                let cut = data.as_bytes().get(..end).unwrap_or_default();
                assert!(layer_words(cut).is_none_or(|w| w <= 2), "{cut:?}");
            }
        }
        for junk in [&b"\xff\xfe(((("[..], b"/OCR BMC Q Q Q EMC", b"/OCR BMC ET EMC", b"/OCR BMC BT BT ET ET EMC", b"/OCR /OCR BMC EMC"] {
            assert_eq!(layer_words(junk), None, "{junk:?}");
        }
    }
}
