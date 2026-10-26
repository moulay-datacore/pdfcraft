//! PdfCraft's own OCR text layers: the invisible text Scan & OCR ▸ Recognize text puts over a
//! scan, which a second run may replace.
//!
//! A layer is one content stream that [`stamp`](crate::stamp) added with the tag [`OCR_TAG`]
//! (`/PCMark /OCR` in the stream dictionary) and that holds nothing but what the recogniser
//! writes (`pdfcraft_ocr::text_layer`): marked content `/OCR BMC … EMC` around one text object
//! in rendering mode 3, `/PCHelv` words placed with `Tm` and shown with `Tj`, inside `q … Q`
//! (files written before #3029 have no `q … Q`). The tag alone is not proof: another writer may
//! keep, copy or forge it, and Edit text rewrites a layer's words in place under the same tag.
//! A corrected word may then be drawn visibly, which the operators show, or stay invisible text
//! in the layer's own font (a line edit), which they don't. So a stream counts only when its
//! operators are a layer and nothing else, and its words are still the ones PdfCraft wrote: a
//! layer records the SHA-256 digest of its content when it is written (`/PCDigest`), and an edit
//! through Edit text marks a layer it changes (`/PCEdited true`), which also covers layers
//! written before they had a digest. Anything else on the page, an edited layer included, is
//! the page's own text and is never taken away; [`Kept`] says why a stream under the tag is
//! kept.

use pdfcraft_cos::{Dict, Document, ObjRef, Object, PdfString, Stream};
use sha2::{Digest, Sha256};

use crate::EditError;

/// The `/PCMark` tag of an OCR text layer's content stream.
pub const OCR_TAG: &str = "OCR";

/// The stream dictionary key of the SHA-256 digest of a layer's decoded content as written.
const DIGEST: &[u8] = b"PCDigest";

/// The stream dictionary key Edit text sets on a layer whose content it changed.
const EDITED: &[u8] = b"PCEdited";

/// The largest decoded stream read as a layer: a dense page's layer is a few hundred kilobytes
/// (about 60 bytes a word).
const MAX_LAYER: usize = 8 * 1024 * 1024;

/// PdfCraft's own OCR text on a page.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OcrLayers {
    /// The content streams that hold it, in content order.
    pub streams: Vec<ObjRef>,
    /// The words in them.
    pub words: usize,
    /// The characters those words show, blanks left out: as many glyphs as text extraction
    /// finds in them (`/PCHelv` is a one-byte font).
    pub glyphs: usize,
}

impl OcrLayers {
    /// Whether the page has any.
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
}

/// The OCR text PdfCraft added to a page: the layers still as it wrote them, which a new
/// reading may replace, and the streams under the same tag that are not (any longer) only that.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OcrText {
    /// The layers still as PdfCraft wrote them.
    pub own: OcrLayers,
    /// The streams tagged [`OCR_TAG`] that are not PdfCraft's OCR text as it wrote it, in content
    /// order, each with why. They are the page's own text, never removed.
    pub kept: Vec<(ObjRef, Kept)>,
}

/// Why a content stream tagged [`OCR_TAG`] is not PdfCraft's OCR text as it wrote it, and so is
/// kept as the page's own text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kept {
    /// It was changed since PdfCraft wrote it: Edit text marked it, or its content is no longer
    /// what its digest records (a word corrected, visibly or not).
    Edited,
    /// Its content can't be read: it doesn't decode (damaged, or in a filter PdfCraft doesn't
    /// decode), or decodes to more than a layer ever holds (8 MiB). What it holds is unknown.
    Unreadable,
    /// It holds something other than OCR text as PdfCraft writes it, and nothing records that
    /// PdfCraft wrote it and it changed since: another program's stream under the tag (or its
    /// digest, which PdfCraft writes as a string, is something else), or a layer from before
    /// layers had a digest that something changed without marking it.
    Foreign,
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

/// What `o`, one of a page's content streams, is when it is tagged [`OCR_TAG`]: a layer still as
/// PdfCraft wrote it, with its words and the glyphs they show, or why it isn't.
fn tagged_text(doc: &Document, o: &Object) -> Option<Result<(usize, usize), Kept>> {
    if crate::stream_tag(doc, o).as_deref() != Some(OCR_TAG) {
        return None;
    }
    let Object::Stream(stream) = &*doc.resolve(o) else { return None };
    Some(unedited(stream))
}

/// The words and glyphs of `stream` when it is a layer PdfCraft wrote and nothing has changed
/// it since (see the module documentation), else why it isn't.
fn unedited(stream: &Stream) -> Result<(usize, usize), Kept> {
    if stream.dict.contains(EDITED) {
        return Err(Kept::Edited);
    }
    // Strictly: a layer read only in part (damaged) is not known to be one.
    let data = stream.decoded_strict_within(MAX_LAYER).map_err(|_| Kept::Unreadable)?;
    match stream.dict.get(DIGEST) {
        // Written before layers had a digest: the operators alone tell.
        None => {}
        Some(Object::String(s)) if s.bytes == digest(&data) => {}
        // PdfCraft wrote it, and it has changed since.
        Some(Object::String(_)) => return Err(Kept::Edited),
        Some(_) => return Err(Kept::Foreign),
    }
    layer_text(&data).ok_or(Kept::Foreign)
}

/// The SHA-256 digest of `data`.
fn digest(data: &[u8]) -> Vec<u8> {
    Sha256::digest(data).to_vec()
}

/// The stream dictionary of a new OCR text layer holding `content`: tagged [`OCR_TAG`], with the
/// digest of `content`.
pub(crate) fn layer_dict(content: &[u8]) -> Dict {
    let mut dict = crate::tagged(OCR_TAG);
    dict.set(DIGEST.to_vec(), Object::String(PdfString { bytes: digest(content), hex: true }));
    dict
}

/// Mark `dict`, the dictionary of a content stream whose bytes an edit changed, as no longer
/// only PdfCraft's OCR text when it is tagged [`OCR_TAG`]: a line edit keeps a corrected word
/// invisible and in the layer's font, which the operators don't show, nor anything else in a
/// layer written before layers had a digest. Other streams are left as they are.
pub(crate) fn mark_edited(dict: &mut Dict) {
    if dict.name(b"PCMark") == Some(OCR_TAG.as_bytes()) {
        dict.set(EDITED.to_vec(), Object::Bool(true));
    }
}

/// The words in `data`, and the glyphs they show, when it is an OCR text layer as PdfCraft
/// writes it (see the module documentation), else `None`.
pub(crate) fn layer_text(data: &[u8]) -> Option<(usize, usize)> {
    let number = |o: &Object| o.as_f64().filter(|v| v.is_finite());
    let mut ops = pdfcraft_content::ops(data);
    let (mut words, mut glyphs, mut depth) = (0usize, 0usize, 0usize);
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
            b"Tj" if in_text && invisible => {
                let [Object::String(s)] = operands else { return None };
                words = words.checked_add(1)?;
                // One byte, one glyph (WinAnsiEncoding); text extraction leaves out the blanks
                // (tab, line ends, space, no-break space).
                glyphs = glyphs.checked_add(s.bytes.iter().filter(|b| !matches!(b, 0x09..=0x0D | 0x20 | 0xA0)).count())?;
            }
            b"EMC" if operands.is_empty() && depth == 0 && !in_text => closed = true,
            _ => return None,
        }
    }
    (closed && ops.skipped() == 0 && ops.ambiguous_images() == 0).then_some((words, glyphs))
}

/// The OCR text PdfCraft added to `page`. Only the streams tagged [`OCR_TAG`] are decoded.
fn text_on(doc: &Document, page: &pdfcraft_model::Page) -> OcrText {
    let mut found = OcrText::default();
    for o in content_list(doc, page) {
        // A layer is a stream, which is always indirect: one listed directly is no layer.
        let Object::Ref(r) = o else { continue };
        match tagged_text(doc, &o) {
            Some(Ok((words, glyphs))) => {
                found.own.streams.push(r);
                found.own.words = found.own.words.saturating_add(words);
                found.own.glyphs = found.own.glyphs.saturating_add(glyphs);
            }
            Some(Err(why)) => found.kept.push((r, why)),
            None => {}
        }
    }
    found
}

/// PdfCraft's own OCR text layers on `page` (0-based).
pub fn own_ocr_layers(doc: &Document, page: usize) -> Result<OcrLayers, EditError> {
    let p = crate::page_list(doc).into_iter().nth(page).ok_or(EditError::NoSuchPage(page))?;
    Ok(text_on(doc, &p).own)
}

/// The pages (0-based) that have PdfCraft's own OCR text layers, in order, with the layers: one
/// walk of the page tree, so it is cheap on a long document.
pub fn own_ocr_pages(doc: &Document) -> Vec<(usize, OcrLayers)> {
    crate::page_list(doc).iter().enumerate().map(|(i, p)| (i, text_on(doc, p).own)).filter(|(_, l)| !l.is_empty()).collect()
}

/// Of `pages` (0-based; each once, in order; pages past the end have none), those that have
/// streams tagged [`OCR_TAG`], with what they are ([`OcrText`]): OCR text PdfCraft added as it
/// wrote it, or kept, and why. One walk of the page tree, decoding only those pages' streams
/// tagged [`OCR_TAG`].
pub fn ocr_text_on(doc: &Document, pages: &[usize]) -> Vec<(usize, OcrText)> {
    let all = crate::page_list(doc);
    pages
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<usize>>()
        .into_iter()
        .filter_map(|i| Some((i, text_on(doc, all.get(i)?))))
        .filter(|(_, t)| !t.own.is_empty() || !t.kept.is_empty())
        .collect()
}

/// Remove PdfCraft's own OCR text layers from each of `pages` (0-based), with the `q … Q`
/// wrapper the first one may have put around the page's content when nothing after it still
/// needs it; the rest of each page, edited layers included, is left as it is. One walk of the
/// page tree. Returns what was removed, for each page that had some.
pub fn remove_own_ocr_layers(doc: &mut Document, pages: &[usize]) -> Result<Vec<(usize, OcrLayers)>, EditError> {
    let all = crate::page_list(doc);
    crate::check(pages, all.len())?;
    let mut removed = Vec::new();
    // Each page once, in order.
    for i in pages.iter().copied().collect::<std::collections::BTreeSet<usize>>() {
        let Some(p) = all.get(i) else { continue };
        let found = text_on(doc, p).own;
        if found.is_empty() {
            continue;
        }
        crate::remove_streams(doc, p, |_, o| matches!(o, Object::Ref(r) if found.streams.contains(r)))?;
        removed.push((i, found));
    }
    Ok(removed)
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

    /// The streams and words of `layers`.
    fn counts(layers: &OcrLayers) -> (usize, usize) {
        (layers.streams.len(), layers.words)
    }

    /// Remove the layers of page 0; the streams and words removed.
    fn remove(doc: &mut Document) -> (usize, usize) {
        let removed = remove_own_ocr_layers(doc, &[0]).unwrap();
        assert!(removed.iter().all(|(page, l)| *page == 0 && !l.is_empty()), "{removed:?}");
        removed.first().map_or((0, 0), |(_, l)| counts(l))
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
        assert_eq!(layer_text(LAYER.as_bytes()), Some((2, 10)));
        assert_eq!(layer_text(OLD_LAYER.as_bytes()), Some((1, 7)));
        // A layer whose words were all deleted is still a layer.
        assert_eq!(layer_text(b"/OCR BMC\nq\nBT\n3 Tr\n/PCHelv 1 Tf\nET\nQ\nEMC\n"), Some((0, 0)));
        // Blanks show no glyph to text extraction: tab, line ends, space and no-break space.
        assert_eq!(layer_text(b"/OCR BMC\nBT\n3 Tr\n/PCHelv 1 Tf\n1 0 0 1 0 0 Tm (a\\tb\\r c\\240d\\205) Tj\nET\nEMC\n"), Some((1, 5)));
        let doc = page(&[("", "BT /F1 9 Tf 72 720 Td (Real text) Tj ET"), (OCR_TAG, LAYER), (OCR_TAG, OLD_LAYER)]);
        let layers = own_ocr_layers(&doc, 0).unwrap();
        assert_eq!((counts(&layers), layers.glyphs), ((2, 3), 17));
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
            assert_eq!(layer_text(data.as_bytes()), None, "{data:?}");
            let mut doc = page(&[(OCR_TAG, data)]);
            // Nothing records that PdfCraft wrote it: not "edited", someone else's.
            let text = ocr_text_on(&doc, &[0]);
            assert!(
                matches!(text.as_slice(), [(0, t)] if t.own.is_empty() && matches!(t.kept.as_slice(), [(_, Kept::Foreign)])),
                "{data:?}: {text:?}"
            );
            let before = streams(&doc);
            assert_eq!(remove(&mut doc), (0, 0), "{data:?}");
            assert_eq!(streams(&doc), before);
        }
        // A layer without the tag is someone else's.
        let mut doc = page(&[("", LAYER), ("HeaderFooter", OLD_LAYER)]);
        assert_eq!(remove(&mut doc), (0, 0));
        assert_eq!(streams(&doc).len(), 2);
    }

    /// `doc` saved and opened again.
    fn reopen(doc: &Document) -> Document {
        Document::open(Arc::new(write_incremental(doc, &SaveOptions::default()).unwrap())).unwrap()
    }

    /// The stream `r` with `change` made to its dictionary and data, kept under the same number.
    fn rewrite(doc: &mut Document, r: ObjRef, change: impl FnOnce(&mut Dict, &mut Vec<u8>)) {
        let Object::Stream(stream) = &*doc.get(r) else { panic!("{r:?} is no stream") };
        let (mut dict, mut data) = (stream.dict.clone(), stream.decoded().unwrap());
        dict.remove(b"Length");
        change(&mut dict, &mut data);
        doc.set(r, Object::Stream(Stream::flate(dict, &data)));
    }

    /// Correct `from` to `to` in the line of page 0 that has it, as Edit text's line edit does.
    fn correct(doc: &mut Document, from: &str, to: &str) {
        let lines = crate::text_lines(doc, 0).unwrap();
        let (line, text) = lines.iter().enumerate().find(|(_, l)| l.text.contains(from)).map(|(i, l)| (i, l.text.replace(from, to))).unwrap();
        crate::replace_line(doc, 0, line, &text).unwrap();
    }

    /// The acceptance defect: Edit text's line edit rewrites a word of a layer in place, as
    /// invisible text in the layer's own font, so the stream was still a layer by its operators
    /// alone and reading the page again with replace_ocr dropped the correction. A layer that
    /// changed is the user's text, kept: in the format written today, and in one written before
    /// layers had a digest.
    #[test]
    fn a_layer_corrected_in_place_is_the_user_s_text() {
        let original = "BT /F1 9 Tf 72 720 Td (Real text) Tj ET";
        let mut stamped = page(&[("", original)]);
        let layer = crate::stamp(&mut stamped, 0, OCR_TAG, LAYER.as_bytes().to_vec()).unwrap();
        assert_eq!(counts(&own_ocr_layers(&reopen(&stamped), 0).unwrap()), (1, 2), "the digest is saved with the layer");
        let mut legacy = stamped.clone();
        rewrite(&mut legacy, layer, |dict, _| {
            dict.remove(DIGEST);
        });
        for (format, mut doc) in [("with a digest", stamped), ("from before digests", legacy)] {
            assert_eq!(counts(&own_ocr_layers(&doc, 0).unwrap()), (1, 2), "{format}");
            correct(&mut doc, "Hello", "Hallo");
            // Still invisible text in /PCHelv and nothing else: the operators alone can't tell.
            let after = streams(&doc);
            let corrected = after.iter().find(|s| s.contains("Hallo")).unwrap();
            assert!(layer_text(corrected.as_bytes()).is_some(), "{format}: {corrected}");
            assert_eq!(own_ocr_layers(&doc, 0).unwrap(), OcrLayers::default(), "{format}");
            assert!(own_ocr_pages(&doc).is_empty() && own_ocr_pages(&reopen(&doc)).is_empty(), "{format}");
            let text = ocr_text_on(&doc, &[0, 0, 7]);
            assert!(
                matches!(text.as_slice(), [(0, t)] if t.own.is_empty() && matches!(t.kept.as_slice(), [(_, Kept::Edited)])),
                "{format}: {text:?}"
            );
            assert_eq!(remove(&mut doc), (0, 0), "{format}");
            assert_eq!(streams(&doc), after, "{format}: the correction stays");
        }
    }

    /// Any change to a layer's content, by any writer that keeps its dictionary, makes it the
    /// page's own text: the digest no longer matches, or isn't one. A layer written before
    /// layers had a digest is told by its operators alone, unless Edit text marked it.
    #[test]
    fn a_layer_whose_content_changed_is_not_taken() {
        let mut doc = page(&[]);
        let layer = crate::stamp(&mut doc, 0, OCR_TAG, LAYER.as_bytes().to_vec()).unwrap();
        type Change = fn(&mut Dict, &mut Vec<u8>);
        let changes: [(&str, Change, Kept); 5] = [
            ("a word changed", |_, data| *data = LAYER.replace("(world)", "(World)").into_bytes(), Kept::Edited),
            ("a word deleted", |_, data| *data = LAYER.replace("30 0 0 12 120 700 Tm (world) Tj\n", "").into_bytes(), Kept::Edited),
            ("the digest replaced", |dict, _| dict.set(DIGEST.to_vec(), Object::String(PdfString::literal(vec![0; 32]))), Kept::Edited),
            // PdfCraft writes its digest as a string: this one is someone else's.
            ("the digest not a string", |dict, _| dict.set(DIGEST.to_vec(), Object::Int(1)), Kept::Foreign),
            ("marked edited", |dict, _| dict.set(EDITED.to_vec(), Object::Bool(true)), Kept::Edited),
        ];
        for (what, change, why) in changes {
            let mut changed = doc.clone();
            rewrite(&mut changed, layer, change);
            let text = ocr_text_on(&changed, &[0]);
            assert!(matches!(text.as_slice(), [(0, t)] if t.own.is_empty() && t.kept == [(layer, why)]), "{what}: {text:?}");
            assert_eq!(remove(&mut changed), (0, 0), "{what}");
        }
        // Only the content counts: compressed differently, a layer is still PdfCraft's.
        let mut recoded = doc.clone();
        let Object::Stream(stream) = &*doc.get(layer) else { panic!() };
        let mut dict = stream.dict.clone();
        dict.remove(b"Filter");
        dict.remove(b"Length");
        recoded.set(layer, Object::Stream(Stream::from_raw(dict, LAYER.as_bytes().to_vec())));
        assert_eq!(counts(&own_ocr_layers(&recoded, 0).unwrap()), (1, 2));

        // Before digests, only the operators tell: a word another writer changed in place goes
        // unseen (Edit text marks what it changes, see above).
        let mut legacy = page(&[(OCR_TAG, LAYER)]);
        let r = own_ocr_layers(&legacy, 0).unwrap().streams[0];
        rewrite(&mut legacy, r, |_, data| *data = LAYER.replace("(world)", "(World)").into_bytes());
        assert_eq!(counts(&own_ocr_layers(&legacy, 0).unwrap()), (1, 2));
    }

    /// The wording defect: a stream under the tag whose content couldn't be read was said to be
    /// OCR text edited since it was added. What it holds is unknown, so it is kept, and said to be
    /// unreadable: damaged data (which a tolerant decode reads in part), a filter nothing here
    /// decodes, and a layer past the 8 MiB a layer is read within. One just within it is a layer,
    /// and a layer without its digest is told by its operators (another writer's content beside
    /// a copied tag and digest is not a layer).
    #[test]
    fn a_tagged_stream_that_cannot_be_read_is_kept_as_unreadable() {
        let mut doc = page(&[]);
        let layer = crate::stamp(&mut doc, 0, OCR_TAG, LAYER.as_bytes().to_vec()).unwrap();
        // What the page's OCR text is with `stream` in the layer's place, and what removing takes.
        let with = |stream: Stream| {
            let mut changed = doc.clone();
            changed.set(layer, Object::Stream(stream));
            let text = ocr_text_on(&changed, &[0]).into_iter().next().map(|(_, t)| (counts(&t.own), t.kept));
            (text, remove(&mut changed))
        };
        let kept = |why: Kept| (Some(((0, 0), vec![(layer, why)])), (0, 0));
        let padded = |len: usize| {
            let mut data = LAYER.as_bytes().to_vec();
            data.resize(len, b' ');
            data
        };
        let whole = Stream::flate(layer_dict(LAYER.as_bytes()), LAYER.as_bytes());
        assert_eq!(with(whole.clone()), (Some(((1, 2), Vec::new())), (1, 2)), "as written, it is a layer");

        let cut = Stream::from_raw(whole.dict.clone(), whole.raw.get(..whole.raw.len() / 2).unwrap_or_default().to_vec());
        assert!(cut.decoded().is_ok_and(|d| d.len() < LAYER.len() && LAYER.as_bytes().starts_with(&d)), "read in part");
        assert_eq!(with(cut), kept(Kept::Unreadable), "damaged");
        let mut image = layer_dict(LAYER.as_bytes());
        image.set(b"Filter".to_vec(), Object::name("JBIG2Decode"));
        assert_eq!(with(Stream::from_raw(image, LAYER.as_bytes().to_vec())), kept(Kept::Unreadable), "a filter not decoded here");
        let too_large = padded(MAX_LAYER + 1);
        assert_eq!(with(Stream::flate(layer_dict(&too_large), &too_large)), kept(Kept::Unreadable), "too large");
        let largest = padded(MAX_LAYER);
        assert_eq!(with(Stream::flate(layer_dict(&largest), &largest)), (Some(((1, 2), Vec::new())), (1, 2)), "just within");

        let other = b"BT /F1 9 Tf 72 700 Td (Another program) Tj ET".to_vec();
        assert_eq!(with(Stream::flate(layer_dict(&other), &other)), kept(Kept::Foreign), "not a layer, whatever its digest");
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
        assert_eq!(remove(&mut doc), (1, 2));
        assert_eq!(streams(&doc), [original], "the content list is back to what it was");

        // Content drawn after the layer stays, and so does the wrapper it relies on.
        let mut doc = page(&[("", original)]);
        crate::stamp(&mut doc, 0, OCR_TAG, LAYER.as_bytes().to_vec()).unwrap();
        crate::stamp(&mut doc, 0, "Probe", b"q 0 0 10 10 re f Q\n".to_vec()).unwrap();
        let before = streams(&doc);
        remove(&mut doc);
        let after = streams(&doc);
        let without: Vec<&String> = before.iter().filter(|s| s.as_str() != LAYER).collect();
        assert_eq!(after.iter().collect::<Vec<_>>(), without);
        assert!(after.iter().any(|s| s.contains("re f")) && after.iter().any(|s| s == original));

        // Old- and new-format layers beside an edited one: only the edited one stays.
        let edited = LAYER.replace("(world) Tj", "(world) Tj 0 Tr (shown) Tj");
        let mut doc = page(&[("", original), (OCR_TAG, OLD_LAYER), (OCR_TAG, &edited), (OCR_TAG, LAYER)]);
        assert_eq!(remove(&mut doc), (2, 3));
        assert_eq!(streams(&doc), [original.to_string(), edited]);
        assert_eq!(own_ocr_layers(&doc, 0).unwrap(), OcrLayers::default());
    }

    /// The reviewer's defect: listing the pages with OCR text walked the page tree once for every
    /// page, so opening Recognize Text on a 4000-page scan froze the app for most of a minute.
    /// Finding them, and removing them from many pages, is one walk; the layers' streams are
    /// named, so a later change to the page can be told apart.
    #[test]
    fn layers_on_many_pages_are_found_and_removed_in_one_walk() {
        const PAGES: usize = 5000;
        let mut doc = page(&[("", "BT /F1 9 Tf (p) Tj ET")]);
        let first = crate::page_list(&doc).remove(0);
        let content = first.dict.get(b"Contents").cloned().unwrap();
        let pages_ref = match doc.get(first.obj).as_dict().and_then(|d| d.get(b"Parent")) {
            Some(Object::Ref(r)) => *r,
            other => panic!("{other:?}"),
        };
        let mut kids = vec![Object::Ref(first.obj)];
        for _ in 1..PAGES {
            let mut d = Dict::new();
            d.set(b"Type".to_vec(), Object::name("Page"));
            d.set(b"Parent".to_vec(), Object::Ref(pages_ref));
            d.set(b"Contents".to_vec(), content.clone());
            kids.push(Object::Ref(doc.add(Object::Dict(d))));
        }
        doc.update_dict(pages_ref, |d| {
            d.set(b"Kids".to_vec(), Object::Array(kids));
            d.set(b"Count".to_vec(), Object::Int(PAGES as i64));
        })
        .unwrap();
        let mut layered = Vec::new();
        for page in [0, 2500, PAGES - 1] {
            layered.push(crate::stamp(&mut doc, page, OCR_TAG, LAYER.as_bytes().to_vec()).unwrap());
        }
        let started = std::time::Instant::now();
        let found = own_ocr_pages(&doc);
        assert_eq!(found.iter().map(|(p, l)| (*p, l.words)).collect::<Vec<_>>(), [(0, 2), (2500, 2), (PAGES - 1, 2)]);
        assert_eq!(found.iter().flat_map(|(_, l)| l.streams.clone()).collect::<Vec<_>>(), layered);
        let all: Vec<usize> = (0..PAGES).rev().collect();
        let removed = remove_own_ocr_layers(&mut doc, &all).unwrap();
        assert_eq!(removed, found, "each page once, in order");
        assert!(own_ocr_pages(&doc).is_empty());
        // Quadratic, this took minutes; linear, well under a second in a release build.
        assert!(started.elapsed() < std::time::Duration::from_secs(20), "{:?}", started.elapsed());
        assert_eq!(remove_own_ocr_layers(&mut doc, &[PAGES]), Err(EditError::NoSuchPage(PAGES)));
    }

    /// The check reads untrusted content: every cut of a layer, and bytes that are no content at
    /// all, come back as an answer, never a panic.
    #[test]
    fn any_bytes_are_answered() {
        for data in [LAYER, OLD_LAYER] {
            for end in 0..data.len() {
                let cut = data.as_bytes().get(..end).unwrap_or_default();
                assert!(layer_text(cut).is_none_or(|(w, g)| w <= 2 && g <= 10), "{cut:?}");
            }
        }
        for junk in [&b"\xff\xfe(((("[..], b"/OCR BMC Q Q Q EMC", b"/OCR BMC ET EMC", b"/OCR BMC BT BT ET ET EMC", b"/OCR /OCR BMC EMC"] {
            assert_eq!(layer_text(junk), None, "{junk:?}");
        }
    }
}
