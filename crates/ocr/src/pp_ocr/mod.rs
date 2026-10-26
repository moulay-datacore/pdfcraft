//! Accented-Latin text recognition with a PaddleOCR (PP-OCR) CTC recognition model.
//!
//! ocrs finds the words on the page and groups them into lines; this module reads them. Each
//! line is cut out of the page upright, scaled to 48 pixels high and read by the model, which
//! knows the accented letters ocrs's own model does not (é è ê à â ç ù û ô î ï ë œ « » …).
//!
//! The model contract (PaddleOCR's `CTCLabelDecode`, `use_space_char: true`):
//! - input `[N, 3, 48, W]` f32, the same grey value in all three channels, `v / 127.5 - 1`
//!   (white is +1); a batch is padded on the right with 0 (mid-grey), as PaddleOCR pads;
//! - output `[N, T, C]` softmax probabilities, batch first, `T = W / 8`;
//! - `C` is the dictionary's entries plus the blank and the space ([`dict`]).
//!
//! Every input is untrusted (the page pixels, and the model and dictionary files, which a user
//! can point `PDFCRAFT_MODELS` at). The model file must be the one PdfCraft ships, byte for byte
//! (its SHA-256), before rten parses it: rten's ONNX reader recurses without a depth limit and
//! sizes some tensors from the graph, so a crafted file could overflow the stack or exhaust
//! memory, which abort the process rather than panic. The dictionary, which PdfCraft's own code
//! parses, is size-capped, and the two are checked against each other when loaded; every shape
//! the model returns is checked, and lines are cut out and read a batch at a time, within a
//! memory budget, so a crowded page is read in full.

#![deny(clippy::indexing_slicing)]

mod decode;
pub(crate) mod dict;
pub(crate) mod frame;

use std::io::Read;
use std::path::Path;

use rten::Dimension;
use rten_imageproc::RotatedRect;
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use sha2::{Digest, Sha256};

use decode::{greedy, snap, spans, words_of};
use dict::Charset;
use frame::{Crop, Gray, LineFrame, Margins, REC_HEIGHT, crop, crop_width, direction, line_directions, oriented, split_at_gaps};

/// The narrowest batch PaddleOCR feeds (its `rec_image_shape` width); narrower crops are padded.
const MIN_WIDTH: usize = 320;
/// Lines per batch at most.
const MAX_BATCH: usize = 8;
/// Model output floats per batch (64 MB): with PP-OCRv6's 18 710 classes, about 7000 px of
/// line. A batch always holds at least one line, which the dictionary cap keeps below this.
/// Lines are cut out a batch at a time, so a page's memory doesn't grow with its lines.
const OUTPUT_BUDGET: usize = 16 << 20;
/// A model file larger than this is refused unread (PP-OCRv6 small is 21 MB).
const MAX_MODEL_BYTES: u64 = 256 << 20;
/// A gap between two words wider than this many word heights splits the line.
const GAP_FACTOR: f32 = 2.0;
/// Lines read with a lower mean probability are dropped (PaddleOCR's `drop_score`): logos,
/// signatures and dirt.
const MIN_CONFIDENCE: f32 = 0.5;
/// Two words read from different crops are the same ink read twice when this much of the
/// smaller one's box lies in the other's.
const SAME_INK: f32 = 0.5;

/// A quad in image pixels: top-left, top-right, bottom-right, bottom-left as read.
pub(crate) type Quad = [[f32; 2]; 4];

/// A recognised word.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RecWord {
    pub text: String,
    /// Its box, skewed like the line it is on.
    pub quad: Quad,
    /// The lowest probability among its characters.
    pub confidence: f32,
    /// Whether its box came from ocrs's detection boxes (`true`) or from its time steps alone.
    pub snapped: bool,
}

/// A recognised line (or the part of one between two wide gaps).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RecLine {
    pub words: Vec<RecWord>,
    /// The mean probability of its characters.
    pub confidence: f32,
    /// The frame of the crop it was read from.
    pub frame: LineFrame,
}

/// The output of a model run: `[N, T, C]` and its values, row-major.
pub(crate) struct Output {
    pub shape: [usize; 3],
    pub data: Vec<f32>,
}

/// What the recognizer needs from a model: tests use fakes.
pub(crate) trait LineModel: Send + Sync {
    /// Run a batch `[N, 3, 48, W]`.
    fn run(&self, input: NdTensor<f32, 4>) -> Result<Output, String>;
}

impl LineModel for rten::Model {
    fn run(&self, input: NdTensor<f32, 4>) -> Result<Output, String> {
        let value = self.run_one(input.into(), None).map_err(|e| e.to_string())?;
        let out: NdTensor<f32, 3> = value.try_into().map_err(|e| format!("its output is not a 3-D float tensor ({e:?})"))?;
        Ok(Output { shape: out.shape(), data: out.into_data() })
    }
}

/// A loaded recognition model and its classes.
pub(crate) struct Recognizer {
    model: Box<dyn LineModel>,
    charset: Charset,
}

/// `path`'s bytes, refusing a file larger than `cap` before reading it all.
fn read_capped(path: &Path, cap: u64) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.take(cap.saturating_add(1)).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if u64::try_from(bytes.len()).map_or(true, |n| n > cap) {
        return Err(format!("the file is larger than {} MB", cap >> 20));
    }
    Ok(bytes)
}

/// The SHA-256 of `bytes`, in lower-case hex.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Check what `model` declares about its input and output against the contract: one input
/// `[N, 3, 48, W]` and one output `[N, T, classes]`. Dimensions the model leaves open pass;
/// [`probe`] then checks what it returns.
fn check_signature(model: &rten::Model, classes: usize) -> Result<(), String> {
    let (inputs, outputs) = (model.input_ids().len(), model.output_ids().len());
    if inputs != 1 || outputs != 1 {
        return Err(format!("the model has {inputs} inputs and {outputs} outputs, not one of each"));
    }
    let fixed = |d: Option<&Dimension>| match d {
        Some(Dimension::Fixed(n)) => Some(*n),
        _ => None,
    };
    if let Some(shape) = model.input_shape(0) {
        if shape.len() != 4 {
            return Err(format!("the model's input has {} dimensions, not 4 ([lines, 3, 48, width])", shape.len()));
        }
        if let Some(c) = fixed(shape.get(1)).filter(|c| *c != 3) {
            return Err(format!("the model reads {c} colour channels, not 3"));
        }
        if let Some(h) = fixed(shape.get(2)).filter(|h| *h != REC_HEIGHT) {
            return Err(format!("the model reads lines {h} pixels high, not {REC_HEIGHT}"));
        }
    }
    let output = model.output_ids().first().and_then(|id| model.node_info(*id)).and_then(|n| n.shape());
    if let Some(shape) = output {
        if shape.len() != 3 {
            return Err(format!("the model's output has {} dimensions, not 3 ([lines, steps, classes])", shape.len()));
        }
        if let Some(c) = fixed(shape.get(2)).filter(|c| *c != classes) {
            return Err(class_mismatch(c, classes));
        }
    }
    Ok(())
}

fn class_mismatch(got: usize, classes: usize) -> String {
    format!(
        "the model has {got} classes but its dictionary makes {classes} ({} entries, the blank and the space): they don't belong together",
        classes.saturating_sub(2)
    )
}

/// Run `model` once on a blank line and check what it returns: `[1, T, classes]` with
/// `1 ≤ T ≤ 320`.
fn probe(model: &dyn LineModel, classes: usize) -> Result<(), String> {
    let input = NdTensor::try_from_data([1, 3, REC_HEIGHT, MIN_WIDTH], vec![0.0; 3 * REC_HEIGHT * MIN_WIDTH]).map_err(|e| format!("{e:?}"))?;
    let out = model.run(input).map_err(|e| format!("the model doesn't run: {e}"))?;
    let [n, t, c] = out.shape;
    if c != classes {
        return Err(class_mismatch(c, classes));
    }
    if n != 1 || t == 0 || t > MIN_WIDTH {
        return Err(format!("the model returned {:?} for one line {MIN_WIDTH} pixels wide", out.shape));
    }
    Ok(())
}

/// The classes of the dictionary at `dict`, restricted to what the text layer can hold.
pub(crate) fn decoding_charset(dict: &Path) -> Result<Charset, String> {
    let mut charset = Charset::read(dict)?;
    charset.restrict(|c| pdfcraft_fonts::win_ansi_byte(c).is_some());
    Ok(charset)
}

impl Recognizer {
    /// Load the model at `model` (ONNX or .rten), which must have the SHA-256 `sha256` (lower-case
    /// hex), and its dictionary at `dict`, and check that they work together. A file with another
    /// hash is refused before it is parsed (see the module documentation). Decoding is restricted
    /// to the characters the OCR text layer can hold (Helvetica in WinAnsiEncoding), so recognised
    /// text is never written as `?`.
    pub(crate) fn load(model: &Path, dict: &Path, sha256: &str) -> Result<Recognizer, String> {
        let named = |p: &Path, e: String| format!("{}: {e}", p.display());
        let charset = decoding_charset(dict).map_err(|e| named(dict, e))?;
        let bytes = read_capped(model, MAX_MODEL_BYTES).map_err(|e| named(model, e))?;
        let got = sha256_hex(&bytes);
        if !got.eq_ignore_ascii_case(sha256) {
            return Err(named(model, format!("not the model PdfCraft ships (its SHA-256 is {got}, not {sha256}), so it isn't read")));
        }
        // From memory, so the model can't make the loader read other files (ONNX external data).
        let loaded = rten::Model::load(bytes).map_err(|e| named(model, e.to_string()))?;
        check_signature(&loaded, charset.classes()).map_err(|e| named(model, e))?;
        Self::with_model(Box::new(loaded), charset).map_err(|e| named(model, e))
    }

    /// A recognizer over an already loaded model, which is probed first.
    fn with_model(model: Box<dyn LineModel>, charset: Charset) -> Result<Recognizer, String> {
        probe(model.as_ref(), charset.classes())?;
        Ok(Recognizer { model, charset })
    }

    /// The characters it may emit.
    #[cfg(test)]
    pub(crate) fn allowed_chars(&self) -> impl Iterator<Item = char> + '_ {
        self.charset.allowed_chars()
    }

    /// Read the lines ocrs found in `image` (`OcrEngine::find_text_lines`: words left to right,
    /// lines in reading order). A line split at a wide gap gives several lines.
    pub(crate) fn recognize(&self, image: &Gray, lines: &[Vec<RotatedRect>]) -> Result<Vec<RecLine>, String> {
        let frames = self.frames(lines);
        let mut read: Vec<RecLine> = self.read(image, frames)?.into_iter().flatten().filter(|l| l.confidence >= MIN_CONFIDENCE).collect();
        dedupe(&mut read);
        Ok(merge_fragments(read))
    }

    /// The frames of the lines, split at wide gaps, each along its direction. Lines that can't
    /// be framed, or cut out, are left out.
    fn frames(&self, lines: &[Vec<RotatedRect>]) -> Vec<LineFrame> {
        let parts: Vec<Vec<RotatedRect>> = lines.iter().flat_map(|l| split_at_gaps(l, GAP_FACTOR)).map(|p| oriented(&p)).collect();
        let dirs: Vec<_> = parts.iter().map(|p| direction(p)).collect();
        parts
            .iter()
            .zip(line_directions(&dirs))
            .filter_map(|(part, d)| LineFrame::along(part, d?, Margins::default()))
            .filter(|f| crop_width(f).is_some())
            .collect()
    }

    /// Cut `frames` out of `image` and read them, a batch at a time, one entry per frame in
    /// order: `None` when nothing was read in it.
    fn read(&self, image: &Gray, frames: Vec<LineFrame>) -> Result<Vec<Option<RecLine>>, String> {
        let widths: Vec<usize> = frames.iter().map(|f| crop_width(f).unwrap_or(0)).collect();
        let mut frames: Vec<Option<LineFrame>> = frames.into_iter().map(Some).collect();
        let mut order: Vec<usize> = (0..widths.len()).collect();
        order.sort_by_key(|&i| widths.get(i).copied().unwrap_or(0));
        let mut out: Vec<Option<RecLine>> = vec![None; widths.len()];
        let classes = self.charset.classes();
        let mut start = 0;
        while start < order.len() {
            // Lines of similar width together, as many as fit the output budget.
            let mut end = start;
            let mut batch_w = MIN_WIDTH;
            while let Some(&i) = order.get(end) {
                let w = widths.get(i).copied().unwrap_or(0).max(batch_w).next_multiple_of(8);
                let n = end.saturating_sub(start).saturating_add(1);
                let cost = n.saturating_mul(w / 8).saturating_mul(classes);
                if end > start && (n > MAX_BATCH || cost > OUTPUT_BUDGET) {
                    break;
                }
                batch_w = w;
                end = end.saturating_add(1);
            }
            // Cut out now, so only one batch's crops are in memory.
            let ids = order.get(start..end).unwrap_or(&[]);
            let (ids, batch): (Vec<usize>, Vec<Crop>) = ids.iter().filter_map(|&i| Some((i, crop(image, frames.get_mut(i)?.take()?)?))).unzip();
            let batch: Vec<&Crop> = batch.iter().collect();
            for (i, line) in ids.into_iter().zip(self.read_batch(&batch, batch_w)?) {
                if let Some(slot) = out.get_mut(i) {
                    *slot = line;
                }
            }
            start = end;
        }
        Ok(out)
    }

    fn read_batch(&self, batch: &[&Crop], width: usize) -> Result<Vec<Option<RecLine>>, String> {
        let n = batch.len();
        let plane = REC_HEIGHT.saturating_mul(width);
        let mut input = vec![0.0_f32; n.saturating_mul(3).saturating_mul(plane)];
        for (b, c) in batch.iter().enumerate() {
            for ch in 0..3 {
                let base = b.saturating_mul(3).saturating_add(ch).saturating_mul(plane);
                for (row, src) in c.pixels.chunks_exact(c.width.max(1)).enumerate() {
                    let at = base.saturating_add(row.saturating_mul(width));
                    if let Some(dst) = input.get_mut(at..at.saturating_add(src.len())) {
                        dst.copy_from_slice(src);
                    }
                }
            }
        }
        let input = NdTensor::try_from_data([n, 3, REC_HEIGHT, width], input).map_err(|e| format!("{e:?}"))?;
        let out = self.model.run(input).map_err(|e| format!("the recognition model failed: {e}"))?;
        let [bn, steps, classes] = out.shape;
        let per_line = steps.checked_mul(classes);
        if bn != n || classes != self.charset.classes() || steps == 0 || per_line.and_then(|p| p.checked_mul(n)) != Some(out.data.len()) {
            return Err(format!("the recognition model returned {:?} for {n} lines {width} pixels wide", out.shape));
        }
        let per_line = per_line.unwrap_or(0);
        let stride = width as f32 / steps as f32;
        Ok(batch
            .iter()
            .enumerate()
            .map(|(b, c)| {
                let rows = b.checked_mul(per_line).and_then(|s| out.data.get(s..s.saturating_add(per_line))).unwrap_or(&[]);
                // Steps over the padding read nothing.
                let valid = ((c.width as f32 / stride).ceil() as usize).min(steps);
                let read = greedy(rows, classes, valid, &self.charset);
                self.line(c, &read, stride)
            })
            .collect())
    }

    /// The words of one crop from its decoded steps.
    fn line(&self, c: &Crop, steps: &[decode::Step], stride: f32) -> Option<RecLine> {
        let words = words_of(steps, &self.charset);
        if words.is_empty() {
            return None;
        }
        let f = &c.frame;
        let spans = spans(&words, stride, c.sx, f.length);
        let snapped = snap(&spans, &f.words, stride / c.sx);
        let quad = |[u0, u1, v0, v1]: [f32; 4]| [f.point(u0, v0), f.point(u1, v0), f.point(u1, v1), f.point(u0, v1)];
        let (mut p_sum, mut p_n) = (0.0_f32, 0_usize);
        let mut out = Vec::with_capacity(words.len());
        for (w, (span, snap)) in words.iter().zip(spans.iter().zip(snapped)) {
            let text: String = w.iter().filter_map(|s| self.charset.char_of(s.class)).collect();
            p_sum += w.iter().map(|s| s.p).sum::<f32>();
            p_n = p_n.saturating_add(w.len());
            let ctc = [span[0], span[1], f.ink[0], f.ink[1]];
            out.push(RecWord {
                text,
                quad: quad(snap.unwrap_or(ctc)),
                confidence: w.iter().map(|s| s.p).fold(1.0, f32::min),
                snapped: snap.is_some(),
            });
        }
        Some(RecLine { words: out, confidence: if p_n > 0 { p_sum / p_n as f32 } else { 0.0 }, frame: f.clone() })
    }
}

/// The axis-aligned box `[left, top, right, bottom]` around `quads`.
pub(crate) fn bounds<'a>(quads: impl IntoIterator<Item = &'a Quad>) -> [f32; 4] {
    let mut b = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    for p in quads.into_iter().flatten() {
        b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
    }
    b
}

/// The centre of `q`.
pub(crate) fn centre(q: &Quad) -> [f32; 2] {
    let [x, y] = q.iter().fold([0.0, 0.0], |s, p| [s[0] + p[0], s[1] + p[1]]);
    [x / 4.0, y / 4.0]
}

/// How much of the smaller of `a` and `b` lies in the other, from 0 to 1, measured along
/// `a`'s own axes (words on one line share its skew).
#[cfg(test)]
fn overlap(a: &Quad, b: &Quad) -> f32 {
    overlap_and_areas(a, b).0
}

/// [`overlap`], and the areas of `a` and `b` measured the same way (0 for a degenerate quad).
fn overlap_and_areas(a: &Quad, b: &Quad) -> (f32, f32, f32) {
    let [tl, tr, _, bl] = *a;
    let (d, n) = ([tr[0] - tl[0], tr[1] - tl[1]], [bl[0] - tl[0], bl[1] - tl[1]]);
    let (ld, ln) = (d[0].hypot(d[1]), n[0].hypot(n[1]));
    if !(ld > 1e-3 && ln > 1e-3 && ld.is_finite() && ln.is_finite()) {
        return (0.0, 0.0, 0.0);
    }
    let (d, n) = ([d[0] / ld, d[1] / ld], [n[0] / ln, n[1] / ln]);
    let extent = |q: &Quad| {
        let mut e = [f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY];
        for p in q {
            let r = [p[0] - tl[0], p[1] - tl[1]];
            let (u, v) = (r[0] * d[0] + r[1] * d[1], r[0] * n[0] + r[1] * n[1]);
            e = [e[0].min(u), e[1].max(u), e[2].min(v), e[3].max(v)];
        }
        e
    };
    let (ea, eb) = (extent(a), extent(b));
    let area = |e: [f32; 4]| (e[1] - e[0]).max(0.0) * (e[3] - e[2]).max(0.0);
    let inter = area([ea[0].max(eb[0]), ea[1].min(eb[1]), ea[2].max(eb[2]), ea[3].min(eb[3])]);
    let (area_a, area_b) = (area(ea), area(eb));
    let smaller = area_a.min(area_b);
    let f = inter / smaller;
    (if smaller > 0.0 && f.is_finite() { f.min(1.0) } else { 0.0 }, area_a, area_b)
}

/// Of two words read from the same ink, whether `a` is the one to keep: the one snapped to a
/// detection box (the box belongs to its crop's line); else the larger one when it holds more
/// text (the other is part of it, such as a mark it was read glued to: "Bonjourà" and "à"),
/// whatever their confidences; else the more confident one, else `a`, the first.
fn keep_first(a: &RecWord, b: &RecWord, area_a: f32, area_b: f32) -> bool {
    if a.snapped != b.snapped {
        return a.snapped;
    }
    let (ca, cb) = (a.text.chars().count(), b.text.chars().count());
    if area_a > area_b && ca > cb {
        return true;
    }
    if area_b > area_a && cb > ca {
        return false;
    }
    a.confidence >= b.confidence
}

/// Drop words read twice. Where a text line comes in several crops (ocrs split it, or it was
/// split at a wide gap), the crops' margins overlap, and a word in the overlap, such as a small
/// "à" with no detection box of its own, is read in both ("à à Châlons"). Of two words from
/// different crops on the same ink, one stays ([`keep_first`]). A line left without words goes.
pub(crate) fn dedupe(lines: &mut Vec<RecLine>) {
    let boxes: Vec<[f32; 4]> = lines.iter().map(|l| bounds(l.words.iter().map(|w| &w.quad))).collect();
    let mut gone: Vec<Vec<bool>> = lines.iter().map(|l| vec![false; l.words.len()]).collect();
    let meet = |a: [f32; 4], b: [f32; 4]| a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3];
    for (i, (li, bi)) in lines.iter().zip(&boxes).enumerate() {
        for (j, (lj, bj)) in lines.iter().zip(&boxes).enumerate().skip(i.saturating_add(1)) {
            if !meet(*bi, *bj) {
                continue;
            }
            for (a, wa) in li.words.iter().enumerate() {
                for (b, wb) in lj.words.iter().enumerate() {
                    let gone_a = gone.get(i).and_then(|g| g.get(a)).copied().unwrap_or(true);
                    let gone_b = gone.get(j).and_then(|g| g.get(b)).copied().unwrap_or(true);
                    if gone_a || gone_b {
                        continue;
                    }
                    let (shared, area_a, area_b) = overlap_and_areas(&wa.quad, &wb.quad);
                    if shared < SAME_INK {
                        continue;
                    }
                    let (line, word) = if keep_first(wa, wb, area_a, area_b) { (j, b) } else { (i, a) };
                    if let Some(g) = gone.get_mut(line).and_then(|g| g.get_mut(word)) {
                        *g = true;
                    }
                }
            }
        }
    }
    for (line, g) in lines.iter_mut().zip(&gone) {
        let mut flags = g.iter();
        line.words.retain(|_| !flags.next().copied().unwrap_or(false));
    }
    lines.retain(|l| !l.words.is_empty());
}

/// Whether frame position (`u`, `v`) is on `frame`'s own baseline: across the line within one
/// of the detection boxes beside it (with a tenth of its height to spare), those within two
/// typical box heights of `u` along the line. A box more than twice as tall as is typical (a
/// logo, a large heading that ocrs chained to the line) doesn't count. A line's ink band is not
/// enough: such a box makes the band cover the lines beside it.
fn on_baseline(frame: &LineFrame, u: f32, v: f32) -> bool {
    let height = |b: &[f32; 4]| (b[3] - b[2]).max(0.0);
    let mut heights: Vec<f32> = frame.words.iter().map(height).filter(|h| h.is_finite()).collect();
    heights.sort_by(f32::total_cmp);
    // The lower median: of a logo and one word, the word's.
    let Some(typical) = heights.get(heights.len().saturating_sub(1) / 2).copied() else { return false };
    let reach = 2.0 * typical;
    frame.words.iter().any(|b| {
        let spare = 0.1 * height(b);
        height(b) <= 2.0 * typical && b[0] - reach <= u && u <= b[1] + reach && b[2] - spare <= v && v <= b[3] + spare
    })
}

/// Put fragments back into their lines. ocrs sometimes makes a few words of a text line a line
/// of their own (words a little off the baseline, such as "; año,"), which then comes after the
/// line it is part of and breaks the reading order. A line whose every word is centred inside a
/// longer line's frame, along it and across it within its ink and [`on_baseline`], joins the
/// longest such line, and that line's words are put in order along it. Lines one above the
/// other, beside a tall box chained into a line, or side by side past the line's margin, stay
/// apart.
pub(crate) fn merge_fragments(lines: Vec<RecLine>) -> Vec<RecLine> {
    // A strict order, longer and then earlier, so following hosts always ends at a line that
    // has none.
    let longer =
        |(i, a): (usize, &RecLine), (j, b): (usize, &RecLine)| a.frame.length > b.frame.length || (a.frame.length == b.frame.length && i < j);
    let holds = |host: &RecLine, frag: &RecLine| {
        frag.words.iter().all(|w| {
            let [u, v] = host.frame.locate(centre(&w.quad));
            (0.0..=host.frame.length).contains(&u) && (host.frame.ink[0]..=host.frame.ink[1]).contains(&v) && on_baseline(&host.frame, u, v)
        })
    };
    let host: Vec<Option<usize>> = lines
        .iter()
        .enumerate()
        .map(|frag| {
            let hosts = lines.iter().enumerate().filter(|&host| longer(host, frag) && holds(host.1, frag.1));
            hosts.reduce(|best, h| if longer(h, best) { h } else { best }).map(|(i, _)| i)
        })
        .collect();
    let n = lines.len();
    let root = |mut j: usize| {
        for _ in 0..n {
            match host.get(j).copied().flatten() {
                Some(i) => j = i,
                None => break,
            }
        }
        j
    };
    let roots: Vec<usize> = (0..n).map(root).collect();
    let mut slots: Vec<Option<RecLine>> = lines.into_iter().map(Some).collect();
    let mut joined = vec![false; n];
    for (j, &r) in roots.iter().enumerate() {
        if r == j {
            continue;
        }
        let Some(frag) = slots.get_mut(j).and_then(Option::take) else { continue };
        if let Some(Some(line)) = slots.get_mut(r) {
            let chars = |l: &RecLine| l.words.iter().map(|w| w.text.chars().count()).sum::<usize>() as f32;
            let (a, b) = (chars(line), chars(&frag));
            if a + b > 0.0 {
                line.confidence = (line.confidence * a + frag.confidence * b) / (a + b);
            }
            line.words.extend(frag.words);
            if let Some(m) = joined.get_mut(r) {
                *m = true;
            }
        } else if let Some(slot) = slots.get_mut(j) {
            // A root is never a fragment, so its slot is never empty; were it, keep the line.
            *slot = Some(frag);
        }
    }
    for (slot, joined) in slots.iter_mut().zip(joined) {
        if let (Some(RecLine { words, frame, .. }), true) = (slot, joined) {
            let along = |w: &RecWord| frame.locate(centre(&w.quad))[0];
            words.sort_by(|a, b| along(a).total_cmp(&along(b)));
        }
    }
    slots.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests;
