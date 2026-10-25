//! Local, bounded image matching for an invisible OCR row. Geometry comes from the OCR word
//! boxes; the printed face comes from pixels. Width never contributes to the ranking.
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use pdfcraft_cos::{Dict, Document as CosDocument, Object, SaveOptions, Stream};
use pdfcraft_fonts::embedding::{self, FontGrant};
use pdfcraft_render::{PageInfo, PageRenderer, RenderConfig, RenderRequest, Tile};

use crate::{
    Document, guard,
    text_fonts::{Catalogue, FontFile},
};

const MAX_CANDIDATES: usize = 40;
const MAX_CROP_WIDTH: u32 = 2048;
const MAX_CROP_HEIGHT: u32 = 256;
const MAX_CROP_PIXELS: usize = 512 * 1024;
const TIME_BUDGET: Duration = Duration::from_secs(3);
const MIN_SCORE: f64 = 0.80;
const MIN_GAP: f64 = 0.015;

/// The same approved, static faces offered by the desktop document font picker.
pub fn bundled_text_fonts() -> &'static [(String, &'static [u8])] {
    static FONTS: OnceLock<Vec<(String, &'static [u8])>> = OnceLock::new();
    FONTS.get_or_init(|| {
        let mut faces = vec![
            ("Inter Regular".into(), pdfcraft_fonts::INTER_REGULAR),
            ("Inter Medium".into(), include_bytes!("../../../assets/fonts/Inter-Medium.ttf").as_slice()),
            ("Inter SemiBold".into(), include_bytes!("../../../assets/fonts/Inter-SemiBold.ttf").as_slice()),
            ("JetBrains Mono Regular".into(), include_bytes!("../../../assets/fonts/JetBrainsMono-Regular.ttf").as_slice()),
            ("Dancing Script".into(), include_bytes!("../../../assets/fonts/DancingScript.ttf").as_slice()),
        ];
        faces.extend(pdfcraft_fonts::CRAFT_FONTS.iter().map(|face| (face.name(), face.bytes)));
        faces.retain(|(_, bytes)| embedding::validate_document_input(bytes, FontGrant::OpenLicence).is_ok());
        faces
    })
}

#[derive(Clone)]
pub struct OcrFontCandidate {
    /// PostScript identity, independent of localized display names.
    pub face: String,
    pub full_name: String,
    pub weight: u16,
    pub italic: bool,
    /// Symmetric ink F-score with one scan-pixel tolerance, in [0, 1].
    pub score: f64,
    program: Arc<Vec<u8>>,
    grant: FontGrant,
}
impl std::fmt::Debug for OcrFontCandidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OcrFontCandidate")
            .field("face", &self.face)
            .field("full_name", &self.full_name)
            .field("weight", &self.weight)
            .field("italic", &self.italic)
            .field("score", &self.score)
            .finish_non_exhaustive()
    }
}
impl OcrFontCandidate {
    /// The validated snapshot used for matching. Registration still goes through the normal
    /// document-font validator and its original grant.
    pub fn program(&self) -> Arc<Vec<u8>> {
        self.program.clone()
    }
    pub fn grant(&self) -> FontGrant {
        self.grant
    }
}

#[derive(Clone, Debug, Default)]
pub struct OcrFontSuggestion {
    pub candidates: Vec<OcrFontCandidate>,
    /// False means “no clear match”, including a blank scan, weak agreement, ambiguity or a
    /// request that ran out of time before every selected candidate was compared.
    pub confident: bool,
}

/// An immutable snapshot safe to move to a worker. Constructing it does not discover fonts,
/// read font programs, decode an image or rasterize a page.
pub struct OcrFontSuggestRequest {
    bytes: Arc<Vec<u8>>,
    cos: CosDocument,
    config: RenderConfig,
    info: PageInfo,
    page: usize,
    block: pdfcraft_edit::TextBlock,
    catalogue: Option<Arc<Catalogue>>,
}
impl Document {
    pub fn ocr_font_suggest_request(&self, page: usize, block: usize) -> Result<OcrFontSuggestRequest, String> {
        guard(|| {
            let editor = self.editor.as_ref().ok_or("the document can't be read")?;
            let block = editor.text_blocks(page).map_err(|e| e.to_string())?.get(block).cloned().ok_or("the OCR block does not exist")?;
            if !block.invisible {
                return Err("font suggestions require an invisible OCR block".into());
            }
            Ok(OcrFontSuggestRequest {
                bytes: self.bytes.clone(),
                cos: editor.cos.clone(),
                config: self.config.clone(),
                info: self.info.pages.get(page).cloned().ok_or("the OCR page does not exist")?,
                page,
                block,
                catalogue: self.text_fonts.clone(),
            })
        })?
    }

    /// Read-only convenience entry point. Desktop callers move a request to a worker and
    /// call `run` with a cancellation flag instead.
    pub fn ocr_font_suggest(&self, page: usize, block: usize) -> Result<OcrFontSuggestion, String> {
        self.ocr_font_suggest_request(page, block)?.run(&AtomicBool::new(false))
    }
}

fn stopped(cancel: &AtomicBool, start: Instant) -> Result<bool, String> {
    if cancel.load(Ordering::Relaxed) {
        return Err("OCR font suggestion cancelled".into());
    }
    Ok(start.elapsed() >= TIME_BUDGET)
}

enum Source {
    Bundled(&'static [u8]),
    Installed(FontFile),
}
struct CandidateSource {
    source: Source,
    face: String,
    full_name: String,
    weight: u16,
    italic: bool,
}

// Ordinary document families first, each with regular/bold/italic faces. This is a metadata
// priority, never a width score; in particular Times New Roman survives the 40-face cap.
fn family_priority(family: &str) -> usize {
    let key = family.to_ascii_lowercase();
    [
        "times new roman",
        "arial",
        "calibri",
        "cambria",
        "georgia",
        "courier new",
        "helvetica",
        "times",
        "liberation serif",
        "liberation sans",
        "dejavu serif",
        "dejavu sans",
        "verdana",
    ]
    .iter()
    .position(|name| key == *name)
    .unwrap_or(100)
}
impl OcrFontSuggestRequest {
    pub fn run(self, cancel: &AtomicBool) -> Result<OcrFontSuggestion, String> {
        guard(|| self.run_inner(cancel))?
    }

    fn run_inner(&self, cancel: &AtomicBool) -> Result<OcrFontSuggestion, String> {
        let start = Instant::now();
        if stopped(cancel, start)? {
            return Ok(OcrFontSuggestion::default());
        }
        let Some((text, scan)) = self.scan_line(cancel, start)? else {
            return Ok(OcrFontSuggestion::default());
        };
        let mut sources = Vec::new();
        for (_, bytes) in bundled_text_fonts() {
            if let Ok(meta) = embedding::document_metadata(bytes, Some(FontGrant::OpenLicence)) {
                sources.push(CandidateSource {
                    source: Source::Bundled(bytes),
                    face: meta.postscript,
                    full_name: meta.full_name,
                    weight: meta.weight,
                    italic: meta.italic,
                });
            }
        }
        // Keep the five Latin bundled choices in every comparison; installed common
        // families fill the remaining slots before optional craft-fonts.
        let extra_bundled = if sources.len() > 5 { sources.split_off(5) } else { Vec::new() };
        if let Some(catalogue) = &self.catalogue {
            let mut installed: Vec<_> = catalogue.files().filter(|face| (350..=750).contains(&face.weight)).cloned().collect();
            installed.sort_by(|a, b| {
                family_priority(&a.family)
                    .cmp(&family_priority(&b.family))
                    .then(a.family.cmp(&b.family))
                    .then(a.italic.cmp(&b.italic))
                    .then(a.weight.abs_diff(400).cmp(&b.weight.abs_diff(400)))
                    .then(a.face.cmp(&b.face))
            });
            sources.extend(installed.into_iter().map(|face| CandidateSource {
                face: face.face.clone(),
                full_name: face.full_name.clone(),
                weight: face.weight,
                italic: face.italic,
                source: Source::Installed(face),
            }));
        }
        sources.extend(extra_bundled);
        let mut seen = std::collections::BTreeSet::new();
        sources.retain(|s| seen.insert(s.face.clone()));
        sources.truncate(MAX_CANDIDATES);
        let mut ranked = Vec::new();
        let mut complete = true;
        for source in sources {
            if stopped(cancel, start)? {
                complete = false;
                break;
            }
            let (bytes, grant) = match source.source {
                Source::Bundled(bytes) => (bytes.to_vec(), FontGrant::OpenLicence),
                Source::Installed(file) => {
                    let Some(catalogue) = &self.catalogue else { continue };
                    let Some(bytes) = catalogue.suggestion_program(&file) else { continue };
                    (bytes, catalogue.grant())
                }
            };
            if !embedding::covers(&bytes, &text) {
                continue;
            }
            let Ok(mask) = render_face(&bytes, grant, &text, scan.height) else { continue };
            if stopped(cancel, start)? {
                complete = false;
                break;
            }
            let score = ink_score(&scan, &mask);
            ranked.push(OcrFontCandidate {
                face: source.face,
                full_name: source.full_name,
                weight: source.weight,
                italic: source.italic,
                score,
                program: Arc::new(bytes),
                grant,
            });
            ranked.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.face.cmp(&b.face)));
            ranked.truncate(3);
        }
        let confident = complete
            && ranked.first().is_some_and(|best| best.score >= MIN_SCORE && ranked.get(1).is_some_and(|next| best.score - next.score >= MIN_GAP));
        Ok(OcrFontSuggestion { candidates: ranked, confident })
    }

    fn scan_line(&self, cancel: &AtomicBool, start: Instant) -> Result<Option<(String, Mask)>, String> {
        let lines = pdfcraft_edit::text_lines(&self.cos, self.page).map_err(|e| e.to_string())?;
        let row = self
            .block
            .rows
            .iter()
            .filter_map(|row| {
                let members: Vec<_> = row.iter().filter_map(|i| lines.get(*i)).collect();
                let text = members.iter().map(|line| line.text.trim()).collect::<Vec<_>>().join(" ");
                let bounds = members.iter().flat_map(|line| line.ink.iter()).copied().reduce(union)?;
                Some((text, bounds))
            })
            .filter(|(text, rect)| (4..=256).contains(&text.chars().count()) && valid_rect(*rect))
            .max_by_key(|(text, _)| text.len());
        let Some((text, rect)) = row else { return Ok(None) };
        if self.block.rotation.abs() > 0.01 {
            return Ok(None);
        }
        // Use the containing scan's pixel density. The renderer applies the image CTM and
        // page /Rotate, including scaled and translated XObjects and safe image decoding.
        let mut scale = 3.0f64;
        if let Ok(images) = pdfcraft_edit::images::reading_images(&self.cos, self.page) {
            let centre = [(rect[0] + rect[2]) * 0.5, (rect[1] + rect[3]) * 0.5];
            if let Some(image) = images.iter().find(|image| !image.is_form && image.contains(centre)) {
                let [a, b, c, d, _, _] = image.matrix;
                let density = (f64::from(image.width) / a.hypot(b).max(1e-6)).max(f64::from(image.height) / c.hypot(d).max(1e-6));
                let view_unit = if self.info.rotation.is_multiple_of(180) {
                    f64::from(self.info.width) / f64::from(self.info.crop[2] - self.info.crop[0]).max(1e-6)
                } else {
                    f64::from(self.info.height) / f64::from(self.info.crop[2] - self.info.crop[0]).max(1e-6)
                };
                if density.is_finite() && view_unit.is_finite() && view_unit > 0.0 {
                    scale = (density / view_unit).clamp(1.0, 8.0);
                }
            }
        }
        let [x0, y0, x1, y1] = rect;
        let h = y1 - y0;
        let margin = h * 0.65;
        let crop = [x0 - h * 0.15, y0 - margin, x1 + h * 0.15, y1 + margin];
        let shown = view_rect(&self.info, crop);
        let core = view_rect(&self.info, rect);
        if !valid_rect(shown) {
            return Ok(None);
        }
        let (w, h) = (shown[2] - shown[0], shown[3] - shown[1]);
        let (max_w, max_h) =
            if self.info.rotation.is_multiple_of(180) { (MAX_CROP_WIDTH, MAX_CROP_HEIGHT) } else { (MAX_CROP_HEIGHT, MAX_CROP_WIDTH) };
        scale = scale.min(f64::from(max_w) / w).min(f64::from(max_h) / h);
        if !scale.is_finite() || scale <= 0.0 {
            return Ok(None);
        }
        let left = (shown[0] * scale).floor().max(0.0);
        let top = (shown[1] * scale).floor().max(0.0);
        let right = (shown[2] * scale).ceil().min(f64::from(self.info.width) * scale);
        let bottom = (shown[3] * scale).ceil().min(f64::from(self.info.height) * scale);
        if right <= left || bottom <= top {
            return Ok(None);
        }
        let tile = Tile { x: left as u32, y: top as u32, w: (right - left) as u32, h: (bottom - top) as u32 };
        if tile.w == 0 || tile.h == 0 || (tile.w as usize).checked_mul(tile.h as usize).is_none_or(|n| n > MAX_CROP_PIXELS) {
            return Ok(None);
        }
        if stopped(cancel, start)? {
            return Ok(None);
        }
        let mut config = self.config.clone();
        config.hide_comments = true;
        config.reading_colors = Default::default();
        let mut renderer = PageRenderer::new(self.bytes.clone(), config);
        let rendered = renderer.render(RenderRequest { page: self.page, scale: scale as f32, tile: Some(tile), ..Default::default() });
        if let Some(error) = rendered.error {
            return Err(error);
        }
        if stopped(cancel, start)? {
            return Ok(None);
        }
        let raw = Mask::from_rgba(rendered.width as usize, rendered.height as usize, &rendered.rgba);
        let Some(raw) = raw else { return Ok(None) };
        let (mask, core_y) = raw.upright(
            self.info.rotation,
            [
                (core[0] * scale - left).max(0.0),
                (core[1] * scale - top).max(0.0),
                (core[2] * scale - left).max(0.0),
                (core[3] * scale - top).max(0.0),
            ],
        );
        Ok(mask.line_band(core_y).map(|mask| (text, mask)))
    }
}

fn union(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]
}
fn valid_rect(r: [f64; 4]) -> bool {
    r.iter().all(|v| v.is_finite()) && r[2] > r[0] && r[3] > r[1]
}
fn view_rect(info: &PageInfo, rect: [f64; 4]) -> [f64; 4] {
    let a = info.user_to_view(rect[0] as f32, rect[1] as f32);
    let b = info.user_to_view(rect[2] as f32, rect[3] as f32);
    [f64::from(a[0].min(b[0])), f64::from(a[1].min(b[1])), f64::from(a[0].max(b[0])), f64::from(a[1].max(b[1]))]
}

#[derive(Clone, Debug)]
struct Mask {
    width: usize,
    height: usize,
    ink: Vec<bool>,
}
impl Mask {
    fn at(&self, x: usize, y: usize) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        y.checked_mul(self.width).and_then(|n| n.checked_add(x)).and_then(|i| self.ink.get(i)).copied().unwrap_or(false)
    }
    fn from_rgba(width: usize, height: usize, rgba: &[u8]) -> Option<Self> {
        let count = width.checked_mul(height)?;
        if count == 0 || count > MAX_CROP_PIXELS || count.checked_mul(4)? > rgba.len() {
            return None;
        }
        let grey: Vec<u8> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .take(count)
            .map(|p| {
                let r = u32::from(*p.first().unwrap_or(&255));
                let g = u32::from(*p.get(1).unwrap_or(&255));
                let b = u32::from(*p.get(2).unwrap_or(&255));
                ((r * 77 + g * 150 + b * 29) / 256) as u8
            })
            .collect();
        let threshold = otsu(&grey)?;
        Some(Self { width, height, ink: grey.into_iter().map(|v| v <= threshold).collect() })
    }
    fn upright(self, rotation: u16, core: [f64; 4]) -> (Self, [f64; 2]) {
        let (w, h) = (self.width, self.height);
        if rotation == 0 {
            return (self, [core[1], core[3]]);
        }
        let (width, height) = if rotation.is_multiple_of(180) { (w, h) } else { (h, w) };
        let mut ink = Vec::with_capacity(self.ink.len());
        for y in 0..height {
            for x in 0..width {
                let (sx, sy) = match rotation {
                    90 => (w.saturating_sub(1).saturating_sub(y), x),
                    180 => (w.saturating_sub(1).saturating_sub(x), h.saturating_sub(1).saturating_sub(y)),
                    270 => (y, h.saturating_sub(1).saturating_sub(x)),
                    _ => (x, y),
                };
                ink.push(self.at(sx, sy));
            }
        }
        let core_y = match rotation {
            90 => [w as f64 - core[2], w as f64 - core[0]],
            180 => [h as f64 - core[3], h as f64 - core[1]],
            270 => [core[0], core[2]],
            _ => [core[1], core[3]],
        };
        (Self { width, height, ink }, core_y)
    }
    fn line_band(&self, core: [f64; 2]) -> Option<Self> {
        let mut bands: Vec<(usize, usize)> = Vec::new();
        let mut begin = None;
        for y in 0..=self.height {
            let ink = y < self.height && (0..self.width).any(|x| self.at(x, y));
            if ink && begin.is_none() {
                begin = Some(y);
            }
            if !ink && let Some(start) = begin.take() {
                bands.push((start, y));
            }
        }
        let mut merged = Vec::new();
        let mut i = 0;
        while let Some(&(a, b)) = bands.get(i) {
            if let Some(&(c, d)) = bands.get(i + 1)
                && b - a <= ((d - c) / 3).max(2)
                && c - b <= (d - c) / 2
            {
                merged.push((a, d));
                i += 2;
            } else {
                merged.push((a, b));
                i += 1;
            }
        }
        let &(top, bottom) = merged.iter().max_by(|(a, b), (c, d)| {
            let overlap = |top: usize, bottom: usize| ((bottom as f64).min(core[1]) - (top as f64).max(core[0])).max(0.0);
            overlap(*a, *b).total_cmp(&overlap(*c, *d))
        })?;
        if bottom.saturating_sub(top) < 5 || bottom as f64 <= core[0] || top as f64 >= core[1] {
            return None;
        }
        self.trim(top, bottom)
    }
    fn trim(&self, top: usize, bottom: usize) -> Option<Self> {
        let left = (0..self.width).find(|x| (top..bottom).any(|y| self.at(*x, y)))?;
        let right = (0..self.width).rev().find(|x| (top..bottom).any(|y| self.at(*x, y)))?.saturating_add(1);
        let width = right.checked_sub(left)?;
        let height = bottom.checked_sub(top)?;
        if width < 8 || height < 5 {
            return None;
        }
        let mut ink = Vec::with_capacity(width.checked_mul(height)?);
        for y in top..bottom {
            for x in left..right {
                ink.push(self.at(x, y));
            }
        }
        let count = ink.iter().filter(|p| **p).count();
        let density = count as f64 / ink.len().max(1) as f64;
        if !(0.015..=0.65).contains(&density) {
            return None;
        }
        Some(Self { width, height, ink })
    }
}
fn otsu(grey: &[u8]) -> Option<u8> {
    let mut histogram = [0usize; 256];
    for value in grey {
        if let Some(n) = histogram.get_mut(usize::from(*value)) {
            *n += 1;
        }
    }
    let min = histogram.iter().position(|n| *n > 0)?;
    let max = histogram.iter().rposition(|n| *n > 0)?;
    if max.saturating_sub(min) < 24 {
        return None;
    }
    let total = grey.len() as f64;
    let sum = grey.iter().map(|v| f64::from(*v)).sum::<f64>();
    let (mut count, mut weight, mut best, mut threshold) = (0.0, 0.0, 0.0, min as u8);
    for (level, n) in histogram.iter().enumerate().take(max) {
        count += *n as f64;
        weight += level as f64 * *n as f64;
        if count == 0.0 || count >= total {
            continue;
        }
        let distance = weight / count - (sum - weight) / (total - count);
        let variance = count * (total - count) * distance * distance;
        if variance > best {
            best = variance;
            threshold = level as u8;
        }
    }
    Some(threshold)
}

fn render_face(bytes: &[u8], grant: FontGrant, text: &str, height: usize) -> Result<Mask, String> {
    let mut cos = pdfcraft_create::blank(8000.0, 256.0, 1).map_err(|e| e.to_string())?;
    let page = pdfcraft_model::pages(&cos).into_iter().next().ok_or("the candidate page is missing")?;
    let name = pdfcraft_edit::text::register_text_font(&mut cos, 0, bytes, grant).map_err(|e| e.to_string())?;
    let page_dict = cos.dict(&Object::Ref(page.obj)).ok_or("candidate page missing")?;
    let resources = page_dict.get(b"Resources").and_then(|o| cos.dict(o)).ok_or("candidate resources missing")?;
    let fonts = resources.get(b"Font").and_then(|o| cos.dict(o)).ok_or("candidate fonts missing")?;
    let font = fonts.get(name.as_bytes()).and_then(|o| cos.dict(o)).ok_or("candidate face missing")?;
    let metrics = pdfcraft_fonts::pdf::Metrics::from_dict(&cos, &font);
    let encoded = metrics.encode(text).ok_or("the face cannot encode this OCR row")?;
    let size = (height as f64 * 1.5).clamp(16.0, 160.0);
    let natural = metrics.codes(&encoded).iter().map(|(code, _)| metrics.width(*code) * metrics.scale * size).sum::<f64>();
    if !natural.is_finite() || natural <= 0.0 || natural > 7900.0 {
        return Err("the candidate row exceeds the raster limit".into());
    }
    let width = (natural + size * 2.0).ceil();
    let page_height = (size * 2.5).ceil();
    let mut hex = String::with_capacity(encoded.len().saturating_mul(2));
    use std::fmt::Write as _;
    for value in encoded {
        let _ = write!(hex, "{value:02X}");
    }
    let content = format!("BT /{name} {size} Tf 1 0 0 1 {size} {size} Tm <{hex}> Tj ET");
    let stream = cos.add(Object::Stream(Stream::from_raw(Dict::new(), content.into_bytes())));
    cos.update_dict(page.obj, |d| {
        d.set(b"Contents", Object::Ref(stream));
        d.set(b"MediaBox", Object::Array(vec![Object::Int(0), Object::Int(0), Object::Real(width), Object::Real(page_height)]));
    })
    .map_err(|e| e.to_string())?;
    let bytes = Arc::new(pdfcraft_cos::write_full(&cos, &SaveOptions::default()).map_err(|e| e.to_string())?);
    let scale = (2048.0 / width).min(256.0 / page_height).min(1.0) as f32;
    let out = PageRenderer::new(bytes, RenderConfig::default()).render(RenderRequest { scale, ..Default::default() });
    if let Some(error) = out.error {
        return Err(error);
    }
    let mask = Mask::from_rgba(out.width as usize, out.height as usize, &out.rgba).ok_or("the candidate has no ink")?;
    mask.line_band([0.0, mask.height as f64]).ok_or_else(|| "the candidate has no text band".into())
}
fn ink_score(scan: &Mask, candidate: &Mask) -> f64 {
    let mut stretched = Vec::with_capacity(scan.ink.len());
    for y in 0..scan.height {
        for x in 0..scan.width {
            let sx = x.saturating_mul(candidate.width) / scan.width.max(1);
            let sy = y.saturating_mul(candidate.height) / scan.height.max(1);
            stretched.push(candidate.at(sx, sy));
        }
    }
    let stretched = Mask { width: scan.width, height: scan.height, ink: stretched };
    let agreement = |a: &Mask, b: &Mask| {
        let mut ink = 0usize;
        let mut hits = 0usize;
        for y in 0..a.height {
            for x in 0..a.width {
                if a.at(x, y) {
                    ink += 1;
                    if (y.saturating_sub(1)..=y.saturating_add(1).min(b.height.saturating_sub(1)))
                        .any(|by| (x.saturating_sub(1)..=x.saturating_add(1).min(b.width.saturating_sub(1))).any(|bx| b.at(bx, by)))
                    {
                        hits += 1;
                    }
                }
            }
        }
        if ink == 0 { 0.0 } else { hits as f64 / ink as f64 }
    };
    let precision = agreement(&stretched, scan);
    let recall = agreement(scan, &stretched);
    if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) }
}
