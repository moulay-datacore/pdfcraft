//! Scan & OCR ▸ Recognize text: render pages, read the words in them and add an invisible text
//! layer ([`Edit::AddOcrText`]), making scanned pages searchable and selectable.
//!
//! Recognition is slow (about a second a page), so it is split from the edit: [`Session::ocr_job`]
//! captures what it needs, [`OcrJob::run`] works anywhere (the UI runs it on a worker thread) and
//! [`Session::apply_ocr`] applies the result as one undoable step.
//!
//! **Languages.** A run reads one language ([`LANGUAGES`]): the one its settings name, else the
//! default for the document ([`Session::ocr_language_for`]: the session's OCR language, which is
//! the user's last choice, then the document's own `/Lang`, then English). A language whose model
//! isn't installed (or can't be loaded) is read as English, and [`OcrOutcome::fallback`] says so:
//! it is never silent.
//!
//! **Reading a page again.** PdfCraft's own OCR text counts as the page's text, so a second run
//! skips the page, or adds only the words the first one missed. With
//! [`OcrSettings::replace_ocr`] it is set aside instead: the page is read as if it weren't there,
//! and the new text replaces it in the same undoable step ([`Edit::ReplaceOcrText`]). Only text
//! that is exactly an OCR layer PdfCraft wrote is replaced ([`pdfcraft_edit::own_ocr_layers`]);
//! an edited layer and every other kind of text stay.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError, TryLockError};

use pdfcraft_cos::{SaveOptions, write_incremental};
use pdfcraft_render::{PageInfo, PageRenderer, RenderConfig, RenderRequest, RequestKind};

pub use pdfcraft_ocr::{
    Availability, ENGLISH, Fallback, FallbackReason, LANGUAGES, Language, LatinModel, Models, Ocr, OcrError, PlacedWord, Reader, language,
};

use crate::{DocId, Edit, EditError, Session};

/// Recognize Text settings.
#[derive(Clone, Debug, PartialEq)]
pub struct OcrSettings {
    /// Resolution pages are rendered at for recognition (Acrobat's "Downsample to" choices).
    pub dpi: f32,
    /// The language to read: a code from [`LANGUAGES`] or a language tag such as `fr-CA`
    /// ([`language`]). `None` reads the document's default ([`Session::ocr_language_for`]). A
    /// language text recognition doesn't read is refused, never read as another one.
    pub language: Option<String>,
    /// Leave pages that already have text alone (Acrobat reports "page contains renderable
    /// text" and skips them). When `false`, such pages are read too (a picture on a page that
    /// also has text, #1028) and only the words that aren't already there are added.
    pub skip_text_pages: bool,
    /// What counts as the page's text when `skip_text_pages` is on: only its own (page content
    /// and form fields). Comments and text boxes are left out, and are not drawn into the picture
    /// that is read, so a scan with a few comments on it is still a scan. `false`: everything
    /// the page shows.
    pub native_text_only: bool,
    /// Replace PdfCraft's own earlier OCR text ([`pdfcraft_edit::own_ocr_layers`]): it doesn't
    /// count as the page's text, so a page whose only text it is is read again, and the new
    /// words take its place in the same undoable step. A page where nothing is recognised keeps
    /// it, and so does one read as English because its language's model is missing (that would
    /// lose its accents). Text added any other way, an OCR layer edited since included, stays
    /// and counts.
    pub replace_ocr: bool,
}

/// A recognised word whose box is at least this much covered by the page's own text is that
/// text read again, not a word in a picture.
const ALREADY_ON_PAGE: f32 = 0.5;

/// The long side of an A4 sheet in points (a letter page's, 792, is shorter): "Downsample to"
/// is a resolution on paper this size or smaller.
const SHEET: f32 = 842.0;

/// How much of `word` the `text` boxes cover, from 0 to 1. Boxes are `[x0, y0, x1, y1]` in page
/// view space. Overlaps between the text boxes are counted twice (glyphs rarely overlap), so the
/// sum is capped at the whole word.
fn covered(word: [f32; 4], text: &[[f32; 4]]) -> f32 {
    // `max(0.0)` also turns a NaN side into 0: a degenerate box covers nothing.
    let area = |[x0, y0, x1, y1]: [f32; 4]| (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let total = area(word);
    if !total.is_finite() || total <= 0.0 {
        return 0.0;
    }
    let inside: f32 =
        text.iter().map(|g| area([word[0].max(g[0]), word[1].max(g[1]), word[2].min(g[2]), word[3].min(g[3])])).filter(|a| a.is_finite()).sum();
    (inside / total).min(1.0)
}

impl Default for OcrSettings {
    fn default() -> Self {
        OcrSettings { dpi: 300.0, language: None, skip_text_pages: true, native_text_only: false, replace_ocr: false }
    }
}

/// The language `tag` names ([`language`]), or why it can't be read.
pub fn resolve_language(tag: &str) -> Result<&'static Language, String> {
    language(tag).ok_or_else(|| {
        // The tag is untrusted input: quote a short piece of it.
        let shown: String = tag.chars().take(32).collect();
        let codes: Vec<&str> = LANGUAGES.iter().map(|l| l.code).collect();
        format!("text recognition doesn't read the language {shown:?}; use one of {}", codes.join(", "))
    })
}

/// What recognition found on one page.
#[derive(Clone, Debug, PartialEq)]
pub struct OcrPage {
    pub page: usize,
    pub words: Vec<PlacedWord>,
    /// Words recognised but left out because the page's own text already has them there.
    pub already_on_page: usize,
    /// Why the page was not read, if it was skipped.
    pub skipped: Option<String>,
    /// With [`OcrSettings::replace_ocr`]: the number of words of PdfCraft's earlier OCR text that
    /// `words` replace. `None` when they replace nothing (the page had none, or nothing was
    /// recognised on it, which keeps it).
    pub replaced: Option<usize>,
}

impl OcrPage {
    pub fn text(&self) -> String {
        self.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
    }
}

/// What [`OcrJob::run`] did.
#[derive(Clone, Debug, PartialEq)]
pub struct OcrOutcome {
    pub pages: Vec<OcrPage>,
    /// The language the pages were read as: the one asked for, or English when it fell back.
    pub language: &'static Language,
    /// Set when the language asked for was read as English instead, and why.
    pub fallback: Option<Fallback>,
}

impl OcrOutcome {
    pub fn words(&self) -> usize {
        self.pages.iter().map(|p| p.words.len()).sum()
    }

    /// The pages whose earlier OCR text the new text replaces.
    pub fn replaced_pages(&self) -> usize {
        self.pages.iter().filter(|p| p.replaced.is_some()).count()
    }
}

/// A model file as the recogniser cache knows it: its path, and its size and modification time
/// when it can be read, so a file replaced or repaired while the app runs is loaded again.
type FileStamp = (PathBuf, Option<(u64, Option<std::time::SystemTime>)>);

/// The model files of `models`, stamped.
fn stamps(models: &Models) -> Vec<FileStamp> {
    let mut files = vec![models.detection.clone()];
    files.extend(models.recognition.clone());
    if let Some(latin) = &models.latin {
        files.extend([latin.model.clone(), latin.dict.clone()]);
    }
    files
        .into_iter()
        .map(|path| {
            let stamp = std::fs::metadata(&path).ok().map(|m| (m.len(), m.modified().ok()));
            (path, stamp)
        })
        .collect()
}

/// The recognisers loaded, newest last, each with the files it was loaded from.
static ENGINES: Mutex<Vec<(Vec<FileStamp>, Arc<Ocr>)>> = Mutex::new(Vec::new());

/// The most recognisers kept loaded. One set of model files is the rule; another appears when the
/// files change (models installed or replaced) or a caller names its own ([`engine_for`]).
const LOADED: usize = 3;

/// The recogniser for the models [`Models::find`] finds, loaded once (it takes a moment) and
/// shared: see [`engine_for`].
pub fn engine() -> Result<Arc<Ocr>, String> {
    engine_for(&Models::find().ok_or_else(|| OcrError::NoModels.to_string())?)
}

/// The recogniser for `models`, loaded once and shared. It is loaded again when one of its
/// files changes, so a model installed, repaired or replaced while the app runs is picked up
/// (a recogniser remembers a model that failed to load). Each language's reader inside it is
/// loaded the first time that language is read.
pub fn engine_for(models: &Models) -> Result<Arc<Ocr>, String> {
    let key = stamps(models);
    // Loaded under the lock: two runs starting together load the models once.
    let mut loaded = ENGINES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, ocr)) = loaded.iter().find(|(k, _)| *k == key) {
        return Ok(ocr.clone());
    }
    // The model files are untrusted (PDFCRAFT_MODELS can point anywhere): a panic while loading
    // them is an error, not a crash.
    let ocr = Arc::new(
        crate::guard(|| Ocr::load(models)).map_err(|p| format!("loading the text recognition models failed: {p}"))?.map_err(|e| e.to_string())?,
    );
    if loaded.len() >= LOADED {
        loaded.remove(0);
    }
    loaded.push((key, ocr.clone()));
    Ok(ocr)
}

/// Whether the recognition models are installed: at least one language can be read.
pub fn available() -> bool {
    Models::find().is_some()
}

/// Whether each of [`LANGUAGES`] can be read, in order, with `models` (the models
/// [`Models::find`] finds when `None`): from the files installed and, once a recogniser for them
/// is loaded, from whether each reader loaded. Cheap enough to ask every frame: it never waits
/// for models being loaded.
pub fn statuses(models: Option<&Models>) -> Vec<(&'static Language, Availability)> {
    let found;
    let models = match models {
        Some(m) => m,
        None => match Models::find() {
            Some(m) => {
                found = m;
                &found
            }
            None => return LANGUAGES.iter().map(|l| (l, Availability::Missing)).collect(),
        },
    };
    let key = stamps(models);
    let loaded = match ENGINES.try_lock() {
        Ok(engines) => engines.iter().find(|(k, _)| *k == key).map(|(_, o)| o.clone()),
        Err(TryLockError::Poisoned(p)) => p.into_inner().iter().find(|(k, _)| *k == key).map(|(_, o)| o.clone()),
        Err(TryLockError::WouldBlock) => None,
    };
    LANGUAGES.iter().map(|l| (l, loaded.as_ref().map_or_else(|| models.availability(l), |o| o.availability(l)))).collect()
}

/// Whether `language` can be read with the models [`Models::find`] finds ([`statuses`]).
pub fn availability(language: &Language) -> Availability {
    statuses(None).into_iter().find(|(l, _)| *l == language).map_or(Availability::Missing, |(_, a)| a)
}

/// A document's bytes without PdfCraft's own OCR text, and the words that text had per page.
type SetAside = (Arc<Vec<u8>>, BTreeMap<usize, usize>);

/// Everything recognition needs from a document, detached from the session.
pub struct OcrJob {
    bytes: Arc<Vec<u8>>,
    password: Option<String>,
    infos: Vec<PageInfo>,
    /// Per page: the render scale (pixels per point) that shows its images at their own
    /// resolution, if it has any.
    native: Vec<Option<f32>>,
    pub pages: Vec<usize>,
    pub settings: OcrSettings,
    /// The language the pages are read as: the settings', else the document's default.
    pub language: &'static Language,
    /// With [`OcrSettings::replace_ocr`]: per page that has PdfCraft's own OCR text, its number
    /// of words. `bytes` is the document without it, so the page is read as if it weren't there.
    replacing: BTreeMap<usize, usize>,
}

impl OcrJob {
    /// Read the pages. `progress(done, total)` is called before each page; returning `false`
    /// stops (the pages read so far are returned).
    pub fn run(self, ocr: &Ocr, mut progress: impl FnMut(usize, usize) -> bool) -> OcrOutcome {
        let config =
            RenderConfig { password: self.password.as_deref().map(Arc::from), hide_comments: self.settings.native_text_only, ..Default::default() };
        let mut r = PageRenderer::new(self.bytes.clone(), config);
        let mut out = Vec::new();
        let mut fallback = None;
        let total = self.pages.len();
        for (done, &page) in self.pages.iter().enumerate() {
            if !progress(done, total) {
                break;
            }
            let Some(info) = self.infos.get(page) else { continue };
            let skip = |why: &str| OcrPage { page, words: Vec::new(), already_on_page: 0, skipped: Some(why.into()), replaced: None };
            // The boxes of the page's own text (an earlier OCR layer's included, unless it is
            // being replaced), in view space.
            let existing: Vec<[f32; 4]> = r
                .render(RenderRequest { page, kind: RequestKind::Text, scale: 1.0, ..Default::default() })
                .text
                .map(|t| t.glyphs.iter().filter(|g| !g.text.trim().is_empty()).map(|g| g.rect).collect())
                .unwrap_or_default();
            if self.settings.skip_text_pages && !existing.is_empty() {
                out.push(skip("the page already contains text"));
                continue;
            }
            let shot = r.render(RenderRequest { page, scale: self.scale(page, info), ..Default::default() });
            if let Some(e) = shot.error {
                out.push(skip(&e));
                continue;
            }
            // One page the recogniser can't take (a panic in it) skips that page, not the run.
            let read = crate::guard(|| ocr.recognize_in(&shot.rgba, shot.width, shot.height, self.language))
                .unwrap_or_else(|panic| Err(OcrError::Recognize(panic)));
            let recognition = match read {
                Ok(r) => r,
                Err(e) => {
                    out.push(skip(&e.to_string()));
                    continue;
                }
            };
            // Replacing never trades the earlier text for a poorer reading: read as English
            // instead of the language asked for, the page keeps what it has.
            if let (Some(f), true) = (&recognition.fallback, self.replacing.contains_key(&page)) {
                out.push(skip(&format!(
                    "its earlier recognized text was kept rather than replaced by a reading in English instead of {}",
                    f.from.name
                )));
                fallback = fallback.or(recognition.fallback);
                continue;
            }
            if fallback.is_none() {
                fallback = recognition.fallback;
            }
            let scale = shot.width as f32 / info.width.max(1e-3);
            let to_user = |x: f32, y: f32| {
                let [u, v] = info.view_to_user(x / scale, y / scale);
                [u as f64, v as f64]
            };
            // Words the page's own text already has are not added a second time.
            let (fresh, already): (Vec<_>, Vec<_>) =
                recognition.lines.iter().flat_map(|l| &l.words).partition(|w| covered(w.rect.map(|v| v / scale), &existing) < ALREADY_ON_PAGE);
            let words: Vec<PlacedWord> = fresh.into_iter().map(|w| PlacedWord::place(w, to_user)).collect();
            // Nothing recognised keeps the earlier text: an empty result replaces nothing.
            let replaced = self.replacing.get(&page).copied().filter(|_| !words.is_empty());
            out.push(OcrPage { page, words, already_on_page: already.len(), skipped: None, replaced });
        }
        progress(total, total);
        let language = if fallback.is_some() { ENGLISH } else { self.language };
        OcrOutcome { pages: out, language, fallback }
    }

    /// The render scale (pixels per point) `page` is read at. A scan is read at its own
    /// resolution (enlarging it only blurs the letters), at most the chosen one.
    fn scale(&self, page: usize, info: &PageInfo) -> f32 {
        let wanted = self.settings.dpi.clamp(72.0, 600.0) / 72.0;
        let Some(native) = self.native.get(page).copied().flatten() else { return wanted };
        // A scan whose page is larger than a sheet (often a box given in pixels, one point per
        // pixel, #1848) is measured as if shrunk to one: otherwise 1 px/pt is already its full
        // image, and no "Downsample to" choice could ask for less.
        let fit = (SHEET / info.width.max(info.height).max(1e-3)).min(1.0);
        native.clamp(fit, wanted * fit)
    }
}

impl Session {
    /// Recognize Text's language when a run names none: the user's last choice (the interface
    /// remembers it), or `None` to follow each document's language. Refuses a language text
    /// recognition doesn't read, keeping the previous one.
    pub fn set_ocr_language(&mut self, tag: Option<&str>) -> Result<(), String> {
        self.ocr_language = tag.map(resolve_language).transpose()?;
        Ok(())
    }

    /// See [`Session::set_ocr_language`].
    pub fn ocr_language(&self) -> Option<&'static Language> {
        self.ocr_language
    }

    /// The language document `id` is read as when a run names none: the session's OCR language
    /// (the user's last choice), else the document's own (its `/Lang`, when text recognition
    /// reads it), else `interface` (the interface language; automation passes none, so its
    /// runs don't depend on where they run), else English.
    pub fn ocr_language_for(&self, id: DocId, interface: Option<&str>) -> &'static Language {
        let document = self.get(id).and_then(|d| d.initial_view().language);
        pdfcraft_ocr::first_language([self.ocr_language.map(|l| l.code), document.as_deref(), interface])
    }

    /// Of `pages` (0-based; empty = all), those that have PdfCraft's own OCR text, which
    /// [`OcrSettings::replace_ocr`] replaces. It reads their OCR streams: ask once per change.
    pub fn own_ocr_pages(&self, id: DocId, pages: &[usize]) -> Vec<usize> {
        let Some(editor) = self.get(id).and_then(|d| d.editor.as_ref()) else { return Vec::new() };
        let all = self.get(id).map_or(0, |d| d.info.pages.len());
        let pages: Vec<usize> = if pages.is_empty() { (0..all).collect() } else { pages.iter().copied().filter(|p| *p < all).collect() };
        pages.into_iter().filter(|&p| pdfcraft_edit::own_ocr_layers(&editor.cos, p).is_ok_and(|l| l.streams > 0)).collect()
    }

    /// Capture what recognising `pages` (0-based; empty = all) of document `id` needs. Refuses a
    /// language text recognition doesn't read.
    pub fn ocr_job(&self, id: DocId, pages: &[usize], settings: OcrSettings) -> Result<OcrJob, String> {
        let doc = self.get(id).ok_or("no such document")?;
        let language = match settings.language.as_deref() {
            Some(tag) => resolve_language(tag)?,
            None => self.ocr_language_for(id, None),
        };
        let all = doc.info.pages.len();
        let pages: Vec<usize> = if pages.is_empty() { (0..all).collect() } else { pages.iter().copied().filter(|p| *p < all).collect() };
        let native = (0..all)
            .map(|p| {
                if !pages.contains(&p) {
                    return None;
                }
                doc.page_images(p)
                    .iter()
                    .filter(|i| i.width > 1 && i.height > 1)
                    .map(|i| {
                        let m = i.matrix;
                        let (w, h) = (m[0].hypot(m[1]), m[2].hypot(m[3]));
                        (i.width as f64 / w.max(1e-6)).max(i.height as f64 / h.max(1e-6)) as f32
                    })
                    .reduce(f32::max)
            })
            .collect();
        let (bytes, replacing) = if settings.replace_ocr { self.without_own_ocr(id, &pages)? } else { (doc.bytes.clone(), BTreeMap::new()) };
        Ok(OcrJob { bytes, password: doc.password.clone(), infos: doc.info.pages.clone(), native, pages, settings, language, replacing })
    }

    /// The words of PdfCraft's own OCR text on each of `pages` that has some, and document
    /// `id`'s bytes without it, to read those pages as if it weren't there. The document itself
    /// is not changed.
    fn without_own_ocr(&self, id: DocId, pages: &[usize]) -> Result<SetAside, String> {
        let doc = self.get(id).ok_or("no such document")?;
        let Some(editor) = doc.editor.as_ref() else { return Ok((doc.bytes.clone(), BTreeMap::new())) };
        let found: BTreeMap<usize, usize> = pages
            .iter()
            .filter_map(|&p| pdfcraft_edit::own_ocr_layers(&editor.cos, p).ok().filter(|l| l.streams > 0).map(|l| (p, l.words)))
            .collect();
        if found.is_empty() {
            return Ok((doc.bytes.clone(), found));
        }
        let set_aside = crate::guard(|| {
            let mut cos = editor.cos.clone();
            for &page in found.keys() {
                pdfcraft_edit::remove_own_ocr_layers(&mut cos, page).map_err(|e| e.to_string())?;
            }
            write_incremental(&cos, &SaveOptions::default()).map_err(|e| e.to_string())
        });
        let bytes = set_aside.and_then(|r| r).map_err(|e| format!("the earlier recognized text could not be set aside: {e}"))?;
        Ok((Arc::new(bytes), found))
    }

    /// Add the words found by [`OcrJob::run`] as one undoable step, replacing PdfCraft's earlier
    /// OCR text on the pages that say so ([`OcrPage::replaced`]). Returns the number of words.
    pub fn apply_ocr(&mut self, id: DocId, found: &[OcrPage]) -> Result<usize, EditError> {
        let edits: Vec<Edit> = found
            .iter()
            .filter(|p| !p.words.is_empty())
            .map(|p| match p.replaced {
                Some(_) => Edit::ReplaceOcrText { page: p.page, words: p.words.clone() },
                None => Edit::AddOcrText { page: p.page, words: p.words.clone() },
            })
            .collect();
        let words = found.iter().map(|p| p.words.len()).sum();
        if !edits.is_empty() {
            self.apply(id, Edit::Batch { label: "Recognize text".into(), edits })?;
        }
        Ok(words)
    }

    /// Recognise text on `pages` and add it, in one call (the CLI and agents use this).
    pub fn recognize_text(&mut self, id: DocId, pages: &[usize], settings: OcrSettings) -> Result<OcrOutcome, String> {
        if let Some(why) = self.get(id).ok_or("no such document")?.read_only_reason.clone() {
            return Err(why);
        }
        let job = self.ocr_job(id, pages, settings)?;
        let ocr = engine()?;
        let found = job.run(&ocr, |_, _| true);
        self.apply_ocr(id, &found.pages).map_err(|e| e.to_string())?;
        Ok(found)
    }
}

/// What [`recognize_file`] did to one file.
#[derive(Clone, Debug, PartialEq)]
pub struct FileResult {
    /// The new file (incrementally saved), or the original when nothing was recognised.
    pub bytes: Arc<Vec<u8>>,
    pub pages: Vec<OcrPage>,
    /// As in [`OcrOutcome`]: the language read, and the fallback to English, if any.
    pub language: &'static Language,
    pub fallback: Option<Fallback>,
}

impl FileResult {
    pub fn words(&self) -> usize {
        self.pages.iter().map(|p| p.words.len()).sum()
    }
}

/// Recognize text in multiple files: one PDF's bytes in, the searchable PDF out. `progress`
/// works as for [`OcrJob::run`]. Without a language in `settings`, each file is read in its own
/// (its `/Lang`), else in `interface` (the interface language: the UI passes its own, automation
/// none), else English ([`Session::ocr_language_for`]).
pub fn recognize_file(
    name: &str,
    bytes: Arc<Vec<u8>>,
    password: Option<&str>,
    mut settings: OcrSettings,
    interface: Option<&str>,
    ocr: &Ocr,
    progress: impl FnMut(usize, usize) -> bool,
) -> Result<FileResult, String> {
    let mut s = Session::new();
    let id = s.open(name, None, bytes.clone(), password).map_err(|e| e.to_string())?;
    if let Some(why) = s.get(id).and_then(|d| d.read_only_reason.clone()) {
        return Err(why);
    }
    if settings.language.is_none() {
        settings.language = Some(s.ocr_language_for(id, interface).code.to_string());
    }
    let job = s.ocr_job(id, &[], settings)?;
    let OcrOutcome { pages, language, fallback } = job.run(ocr, progress);
    if s.apply_ocr(id, &pages).map_err(|e| e.to_string())? == 0 {
        return Ok(FileResult { bytes, pages, language, fallback });
    }
    // Checked like a save before the caller writes it (#2166).
    let bytes = s.save_bytes(id).map_err(|e| e.to_string())?;
    s.check_save(id, &bytes).map_err(|e| e.to_string())?;
    Ok(FileResult { bytes, pages, language, fallback })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export;

    /// #1028: a recognised word over the page's own text is that text again; one in an empty
    /// part of the page (a picture) is new.
    #[test]
    fn words_over_existing_text_count_as_already_there() {
        // Glyph em boxes of "Total" at 12 pt, five letters side by side.
        let glyphs: Vec<[f32; 4]> = (0..5).map(|i| [100.0 + 7.0 * i as f32, 50.0, 107.0 + 7.0 * i as f32, 62.0]).collect();
        // OCR's tight box around the same ink.
        assert!(covered([100.5, 52.0, 134.0, 61.0], &glyphs) >= ALREADY_ON_PAGE);
        // A word in a picture further down, and one beside the text on the same line.
        assert_eq!(covered([100.0, 200.0, 140.0, 212.0], &glyphs), 0.0);
        assert_eq!(covered([140.0, 52.0, 170.0, 61.0], &glyphs), 0.0);
        // Touching the text's last letter is not enough.
        assert!(covered([131.0, 52.0, 170.0, 61.0], &glyphs) < ALREADY_ON_PAGE);
        // A page without text has nothing to cover a word.
        assert_eq!(covered([100.5, 52.0, 134.0, 61.0], &[]), 0.0);
    }

    /// #1848: an A4 scan at 300 dpi whose page box is its pixel size (2480 x 3508 pt, one pixel
    /// per point) gets a smaller bitmap for each lower "Downsample to" choice, and a low-resolution
    /// letter page is still never shrunk below one pixel per point.
    #[test]
    fn downsample_shrinks_a_page_box_given_in_pixels() {
        let mut s = Session::new();
        let page = |w: u32, h: u32| export::encode_png(w, h, &vec![255; (w * h * 4) as usize]).unwrap();
        let bytes = s.create_from_images(&[("a4.png".into(), page(2480, 3508)), ("letter.png".into(), page(612, 792))]).unwrap();
        let id = s.open("scan.pdf", None, bytes, None).unwrap();
        let mut r = PageRenderer::new(s.get(id).unwrap().bytes.clone(), RenderConfig::default());
        for (dpi, a4) in [(72.0, [595, 842]), (150.0, [1240, 1754]), (300.0, [2480, 3508]), (600.0, [2480, 3508])] {
            let job = s.ocr_job(id, &[], OcrSettings { dpi, ..Default::default() }).unwrap();
            for (page, want) in [(0, a4), (1, [612, 792])] {
                let scale = job.scale(page, &job.infos[page]);
                let shot = r.render(RenderRequest { page, scale, ..Default::default() });
                assert!(shot.error.is_none(), "{:?}", shot.error);
                let near = |got: u32, want: u32| got.abs_diff(want) <= 1;
                assert!(near(shot.width, want[0]) && near(shot.height, want[1]), "{dpi} dpi, page {page}: {}x{}", shot.width, shot.height);
            }
        }
    }

    /// A one-page scan: a plain light picture covering the page (no text of its own).
    fn scan(s: &mut Session, pages: usize) -> DocId {
        let png = export::encode_png(200, 200, &[250; 200 * 200 * 4]).unwrap();
        let images: Vec<(String, Vec<u8>)> = (0..pages).map(|i| (format!("page{i}.png"), png.clone())).collect();
        let bytes = s.create_from_images(&images).unwrap();
        s.open("scan.pdf", None, bytes, None).unwrap()
    }

    fn word(text: &str, x: f64, y: f64) -> PlacedWord {
        PlacedWord { text: text.into(), origin: [x, y], across: [6.0 * text.len() as f64, 0.0], up: [0.0, 8.0] }
    }

    /// The text of each page of `bytes`, as the renderer extracts it.
    fn texts(bytes: &Arc<Vec<u8>>) -> Vec<String> {
        let mut r = PageRenderer::new(bytes.clone(), RenderConfig::default());
        let pages = pdfcraft_render::inspect(bytes.clone(), None).map_or(0, |i| i.pages.len());
        (0..pages)
            .map(|page| {
                let t = r.render(RenderRequest { page, kind: RequestKind::Text, scale: 1.0, ..Default::default() }).text;
                t.map(|t| t.plain_text().split_whitespace().collect::<Vec<_>>().join(" ")).unwrap_or_default()
            })
            .collect()
    }

    /// The language a run reads: the one it names (a tag works too, an unknown one is refused),
    /// else the session's (the user's last choice), else the document's `/Lang`, else English;
    /// the interface's language comes after the document's.
    #[test]
    fn the_language_is_the_run_s_then_the_user_s_then_the_document_s() {
        let mut s = Session::new();
        let id = scan(&mut s, 1);
        let read = |s: &Session, language: Option<&str>| s.ocr_job(id, &[], OcrSettings { language: language.map(Into::into), ..Default::default() });
        assert_eq!(read(&s, None).unwrap().language, ENGLISH, "a document without /Lang");
        assert_eq!(s.ocr_language_for(id, Some("pt-BR")).code, "pt", "then the interface's");
        assert_eq!(s.ocr_language_for(id, Some("ja")), ENGLISH, "an interface language OCR doesn't read");

        let mut view = s.get(id).unwrap().initial_view();
        view.language = Some("fr-FR".into());
        s.apply(id, Edit::SetInitialView(Box::new(view))).unwrap();
        assert_eq!(read(&s, None).unwrap().language.code, "fr", "the document's /Lang");
        assert_eq!(s.ocr_language_for(id, Some("de")).code, "fr", "before the interface's");

        s.set_ocr_language(Some("DE-at")).unwrap();
        assert_eq!(s.ocr_language().map(|l| l.code), Some("de"));
        assert_eq!(read(&s, None).unwrap().language.code, "de", "the user's choice comes first");
        assert_eq!(read(&s, Some("es-419")).unwrap().language.code, "es", "the run's own beats it");

        let refused = read(&s, Some("xx")).err().unwrap();
        assert!(refused.contains("\"xx\"") && refused.contains("en, fr, de, es, it, pt, nl"), "{refused}");
        let long = "x".repeat(10_000);
        assert!(read(&s, Some(&long)).err().unwrap().len() < 200, "an untrusted tag is quoted short");
        assert!(s.set_ocr_language(Some("pl")).is_err(), "Polish needs a Unicode text layer first");
        assert_eq!(s.ocr_language().map(|l| l.code), Some("de"), "a refused choice keeps the last one");
        s.set_ocr_language(None).unwrap();
        assert_eq!(read(&s, None).unwrap().language.code, "fr", "back to the document's");
    }

    /// Re-reading with `replace_ocr`: PdfCraft's own OCR text is set aside for the reading (the
    /// document is untouched until the result is applied), and applying replaces it in one
    /// undoable step that leaves everything else on the page as it was. No models needed: the
    /// result is made up.
    #[test]
    fn replacing_sets_the_own_ocr_text_aside_and_swaps_it_in_one_step() {
        let mut s = Session::new().with_clock(|| 1_700_000_000);
        let id = scan(&mut s, 2);
        let (w, h) = (s.get(id).unwrap().info.pages[0].width as f64, s.get(id).unwrap().info.pages[0].height as f64);
        s.apply(id, Edit::AddOcrText { page: 0, words: vec![word("Odl", 10.0, 100.0), word("wrds", 40.0, 100.0)] }).unwrap();
        let note = crate::AddedText { rect: [10.0, h * 0.6, w - 10.0, h - 10.0], text: "Invoice".into(), size: 12.0, ..Default::default() };
        s.apply(id, Edit::AddText { page: 1, text: note }).unwrap();
        s.apply(id, Edit::AddOcrText { page: 1, words: vec![word("scan", 10.0, 20.0)] }).unwrap();
        let before = s.get(id).unwrap().bytes.clone();
        assert_eq!(texts(&before), ["Odl wrds", "Invoice scan"]);
        // What `text_layer` writes is what `own_ocr_layers` finds.
        assert_eq!(s.own_ocr_pages(id, &[]), [0, 1]);
        assert_eq!(s.own_ocr_pages(id, &[1, 9]), [1]);

        let job = s.ocr_job(id, &[], OcrSettings { replace_ocr: true, ..Default::default() }).unwrap();
        assert_eq!(job.replacing, BTreeMap::from([(0, 2), (1, 1)]));
        assert_eq!(texts(&job.bytes), ["", "Invoice"], "read as if the OCR text weren't there; the added text stays");
        assert!(Arc::ptr_eq(&s.get(id).unwrap().bytes, &before), "the document itself is untouched");
        let plain = s.ocr_job(id, &[], OcrSettings::default()).unwrap();
        assert!(plain.replacing.is_empty() && Arc::ptr_eq(&plain.bytes, &before), "without replace_ocr nothing is set aside");

        // Page 0 is read again; page 1 still has text of its own, so it is skipped and kept.
        let found = [
            OcrPage {
                page: 0,
                words: vec![word("Old", 10.0, 100.0), word("words", 40.0, 100.0)],
                already_on_page: 0,
                skipped: None,
                replaced: Some(2),
            },
            OcrPage { page: 1, words: Vec::new(), already_on_page: 0, skipped: Some("the page already contains text".into()), replaced: None },
        ];
        let labels_before = s.get(id).unwrap().can_undo().map(String::from);
        assert_eq!(s.apply_ocr(id, &found).unwrap(), 2);
        assert_eq!(texts(&s.get(id).unwrap().bytes), ["Old words", "Invoice scan"]);
        assert_eq!(s.own_ocr_pages(id, &[]), [0, 1], "the new text is an OCR layer of PdfCraft's too");
        assert_eq!(s.get(id).unwrap().can_undo(), Some("Recognize text"));
        assert_eq!(s.get(id).unwrap().page_images(0).len(), 1, "the scan stays");

        s.undo(id).unwrap();
        assert_eq!(texts(&s.get(id).unwrap().bytes), ["Odl wrds", "Invoice scan"], "one undo brings the earlier text back");
        assert_eq!(s.get(id).unwrap().can_undo().map(String::from), labels_before);
        s.redo(id).unwrap();
        assert_eq!(texts(&s.get(id).unwrap().bytes), ["Old words", "Invoice scan"]);
    }

    /// Model files are untrusted: missing or damaged ones are an error, never a panic, a
    /// poisoned cache still answers, and what each language can do follows the files.
    #[test]
    fn the_recogniser_cache_survives_bad_files_and_a_poisoned_lock() {
        let dir = std::env::temp_dir().join(format!("pdfcraft-engine-ocr-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let junk = dir.join("junk.rten");
        std::fs::write(&junk, b"not a model").unwrap();
        let models = |detection: PathBuf| Models { detection, recognition: Some(junk.clone()), latin: None };
        assert!(engine_for(&models(dir.join("missing.rten"))).is_err());
        let _ = std::thread::spawn(|| {
            let _held = ENGINES.lock();
            panic!("a worker died holding the cache");
        })
        .join();
        assert!(engine_for(&models(junk.clone())).is_err(), "a damaged file is an error");
        use Availability::*;
        let states = |m: &Models| statuses(Some(m)).into_iter().map(|(l, a)| (l.code, a)).collect::<Vec<_>>();
        let english = states(&models(junk.clone()));
        assert_eq!(english.first(), Some(&("en", Ready)));
        assert!(english.iter().skip(1).all(|(_, a)| *a == FallsBackToEnglish), "{english:?}");
        let latin =
            Models { detection: junk.clone(), recognition: None, latin: Some(pdfcraft_ocr::LatinModel { model: junk.clone(), dict: junk.clone() }) };
        let latin = states(&latin);
        assert_eq!(latin.first(), Some(&("en", Missing)));
        assert!(latin.iter().skip(1).all(|(_, a)| *a == Ready), "{latin:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn coverage_of_overlapping_or_broken_boxes_stays_between_zero_and_one() {
        // A glyph drawn twice (fake bold) does not cover more than the whole word.
        let twice = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]];
        assert_eq!(covered([0.0, 0.0, 10.0, 10.0], &twice), 1.0);
        // Empty, inverted and non-finite boxes cover nothing and are covered by nothing.
        for word in [[5.0, 5.0, 5.0, 9.0], [9.0, 0.0, 1.0, 9.0], [f32::NAN, 0.0, 9.0, 9.0], [0.0, 0.0, f32::INFINITY, 9.0]] {
            assert_eq!(covered(word, &twice), 0.0, "{word:?}");
        }
        let broken = [[f32::NAN; 4], [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY]];
        let c = covered([0.0, 0.0, 10.0, 10.0], &broken);
        assert!((0.0..=1.0).contains(&c), "{c}");
    }
}
