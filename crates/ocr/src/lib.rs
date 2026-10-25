//! pdfcraft-ocr — Scan & OCR ▸ Recognize text (L4).
//!
//! The caller renders a page to pixels; [`Ocr::recognize_in`] finds the words in them, read as
//! a [`Language`], and [`text_layer`] turns words placed in user space into page content:
//! invisible text (rendering mode 3) over each word, so the page becomes a searchable image
//! (Acrobat's "Searchable Image (Exact)": the image is left untouched).
//!
//! The models are fetched by `cargo xtask models` (see ATTRIBUTION.toml) and found by
//! [`Models::find`]. The ocrs engine (MIT/Apache-2.0) and its detection model (CC-BY-SA-4.0)
//! find the words and group them into lines for every language. Then:
//! - English is read by ocrs's recognition model (CC-BY-SA-4.0): the Latin alphabet without
//!   accents;
//! - French, German, Spanish, Italian, Portuguese and Dutch are read by PaddleOCR's PP-OCRv6
//!   recognition model (Apache-2.0), accents included, restricted to the characters the text
//!   layer's WinAnsiEncoding can hold.
//!
//! A language whose model is missing is read as English instead, and the result says so
//! ([`Recognition::fallback`]): it is never silent.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod pp_ocr;

use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub use pdfcraft_fonts::helvetica_width;

/// The model files, as named in ATTRIBUTION.toml.
pub const DETECTION_MODEL: &str = "text-detection.rten";
pub const RECOGNITION_MODEL: &str = "text-recognition.rten";
/// PaddleOCR's PP-OCRv6 small recognition model, the official ONNX file unchanged, and its
/// character dictionary: they read accented Latin. Their names say which model they are, so a
/// newer model comes with new names and a mismatched pair can't be installed by accident.
pub const LATIN_MODEL: &str = "PP-OCRv6_small_rec.onnx";
pub const LATIN_DICT: &str = "ppocrv6_dict.txt";

/// The recognition model's alphabet, as it was trained (ocrs-models' `DEFAULT_ALPHABET`): its
/// output classes follow this order. The ocrs crate's own copy has an `E` where the training had
/// `€`, which made a scanned "€250" read as "E250".
pub const ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~€ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Which recognizer reads a language.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reader {
    /// ocrs's recognition model ([`RECOGNITION_MODEL`]): the Latin alphabet without accents.
    Ocrs,
    /// PaddleOCR's PP-OCRv6 model ([`LATIN_MODEL`]), restricted to WinAnsiEncoding.
    PpOcr,
}

/// A language text recognition reads.
#[derive(Debug, PartialEq, Eq)]
pub struct Language {
    /// ISO 639-1 code.
    pub code: &'static str,
    /// Its name in English; the interface translates it.
    pub name: &'static str,
    pub reader: Reader,
    /// The letters and signs beyond ASCII it is written with. Every one fits the text layer
    /// (Helvetica in WinAnsiEncoding) and the model's dictionary, which the tests check. A
    /// language with letters outside WinAnsi (Polish, Czech, Romanian, Turkish) needs a text
    /// layer in a Unicode font first.
    pub letters: &'static str,
}

const EN: Language = Language { code: "en", name: "English", reader: Reader::Ocrs, letters: "" };

/// English: what a language whose model is missing is read as.
pub const ENGLISH: &Language = &EN;

/// The languages text recognition reads, English first. Spanish has no "¡" (the model's
/// dictionary lacks it) and German no low quotation mark "„" (likewise).
pub const LANGUAGES: &[Language] = &[
    EN,
    Language {
        code: "fr", name: "French", reader: Reader::PpOcr, letters: "àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ«»’€°"
    },
    Language { code: "de", name: "German", reader: Reader::PpOcr, letters: "äöüßÄÖÜ€°" },
    Language { code: "es", name: "Spanish", reader: Reader::PpOcr, letters: "áéíóúüñÁÉÍÓÚÜÑ¿ºª€°" },
    Language { code: "it", name: "Italian", reader: Reader::PpOcr, letters: "àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ€°" },
    Language { code: "pt", name: "Portuguese", reader: Reader::PpOcr, letters: "áâãàçéêíóôõúüÁÂÃÀÇÉÊÍÓÔÕÚÜºª€°" },
    Language { code: "nl", name: "Dutch", reader: Reader::PpOcr, letters: "áéíóúàèëïöüÁÉÍÓÚÈËÏÖÜ€°" },
];

/// The language `tag` names: an ISO 639-1 code or a BCP 47 tag ("fr", "fr-FR", "FR_ca"), by
/// its primary subtag, ignoring case. `None` for a language text recognition doesn't read.
pub fn language(tag: &str) -> Option<&'static Language> {
    // A language tag is short; anything longer is not one (and isn't scanned).
    if tag.len() > 64 {
        return None;
    }
    let primary = tag.trim().split(['-', '_']).next()?;
    LANGUAGES.iter().find(|l| l.code.eq_ignore_ascii_case(primary))
}

/// The first of `tags` that names a language text recognition reads, else English. Callers
/// pass, in order: the user's last choice, the document's `/Lang`, the interface language.
pub fn first_language<'a>(tags: impl IntoIterator<Item = Option<&'a str>>) -> &'static Language {
    tags.into_iter().flatten().find_map(language).unwrap_or(ENGLISH)
}

/// Whether a language can be read as itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Availability {
    /// Its recognizer is installed.
    Ready,
    /// Its recognizer is missing (or can't be loaded) but English is installed: it would be
    /// read as English, without its accents.
    FallsBackToEnglish,
    /// Nothing that could read it is installed.
    Missing,
}

fn availability_of(language: &Language, ready: bool, english: bool) -> Availability {
    match (ready, english, language.reader) {
        (true, _, _) => Availability::Ready,
        (false, true, Reader::PpOcr) => Availability::FallsBackToEnglish,
        _ => Availability::Missing,
    }
}

/// Whether `language` can be read with the models [`Models::find`] finds.
pub fn availability(language: &Language) -> Availability {
    Models::find().map_or(Availability::Missing, |m| m.availability(language))
}

#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    /// Shown to end users (release packages ship the models; a source build fetches them with
    /// `cargo xtask models`). The UI catalogs key this exact text.
    #[error("the text recognition models are not installed (reinstall PdfCraft, or set PDFCRAFT_MODELS to the folder that holds them)")]
    NoModels,
    #[error("loading {0}: {1}")]
    Load(String, String),
    #[error("text recognition failed: {0}")]
    Recognize(String),
    #[error("the image is empty")]
    EmptyImage,
    #[error("the image's pixels don't match its size")]
    BadImage,
    #[error("the image is too large to read ({0} pixels)")]
    TooLarge(u64),
}

/// The PP-OCR model and its dictionary, which belong together.
#[derive(Clone, Debug, PartialEq)]
pub struct LatinModel {
    pub model: PathBuf,
    pub dict: PathBuf,
}

/// Where the model files are.
#[derive(Clone, Debug, PartialEq)]
pub struct Models {
    pub detection: PathBuf,
    /// ocrs's recognition model, which reads English.
    pub recognition: Option<PathBuf>,
    /// The PP-OCR model, which reads the other languages.
    pub latin: Option<LatinModel>,
}

impl Models {
    /// The models in `dir`: the detection model and at least one recognizer.
    pub fn in_dir(dir: &Path) -> Option<Models> {
        Self::in_dirs(&[dir.to_path_buf()])
    }

    /// The models in `dirs`, each from the first directory that has it: the detection model,
    /// ocrs's recognition model, and the PP-OCR model together with its dictionary (one
    /// directory has both). `None` without the detection model or without any recognizer.
    pub fn in_dirs(dirs: &[PathBuf]) -> Option<Models> {
        let first = |files: &[&str]| dirs.iter().find(|d| files.iter().all(|f| d.join(f).is_file()));
        let detection = first(&[DETECTION_MODEL])?.join(DETECTION_MODEL);
        let recognition = first(&[RECOGNITION_MODEL]).map(|d| d.join(RECOGNITION_MODEL));
        let latin = first(&[LATIN_MODEL, LATIN_DICT]).map(|d| LatinModel { model: d.join(LATIN_MODEL), dict: d.join(LATIN_DICT) });
        (recognition.is_some() || latin.is_some()).then_some(Models { detection, recognition, latin })
    }

    /// Look for the models: `$PDFCRAFT_MODELS`, then where the release packages install them
    /// beside the executable ([`Models::dirs_beside_exe`]), then the source tree's `assets/models/`.
    pub fn find() -> Option<Models> {
        Self::in_dirs(&Self::search_dirs())
    }

    /// Whether `language` can be read with these models (whether their files are there; a file
    /// that can't be loaded shows when it is: [`Ocr::availability`]).
    pub fn availability(&self, language: &Language) -> Availability {
        let ready = match language.reader {
            Reader::Ocrs => self.recognition.is_some(),
            Reader::PpOcr => self.latin.is_some(),
        };
        availability_of(language, ready, self.recognition.is_some())
    }

    /// The directories [`Models::find`] looks in, in order.
    pub fn search_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();
        if let Some(d) = std::env::var_os("PDFCRAFT_MODELS") {
            dirs.push(PathBuf::from(d));
        }
        if let Some(exe) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
            dirs.extend(Self::dirs_beside_exe(&exe));
        }
        dirs.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models"));
        dirs
    }

    /// Where the release packages put the models, relative to the directory holding the
    /// executable (packaging/*):
    /// - `models/` beside it: the Windows MSI and portable zip;
    /// - `../Resources/models`: the macOS app bundle (`Contents/MacOS` → `Contents/Resources`);
    /// - `../share/pdfcraft/models`: the FHS-style trees (`bin/` → `share/`) of the Linux
    ///   deb/rpm/tar.gz/AppImage/Flatpak and the FreeBSD tarball, wherever they are installed.
    pub fn dirs_beside_exe(exe_dir: &Path) -> Vec<PathBuf> {
        vec![exe_dir.join("models"), exe_dir.join("../Resources/models"), exe_dir.join("../share/pdfcraft/models")]
    }
}

/// A recognised word: its text and its box in image pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    /// The axis-aligned box around [`Word::quad`], `[left, top, right, bottom]`.
    pub rect: [f32; 4],
    /// The box as read: top-left, top-right, bottom-right and bottom-left as the word reads.
    /// On a skewed scan it is skewed like the line (the English recognizer reads level boxes).
    pub quad: [[f32; 2]; 4],
}

impl Word {
    /// A word whose box is the level rectangle `[left, top, right, bottom]`.
    pub fn from_rect(text: String, rect: [f32; 4]) -> Word {
        let [l, t, r, b] = rect;
        Word { text, rect, quad: [[l, t], [r, t], [r, b], [l, b]] }
    }

    /// A word whose box is `quad` (top-left, top-right, bottom-right, bottom-left).
    pub fn from_quad(text: String, quad: [[f32; 2]; 4]) -> Word {
        Word { text, rect: pp_ocr::bounds(&[quad]), quad }
    }
}

/// A recognised line of words, in reading order.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Line {
    pub words: Vec<Word>,
}

impl Line {
    pub fn text(&self) -> String {
        self.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
    }
}

/// Why a language was read as English.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FallbackReason {
    /// Its model isn't installed.
    NotInstalled,
    /// Its model is installed but can't be used (the files are damaged, or don't belong
    /// together): why.
    Unusable(String),
}

/// A language read as English instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fallback {
    /// The language asked for.
    pub from: &'static Language,
    pub reason: FallbackReason,
}

/// What [`Ocr::recognize_in`] read.
#[derive(Clone, Debug, PartialEq)]
pub struct Recognition {
    pub lines: Vec<Line>,
    /// The language the text was read as: the one asked for, or English when it fell back.
    pub language: &'static Language,
    /// Set when the language asked for was read as English, and why.
    pub fallback: Option<Fallback>,
}

/// The largest image read (a 600-dpi A3 page is 70 million pixels).
const MAX_IMAGE_PIXELS: usize = 100_000_000;

/// The message of a caught panic.
fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| (*s).to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "unknown error".into())
}

/// Run `f`, turning a panic inside ocrs or rten (which have internal `expect`s) into an error:
/// the page is skipped, the run goes on.
fn guarded<T>(f: impl FnOnce() -> Result<T, OcrError>) -> Result<T, OcrError> {
    std::panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| Err(OcrError::Recognize(panic_message(p.as_ref()))))
}

/// `pixels` (grey, RGB or RGBA, `width` × `height`) as grey, BT.601 like ocrs.
fn grey(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, OcrError> {
    let area = usize::try_from(width).ok().zip(usize::try_from(height).ok()).and_then(|(w, h)| w.checked_mul(h)).unwrap_or(usize::MAX);
    if area == 0 || pixels.is_empty() {
        return Err(OcrError::EmptyImage);
    }
    if area > MAX_IMAGE_PIXELS {
        return Err(OcrError::TooLarge(u64::from(width).saturating_mul(u64::from(height))));
    }
    let luma = |p: &[u8]| match p {
        [r, g, b, ..] => u8::try_from((299 * u32::from(*r) + 587 * u32::from(*g) + 114 * u32::from(*b) + 500) / 1000).unwrap_or(u8::MAX),
        _ => u8::MAX,
    };
    match (pixels.len() / area, pixels.len() % area) {
        (1, 0) => Ok(pixels.to_vec()),
        (3, 0) => Ok(pixels.chunks_exact(3).map(luma).collect()),
        (4, 0) => Ok(pixels.chunks_exact(4).map(luma).collect()),
        _ => Err(OcrError::BadImage),
    }
}

/// `pixels` without their alpha when they are RGBA, `width` × `height` (ocrs wants 1 or 3
/// channels); anything else as it is.
fn without_alpha(pixels: &[u8], width: u32, height: u32) -> std::borrow::Cow<'_, [u8]> {
    // Checked: `width * height * 4` overflowed u32 for a large enough image.
    let rgba = usize::try_from(width).ok().zip(usize::try_from(height).ok()).and_then(|(w, h)| w.checked_mul(h)?.checked_mul(4));
    if Some(pixels.len()) == rgba {
        pixels.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect::<Vec<u8>>().into()
    } else {
        pixels.into()
    }
}

/// A loaded recogniser. Loading takes a moment; keep one and reuse it. The PP-OCR model is
/// loaded the first time a language needs it.
pub struct Ocr {
    /// Finds words and lines; reads English when ocrs's recognition model is installed.
    engine: ocrs::OcrEngine,
    english: bool,
    latin_files: Option<LatinModel>,
    latin: OnceLock<Result<pp_ocr::Recognizer, String>>,
}

impl Ocr {
    pub fn load(models: &Models) -> Result<Ocr, OcrError> {
        let load = |p: &Path| rten::Model::load_file(p).map_err(|e| OcrError::Load(p.display().to_string(), e.to_string()));
        let detection_model = Some(load(&models.detection)?);
        let recognition_model = models.recognition.as_deref().map(load).transpose()?;
        let english = recognition_model.is_some();
        let params = ocrs::OcrEngineParams { detection_model, recognition_model, alphabet: Some(ALPHABET.into()), ..Default::default() };
        let engine = ocrs::OcrEngine::new(params).map_err(|e| OcrError::Load("ocr engine".into(), e.to_string()))?;
        Ok(Ocr { engine, english, latin_files: models.latin.clone(), latin: OnceLock::new() })
    }

    /// Load the models found by [`Models::find`].
    pub fn find() -> Result<Ocr, OcrError> {
        Self::load(&Models::find().ok_or(OcrError::NoModels)?)
    }

    /// Whether `language` can be read as itself, as far as is known: a PP-OCR model that
    /// failed to load counts as missing.
    pub fn availability(&self, language: &Language) -> Availability {
        let ready = match language.reader {
            Reader::Ocrs => self.english,
            Reader::PpOcr => self.latin_files.is_some() && self.latin.get().is_none_or(Result::is_ok),
        };
        availability_of(language, ready, self.english)
    }

    /// The PP-OCR recognizer, loaded and checked the first time it is needed.
    fn latin(&self) -> Result<&pp_ocr::Recognizer, FallbackReason> {
        let files = self.latin_files.as_ref().ok_or(FallbackReason::NotInstalled)?;
        let loaded = self.latin.get_or_init(|| {
            std::panic::catch_unwind(AssertUnwindSafe(|| pp_ocr::Recognizer::load(&files.model, &files.dict)))
                .unwrap_or_else(|p| Err(format!("{}: {}", files.model.display(), panic_message(p.as_ref()))))
        });
        loaded.as_ref().map_err(|e| FallbackReason::Unusable(e.clone()))
    }

    /// Recognise the text in an RGBA (or RGB, or grey) image, `width` × `height` pixels, as
    /// `language`. A language whose model is missing or unusable is read as English when
    /// English is installed, and [`Recognition::fallback`] says so.
    pub fn recognize_in(&self, pixels: &[u8], width: u32, height: u32, language: &'static Language) -> Result<Recognition, OcrError> {
        if language.reader == Reader::Ocrs {
            return Ok(Recognition { lines: self.recognize(pixels, width, height)?, language, fallback: None });
        }
        match self.latin() {
            Ok(rec) => Ok(Recognition { lines: guarded(|| self.read_latin(rec, pixels, width, height))?, language, fallback: None }),
            Err(reason) if self.english => Ok(Recognition {
                lines: self.recognize(pixels, width, height)?,
                language: ENGLISH,
                fallback: Some(Fallback { from: language, reason }),
            }),
            Err(FallbackReason::NotInstalled) => Err(OcrError::NoModels),
            Err(FallbackReason::Unusable(why)) => Err(OcrError::Load(format!("the {} recognition model", language.name), why)),
        }
    }

    /// Recognise English text in an RGBA (or RGB, or grey) image, `width` × `height` pixels,
    /// with ocrs's recognition model.
    pub fn recognize(&self, pixels: &[u8], width: u32, height: u32) -> Result<Vec<Line>, OcrError> {
        if !self.english {
            return Err(OcrError::NoModels);
        }
        guarded(|| self.read_english(pixels, width, height))
    }

    fn read_english(&self, pixels: &[u8], width: u32, height: u32) -> Result<Vec<Line>, OcrError> {
        if width == 0 || height == 0 || pixels.is_empty() {
            return Err(OcrError::EmptyImage);
        }
        // ocrs multiplies the size out unchecked: check it here.
        let area = usize::try_from(width).ok().zip(usize::try_from(height).ok()).and_then(|(w, h)| w.checked_mul(h));
        if !area.is_some_and(|a| [1, 3, 4].iter().any(|c| a.checked_mul(*c) == Some(pixels.len()))) {
            return Err(OcrError::BadImage);
        }
        let rgb = without_alpha(pixels, width, height);
        let err = |e: &dyn std::fmt::Display| OcrError::Recognize(e.to_string());
        let source = ocrs::ImageSource::from_bytes(&rgb, (width, height)).map_err(|e| err(&e))?;
        let input = self.engine.prepare_input(source).map_err(|e| err(&e))?;
        let found = self.engine.detect_words(&input).map_err(|e| err(&e))?;
        let lines = self.engine.find_text_lines(&input, &found);
        let read = self.engine.recognize_text(&input, &lines).map_err(|e| err(&e))?;
        use ocrs::TextItem;
        Ok(read
            .into_iter()
            .flatten()
            .map(|line| Line {
                words: line
                    .words()
                    .map(|w| {
                        let r = w.bounding_rect();
                        Word::from_rect(w.to_string(), [r.left() as f32, r.top() as f32, r.right() as f32, r.bottom() as f32])
                    })
                    .filter(|w| !w.text.trim().is_empty())
                    .collect(),
            })
            .filter(|l| !l.words.is_empty())
            .collect())
    }

    fn read_latin(&self, rec: &pp_ocr::Recognizer, pixels: &[u8], width: u32, height: u32) -> Result<Vec<Line>, OcrError> {
        let grey = grey(pixels, width, height)?;
        let (w, h) = (usize::try_from(width).unwrap_or(0), usize::try_from(height).unwrap_or(0));
        let image = pp_ocr::frame::Gray::new(&grey, w, h).ok_or(OcrError::BadImage)?;
        let err = |e: &dyn std::fmt::Display| OcrError::Recognize(e.to_string());
        let source = ocrs::ImageSource::from_bytes(&grey, (width, height)).map_err(|e| err(&e))?;
        let input = self.engine.prepare_input(source).map_err(|e| err(&e))?;
        let found = self.engine.detect_words(&input).map_err(|e| err(&e))?;
        let lines = self.engine.find_text_lines(&input, &found);
        let read = rec.recognize(&image, &lines).map_err(OcrError::Recognize)?;
        Ok(read.into_iter().map(|l| Line { words: l.words.into_iter().map(|w| Word::from_quad(w.text, w.quad)).collect() }).collect())
    }
}

/// A word placed on a page, in PDF user space: the bottom-left corner of its box, and the
/// vectors along its bottom edge (`across`, left → right as read) and its left edge (`up`).
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedWord {
    pub text: String,
    pub origin: [f64; 2],
    pub across: [f64; 2],
    pub up: [f64; 2],
}

impl PlacedWord {
    /// Place a pixel-space word with `to_user`, which maps an image pixel (x right, y down) to
    /// user space.
    /// The word's box is its quad, so a skewed word is placed skewed.
    pub fn place(word: &Word, to_user: impl Fn(f32, f32) -> [f64; 2]) -> PlacedWord {
        let [tl, _, br, bl] = word.quad;
        let o = to_user(bl[0], bl[1]);
        let br = to_user(br[0], br[1]);
        let tl = to_user(tl[0], tl[1]);
        PlacedWord { text: word.text.clone(), origin: o, across: [br[0] - o[0], br[1] - o[1]], up: [tl[0] - o[0], tl[1] - o[1]] }
    }
}

/// Helvetica's ascender and descender (em fractions): a word box spans about this much.
const BOX_EM: f64 = 0.93;
const DESCENT: f64 = 0.21;
/// Helvetica's cap height and x-height (em fractions).
const CAP_HEIGHT: f64 = 0.718;
const X_HEIGHT: f64 = 0.523;

/// How far `text`'s ink reaches above and below the baseline, in em fractions: a word box is
/// tight around the ink, so "DT" spans the cap height only, "sun" the x-height, and "gap" reaches
/// down to the descender. Text without letters or digits keeps the full ascender-descender span.
fn ink_extent(text: &str) -> (f64, f64) {
    if !text.chars().any(char::is_alphanumeric) {
        return (BOX_EM - DESCENT, -DESCENT);
    }
    let tall = text
        .chars()
        .any(|c| c.is_uppercase() || c.is_ascii_digit() || "bdfhijklt!?/\\()[]{}|'\"#$%&@".contains(c) || (c.is_alphabetic() && !c.is_ascii()));
    // The cedilla of ç and Ç reaches below the baseline (#4659).
    let low = text.chars().any(|c| "gjpqyçÇ()[]{}|_".contains(c));
    (if tall { CAP_HEIGHT } else { X_HEIGHT }, if low { -DESCENT } else { 0.0 })
}

fn num(v: f64) -> String {
    let s = format!("{:.4}", if v.abs() < 1e-9 { 0.0 } else { v });
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.into() }
}

/// Page content that writes `words` as invisible text in `/PCHelv` (standard Helvetica,
/// WinAnsiEncoding), each word stretched to its box so selection and search highlight the
/// right place. Marked content `/OCR` so it can be told apart from the page's own text.
pub fn text_layer(words: &[PlacedWord]) -> Vec<u8> {
    // `3 Tr` and the font are graphics state, which `ET` doesn't reset: without `q … Q` they
    // would stay set for every later content stream on the page, so text added after OCR (a
    // header or footer) drew invisibly (#3029).
    let mut out = b"/OCR BMC\nq\nBT\n3 Tr\n/PCHelv 1 Tf\n".to_vec();
    for w in words {
        let width = helvetica_width(&w.text, 1.0);
        let height = w.up[0].hypot(w.up[1]);
        if width <= 0.0 || height <= 0.0 {
            continue;
        }
        // The em square: the word's own ink (cap height or x-height, descenders or not) spans the
        // box, so the text is as tall as the scanned word; its width is stretched to fit the word.
        let (top, bottom) = ink_extent(&w.text);
        let em = [w.up[0] / (top - bottom), w.up[1] / (top - bottom)];
        let ax = [w.across[0] / width, w.across[1] / width];
        let o = [w.origin[0] - em[0] * bottom, w.origin[1] - em[1] * bottom];
        let m = [ax[0], ax[1], em[0], em[1], o[0], o[1]];
        let nums: Vec<String> = m.iter().map(|v| num(*v)).collect();
        out.extend_from_slice(nums.join(" ").as_bytes());
        out.extend_from_slice(b" Tm ");
        out.extend(pdfcraft_fonts::literal(&pdfcraft_fonts::win_ansi(&w.text)));
        out.extend_from_slice(b" Tj\n");
    }
    out.extend_from_slice(b"ET\nQ\nEMC\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn places_words_through_the_mapping() {
        // 2 pixels per point, page 612 × 792, y flipped.
        let to_user = |x: f32, y: f32| [x as f64 / 2.0, 792.0 - y as f64 / 2.0];
        let w = PlacedWord::place(&Word::from_rect("Hello".into(), [100.0, 200.0, 300.0, 240.0]), to_user);
        assert_eq!(w.origin, [50.0, 672.0]);
        assert_eq!(w.across, [100.0, 0.0]);
        assert_eq!(w.up, [0.0, 20.0]);
    }

    /// The text layer is Helvetica in WinAnsiEncoding: every character the model can recognise
    /// must be one it can hold, or recognised text would be saved as `?`. A model with a wider
    /// alphabet (Chinese, for example) needs a text layer in another font first.
    #[test]
    fn the_model_alphabet_fits_the_text_layer_font() {
        assert_eq!(pdfcraft_fonts::first_undrawable(ALPHABET), None);
    }

    #[test]
    fn text_layer_is_invisible_text_fitted_to_each_box() {
        let w = PlacedWord { text: "Hi".into(), origin: [10.0, 20.0], across: [30.0, 0.0], up: [0.0, 9.3] };
        let s = String::from_utf8(text_layer(&[w])).unwrap();
        assert!(s.contains("3 Tr") && s.contains("/PCHelv 1 Tf"), "{s}");
        let width = helvetica_width("Hi", 1.0);
        // "Hi" has no descender: its box spans the cap height, from the baseline up.
        assert!(s.contains(&format!("{} 0 0 {} 10 20 Tm (Hi) Tj", num(30.0 / width), num(9.3 / CAP_HEIGHT))), "{s}");
        assert!(s.starts_with("/OCR BMC") && s.ends_with("EMC\n"));
    }

    /// #3029: the layer's invisible rendering mode and font end with the layer. Every operator
    /// that sets them sits inside a `q … Q` pair that the layer closes itself, so content drawn
    /// after it on the page starts from the graphics state it had before.
    #[test]
    fn the_layer_restores_the_graphics_state_it_changes() {
        let w = PlacedWord { text: "Hi".into(), origin: [10.0, 20.0], across: [30.0, 0.0], up: [0.0, 9.3] };
        let s = String::from_utf8(text_layer(&[w])).unwrap();
        let mut depth = 0_i32;
        for token in s.split_ascii_whitespace() {
            match token {
                "q" => depth += 1,
                "Q" => depth -= 1,
                "Tr" | "Tf" | "Tm" => assert!(depth > 0, "`{token}` outside q … Q in {s:?}"),
                _ => {}
            }
            assert!(depth >= 0, "unbalanced Q in {s:?}");
        }
        assert_eq!(depth, 0, "the layer leaves a q open: {s:?}");
    }

    /// A word box is tight around the ink, so the text's size and baseline come from the letters:
    /// capitals, lowercase without ascenders, descenders.
    #[test]
    fn the_text_is_as_tall_as_the_scanned_word() {
        assert_eq!(ink_extent("DT"), (CAP_HEIGHT, 0.0));
        assert_eq!(ink_extent("402337"), (CAP_HEIGHT, 0.0));
        assert_eq!(ink_extent("sun"), (X_HEIGHT, 0.0));
        assert_eq!(ink_extent("gap"), (X_HEIGHT, -DESCENT));
        assert_eq!(ink_extent("français"), (CAP_HEIGHT, -DESCENT));
        assert_eq!(ink_extent("ÇA"), (CAP_HEIGHT, -DESCENT));
        assert_eq!(ink_extent("Gallus.Ze@camtel.cm"), (CAP_HEIGHT, 0.0));
        assert_eq!(ink_extent("Corporation"), (CAP_HEIGHT, -DESCENT));
        assert_eq!(ink_extent("--"), (BOX_EM - DESCENT, -DESCENT));
        let w = PlacedWord { text: "gap".into(), origin: [0.0, 100.0], across: [20.0, 0.0], up: [0.0, 7.3] };
        let s = String::from_utf8(text_layer(&[w])).unwrap();
        let em = 7.3 / (X_HEIGHT + DESCENT);
        assert!(s.contains(&format!(" 0 0 {} 0 {} Tm (gap) Tj", num(em), num(100.0 + em * DESCENT))), "{s}");
    }

    #[test]
    fn rotated_pages_keep_the_reading_direction() {
        // A page turned 90°: image x runs up the page.
        let to_user = |x: f32, y: f32| [y as f64, x as f64];
        let w = PlacedWord::place(&Word::from_rect("Up".into(), [0.0, 0.0, 50.0, 10.0]), to_user);
        assert_eq!(w.across, [0.0, 50.0]);
        assert_eq!(w.up, [-10.0, 0.0]);
    }

    /// #103: every release package's layout puts the models where the installed app looks.
    /// Mirrors packaging/{windows/package.ps1,macos/package.sh,linux/package.sh,freebsd/package.sh}.
    #[test]
    fn finds_the_models_in_every_release_package_layout() {
        let base = std::env::temp_dir().join(format!("pdfcraft-ocr-layouts-{}", std::process::id()));
        let layouts = [
            ("windows", "PdfCraft", "PdfCraft/models"),
            ("macos", "PdfCraft.app/Contents/MacOS", "PdfCraft.app/Contents/Resources/models"),
            ("linux-deb-rpm", "usr/bin", "usr/share/pdfcraft/models"),
            ("linux-tar", "pdfcraft-1.0.0-linux-x86_64/bin", "pdfcraft-1.0.0-linux-x86_64/share/pdfcraft/models"),
            ("appimage", "PdfCraft.AppDir/usr/bin", "PdfCraft.AppDir/usr/share/pdfcraft/models"),
            ("flatpak", "app/bin", "app/share/pdfcraft/models"),
            ("freebsd", "usr/local/bin", "usr/local/share/pdfcraft/models"),
        ];
        for (name, exe_dir, models_dir) in layouts {
            let root = base.join(name);
            let (exe_dir, models_dir) = (root.join(exe_dir), root.join(models_dir));
            std::fs::create_dir_all(&exe_dir).unwrap();
            std::fs::create_dir_all(&models_dir).unwrap();
            let found = || Models::in_dirs(&Models::dirs_beside_exe(&exe_dir));
            assert_eq!(found(), None, "{name}: no models yet");
            for file in [DETECTION_MODEL, RECOGNITION_MODEL, LATIN_MODEL, LATIN_DICT] {
                std::fs::write(models_dir.join(file), b"model").unwrap();
            }
            let models = found().unwrap_or_else(|| panic!("{name}: models in {} not found", models_dir.display()));
            let at = |p: &Path, file: &str| assert_eq!(p.canonicalize().unwrap(), models_dir.join(file).canonicalize().unwrap(), "{name}");
            at(&models.detection, DETECTION_MODEL);
            at(models.recognition.as_deref().unwrap(), RECOGNITION_MODEL);
            let latin = models.latin.as_ref().unwrap_or_else(|| panic!("{name}: no accented-Latin model"));
            at(&latin.model, LATIN_MODEL);
            at(&latin.dict, LATIN_DICT);
            assert!(LANGUAGES.iter().all(|l| models.availability(l) == Availability::Ready), "{name}");
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Which languages can be read follows the files installed, wherever each one is found:
    /// a language without its model falls back to English when English is there.
    #[test]
    fn availability_follows_the_installed_files() {
        let base = std::env::temp_dir().join(format!("pdfcraft-ocr-availability-{}", std::process::id()));
        let dir = |name: &str, files: &[&str]| {
            let d = base.join(name);
            std::fs::create_dir_all(&d).unwrap();
            for f in files {
                std::fs::write(d.join(f), b"model").unwrap();
            }
            d
        };
        let (en, fr) = (ENGLISH, language("fr").unwrap());
        let status = |dirs: &[PathBuf]| Models::in_dirs(dirs).map(|m| (m.availability(en), m.availability(fr)));
        use Availability::*;
        assert_eq!(status(&[dir("none", &[])]), None);
        assert_eq!(status(&[dir("detection", &[DETECTION_MODEL])]), None);
        assert_eq!(status(&[dir("no-detection", &[RECOGNITION_MODEL, LATIN_MODEL, LATIN_DICT])]), None);
        assert_eq!(status(&[dir("english", &[DETECTION_MODEL, RECOGNITION_MODEL])]), Some((Ready, FallsBackToEnglish)));
        assert_eq!(status(&[dir("latin", &[DETECTION_MODEL, LATIN_MODEL, LATIN_DICT])]), Some((Missing, Ready)));
        // The model without its dictionary is not the model.
        assert_eq!(status(&[dir("no-dict", &[DETECTION_MODEL, RECOGNITION_MODEL, LATIN_MODEL])]), Some((Ready, FallsBackToEnglish)));
        assert_eq!(status(&[dir("dict-only", &[DETECTION_MODEL, LATIN_DICT])]), None);
        // A folder with only the accented-Latin model (`PDFCRAFT_MODELS`) beside the packaged ones.
        let extra = dir("extra", &[LATIN_MODEL, LATIN_DICT]);
        let packaged = dir("packaged", &[DETECTION_MODEL, RECOGNITION_MODEL]);
        let m = Models::in_dirs(&[extra.clone(), packaged.clone()]).unwrap();
        assert_eq!((m.availability(en), m.availability(fr)), (Ready, Ready));
        assert_eq!((m.detection.parent(), m.latin.unwrap().model.parent()), (Some(packaged.as_path()), Some(extra.as_path())));
        // Earlier folders win, file by file.
        let m = Models::in_dirs(&[dir("first", &[DETECTION_MODEL]), packaged.clone()]).unwrap();
        assert_eq!(m.detection.parent(), Some(base.join("first").as_path()));
        assert_eq!(m.recognition.as_deref().and_then(Path::parent), Some(packaged.as_path()));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn languages_are_unique_and_fit_the_text_layer() {
        assert_eq!(LANGUAGES.first(), Some(ENGLISH));
        let mut codes: Vec<&str> = LANGUAGES.iter().map(|l| l.code).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), LANGUAGES.len());
        assert_eq!(codes, ["de", "en", "es", "fr", "it", "nl", "pt"]);
        for l in LANGUAGES {
            assert_eq!(l.reader == Reader::Ocrs, l.code == "en", "{}", l.code);
            assert_eq!(l.code.len(), 2);
            assert!(!l.name.is_empty());
            // The text layer can hold every letter the language is written with.
            assert_eq!(pdfcraft_fonts::first_undrawable(l.letters), None, "{}", l.code);
            assert!(l.reader == Reader::Ocrs || !l.letters.is_empty(), "{}", l.code);
        }
    }

    #[test]
    fn language_tags_map_by_their_primary_subtag() {
        let code = |t: &str| language(t).map(|l| l.code);
        for (tag, want) in [
            ("fr", "fr"),
            ("fr-FR", "fr"),
            ("FR_ca", "fr"),
            (" de ", "de"),
            ("pt-BR", "pt"),
            ("EN", "en"),
            ("nl-BE", "nl"),
            ("es-419", "es"),
            ("it-CH", "it"),
        ] {
            assert_eq!(code(tag), Some(want), "{tag}");
        }
        for tag in ["", "-", "xx", "f", "fra", "fr fr", "pl", "zh-Hans"] {
            assert_eq!(code(tag), None, "{tag}");
        }
        assert_eq!(code(&format!("fr-{}", "x".repeat(100))), None);
        assert_eq!(first_language([None, Some("xx"), Some("de-AT"), Some("fr")]).code, "de");
        assert_eq!(first_language([None, Some("pl")]), ENGLISH);
        assert_eq!(first_language([]), ENGLISH);
    }

    proptest::proptest! {
        #[test]
        fn language_lookup_is_total(tag in ".{0,80}") {
            if let Some(l) = language(&tag) {
                proptest::prop_assert!(tag.trim().to_ascii_lowercase().starts_with(l.code));
            }
        }
    }

    /// The models are fetched by `cargo xtask models` from ATTRIBUTION.toml: every file the
    /// code looks for must be an entry there, as a model.
    #[test]
    fn every_model_file_is_attributed() {
        let manifest = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ATTRIBUTION.toml")).unwrap();
        for file in [DETECTION_MODEL, RECOGNITION_MODEL, LATIN_MODEL, LATIN_DICT] {
            let entry = manifest
                .split("[[")
                .find(|e| e.contains(&format!("file = \"{file}\"")))
                .unwrap_or_else(|| panic!("{file} is not in ATTRIBUTION.toml"));
            assert!(entry.contains("kind = \"model\""), "{file}: {entry}");
        }
    }

    #[test]
    fn images_are_checked_before_they_are_read() {
        assert_eq!(grey(&[10, 20, 30], 3, 1).unwrap(), [10, 20, 30]);
        assert_eq!(grey(&[255, 255, 255, 0, 0, 0, 0, 255], 2, 1).unwrap(), [255, 0]);
        assert_eq!(grey(&[255, 0, 0], 1, 1).unwrap(), [76]);
        assert!(matches!(grey(&[0; 5], 2, 1), Err(OcrError::BadImage)));
        assert!(matches!(grey(&[0; 8], 2, 2), Err(OcrError::BadImage)));
        assert!(matches!(grey(&[], 2, 1), Err(OcrError::EmptyImage)));
        assert!(matches!(grey(&[0; 4], 0, 4), Err(OcrError::EmptyImage)));
        assert!(matches!(grey(&[0; 4], u32::MAX, u32::MAX), Err(OcrError::TooLarge(_))));
        // RGBA loses its alpha for ocrs; a size whose RGBA length overflows passes through as it
        // is (it used to overflow `width * height * 4` and panic).
        assert_eq!(&*without_alpha(&[1, 2, 3, 4, 5, 6, 7, 8], 2, 1), [1, 2, 3, 5, 6, 7]);
        assert_eq!(&*without_alpha(&[1, 2, 3], 1, 1), [1, 2, 3]);
        assert_eq!(&*without_alpha(&[1, 2, 3, 4], u32::MAX, u32::MAX), [1, 2, 3, 4]);
    }

    /// The message tells end users what to do; release builds have no cargo.
    #[test]
    fn the_alphabet_is_the_trained_one() {
        let chars: Vec<char> = ALPHABET.chars().collect();
        assert_eq!(chars.len(), 96);
        // The euro sign follows the ASCII punctuation (class 44; 0 is the CTC blank), not a second E.
        assert_eq!(chars.iter().position(|&c| c == '€'), Some(43));
        assert_eq!(chars.iter().filter(|&&c| c == 'E').count(), 1);
    }

    #[test]
    fn missing_models_message_is_for_end_users() {
        let message = OcrError::NoModels.to_string();
        assert!(!message.contains("cargo") && message.contains("PDFCRAFT_MODELS"), "{message}");
    }

    /// End to end on rendered text, when the models are installed (`cargo xtask models`).
    #[test]
    fn reads_rendered_text() {
        let Some(models) = Models::find() else {
            eprintln!("skipped: OCR models not installed");
            return;
        };
        let ocr = Ocr::load(&models).unwrap();
        let (w, h, px) = test_image::hello();
        let lines = ocr.recognize(&px, w, h).unwrap();
        let text: Vec<String> = lines.iter().map(Line::text).collect();
        assert!(text.iter().any(|t| t.contains("HELL") && t.contains("WOR")), "{text:?}");
        let word = lines.iter().flat_map(|l| &l.words).find(|w| w.text.contains("HELL")).unwrap();
        assert!(word.rect[0] >= 10.0 && word.rect[0] < 60.0, "{word:?}");
    }

    /// The models `Models::find` finds (`PDFCRAFT_MODELS`, or `cargo xtask models`); the tests
    /// that need them skip without.
    fn installed(what: &str, has: impl Fn(&Models) -> bool) -> Option<Models> {
        let m = Models::find().filter(|m| has(m));
        if m.is_none() {
            eprintln!("skipped: {what} not installed");
        }
        m
    }

    /// Every letter the languages are written with is one the model can emit once restricted
    /// to the text layer's encoding: the restriction takes nothing a language needs.
    #[test]
    fn every_language_letter_survives_the_restriction() {
        let Some(models) = installed("the accented-Latin model", |m| m.latin.is_some()) else { return };
        let Some(latin) = models.latin else { return };
        let cs = pp_ocr::decoding_charset(&latin.dict).unwrap();
        let allowed: std::collections::HashSet<char> = cs.allowed_chars().collect();
        for l in LANGUAGES.iter().filter(|l| l.reader == Reader::PpOcr) {
            let ascii = (' '..='~').filter(|c| *c != ' ');
            let missing: String = l.letters.chars().chain(ascii).filter(|c| !allowed.contains(c)).collect();
            assert_eq!(missing, "", "{}", l.code);
        }
        // Nothing outside WinAnsi is left (no "?" in the text layer).
        assert_eq!(pdfcraft_fonts::first_undrawable(&allowed.iter().collect::<String>()), None);
    }

    /// Lines in French, German, Spanish, Portuguese and Italian, drawn in 20-pt Helvetica.
    const ACCENTED: [&str; 5] = [
        "Élève, garçon, cœur, Noël, déjà, où, être.",
        "« Bien sûr », dit-il : n° 42, 37,5 °C.",
        "Ça coûte 49,90 € ; le reçu à l'hôtel.",
        "Größe, Übung, Straße ; año, señor.",
        "Ação, informação ; perché, città.",
    ];

    /// End to end on rendered accented text, level and skewed 2°, when the models are installed.
    #[test]
    fn reads_accented_latin() {
        let Some(models) = installed("the accented-Latin model", |m| m.latin.is_some()) else { return };
        let ocr = Ocr::load(&models).unwrap();
        let fr = language("fr").unwrap();
        assert_eq!(ocr.availability(fr), Availability::Ready);
        for deg in [0.0, 2.0] {
            let (w, h, px) = test_image::rendered(&ACCENTED, deg);
            let got = ocr.recognize_in(&px, w, h, fr).unwrap();
            assert_eq!((got.language, &got.fallback), (fr, &None));
            let text = got.lines.iter().map(Line::text).collect::<Vec<_>>().join("\n");
            eprintln!("read at {deg}°:\n{text}");
            assert!(!text.contains('?'), "{deg}°: {text}");
            let words: std::collections::HashSet<String> =
                text.split_whitespace().map(|w| w.trim_matches(|c: char| ",.;:".contains(c)).replace('’', "'")).collect();
            for want in [
                "Élève",
                "garçon",
                "cœur",
                "Noël",
                "déjà",
                "où",
                "être",
                "sûr",
                "n°",
                "°C",
                "coûte",
                "€",
                "reçu",
                "à",
                "l'hôtel",
                "Größe",
                "Übung",
                "Straße",
                "año",
                "señor",
                "Ação",
                "informação",
                "perché",
                "città",
            ] {
                assert!(words.contains(want), "{deg}°: {want:?} not read in {text}");
            }
            assert!(text.contains('«') && text.contains('»') && text.to_lowercase().contains("ça"), "{deg}°: {text}");
            // Line for line, in order: a word ocrs gave a line of its own is back in its line.
            let squeeze = |t: &str| t.chars().filter(|c| !c.is_whitespace()).map(|c| if c == '’' { '\'' } else { c }).collect::<String>();
            let read: Vec<String> = got.lines.iter().map(|l| squeeze(&l.text())).collect();
            assert_eq!(read, ACCENTED.map(squeeze), "{deg}°: {text}");
            // The boxes follow the text's skew, and sit where it was drawn: the first word at
            // 40 pt from the left edge, turned about its start.
            for word in got.lines.iter().flat_map(|l| &l.words) {
                let [tl, tr, _, _] = word.quad;
                let angle = (tr[1] - tl[1]).atan2(tr[0] - tl[0]).to_degrees();
                assert!((angle - deg as f32).abs() < 0.5, "{deg}°: {word:?} is at {angle}°");
            }
            let first = got.lines.first().and_then(|l| l.words.first()).unwrap();
            let left = 40.0 * test_image::DPI / 72.0;
            assert!((first.quad[3][0] - left).abs() < 15.0, "{deg}°: {first:?} should start at x = {left}");
        }
    }

    /// A language whose model is missing, or damaged, is read as English, and the result says
    /// so; with no recognizer at all it is an error, never an empty page.
    #[test]
    fn a_language_without_its_model_is_read_as_english_and_says_so() {
        let Some(mut models) = installed("the English models", |m| m.recognition.is_some()) else { return };
        let (w, h, px) = test_image::hello();
        let fr = language("fr").unwrap();
        models.latin = None;
        let ocr = Ocr::load(&models).unwrap();
        assert_eq!(ocr.availability(fr), Availability::FallsBackToEnglish);
        let got = ocr.recognize_in(&px, w, h, fr).unwrap();
        assert_eq!((got.language, got.fallback), (ENGLISH, Some(Fallback { from: fr, reason: FallbackReason::NotInstalled })));
        assert!(got.lines.iter().any(|l| l.text().contains("HELL")), "{:?}", got.lines);
        // English as asked: read as before.
        let english = ocr.recognize_in(&px, w, h, ENGLISH).unwrap();
        assert_eq!((english.language, english.fallback, english.lines), (ENGLISH, None, ocr.recognize(&px, w, h).unwrap()));
        // Installed but damaged: the same, with why.
        let dir = std::env::temp_dir().join(format!("pdfcraft-ocr-damaged-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, dict) = (dir.join(LATIN_MODEL), dir.join(LATIN_DICT));
        std::fs::write(&model, b"not a model").unwrap();
        std::fs::write(&dict, "a\nb\n").unwrap();
        models.latin = Some(LatinModel { model, dict });
        let ocr = Ocr::load(&models).unwrap();
        assert_eq!(ocr.availability(fr), Availability::Ready, "not known until it is loaded");
        let got = ocr.recognize_in(&px, w, h, fr).unwrap();
        let reason = got.fallback.map(|f| f.reason);
        assert!(matches!(&reason, Some(FallbackReason::Unusable(why)) if why.contains(LATIN_MODEL)), "{reason:?}");
        assert_eq!(ocr.availability(fr), Availability::FallsBackToEnglish);
        // Neither recognizer: an error.
        models.recognition = None;
        models.latin = None;
        let ocr = Ocr::load(&models).unwrap();
        assert!(matches!(ocr.recognize_in(&px, w, h, fr), Err(OcrError::NoModels)));
        assert!(matches!(ocr.recognize(&px, w, h), Err(OcrError::NoModels)));
        assert_eq!((ocr.availability(ENGLISH), ocr.availability(fr)), (Availability::Missing, Availability::Missing));
        // A huge image, or pixels that don't match the size, are refused before ocrs sees them.
        let ocr = Ocr::load(&Models::find().unwrap()).unwrap();
        assert!(matches!(ocr.recognize(&[0; 16], u32::MAX, u32::MAX), Err(OcrError::BadImage)));
        assert!(matches!(ocr.recognize(&[0; 16], 2, 4), Err(OcrError::BadImage)));
        assert!(matches!(ocr.recognize_in(&[0; 16], u32::MAX, u32::MAX, fr), Err(OcrError::TooLarge(_)) | Ok(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    mod test_image {
        /// The resolution pages are rendered at, as Recognize Text reads scans.
        pub const DPI: f32 = 300.0;

        /// A one-page PDF, 612 × 300 pt, drawing `lines` in 20-pt Helvetica (WinAnsiEncoding),
        /// 40 pt apart from (40, 250), turned `deg` degrees clockwise about that point; rendered
        /// at [`DPI`].
        pub fn rendered(lines: &[&str], deg: f64) -> (u32, u32, Vec<u8>) {
            let (s, c) = deg.to_radians().sin_cos();
            let mut content = format!("q {c:.6} {:.6} {s:.6} {c:.6} 40 250 cm BT /F1 20 Tf", -s).into_bytes();
            for (i, line) in lines.iter().enumerate() {
                content.extend_from_slice(if i == 0 { b" 0 0 Td " } else { b" 0 -40 Td " });
                content.extend(pdfcraft_fonts::literal(&pdfcraft_fonts::win_ansi(line)));
                content.extend_from_slice(b" Tj");
            }
            content.extend_from_slice(b" ET Q");
            let objects = [
                b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
                b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 300] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".to_vec(),
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".to_vec(),
                [format!("<< /Length {} >>\nstream\n", content.len()).into_bytes(), content, b"\nendstream".to_vec()].concat(),
            ];
            let mut pdf = b"%PDF-1.4\n".to_vec();
            let mut offsets = Vec::new();
            for (i, o) in objects.iter().enumerate() {
                offsets.push(pdf.len());
                pdf.extend(format!("{} 0 obj\n", i + 1).bytes());
                pdf.extend_from_slice(o);
                pdf.extend_from_slice(b"\nendobj\n");
            }
            let xref = pdf.len();
            pdf.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
            for o in offsets {
                pdf.extend(format!("{o:010} 00000 n \n").bytes());
            }
            pdf.extend(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objects.len() + 1).bytes());
            let mut r = pdfcraft_render::PageRenderer::new(std::sync::Arc::new(pdf), pdfcraft_render::RenderConfig::default());
            let shot = r.render(pdfcraft_render::RenderRequest { page: 0, scale: DPI / 72.0, ..Default::default() });
            assert!(shot.error.is_none(), "{:?}", shot.error);
            (shot.width, shot.height, shot.rgba.to_vec())
        }

        /// "HELLO WORLD" drawn with blocky 5×7 letters, 4 pixels per dot, black on white.
        pub fn hello() -> (u32, u32, Vec<u8>) {
            const GLYPHS: &[(char, [&str; 7])] = &[
                ('H', ["X...X", "X...X", "X...X", "XXXXX", "X...X", "X...X", "X...X"]),
                ('E', ["XXXXX", "X....", "X....", "XXXX.", "X....", "X....", "XXXXX"]),
                ('L', ["X....", "X....", "X....", "X....", "X....", "X....", "XXXXX"]),
                ('O', [".XXX.", "X...X", "X...X", "X...X", "X...X", "X...X", ".XXX."]),
                ('W', ["X...X", "X...X", "X...X", "X.X.X", "X.X.X", "XX.XX", "X...X"]),
                ('R', ["XXXX.", "X...X", "X...X", "XXXX.", "X.X..", "X..X.", "X...X"]),
                ('D', ["XXXX.", "X...X", "X...X", "X...X", "X...X", "X...X", "XXXX."]),
            ];
            let text = "HELLO WORLD";
            let (dot, x0, y0) = (4u32, 40u32, 40u32);
            let (w, h) = (x0 * 2 + text.len() as u32 * 6 * dot, y0 * 2 + 7 * dot);
            let mut px = vec![255u8; (w * h * 4) as usize];
            for (i, c) in text.chars().enumerate() {
                let Some((_, rows)) = GLYPHS.iter().find(|g| g.0 == c) else { continue };
                for (ry, row) in rows.iter().enumerate() {
                    for (rx, b) in row.bytes().enumerate() {
                        if b != b'X' {
                            continue;
                        }
                        for dy in 0..dot {
                            for dx in 0..dot {
                                let x = x0 + (i as u32 * 6 + rx as u32) * dot + dx;
                                let y = y0 + ry as u32 * dot + dy;
                                let o = ((y * w + x) * 4) as usize;
                                px[o..o + 3].copy_from_slice(&[0, 0, 0]);
                            }
                        }
                    }
                }
            }
            (w, h, px)
        }
    }
}
