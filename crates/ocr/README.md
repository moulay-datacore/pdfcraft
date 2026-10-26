# pdfcraft-ocr

Layer L4. Scan & OCR ▸ Recognize text: finds the words in a rendered page and writes them back
as an invisible text layer, so a scan becomes searchable and selectable (Acrobat's "Searchable
Image (Exact)": the image is left untouched).

## Recognizers

ocrs's detection model finds the words and groups them into lines for every language. The
words are then read by one of two recognizers, picked by the language:

| Languages | Recognizer | Model files | Licence |
|---|---|---|---|
| English (`en`) | ocrs's recognition model: the Latin alphabet without accents | `text-detection.rten`, `text-recognition.rten` | CC-BY-SA-4.0 (Robert Knight) |
| French, German, Spanish, Italian, Portuguese, Dutch (`fr de es it pt nl`) | PaddleOCR's PP-OCRv6 small recognition model, read by `src/pp_ocr` | `PP-OCRv6_small_rec.onnx`, `ppocrv6_dict.txt` | Apache-2.0 (PaddlePaddle Authors), unmodified |

`cargo xtask models [DIR]` fetches every `kind = "model"` file in `ATTRIBUTION.toml`, checks its
SHA-256, and writes it with its licence text and an `ATTRIBUTION.txt`. The files are never
committed. `Models::find` looks in `$PDFCRAFT_MODELS`, then where the release packages install
them beside the executable (`Models::dirs_beside_exe`), then `assets/models/`. Each file comes
from the first of those directories that has it; the PP-OCR model and its dictionary must be in
the same one. The ONNX file loads through rten's `onnx_format` feature, as PaddlePaddle exported
it.

## API

- `LANGUAGES`, `language(tag)` (a code or BCP 47 tag, by its primary subtag) and
  `first_language(tags)` (the first known one, else `ENGLISH`).
- `availability(language)`, `Models::availability` and `Ocr::availability`: `Ready`,
  `FallsBackToEnglish` (the language's model is missing, or failed to load) or `Missing`.
- `Ocr::recognize_in(pixels, width, height, language)` returns a `Recognition`: the lines, the
  language they were read as, and a `Fallback` (with a `FallbackReason`) when a language was
  read as English. `Ocr::recognize` reads English, as before.
- `Word::quad` is the word's box as read, skewed like its line; `Word::rect` is the level box
  around it. `PlacedWord::place` places the quad, so text on a skewed scan is laid skewed.
- `text_layer` writes placed words as invisible Helvetica in WinAnsiEncoding.

The PP-OCR recognizer is loaded the first time a language needs it, and checked then: a model
that isn't the file PdfCraft ships (`LATIN_MODEL_SHA256`), or a dictionary that is damaged or
doesn't belong with the model, makes that language fall back to English with the reason. Until
that first load the file is judged by its presence alone, so `Ready` before a run means
"installed", not "known to load". Likewise, an ocrs recognition model that fails to load leaves
English unavailable (`Missing`, with the reason when it is read) while the PP-OCR model still
reads the other languages; it is an error only when nothing else is installed.

## How `pp_ocr` reads a page

1. ocrs's lines are split where two words are more than two word heights apart (columns, tab
   stops).
2. Each part gets a direction: its own when it is long enough to measure and leans away from the
   page's skew by a degree or more (a stamp), the page's otherwise. The page's skew is the
   lower median of the long lines within ±3° of level; more only when most of the page's lines
   lean within 3° of it: three or more long lines agreeing on it, or, with no long line near
   level (a receipt or a form of short lines), at least three lines in all. Otherwise it is the
   median of every line, clamped to ±3°. One or two leaning lines (a stamp, a logo chained to a
   line) never set it, and a page of only one or two short lines is read within ±3° of level.
3. The part is cut out upright, 48 pixels high, and read in batches; greedy CTC decoding only
   emits characters the text layer can hold (WinAnsi), so nothing is written as `?`.
4. Words are placed from their time steps and snapped to ocrs's detection boxes they overlap.
   Then the same ink read in two crops is kept once (the snapped word, else the one that holds
   the other's text, else the more confident), and a line that lies on a longer line's baseline
   (ocrs sometimes makes a few words a line of their own) joins it, in order. Lines beside a
   tall box that ocrs chained into a line stay apart.

Every input is untrusted. The PP-OCR model file must match the SHA-256 of the official file
before rten parses it: rten's ONNX reader recurses without a depth limit and sizes some tensors
from the graph, so a crafted file under that name could overflow the stack or exhaust memory,
which abort the process (no panic to catch). The dictionary, which this crate parses itself, is
size-capped, and it and the model are checked against each other and against the model contract
when loaded. Every output shape is checked. Lines are cut out and read a batch at a time, within
a memory budget, so a crowded page is read in full. A panic inside ocrs or rten becomes an error
for that page; an abort does not, which is why the model is pinned. ocrs's own `.rten` models
are loaded as before, unpinned (a separate, pre-existing surface).

## Adding a language

A language whose letters all fit WinAnsiEncoding and the PP-OCRv6 dictionary takes one entry in
`LANGUAGES`, with its letters beyond ASCII. The tests check both. A language with letters outside
WinAnsi (Polish, Czech, Romanian, Turkish) needs a text layer in a Unicode font first.

## Limits

- No upside-down or vertical lines.
- A page of one or two short lines (a label) scanned more than 3° askew is read at 3°: one or
  two word boxes don't measure a skew well enough to trust more. On a page of short lines only,
  the skew is their median, good to about a degree.
- Faint text and stamps read poorly.
- The PP-OCRv6 dictionary lacks `¡` and `„`; `’` is often read as `'`.
- PP-OCR reading is about 1.6× slower than ocrs's, and the model adds about 21 MB to the
  packages.
