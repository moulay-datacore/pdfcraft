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

The PP-OCR recognizer is loaded the first time a language needs it, and checked then: a model or
dictionary that is damaged, or that doesn't belong with the other, makes that language fall
back to English with the reason.

## How `pp_ocr` reads a page

1. ocrs's lines are split where two words are more than two word heights apart (columns, tab
   stops).
2. Each part gets a direction: its own when it is long enough to measure and leans away from the
   page's skew by a degree or more (a stamp), the page's otherwise (the median of the lines it
   is sure of). With no sure line, the median of the guesses, clamped to ±3°.
3. The part is cut out upright, 48 pixels high, and read in batches; greedy CTC decoding only
   emits characters the text layer can hold (WinAnsi), so nothing is written as `?`.
4. Words are placed from their time steps and snapped to ocrs's detection boxes they overlap.
   Then the same ink read in two crops is kept once, and a line that lies inside a longer line
   (ocrs sometimes makes a few words a line of their own) joins it, in order.

Every input is untrusted. The model and dictionary files are size-capped, and checked against
each other and against the model contract when loaded. Every output shape is checked. Memory is
capped per batch and per page. A panic inside ocrs or rten becomes an error for that page.

## Adding a language

A language whose letters all fit WinAnsiEncoding and the PP-OCRv6 dictionary takes one entry in
`LANGUAGES`, with its letters beyond ASCII. The tests check both. A language with letters outside
WinAnsi (Polish, Czech, Romanian, Turkish) needs a text layer in a Unicode font first.

## Limits

- No upside-down or vertical lines.
- Faint text and stamps read poorly.
- The PP-OCRv6 dictionary lacks `¡` and `„`; `’` is often read as `'`.
- PP-OCR reading is about 1.6× slower than ocrs's, and the model adds about 21 MB to the
  packages.
