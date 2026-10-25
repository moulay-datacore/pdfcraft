# Editing existing text and source subsets

Click a text region in **Edit a PDF → Edit text and images**. One editor contains its
source rows and mixed styles. Embedded TrueType and OpenType/CFF faces are used in memory;
source font programs, colour, size and baseline anchors stay intact. A textless preview
prevents a second copy underneath. The frame shows the live extent as text grows or shrinks.
Rows a text edit doesn't reach keep their original operators, so their kerning, `TJ` word
gaps and glyph widths stay exactly as they were. Only rows whose characters changed, or that
moved because an earlier row wrapped, are rewritten, and formatting a selection rewrites
only the rows it covers. Changing the whole paragraph's formatting, width or position rewrites
every row.

Add Text also opens existing text when it is clicked; a blank-space click starts a new item.
Both text tools retype an added text item with one click. Draft boxes grow to the page edge
before wrapping, and the saved item uses that width. A single visual source row grows across font/style boundaries;
multi-row paragraphs retain their existing wrap width. Retyping uses a preview with only
that added item's old drawing removed, so deleted draft characters do not show underneath.

A subset font contains only some of the original face's glyphs. A new character cannot be
reconstructed from a glyph the PDF does not contain. **Load font…** imports an openly licensed,
embeddable `.ttf` or `.otf` face; a `.ttc`/`.otc` collection uses its first face in
**Load font…**. **Installed fonts…** and automation `text_font_import.face_index` can select
other collection faces (zero based). **Installed fonts…** discovers eligible user-installed
faces on request, off the UI thread; the document-font menu selects resources already on the
page. Select **Keep source fonts** to preserve each run's original face. A matching full face
completes its subsets automatically, including new Chinese characters. When the PDF itself embeds
the complete program of a paragraph's font (not a subset) and its licence allows editable
embedding, a character the font's encoding can't show is drawn with that same program. A
character no available face has is refused with an error that names it, and a CJK character
without a craft-fonts build says so. Replacement text takes the craft-fonts Japanese faces when
they can draw all of it, and the bundled Inter face (`assets/fonts`) for the Latin, Greek and
Cyrillic letters they lack (Uzbek `қизил`, Kazakh `әже`): that face needs no build input, so
these work in every build, the web one included. Choosing another face
is explicit. Fonts with restricted embedding, excluded families or
no open-licence declaration are refused. The desktop Format panel offers **Use matching fonts installed on this computer**, persisted
and off by default. Paragraph automation calls `text_edit` and `text_edit_layout` opt in per call
with `installed: true`. With this grant, a font without an open-licence declaration is then
accepted when its OS/2 `fsType` permits editable embedding, and the user answers for its
licence when sharing the PDF (#2077). Interface fallback fonts are never exported.

Based on #1267 by @komo1iddin (Refs #2077), discovery indexes eligible standalone faces off the desktop UI thread, retaining paths and
metadata only. It reads the sfnt directory and at most 1 MiB of name/OS/2 tables per file,
without a cumulative font-size budget. A matching program is read in full only when needed;
changed files and programs failing the full grant validation are skipped. Exact PostScript matches are preferred, then
matching family, weight and italic names; another family never completes a subset. Existing
glyphs retain the source resource; only missing glyphs use the registered full face. Completion
and text are one undo step. Drafts reuse one generation/page/paragraph session and cached
source-face character coverage. Only a matched paragraph retains validated programs (at most
32 MiB), released when the editor closes or the document changes. New editor font families
are installed after drawing so the next frame can use them. The UI reports the face “from this computer”; automation returns
`installed_completions` for the requested paragraph (including layout previews). Automation scans
only its permitted read root when confined. Collections (`.ttc`/`.otc`) are indexed and permission-checked per face; embedding repacks
only the selected face, with corrected sfnt checksums. Variable instances and ambiguous
or unrecognized style names remain unsupported. Turned text, Hebrew and visual-order Indic
rewrites retain their existing single-face behavior and do not register or report installed completion. Discovery retains the 4096-entry cap, four-level depth cap and 16 MiB file cap. It also
caps collections at 256 faces, the catalogue at 4096 faces and cumulative directory/table
reads at 1 MiB per file. macOS includes `/System/Library/Fonts` and `Supplemental`.


The full selected program, glyph mappings, widths and Unicode text are embedded in the PDF.
Saving and reopening therefore needs no locally installed font. Editing the same text again
reuses the font resource the page already has, and every resource made from one program
shares a single copy of it. The Type 3 fonts generated for Japanese or Hebrew characters the
page's fonts lack are made anew by each edit; after an edit, those nothing on the page shows any
more leave the page's resources, unless some content that could name them can't be read.
Original subset bytes are
preserved. Optional matching faces from `CRAFT_FONTS_DIR` can also complete source subsets;
this works only when that build input actually contains the full face. This change does not
add a font binary or change the release font pin. Fonts shared by releases belong in
`storytold/craft-fonts` under the repository's asset rules.

When mixed-style text needs outline fallback, characters using the same actual fallback face
share Type 3 fonts, with at most 240 distinct glyphs per font. Adjacent characters with the same
font and source style are written as one run. Latin characters that the source font can encode
stay in it; Japanese serif/sans and weight choices remain separate. More than 240 missing glyphs
use additional fonts; a styled paragraph needing more than 15,360 distinct fallback glyphs is
refused with an error. Unsupported glyphs are refused before fallback objects are written.
This does not expand glyph coverage, shape mixed contextual runs or remove old unused fonts
when editing again. The separate single-style fallback path is unchanged.

Rewrapped Chinese and Japanese text, which has no spaces between words, may break between any
two characters. A row never begins with closing punctuation, a closing bracket or quote, a small
kana or the prolonged sound mark, and never ends with an opening bracket or quote or a currency
sign before an amount: the line-start and line-end rules of JIS X 4051, as in Unicode line
breaking (UAX #14). Latin words inside such text are not split. Korean and Thai still break only
at spaces.

Select text within the editor before changing size, colour, font, underline or spacing to
format that selection. Automation supports up to 1024 disjoint UTF-8 byte ranges with the
same properties; invalid or overlapping ranges fail atomically. Untouched source formatting
survives. OpenType shaping uses the selected/source face's substitution and positioning
rules, preserves marks and ligatures, and applies Unicode bidi ordering for a single face.
Each shaped run is one text object: its glyphs are shown in one `TJ` array whose adjustments
place each glyph where shaping put it, and a mark set above or below its letter is raised
with the text rise. Logical text remains available after save/reopen through ToUnicode and ActualText.
Added Arabic text records ActualText per shaped line for copying and searching in typing order.
ActualText already supplies typing order, so the editor preserves it without applying
a second bidi reversal; adjacent visual glyph runs still recover their own reading order.
Only right-to-left ActualText is kept apart this way: left-to-right spans, such as the Latin
ligatures (`fi`, `fl`) Word and Chrome wrap in ActualText mid-word, join their line as before.

A justified paragraph stretches each row but its last to the paragraph's width by widening its
spaces. Word spacing (`Tw`) widens only single-byte code 32, so where the font shows a space with
another code (a composite font, the generated Type 3 fallback) each space is followed by a `TJ`
adjustment instead. Justifying a paragraph with several formats in it is not supported yet: it is
left-aligned.

For headless editing, use `text_font_files` → `text_font_import` → `text_fonts` and pass a
resource to `text_edit.font_resource`. `text_edit.ranges` holds selected range formatting.
`text_edit_layout` accepts resource/ranges too and measures a draft without applying it.
All file inputs respect CLI root confinement; importing a font and applying text are undoable.

On a scanned page, recognised text (Scan & OCR) is an invisible layer over the picture.
The Format panel says **Scanned text: the original font is part of the image** and labels
the face picker **Replacement font**. Helvetica is the default replacement, not a claim
about the printed typeface. Choose a standard family, a bundled/open face, a document font,
or an installed face when **Use matching fonts installed on this computer** is on. Installed
choices use the same `InstalledEmbeddable` permission checks as other text edits.
For scans the face picker changes the whole block even when text is selected; size, colour
and other selection formatting remain local to the selection. Automation range fonts do not
change the document default; top-level `font` or `font_resource` does.

Applying an edit covers each original word's pixels with the sampled paper colour and draws
visible replacement text in the sampled ink colour (or the chosen colours). Each visual row
uses the median source vertical font size and baseline, starts at the first word's left edge,
and uses uniform, unskewed glyphs at 100% horizontal scale. Recognition's per-word stretched
matrices and leading-space offsets are discarded. Text rewraps inside the block's width;
explicit font size, spacing, alignment and width formatting still apply. Ordinary visible
paragraphs retain their existing source-font and geometry behavior.

An explicit OCR replacement face becomes the default for later OCR edits in that open document,
including other pages. It is part of the engine's undo/redo state, applies to previews too,
and changes only after a successful OCR font edit. Ordinary paragraphs and other open documents
keep their own fonts. Deleting the original font’s page does not clear the choice. Save retains it in the open
session; reopening the PDF or reverting
starts without a remembered choice. The PDF contains the chosen font resources and visible edited
text, but no preference setting. Layout previews never change the remembered face.

Automation `text_paragraphs`, `text_edit` and `text_edit_layout` report `ocr` for the source block.
Both edit tools accept standard `font` families (`helvetica`, `times`, `courier`), `font_resource`,
and the existing per-call `installed: true` grant. Import an installed face with
`text_font_import` before choosing its resource. Layout previews and commits use the same
OCR defaults and layout; all font paths retain root confinement.

This applies to any invisible text (rendering mode 3), including another OCR tool's layer.
Words split at gaps of 0.6 em or more, and scanned rows remain separate edit blocks. The cover
is one flat colour: textured/patterned paper, overlapping nearby rows after reflow, separating
photographs from lettering, and automatic serif/sans or typeface detection from the image
remain limitations. No automatic image-font matching is performed.

Explicit marked paragraph/heading scopes prevent unrelated tagged regions from merging.
Compatible geometry groups untagged content, including runs split across content streams
or equivalent graphics transforms. Geometry cannot infer author intent in every untagged
PDF. Semantic structure-tree preservation, legacy Type 1/Type 3 inline font preview,
font collections/variable-font instances, mixed-font/style contextual runs and complete rotated
layout fidelity still need broader coverage. Do not interpret the passing real-file checks
as a guarantee for every PDF.

Private acceptance tests are opt-in and keep inputs unmodified. They exercise actual Chinese
keyboard insertion/deletion on all four pages, source-face/baseline checks and identical
in-memory renders before and after saving/reopening. Synthetic tests cover full TrueType
mapping, OpenType/CFF, Arabic marks/ligatures/mixed bidi, repeated edits, disjoint styles,
invalid ranges, font selection and root confinement. CLI acceptance repeats insertion/deletion
and two saves/reopens on four pages; independent PDFium checks extraction and rendering.
No private document, font or screenshot
is included in the repository.

## Paragraph numbers (automation)

A paragraph's number is its position in a list that is regrouped from geometry on every read,
so an edit that merges two paragraphs, or splits one, moves the others. After `text_paragraphs`
the numbers name the paragraphs *that listing showed*: `text_edit`, `text_edit_layout` and
`page_render`'s `hide_text_block` find the paragraph again by its text and box, so editing paragraph 3 then paragraph 1 still edits the
two the listing showed. A paragraph an edit merged into a neighbour (or changed) is refused, and so is the edited paragraph
itself when the edit merged it into one, with
a message to list the page again, instead of rewriting whichever paragraph took its number
(#2516). The result of an edit names the numbers that are gone in `stale_paragraphs`. Listing
a page again renumbers it. The listings kept are capped (256 pages); once the cap drops
them, a page of an affected document is refused until it is listed again, instead of reading its
numbers as positions. Without a listing, a number is a position in the page's current
paragraphs, as before. The inline editor holds the paragraph a switching click asked for the
same way. What is still missing is an identity that survives save and reopen (design: #2508).
