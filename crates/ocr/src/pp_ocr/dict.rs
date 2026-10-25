//! The PP-OCR recognition model's output classes, read from its character dictionary.
//!
//! PaddleOCR's `CTCLabelDecode` (with `use_space_char`): class 0 is the CTC blank, class `i` in
//! `1..=n` is the dictionary's line `i`, and class `n + 1` is a space. The model has `n + 2`
//! classes.

use std::io::Read;
use std::path::Path;

/// A dictionary file larger than this is not a recognition dictionary (PP-OCRv6's is 75 KB).
pub(crate) const MAX_DICT_BYTES: u64 = 1 << 20;
/// The most entries a dictionary may have (PP-OCRv6's has 18 708). It also bounds the model's
/// output per time step, and so the memory one line's output takes.
pub(crate) const MAX_DICT_ENTRIES: usize = 32_768;

/// The model's classes and which of them the decoder may emit.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Charset {
    /// `chars[i]` is class `i + 1`; class 0, the CTC blank, has no character. The last one is
    /// the space class.
    chars: Vec<char>,
    /// Per class: whether the decoder may emit it. The blank always may.
    allowed: Vec<bool>,
    /// Per class: the first class with the same character. A dictionary may list a character
    /// twice (PP-OCRv5's Latin one lists 64 capitals twice), and the decoder must treat both as
    /// one, or a letter read on two neighbouring steps as either class would come out doubled.
    canonical: Vec<usize>,
}

impl Charset {
    /// Parse a PaddleOCR character dictionary: one character per line, `\n` or `\r\n`, an
    /// optional trailing newline and byte-order mark. Spaces are not trimmed (a line may be a
    /// space). Anything else, an empty line or a line of two characters, is an error: it would
    /// shift every later class.
    pub(crate) fn from_dict(text: &str) -> Result<Charset, String> {
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let body = text.strip_suffix('\n').unwrap_or(text);
        if body.is_empty() {
            return Err("the dictionary is empty".into());
        }
        let mut chars = Vec::new();
        for (i, line) in body.split('\n').enumerate() {
            if chars.len() >= MAX_DICT_ENTRIES {
                return Err(format!("the dictionary has more than {MAX_DICT_ENTRIES} entries"));
            }
            let line = line.strip_suffix('\r').unwrap_or(line);
            let mut it = line.chars();
            match (it.next(), it.next()) {
                (Some(c), None) => chars.push(c),
                _ => return Err(format!("line {} is {:?}, not one character", i.saturating_add(1), line.chars().take(16).collect::<String>())),
            }
        }
        chars.push(' ');
        let classes = chars.len().saturating_add(1);
        let mut first: std::collections::HashMap<char, usize> = std::collections::HashMap::with_capacity(chars.len());
        let mut canonical = Vec::with_capacity(classes);
        canonical.push(0);
        for (i, c) in chars.iter().enumerate() {
            let class = i.saturating_add(1);
            canonical.push(*first.entry(*c).or_insert(class));
        }
        Ok(Charset { chars, allowed: vec![true; classes], canonical })
    }

    /// Read and parse the dictionary file at `path`, refusing one larger than
    /// [`MAX_DICT_BYTES`] before reading it all.
    pub(crate) fn read(path: &Path) -> Result<Charset, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        file.take(MAX_DICT_BYTES.saturating_add(1)).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        if u64::try_from(bytes.len()).map_or(true, |n| n > MAX_DICT_BYTES) {
            return Err(format!("the dictionary is larger than {MAX_DICT_BYTES} bytes"));
        }
        let text = String::from_utf8(bytes).map_err(|_| "the dictionary is not UTF-8 text".to_string())?;
        Self::from_dict(&text)
    }

    /// The number of output classes the model must have: the entries, the space and the blank.
    pub(crate) fn classes(&self) -> usize {
        self.chars.len().saturating_add(1)
    }

    /// Let the decoder emit only the characters `keep` accepts, and spaces. A step whose most
    /// likely class is left out takes the most likely one that is not, as ocrs's
    /// `allowed_chars` does.
    pub(crate) fn restrict(&mut self, keep: impl Fn(char) -> bool) {
        for (c, a) in self.chars.iter().zip(self.allowed.iter_mut().skip(1)) {
            *a = *c == ' ' || keep(*c);
        }
    }

    /// Whether the decoder may emit `class` (false for a class the model doesn't have).
    pub(crate) fn allowed(&self, class: usize) -> bool {
        self.allowed.get(class).copied().unwrap_or(false)
    }

    /// The character of `class`; `None` for the blank and for a class the model doesn't have.
    pub(crate) fn char_of(&self, class: usize) -> Option<char> {
        class.checked_sub(1).and_then(|i| self.chars.get(i)).copied()
    }

    /// The first class with the same character as `class`.
    pub(crate) fn canonical(&self, class: usize) -> usize {
        self.canonical.get(class).copied().unwrap_or(class)
    }

    /// Whether `class` stands for no letter: the blank, or a space between words.
    pub(crate) fn is_space(&self, class: usize) -> bool {
        self.char_of(class).is_none_or(char::is_whitespace)
    }

    /// The characters the decoder may emit, the space included.
    #[cfg(test)]
    pub(crate) fn allowed_chars(&self) -> impl Iterator<Item = char> + '_ {
        self.chars.iter().zip(self.allowed.iter().skip(1)).filter(|(_, a)| **a).map(|(c, _)| *c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_are_the_blank_the_lines_and_a_space() {
        let cs = Charset::from_dict("a\r\né\n \n").unwrap();
        assert_eq!(cs.classes(), 5);
        assert_eq!(
            (cs.char_of(0), cs.char_of(1), cs.char_of(2), cs.char_of(3), cs.char_of(4), cs.char_of(5)),
            (None, Some('a'), Some('é'), Some(' '), Some(' '), None)
        );
        // Without a trailing newline, and with a byte-order mark: the same classes.
        assert_eq!(Charset::from_dict("\u{feff}a\né\n ").unwrap(), cs);
        assert!(cs.is_space(0) && cs.is_space(3) && cs.is_space(4) && !cs.is_space(1));
    }

    /// A line that is not one character would shift every later class, so the dictionary is
    /// refused rather than read wrong.
    #[test]
    fn malformed_dictionaries_are_refused() {
        for bad in ["", "\n", "ab\n", "a\n\nb\n", "a\r\r\nb", "\u{feff}"] {
            assert!(Charset::from_dict(bad).is_err(), "{bad:?}");
        }
        let too_many: String = "x\n".repeat(MAX_DICT_ENTRIES + 1);
        assert!(Charset::from_dict(&too_many).unwrap_err().contains("more than"));
        assert_eq!(Charset::from_dict(&"x\n".repeat(MAX_DICT_ENTRIES)).unwrap().classes(), MAX_DICT_ENTRIES + 2);
    }

    #[test]
    fn dictionary_files_are_capped_and_must_be_utf8() {
        let dir = std::env::temp_dir().join(format!("pdfcraft-ocr-dict-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let big = dir.join("big.txt");
        std::fs::write(&big, "x\n".repeat(MAX_DICT_BYTES as usize / 2 + 1)).unwrap();
        assert!(Charset::read(&big).unwrap_err().contains("larger than"));
        let latin1 = dir.join("latin1.txt");
        std::fs::write(&latin1, b"a\n\xe9\n").unwrap();
        assert!(Charset::read(&latin1).unwrap_err().contains("UTF-8"));
        let good = dir.join("good.txt");
        std::fs::write(&good, "a\né\n").unwrap();
        assert_eq!(Charset::read(&good).unwrap().classes(), 4);
        assert!(Charset::read(&dir.join("missing.txt")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn duplicate_entries_share_a_class() {
        let cs = Charset::from_dict("A\nÀ\nb\nÀ\n").unwrap();
        assert_eq!((cs.canonical(2), cs.canonical(4)), (2, 2));
        assert_eq!((cs.canonical(0), cs.canonical(1), cs.canonical(3)), (0, 1, 3));
        // The space class and a space line are one character too.
        let cs = Charset::from_dict("a\n \n").unwrap();
        assert_eq!(cs.canonical(3), 2);
    }

    #[test]
    fn restriction_keeps_spaces_and_the_blank() {
        let mut cs = Charset::from_dict("a\nł\n中\né\n").unwrap();
        cs.restrict(|c| pdfcraft_fonts::win_ansi_byte(c).is_some());
        assert_eq!((0..cs.classes()).map(|k| cs.allowed(k)).collect::<Vec<_>>(), [true, true, false, false, true, true]);
        assert_eq!(cs.allowed_chars().collect::<String>(), "aé ");
        assert!(!cs.allowed(cs.classes()));
    }
}
