//! Metadata-only document-font discovery. Interface fallbacks never enter this catalogue.
use pdfcraft_fonts::embedding::{DocumentFontMetadata, FontGrant};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAX_FILE: u64 = 16 << 20;
const MAX_TABLE_READS: usize = 1 << 20;
const MAX_TABLES: usize = 256;

#[derive(Clone, Debug)]
pub struct FontFile {
    pub path: PathBuf,
    pub family: String,
    /// PostScript identity, used for matching PDF subset names.
    pub face: String,
    pub full_name: String,
    pub weight: u16,
    pub italic: bool,
    pub fs_type: Option<u16>,
}

/// Standard user-installed font folders. Frontends decide when and where to scan.
pub fn installed_directories() -> Vec<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    if cfg!(target_os = "macos") {
        let mut dirs = Vec::new();
        if let Some(home) = home {
            dirs.push(home.join("Library/Fonts"));
        }
        dirs.extend([PathBuf::from("/Library/Fonts"), PathBuf::from("/System/Library/Fonts"), PathBuf::from("/System/Library/Fonts/Supplemental")]);
        dirs
    } else if cfg!(windows) {
        let mut dirs = std::env::var_os("WINDIR").map(|p| PathBuf::from(p).join("Fonts")).into_iter().collect::<Vec<_>>();
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            dirs.push(PathBuf::from(local).join("Microsoft/Windows/Fonts"));
        }
        dirs
    } else {
        let mut dirs = vec![PathBuf::from("/usr/share/fonts"), PathBuf::from("/usr/local/share/fonts")];
        if let Some(home) = home {
            dirs.extend([home.join(".local/share/fonts"), home.join(".fonts")]);
        }
        dirs
    }
}

/// Bounded, symlink-free metadata scan. Full validation is deferred until a matched face is
/// used. The caller controls roots (automation resolves confinement before calling this).
pub fn discover(roots: &[PathBuf], grant: FontGrant) -> Vec<FontFile> {
    Catalogue::discover(roots, grant).files().cloned().collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileStamp {
    size: u64,
    modified: std::time::SystemTime,
    #[cfg(unix)]
    identity: (u64, u64, i64, i64),
}
impl FileStamp {
    fn new(meta: &std::fs::Metadata) -> Option<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Some(Self {
            size: meta.len(),
            modified: meta.modified().ok()?,
            #[cfg(unix)]
            identity: (meta.dev(), meta.ino(), meta.ctime(), meta.ctime_nsec()),
        })
    }
}

struct IndexedFont {
    file: FontFile,
    metadata: DocumentFontMetadata,
    stamp: FileStamp,
}

fn be16(data: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(data.get(at..at.checked_add(2)?)?.try_into().ok()?))
}
fn be32(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(data.get(at..at.checked_add(4)?)?.try_into().ok()?))
}

/// Read only the directory, name and OS/2 tables, never outlines or character maps.
/// Repack the two small tables so the same skrifa parser/policy handles index and full input.
fn read_metadata<R: Read + Seek>(file: &mut R, len: u64, grant: FontGrant) -> Option<DocumentFontMetadata> {
    let mut header = [0; 12];
    file.read_exact(&mut header).ok()?;
    if header.get(..4) != Some(b"OTTO") && header.get(..4) != Some(&[0, 1, 0, 0]) {
        return None;
    }
    let count = usize::from(be16(&header, 4)?);
    if count == 0 || count > MAX_TABLES {
        return None;
    }
    let mut directory = vec![0; count.checked_mul(16)?];
    file.read_exact(&mut directory).ok()?;
    let mut selected = Vec::new();
    let mut tags = std::collections::BTreeSet::new();
    let mut reads = 0usize;
    for record in directory.as_chunks::<16>().0 {
        let tag: [u8; 4] = record.get(..4)?.try_into().ok()?;
        let offset = u64::from(be32(record, 8)?);
        let size = be32(record, 12)?;
        if !tags.insert(tag) || offset.checked_add(u64::from(size))? > len {
            return None;
        }
        if matches!(&tag, b"name" | b"OS/2") {
            reads = reads.checked_add(usize::try_from(size).ok()?)?;
            if reads > MAX_TABLE_READS {
                return None;
            }
            selected.push((tag, offset, size));
        }
    }
    if tags.contains(b"fvar")
        || !(tags.contains(b"glyf") || tags.contains(b"CFF "))
        || [b"head", b"hhea", b"hmtx", b"cmap", b"name"].iter().any(|tag| !tags.contains(*tag))
    {
        return None;
    }
    let table_count = u16::try_from(selected.len()).ok()?;
    let mut bytes = Vec::with_capacity(44usize.checked_add(reads)?);
    bytes.extend_from_slice(header.get(..4)?);
    bytes.extend_from_slice(&table_count.to_be_bytes());
    bytes.extend_from_slice(&[0; 6]);
    let mut offset = 12usize.checked_add(selected.len().checked_mul(16)?)?;
    for (tag, _, size) in &selected {
        bytes.extend_from_slice(tag);
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(&u32::try_from(offset).ok()?.to_be_bytes());
        bytes.extend_from_slice(&size.to_be_bytes());
        offset = offset.checked_add(usize::try_from(*size).ok()?)?;
    }
    for (_, at, size) in selected {
        file.seek(SeekFrom::Start(at)).ok()?;
        let start = bytes.len();
        bytes.resize(start.checked_add(usize::try_from(size).ok()?)?, 0);
        file.read_exact(bytes.get_mut(start..)?).ok()?;
    }
    pdfcraft_fonts::embedding::document_metadata(&bytes, Some(grant)).ok()
}

fn scan(roots: &[PathBuf], grant: FontGrant) -> Vec<IndexedFont> {
    let mut pending: Vec<_> = roots.iter().take(16).rev().map(|r| (r.clone(), 0usize)).collect();
    let mut files = Vec::new();
    let mut entries = 0usize;
    let mut seen = std::collections::HashSet::new();
    while let Some((path, depth)) = pending.pop() {
        if entries >= 4096 {
            break;
        }
        entries += 1;
        let Ok(meta) = std::fs::symlink_metadata(&path) else { continue };
        if meta.file_type().is_symlink() || !seen.insert(path.clone()) {
            continue;
        }
        if meta.is_dir() && depth < 4 {
            if let Ok(children) = std::fs::read_dir(&path) {
                let remaining = 4096usize.saturating_sub(entries.saturating_add(pending.len()));
                let mut children: Vec<_> = children.take(remaining).filter_map(Result::ok).map(|p| p.path()).collect();
                children.sort();
                pending.extend(children.into_iter().rev().map(|p| (p, depth + 1)));
            }
        } else if meta.is_file() && meta.len() <= MAX_FILE && font_extension(&path) {
            let Some(stamp) = FileStamp::new(&meta) else { continue };
            let Ok(mut input) = std::fs::File::open(&path) else { continue };
            if input.metadata().ok().as_ref().and_then(FileStamp::new).as_ref() != Some(&stamp) {
                continue;
            }
            let Some(metadata) = read_metadata(&mut input, meta.len(), grant) else { continue };
            let file = FontFile {
                path,
                family: metadata.family.clone(),
                face: metadata.postscript.clone(),
                full_name: metadata.full_name.clone(),
                weight: metadata.weight,
                italic: metadata.italic,
                fs_type: metadata.fs_type,
            };
            files.push(IndexedFont { file, metadata, stamp });
        }
    }
    files.sort_by(|a, b| a.file.face.cmp(&b.file.face).then(a.file.path.cmp(&b.file.path)));
    files
}
fn font_extension(path: &Path) -> bool {
    path.extension().and_then(std::ffi::OsStr::to_str).is_some_and(|e| e.eq_ignore_ascii_case("ttf") || e.eq_ignore_ascii_case("otf"))
}

/// An immutable snapshot of eligible installed paths and metadata, with no retained programs.
/// Only matching faces are read in full; each use checks for changes and validates the grant.
#[derive(Default)]
pub struct Catalogue {
    fonts: Vec<IndexedFont>,
    index: std::collections::BTreeMap<String, Vec<usize>>,
    grant: FontGrant,
    families: std::collections::BTreeMap<(String, u16, bool), Vec<usize>>,
}

impl Catalogue {
    pub fn discover(roots: &[PathBuf], grant: FontGrant) -> Self {
        let fonts = crate::guard(|| scan(roots, grant)).unwrap_or_default();
        let mut index: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
        let mut families: std::collections::BTreeMap<_, Vec<usize>> = Default::default();
        for (i, font) in fonts.iter().enumerate() {
            let file = &font.file;
            index.entry(pdfcraft_fonts::embedding::face_key(&file.face)).or_default().push(i);
            families.entry((pdfcraft_fonts::embedding::face_key(&file.family), file.weight, file.italic)).or_default().push(i);
        }
        Self { fonts, index, grant, families }
    }

    pub(crate) fn grant(&self) -> FontGrant {
        self.grant
    }

    pub fn files(&self) -> impl Iterator<Item = &FontFile> {
        self.fonts.iter().map(|font| &font.file)
    }

    fn matching(&self, source: &str) -> impl Iterator<Item = &IndexedFont> {
        let key = pdfcraft_fonts::embedding::face_key(source);
        let family = pdfcraft_fonts::embedding::face_identity(source);
        self.index.get(&key).into_iter().flatten().chain(self.families.get(&family).into_iter().flatten()).filter_map(|i| self.fonts.get(*i))
    }

    fn load(&self, font: &IndexedFont) -> Option<Vec<u8>> {
        let path = &font.file.path;
        let meta = std::fs::symlink_metadata(path).ok()?;
        if !meta.is_file() || FileStamp::new(&meta).as_ref() != Some(&font.stamp) {
            return None;
        }
        let mut input = std::fs::File::open(path).ok()?;
        if FileStamp::new(&input.metadata().ok()?).as_ref() != Some(&font.stamp) {
            return None;
        }
        let mut bytes = Vec::new();
        (&mut input).take(MAX_FILE + 1).read_to_end(&mut bytes).ok()?;
        if bytes.len() as u64 != font.stamp.size
            || FileStamp::new(&input.metadata().ok()?).as_ref() != Some(&font.stamp)
            || FileStamp::new(&std::fs::symlink_metadata(path).ok()?).as_ref() != Some(&font.stamp)
            || pdfcraft_fonts::embedding::validate_document_input(&bytes, self.grant).is_err()
            || pdfcraft_fonts::embedding::document_metadata(&bytes, Some(self.grant)).ok().as_ref() != Some(&font.metadata)
        {
            return None;
        }
        Some(bytes)
    }

    /// At most 32 MiB of matched programs, owned by the current edit/paragraph preview.
    pub(crate) fn matching_programs(&self, missing: &[(String, String)]) -> Vec<(String, Vec<u8>)> {
        let mut programs = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        let mut budget = 32usize << 20;
        for (source, missing) in missing {
            for font in self.matching(source) {
                if seen.contains(&font.file.face) {
                    break;
                }
                if font.stamp.size > budget as u64 {
                    continue;
                }
                let Some(bytes) = self.load(font) else { continue };
                if !pdfcraft_fonts::embedding::covers(&bytes, missing) {
                    continue;
                }
                budget = budget.saturating_sub(bytes.len());
                seen.insert(font.file.face.clone());
                programs.push((font.file.full_name.clone(), bytes));
                break;
            }
        }
        programs
    }

    pub(crate) fn register_programs(
        &self,
        cos: &mut pdfcraft_cos::Document,
        page: usize,
        programs: &[(String, Vec<u8>)],
    ) -> Result<Vec<String>, pdfcraft_edit::EditError> {
        for (_, bytes) in programs {
            pdfcraft_edit::text::register_text_font(cos, page, bytes, self.grant)?;
        }
        Ok(programs.iter().map(|(name, _)| name.clone()).collect())
    }

    /// On a transaction/preview clone only. Original programs and existing glyphs stay intact.
    pub(crate) fn complete(
        &self,
        cos: &mut pdfcraft_cos::Document,
        page: usize,
        block: usize,
        text: &str,
        style: &pdfcraft_edit::BlockStyle,
    ) -> Result<Vec<String>, pdfcraft_edit::EditError> {
        if self.fonts.is_empty() {
            return Ok(Vec::new());
        }
        let missing = pdfcraft_edit::text::missing_source_glyphs(cos, page, block, text, style)?;
        self.register_programs(cos, page, &self.matching_programs(&missing))
    }
}
