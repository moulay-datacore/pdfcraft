//! Cached document-font discovery. Interface fallbacks never enter this catalogue.
use pdfcraft_fonts::embedding::FontGrant;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct FontFile {
    pub path: PathBuf,
    pub family: String,
    pub face: String,
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

/// Bounded, symlink-free scan. Only standalone faces permitted by `grant` are listed.
/// The caller controls roots (automation resolves confinement before calling this).
pub fn discover(roots: &[PathBuf], grant: FontGrant) -> Vec<FontFile> {
    Catalogue::discover(roots, grant).files().cloned().collect()
}

fn scan(roots: &[PathBuf], grant: FontGrant) -> Vec<(FontFile, Arc<Vec<u8>>)> {
    use std::io::Read;
    let mut pending: Vec<_> = roots.iter().take(16).rev().map(|r| (r.clone(), 0usize)).collect();
    let mut files = Vec::new();
    let mut entries = 0usize;
    let mut budget = 64u64 << 20;
    let mut retained = 64usize << 20;
    let mut seen = std::collections::HashSet::new();
    while let Some((path, depth)) = pending.pop() {
        if entries >= 4096 || budget == 0 {
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
        } else if meta.is_file() && meta.len() <= (16 << 20) && font_extension(&path) {
            if meta.len() > budget {
                continue;
            }
            let mut bytes = Vec::new();
            let limit = budget.min((16 << 20) + 1);
            let read = std::fs::File::open(&path).and_then(|f| f.take(limit).read_to_end(&mut bytes));
            budget = budget.saturating_sub(bytes.len() as u64);
            if read.is_err() || pdfcraft_fonts::embedding::validate_document_input(&bytes, grant).is_err() {
                continue;
            }
            if bytes.len() <= retained
                && let Ok((family, face)) = pdfcraft_fonts::embedding::names(&bytes)
            {
                retained -= bytes.len();
                files.push((FontFile { path, family, face }, Arc::new(bytes)));
            }
        }
    }
    files.sort_by(|(a, _), (b, _)| a.face.cmp(&b.face).then(a.path.cmp(&b.path)));
    files
}
fn font_extension(path: &Path) -> bool {
    path.extension().and_then(std::ffi::OsStr::to_str).is_some_and(|e| e.eq_ignore_ascii_case("ttf") || e.eq_ignore_ascii_case("otf"))
}

/// An immutable snapshot of eligible installed faces. File IO and validation happen once,
/// off the desktop UI thread; preview and Apply share these exact bytes, even if a file changes.
#[derive(Default)]
pub struct Catalogue {
    fonts: Vec<(FontFile, Arc<Vec<u8>>)>,
    index: std::collections::BTreeMap<String, Vec<usize>>,
    grant: FontGrant,
    families: std::collections::BTreeMap<(String, u16, bool), Vec<usize>>,
}

impl Catalogue {
    pub fn discover(roots: &[PathBuf], grant: FontGrant) -> Self {
        let fonts = crate::guard(|| scan(roots, grant)).unwrap_or_default();
        let mut index: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
        for (i, (file, _)) in fonts.iter().enumerate() {
            index.entry(pdfcraft_fonts::embedding::face_key(&file.face)).or_default().push(i);
        }
        let mut families: std::collections::BTreeMap<_, Vec<usize>> = Default::default();
        for (i, (file, _)) in fonts.iter().enumerate() {
            families.entry(pdfcraft_fonts::embedding::face_identity(&file.face)).or_default().push(i);
        }
        Self { fonts, index, grant, families }
    }

    pub fn files(&self) -> impl Iterator<Item = &FontFile> {
        self.fonts.iter().map(|(file, _)| file)
    }

    fn matching(&self, source: &str) -> impl Iterator<Item = &(FontFile, Arc<Vec<u8>>)> {
        let key = pdfcraft_fonts::embedding::face_key(source);
        // Match a face, not just a family: a regular face must not complete a bold subset.
        let family = pdfcraft_fonts::embedding::face_identity(source);
        self.index.get(&key).into_iter().flatten().chain(self.families.get(&family).into_iter().flatten()).filter_map(|i| self.fonts.get(*i))
    }

    /// On a transaction/preview clone only. Embed a matching full face when its source
    /// subsets cannot encode the draft. Original programs and existing glyphs remain intact.
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
        let mut used = Vec::new();
        for (source, missing) in pdfcraft_edit::text::missing_source_glyphs(cos, page, block, text, style)? {
            if let Some((file, bytes)) = self.matching(&source).find(|(_, bytes)| pdfcraft_fonts::embedding::covers(bytes, &missing)) {
                pdfcraft_edit::text::register_text_font(cos, page, bytes, self.grant)?;
                if !used.contains(&file.face) {
                    used.push(file.face.clone());
                }
            }
        }
        Ok(used)
    }

    pub(crate) fn preview_fonts(&self, programs: &mut Vec<pdfcraft_edit::text::TextEditorFont>) {
        let faces: std::collections::BTreeSet<_> = programs.iter().map(|p| p.face.clone()).collect();
        let mut budget = (32usize << 20).saturating_sub(programs.iter().fold(0usize, |sum, p| sum.saturating_add(p.bytes.len())));
        for source in faces {
            if pdfcraft_fonts::embedding::face_key(&source).is_empty() {
                continue;
            }
            for (index, (file, bytes)) in self.fonts.iter().enumerate() {
                let resource = format!("PCApprovedInstalled{index}");
                if !pdfcraft_fonts::embedding::same_face_name(&file.face, &source)
                    || bytes.len() > budget
                    || programs.iter().any(|p| p.resource == resource)
                {
                    continue;
                }
                budget -= bytes.len();
                programs.push(pdfcraft_edit::text::TextEditorFont { resource, face: source.clone(), bytes: bytes.as_ref().clone() });
            }
        }
    }
}
