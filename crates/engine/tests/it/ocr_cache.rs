//! The OCR recogniser cache (`ocr::engine_for`, `ocr::statuses`) once a recogniser is loaded.
//! Its own test binary module: no other test here loads a recogniser, so nothing evicts the
//! one under test from the cache, which keeps a few. Synthetic files only.

use std::sync::Arc;

use pdfcraft_engine::ocr::{self, Availability, FallbackReason, LatinModel, Models, OcrSettings};
use pdfcraft_engine::{Session, export};

/// The reviewer's gap: nothing checked the state after a load. A damaged accented-Latin model
/// that a run found shows as read as English from then on (the dialog and `ocr_status` say so),
/// the same files keep the same recogniser, and rewriting the file (a repair, a new install)
/// loads the models again and judges them afresh (needs the English models; skipped without).
#[test]
fn the_cache_remembers_a_damaged_model_and_reloads_a_changed_one() {
    let Some(installed) = Models::find().filter(|m| m.recognition.is_some()) else {
        eprintln!("skipped: the English models are not installed");
        return;
    };
    let dir = std::env::temp_dir().join(format!("pdfcraft-engine-ocr-cache-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (model, dict) = (dir.join(pdfcraft_ocr::LATIN_MODEL), dir.join(pdfcraft_ocr::LATIN_DICT));
    std::fs::write(&model, b"not a model").unwrap();
    std::fs::write(&dict, "a\nb\n").unwrap();
    let damaged = Models { latin: Some(LatinModel { model: model.clone(), dict }), ..installed };
    let fr = ocr::language("fr").unwrap();
    let french = || ocr::statuses(Some(&damaged)).into_iter().find(|(l, _)| *l == fr).map(|(_, a)| a);
    assert_eq!(french(), Some(Availability::Ready), "judged by its files until it is loaded");
    let first = ocr::engine_for(&damaged).unwrap();
    assert_eq!(french(), Some(Availability::Ready), "loaded, but its French reader not yet");

    let png = export::encode_png(64, 64, &[250; 64 * 64 * 4]).unwrap();
    let mut s = Session::new();
    let bytes = s.create_from_images(&[("scan.png".into(), png)]).unwrap();
    let id = s.open("scan.pdf", None, bytes, None).unwrap();
    let job = s.ocr_job(id, &[], OcrSettings { language: Some("fr".into()), ..Default::default() }).unwrap();
    let outcome = job.run(&first, |_, _| true).unwrap();
    assert!(matches!(outcome.fallback.map(|f| f.reason), Some(FallbackReason::Unusable(_))), "the run reads it as English");
    assert_eq!(french(), Some(Availability::FallsBackToEnglish), "and from then on it shows so");
    assert!(Arc::ptr_eq(&first, &ocr::engine_for(&damaged).unwrap()), "the same files: the same recogniser");

    // Rewritten (another size, so another stamp): loaded again, and judged afresh.
    std::fs::write(&model, b"still not a model, but another file").unwrap();
    assert_eq!(french(), Some(Availability::Ready), "the new file is judged by its presence until loaded");
    let second = ocr::engine_for(&damaged).unwrap();
    assert!(!Arc::ptr_eq(&first, &second), "a changed file loads the models again");
    assert_eq!(second.availability(fr), Availability::Ready);
    let _ = std::fs::remove_dir_all(&dir);
}
