use serde_json::Value;
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_pdfcraft-cli");

#[test]
fn cli_batch_reports_partial_failure_and_continues_into_subfolders() {
    let dir = std::env::temp_dir().join(format!("pdfcraft-cli-batch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("input/sub")).unwrap();
    std::fs::write(dir.join("input/0.pdf"), b"broken").unwrap();
    std::fs::write(dir.join("input/sub/page.txt"), "Batch text").unwrap();
    let output = Command::new(BIN)
        .args(["batch", "input", "--root"])
        .arg(&dir)
        .args(["--steps", "convert,optimize", "--out-dir", "out", "--recursive"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["completed"], 1, "{report}");
    assert_eq!(report["failed"], 1);
    let converted = dir.join("out/sub/page.pdf");
    assert!(converted.exists());
    let text = Command::new(BIN).arg("text").arg(converted).output().unwrap();
    assert!(text.status.success());
    assert!(String::from_utf8_lossy(&text.stdout).contains("Batch text"));
    let success = Command::new(BIN).current_dir(&dir).args(["batch", "input/sub", "--steps", "convert", "--out-dir", "other"]).output().unwrap();
    assert!(success.status.success(), "{}", String::from_utf8_lossy(&success.stderr));
    std::fs::remove_dir_all(dir).unwrap();
}

/// The reviewer's gap: nothing checked that the OCR flags reach the batch. Each is an OCR
/// setting, refused without the ocr step before any output is made, and --language is checked
/// against the languages text recognition reads (no models needed).
#[test]
fn cli_batch_passes_the_ocr_flags_on() {
    let dir = std::env::temp_dir().join(format!("pdfcraft-cli-batch-ocr-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("input")).unwrap();
    std::fs::write(dir.join("input/a.pdf"), b"%PDF-1.4").unwrap();
    let batch = |extra: &[&str], steps: &str| {
        Command::new(BIN).arg("batch").arg("input").arg("--root").arg(&dir).args(["--steps", steps, "--out-dir", "out"]).args(extra).output().unwrap()
    };
    for (flag, key) in [(&["--replace-ocr"][..], "replace_ocr"), (&["--language", "fr"], "language"), (&["--ocr-all-pages"], "skip_text_pages")] {
        let output = batch(flag, "optimize");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success() && stderr.contains("settings need the ocr step") && stderr.contains(key), "{flag:?}: {stderr}");
        assert!(!dir.join("out").exists(), "{flag:?}: refused before any output");
    }
    let output = batch(&["--language", "pl"], "ocr");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success() && stderr.contains("doesn't read the language \"pl\""), "{stderr}");
    assert!(!dir.join("out").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_batch_rejects_bad_flags_before_creating_outputs() {
    for args in [
        vec!["batch", "--steps"],
        vec!["batch", "--recursive", "--root"],
        vec!["batch", "--steps", "convert", "--out-dir"],
        vec!["batch", "--steps", "convert", "--steps", "ocr", "--out-dir", "out"],
        vec!["batch", "--unknown"],
    ] {
        let output = Command::new(BIN).args(args).output().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
}
