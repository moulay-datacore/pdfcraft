use std::sync::{Arc, atomic::AtomicBool};
use std::time::Instant;

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("missing input PDF")?;
    let mut session = pdfcraft_engine::Session::new();
    let bytes = Arc::new(std::fs::read(&path).map_err(|e| e.to_string())?);
    let id = session.open("diagnostic.pdf", None, bytes, None).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let catalogue = pdfcraft_engine::text_fonts::Catalogue::discover(
        &pdfcraft_engine::text_fonts::installed_directories(),
        pdfcraft_engine::FontGrant::InstalledEmbeddable,
    );
    println!("catalogue={} discovery_ms={}", catalogue.files().count(), start.elapsed().as_millis());
    session.set_text_font_catalogue(Arc::new(catalogue));
    let doc = session.get(id).ok_or("document missing")?;
    for (label, block) in [("Monsieur", 22), ("clinic", 4)] {
        let start = Instant::now();
        let suggestion = doc.ocr_font_suggest_request(0, block)?.run(&AtomicBool::new(false))?;
        println!("{label} ms={} confident={} {:?}", start.elapsed().as_millis(), suggestion.confident, suggestion.candidates);
    }
    Ok(())
}
