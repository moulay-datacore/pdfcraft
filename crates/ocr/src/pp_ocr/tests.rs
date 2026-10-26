//! The recognizer end to end with fake models (no model files needed), and model loading with
//! small ONNX models built here.

use super::frame::tests::word;
use super::*;

/// "x", "y" and the space: classes 1, 2 and 3 (0 is the blank).
fn charset() -> Charset {
    Charset::from_dict("x\ny\n").unwrap()
}

/// A model that "reads" ink: at each time step whose crop columns are dark in the middle rows
/// it emits "x", elsewhere a space, so each blob of ink comes out as a word "x".
struct InkReader;

impl LineModel for InkReader {
    fn run(&self, input: NdTensor<f32, 4>) -> Result<Output, String> {
        let [n, _, h, w] = input.shape();
        let steps = w / 8;
        let mut data = Vec::with_capacity(n * steps * 4);
        for b in 0..n {
            for t in 0..steps {
                let (mut sum, mut count) = (0.0, 0.0);
                for y in h / 3..h * 2 / 3 {
                    for x in t * 8..t * 8 + 8 {
                        sum += input[[b, 0, y, x]];
                        count += 1.0;
                    }
                }
                let class = if sum / count < 0.3 { 1 } else { 3 };
                data.extend((0..4).map(|k| if k == class { 0.9 } else { 0.1 / 3.0 }));
            }
        }
        Ok(Output { shape: [n, steps, 4], data })
    }
}

/// A model that returns `shape`, whatever it is given.
struct Fixed([usize; 3]);

impl LineModel for Fixed {
    fn run(&self, _: NdTensor<f32, 4>) -> Result<Output, String> {
        let [n, t, c] = self.0;
        Ok(Output { shape: self.0, data: vec![0.0; n * t * c] })
    }
}

fn reader() -> Recognizer {
    Recognizer::with_model(Box::new(InkReader), charset()).unwrap()
}

/// A white page with black parallelograms (`x0..x1`, 20 px tall, centred on a line through
/// (0, `y0`) going down `deg` degrees to the right).
fn page(w: usize, h: usize, y0: f32, deg: f32, blobs: &[(usize, usize)]) -> Vec<u8> {
    let mut px = vec![255u8; w * h];
    let slope = deg.to_radians().tan();
    for &(x0, x1) in blobs {
        for x in x0..x1 {
            let yc = y0 + x as f32 * slope;
            for y in 0..h {
                if (y as f32 + 0.5 - yc).abs() < 10.0 {
                    px[y * w + x] = 0;
                }
            }
        }
    }
    px
}

/// A detection box around blob `x0..x1` on that line, 3 px bigger all round as ocrs makes them.
fn detected(x0: usize, x1: usize, y0: f32, deg: f32) -> RotatedRect {
    let cx = (x0 + x1) as f32 / 2.0;
    word(cx, y0 + cx * deg.to_radians().tan(), (x1 - x0) as f32 + 6.0, 26.0, deg)
}

#[test]
fn words_come_back_where_their_ink_is_and_follow_the_skew() {
    let (w, h, y0, deg) = (900, 200, 60.0, 2.0);
    let blobs = [(60, 200), (240, 300), (340, 520), (560, 700), (740, 840)];
    let px = page(w, h, y0, deg, &blobs);
    let img = Gray::new(&px, w, h).unwrap();
    let line: Vec<RotatedRect> = blobs.iter().map(|&(a, b)| detected(a, b, y0, deg)).collect();
    let lines = reader().recognize(&img, &[line]).unwrap();
    assert_eq!(lines.len(), 1, "{lines:?}");
    let words = &lines[0].words;
    assert_eq!(words.len(), blobs.len(), "{words:?}");
    for (wd, &(a, b)) in words.iter().zip(&blobs) {
        assert_eq!(wd.text, "x");
        assert!(wd.snapped, "{wd:?}");
        // Snapped to its detection box: centred on the blob, as wide as the box.
        let [cx, cy] = centre(&wd.quad);
        let mid = (a + b) as f32 / 2.0;
        assert!((cx - mid).abs() < 3.0 && (cy - (y0 + mid * deg.to_radians().tan())).abs() < 3.0, "{wd:?} vs blob {a}..{b}");
        let [tl, tr, br, bl] = wd.quad;
        let angle = (tr[1] - tl[1]).atan2(tr[0] - tl[0]).to_degrees();
        assert!((angle - deg).abs() < 0.1, "{angle}° for {wd:?}");
        assert!(((br[1] - bl[1]).atan2(br[0] - bl[0]).to_degrees() - deg).abs() < 0.1);
        let width = (tr[0] - tl[0]).hypot(tr[1] - tl[1]);
        assert!((width - ((b - a) as f32 + 6.0)).abs() < 1.0, "{width} for {a}..{b}");
    }
}

/// Draw a black bar `len` × `thick` px centred at (`cx`, `cy`), turned `deg` degrees (down to
/// the right), on a white page `w` px wide.
fn bar(px: &mut [u8], w: usize, (cx, cy): (f32, f32), (len, thick): (f32, f32), deg: f32) {
    let (s, c) = deg.to_radians().sin_cos();
    for (i, p) in px.iter_mut().enumerate() {
        let r = [(i % w) as f32 + 0.5 - cx, (i / w) as f32 + 0.5 - cy];
        if (r[0] * c + r[1] * s).abs() < len / 2.0 && (r[1] * c - r[0] * s).abs() < thick / 2.0 {
            *p = 0;
        }
    }
}

/// The angle of a word's top edge, in degrees.
fn angle_of(wd: &RecWord) -> f32 {
    let [tl, tr, _, _] = wd.quad;
    (tr[1] - tl[1]).atan2(tr[0] - tl[0]).to_degrees()
}

/// The prototype's defect, through the recognizer: a short line's own angle came from a word
/// box or two (a logo's strokes: 18.5°), and its crop and words were tilted with it. It is read
/// at the page's skew, which the long lines measure (0.7°).
#[test]
fn a_short_line_is_read_at_the_page_skew_not_its_own_box_angle() {
    let (w, h, deg) = (1000, 420, 0.7);
    let blobs = [(60, 200), (240, 300), (340, 520), (560, 700), (740, 840)];
    let mut px = vec![255u8; w * h];
    let mut lines: Vec<Vec<RotatedRect>> = Vec::new();
    for y0 in [60.0, 140.0, 220.0] {
        for (p, b) in px.iter_mut().zip(page(w, h, y0, deg, &blobs)) {
            *p = (*p).min(b);
        }
        lines.push(blobs.iter().map(|&(a, b)| detected(a, b, y0, deg)).collect());
    }
    // The short line's ink follows the page, its one detection box leans 18.5°.
    let (cx, cy) = (130.0, 330.0 + 130.0 * deg.to_radians().tan());
    bar(&mut px, w, (cx, cy), (60.0, 20.0), deg);
    lines.push(vec![word(cx, cy, 66.0, 26.0, 18.5)]);
    let img = Gray::new(&px, w, h).unwrap();
    let read = reader().recognize(&img, &lines).unwrap();
    let words: Vec<&RecWord> = read.iter().flat_map(|l| &l.words).collect();
    assert_eq!(words.len(), 16, "{read:?}");
    let short = words.iter().find(|wd| (centre(&wd.quad)[1] - cy).abs() < 10.0).unwrap_or_else(|| panic!("{read:?}"));
    assert!((angle_of(short) - deg).abs() < 0.1, "the short line's word is at {}°: {short:?}", angle_of(short));
    assert!(words.iter().all(|wd| (angle_of(wd) - deg).abs() < 0.1), "{read:?}");
}

/// The reviewer's defect: the page's skew was taken from the one long line on a form, a stamp at
/// 20°, and every two-word label was cut out and laid tilted 20° with it, which read each as one
/// word. The labels are read level, each word on its own, and the stamp at its own angle.
#[test]
fn a_leaning_stamp_does_not_tilt_the_short_lines() {
    let (w, h) = (1400usize, 900usize);
    let mut px = vec![255u8; w * h];
    let mut lines: Vec<Vec<RotatedRect>> = Vec::new();
    for y in [100.0f32, 200.0, 300.0] {
        let mut line = Vec::new();
        for x in [150.0f32, 300.0] {
            bar(&mut px, w, (x, y), (100.0, 20.0), 0.0);
            line.push(word(x, y, 106.0, 26.0, 0.0));
        }
        lines.push(line);
    }
    let a = 20.0f32.to_radians();
    let mut stamp = Vec::new();
    for s in [0.0f32, 200.0, 400.0, 600.0, 800.0] {
        let (x, y) = (200.0 + s * a.cos(), 450.0 + s * a.sin());
        bar(&mut px, w, (x, y), (150.0, 20.0), 20.0);
        stamp.push(word(x, y, 156.0, 26.0, 20.0));
    }
    lines.push(stamp);
    let img = Gray::new(&px, w, h).unwrap();
    let read = reader().recognize(&img, &lines).unwrap();
    assert_eq!(read.iter().map(|l| l.words.len()).collect::<Vec<_>>(), [2, 2, 2, 5], "{read:?}");
    for l in read.iter().take(3) {
        assert!(l.words.iter().all(|wd| angle_of(wd).abs() < 0.1 && wd.snapped), "{l:?}");
    }
    assert!(read[3].words.iter().all(|wd| (angle_of(wd) - 20.0).abs() < 0.1), "{:?}", read[3]);
}

/// The reviewer's defect, through the recognizer: a page of short lines only (a receipt, a form)
/// scanned 6° askew was cut out and laid at 3°, the most a page's skew was guessed at without a
/// long line to measure it. Its lines agree on 6°, and are read along it.
#[test]
fn a_page_of_short_lines_is_read_at_the_skew_they_agree_on() {
    let (w, h, deg) = (900usize, 700usize, 6.0f32);
    let a = deg.to_radians();
    let mut px = vec![255u8; w * h];
    let mut lines: Vec<Vec<RotatedRect>> = Vec::new();
    for y0 in [80.0f32, 180.0, 280.0, 380.0] {
        let mut line = Vec::new();
        for s in [0.0f32, 150.0] {
            let (x, y) = (120.0 + s * a.cos(), y0 + s * a.sin());
            bar(&mut px, w, (x, y), (100.0, 20.0), deg);
            line.push(word(x, y, 106.0, 26.0, deg));
        }
        lines.push(line);
    }
    let img = Gray::new(&px, w, h).unwrap();
    let read = reader().recognize(&img, &lines).unwrap();
    assert_eq!(read.iter().map(|l| l.words.len()).collect::<Vec<_>>(), [2, 2, 2, 2], "{read:?}");
    assert!(read.iter().flat_map(|l| &l.words).all(|wd| (angle_of(wd) - deg).abs() < 0.1 && wd.snapped), "{read:?}");
}

/// The verifier's defect: ocrs split a text line in two, the small "à" between the halves had
/// no detection box of its own, and both halves' crops (each with its margin) read it, so the
/// text layer said "à à Châlons". The same ink read twice is kept once.
#[test]
fn a_word_in_the_overlap_of_two_crops_is_read_once() {
    let (w, h, y0) = (600, 120, 60.0);
    // A word, a small mark with no box, a word: ocrs gives the two words as separate lines.
    let blobs = [(50, 150), (164, 174), (183, 283)];
    let px = page(w, h, y0, 0.0, &blobs);
    let img = Gray::new(&px, w, h).unwrap();
    let lines = [vec![detected(50, 150, y0, 0.0)], vec![detected(183, 283, y0, 0.0)]];
    let rec = reader();
    // Both crops take in the mark.
    let frames = rec.frames(&lines);
    let covers = |f: &LineFrame, x: f32| (f.point(0.0, 0.0)[0]..f.point(f.length, 0.0)[0]).contains(&x);
    assert!(frames.len() == 2 && frames.iter().all(|f| covers(f, 169.0)));
    let read = rec.recognize(&img, &lines).unwrap();
    let on_mark = |l: &[RecLine]| l.iter().flat_map(|l| &l.words).filter(|wd| (centre(&wd.quad)[0] - 169.0).abs() < 8.0).count();
    assert_eq!(on_mark(&read), 1, "{read:?}");
    // Every blob is read, the words keep their order, and the words with boxes kept theirs.
    let xs: Vec<f32> = read.iter().flat_map(|l| &l.words).map(|wd| centre(&wd.quad)[0]).collect();
    assert_eq!(xs.len(), 3, "{read:?}");
    assert!(xs.windows(2).all(|p| p[0] < p[1]), "{xs:?}");
    assert!(read.iter().flat_map(|l| &l.words).filter(|wd| wd.snapped).count() == 2);
}

/// The defect the French end-to-end test showed at 0°: ocrs made "; año," a line of its own,
/// lying inside the line it is part of, and it came out after that line. A fragment joins its
/// line, in order along it; the line below stays apart.
#[test]
fn a_fragment_inside_a_line_joins_it() {
    let (w, h, y0) = (500, 140, 50.0);
    let mut px = page(w, h, y0, 0.0, &[(50, 150), (160, 190), (200, 300)]);
    for (p, b) in px.iter_mut().zip(page(w, h, y0 + 40.0, 0.0, &[(50, 150)])) {
        *p = (*p).min(b);
    }
    let img = Gray::new(&px, w, h).unwrap();
    let lines =
        [vec![detected(50, 150, y0, 0.0), detected(200, 300, y0, 0.0)], vec![detected(160, 190, y0, 0.0)], vec![detected(50, 150, y0 + 40.0, 0.0)]];
    let read = reader().recognize(&img, &lines).unwrap();
    assert_eq!(read.len(), 2, "{read:?}");
    let xs: Vec<f32> = read[0].words.iter().map(|wd| centre(&wd.quad)[0]).collect();
    assert_eq!(xs.len(), 3, "{read:?}");
    assert!(xs.windows(2).all(|p| p[0] < p[1]) && (xs[1] - 175.0).abs() < 3.0, "{xs:?}");
    assert!(read[0].words.iter().all(|wd| wd.snapped && (centre(&wd.quad)[1] - y0).abs() < 3.0), "{read:?}");
    assert_eq!(read[1].words.len(), 1, "{read:?}");
    assert!((centre(&read[1].words[0].quad)[1] - (y0 + 40.0)).abs() < 3.0, "{read:?}");
    assert!(merge_fragments(Vec::new()).is_empty());
}

#[test]
fn dedupe_prefers_the_snapped_then_the_confident_word() {
    let q = |x: f32| [[x, 0.0], [x + 10.0, 0.0], [x + 10.0, 10.0], [x, 10.0]];
    let wd = |x: f32, snapped: bool, confidence: f32, text: &str| RecWord { text: text.into(), quad: q(x), confidence, snapped };
    let frame = LineFrame::of_line(&[word(50.0, 5.0, 100.0, 10.0, 0.0)], Margins::default()).unwrap();
    let line = |words: Vec<RecWord>| RecLine { words, confidence: 0.9, frame: frame.clone() };
    let mut lines =
        vec![line(vec![wd(0.0, false, 0.99, "a"), wd(20.0, true, 0.6, "b")]), line(vec![wd(1.0, true, 0.7, "à"), wd(21.0, false, 0.6, "b")])];
    dedupe(&mut lines);
    let texts: Vec<Vec<&str>> = lines.iter().map(|l| l.words.iter().map(|w| w.text.as_str()).collect()).collect();
    assert_eq!(texts, [vec!["b"], vec!["à"]]);
    // Equal: the first stays. A line left empty goes; words that merely touch both stay.
    let mut lines = vec![line(vec![wd(0.0, false, 0.8, "a")]), line(vec![wd(2.0, false, 0.8, "a"), wd(9.0, false, 0.8, "c")])];
    dedupe(&mut lines);
    assert_eq!(lines.len(), 2);
    assert_eq!((lines[0].words.len(), lines[1].words[0].text.as_str()), (1, "c"));
    let mut lines = vec![line(vec![wd(0.0, false, 0.8, "a")]), line(vec![wd(0.0, false, 0.9, "a")])];
    dedupe(&mut lines);
    assert_eq!((lines.len(), lines[0].words[0].confidence), (1, 0.9));
    // Degenerate quads overlap nothing.
    let flat = [[0.0, 0.0]; 4];
    assert_eq!(overlap(&flat, &q(0.0)), 0.0);
    assert_eq!(overlap(&q(0.0), &[[f32::NAN, 0.0]; 4]), 0.0);
}

/// The reviewer's case: a small mark ("à") that ocrs made a line of its own, snapped to its box,
/// and the host's crop reading it glued to the word before ("Bonjourà"), that word's box widened
/// to take it in. The word holding more text stays, whatever the confidences: dropping it lost
/// "Bonjour". Words of about the same extent still go by confidence.
#[test]
fn dedupe_keeps_the_word_that_holds_the_other() {
    let q = |x0: f32, x1: f32| [[x0, 0.0], [x1, 0.0], [x1, 10.0], [x0, 10.0]];
    let wd = |x0: f32, x1: f32, confidence: f32, text: &str| RecWord { text: text.into(), quad: q(x0, x1), confidence, snapped: true };
    let frame = LineFrame::of_line(&[word(50.0, 5.0, 100.0, 10.0, 0.0)], Margins::default()).unwrap();
    let line = |words: Vec<RecWord>| RecLine { words, confidence: 0.9, frame: frame.clone() };
    let texts = |lines: &[RecLine]| lines.iter().map(|l| l.words.iter().map(|w| w.text.clone()).collect::<Vec<_>>()).collect::<Vec<_>>();
    let mut lines = vec![
        line(vec![wd(0.0, 40.0, 0.95, "Avec"), wd(50.0, 160.0, 0.7, "Bonjourà"), wd(170.0, 220.0, 0.95, "tous")]),
        line(vec![wd(140.0, 160.0, 0.9, "à")]),
    ];
    dedupe(&mut lines);
    assert_eq!(texts(&lines), [vec!["Avec", "Bonjourà", "tous"]]);
    // The other way round, the line with the small word first: the same.
    let mut lines = vec![line(vec![wd(140.0, 160.0, 0.9, "à")]), line(vec![wd(50.0, 160.0, 0.7, "Bonjourà")])];
    dedupe(&mut lines);
    assert_eq!(texts(&lines), [vec!["Bonjourà"]]);
    // Two readings of the same word: the more confident one.
    let mut lines = vec![line(vec![wd(50.0, 160.0, 0.7, "Bonjour")]), line(vec![wd(52.0, 158.0, 0.9, "Bonjouv")])];
    dedupe(&mut lines);
    assert_eq!(texts(&lines), [vec!["Bonjouv"]]);
}

/// The reviewer's letterhead: ocrs chained a tall logo with the first address line beside it, so
/// that line's ink band covered the two address lines below it, beside the logo, and they were
/// merged into it word by word ("CMO Tel: 32 Centre 0185 avenue Medical"). A fragment joins a
/// line only on that line's own baseline, where the boxes beside it are.
#[test]
fn lines_beside_a_tall_box_are_not_merged_into_its_line() {
    let wd = |x: f32, y: f32, w: f32, h: f32, t: &str| RecWord {
        text: t.into(),
        quad: [[x - w / 2.0, y - h / 2.0], [x + w / 2.0, y - h / 2.0], [x + w / 2.0, y + h / 2.0], [x - w / 2.0, y + h / 2.0]],
        confidence: 0.9,
        snapped: true,
    };
    let line = |words: Vec<RecWord>, boxes: &[RotatedRect]| RecLine {
        words,
        confidence: 0.9,
        frame: LineFrame::of_line(boxes, Margins::default()).unwrap(),
    };
    let host = line(
        vec![wd(100.0, 100.0, 100.0, 150.0, "CMO"), wd(300.0, 45.0, 200.0, 30.0, "Centre"), wd(550.0, 45.0, 200.0, 30.0, "Medical")],
        &[word(100.0, 100.0, 100.0, 150.0, 0.0), word(300.0, 45.0, 200.0, 30.0, 0.0), word(550.0, 45.0, 200.0, 30.0, 0.0)],
    );
    let below = |y: f32, a: &str, b: &str| {
        line(vec![wd(250.0, y, 120.0, 30.0, a), wd(420.0, y, 160.0, 30.0, b)], &[word(250.0, y, 120.0, 30.0, 0.0), word(420.0, y, 160.0, 30.0, 0.0)])
    };
    // A one-word line right beside the logo, which is far taller than the line's other boxes.
    let fax = line(vec![wd(185.0, 125.0, 50.0, 20.0, "Fax")], &[word(185.0, 125.0, 50.0, 20.0, 0.0)]);
    let out = merge_fragments(vec![host, below(95.0, "32", "avenue"), below(145.0, "Tel:", "0185"), fax]);
    let texts: Vec<Vec<&str>> = out.iter().map(|l| l.words.iter().map(|w| w.text.as_str()).collect()).collect();
    assert_eq!(texts, [vec!["CMO", "Centre", "Medical"], vec!["32", "avenue"], vec!["Tel:", "0185"], vec!["Fax"]]);
}

/// A crowded page is read in full, a batch at a time. Its crops used to be capped per page,
/// which left the lines past the cap unread, and said nothing.
#[test]
fn every_line_of_a_crowded_page_is_read_a_batch_at_a_time() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    /// Reads an "x" at every step, and keeps the size of the largest input it was given.
    struct Counting(Arc<AtomicUsize>);
    impl LineModel for Counting {
        fn run(&self, input: NdTensor<f32, 4>) -> Result<Output, String> {
            let [n, _, _, w] = input.shape();
            self.0.fetch_max(input.len(), Ordering::Relaxed);
            let steps = w / 8;
            Ok(Output { shape: [n, steps, 4], data: (0..n * steps).flat_map(|_| [0.0, 0.97, 0.01, 0.02]).collect() })
        }
    }
    // 200 lines of 3000 px: crops 4000 px wide, 38 million crop pixels in all.
    let (w, h) = (3100, 6000);
    let px = vec![255u8; w * h];
    let img = Gray::new(&px, w, h).unwrap();
    let lines: Vec<Vec<RotatedRect>> = (0..200).map(|i| vec![word(1550.0, 15.0 + i as f32 * 30.0, 3000.0, 20.0, 0.0)]).collect();
    let largest = Arc::new(AtomicUsize::new(0));
    let rec = Recognizer::with_model(Box::new(Counting(Arc::clone(&largest))), charset()).unwrap();
    assert!(rec.frames(&lines).iter().all(|f| frame::crop_width(f) == Some(frame::MAX_WIDTH)));
    let read = rec.recognize(&img, &lines).unwrap();
    assert_eq!(read.len(), 200);
    assert!(read.iter().all(|l| l.words.len() == 1 && l.words[0].text == "x"));
    // Never more than one batch of crops at a time.
    assert_eq!(largest.load(Ordering::Relaxed), MAX_BATCH * 3 * REC_HEIGHT * frame::MAX_WIDTH);
}

/// The batch memory cap: with a dictionary as large as PP-OCRv6's, a batch of long lines would
/// make hundreds of megabytes of model output, so it holds as many lines as fit
/// [`OUTPUT_BUDGET`], one when they are 4000 px wide; short lines still go [`MAX_BATCH`] at a
/// time.
#[test]
fn a_batch_s_output_stays_within_the_budget() {
    use std::sync::{Arc, Mutex};
    /// Reads blanks, and keeps each batch's line count and output size.
    struct Sizes(Arc<Mutex<Vec<(usize, usize)>>>, usize);
    impl LineModel for Sizes {
        fn run(&self, input: NdTensor<f32, 4>) -> Result<Output, String> {
            let [n, _, _, w] = input.shape();
            let (steps, classes) = (w / 8, self.1);
            self.0.lock().unwrap().push((n, n * steps * classes));
            let mut data = vec![0.0; n * steps * classes];
            data.iter_mut().step_by(classes).for_each(|p| *p = 1.0);
            Ok(Output { shape: [n, steps, classes], data })
        }
    }
    // 20 000 entries, as many as PP-OCRv6's 18 708 and more.
    let dict: String = (0x4E00u32..0x4E00 + 20_000).filter_map(char::from_u32).flat_map(|c| [c, '\n']).collect();
    let cs = Charset::from_dict(&dict).unwrap();
    let classes = cs.classes();
    assert_eq!(classes, 20_002);
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let rec = Recognizer::with_model(Box::new(Sizes(Arc::clone(&sizes), classes)), cs).unwrap();
    let px = vec![255u8; 3100 * 400];
    let img = Gray::new(&px, 3100, 400).unwrap();
    let rows = |n: usize, width: f32| -> Vec<Vec<RotatedRect>> {
        (0..n).map(|i| vec![word(20.0 + width / 2.0, 15.0 + i as f32 * 30.0, width, 20.0, 0.0)]).collect()
    };
    sizes.lock().unwrap().clear();
    rec.recognize(&img, &rows(4, 3000.0)).unwrap();
    let long = sizes.lock().unwrap().clone();
    assert_eq!(long.iter().map(|b| b.0).collect::<Vec<_>>(), [1, 1, 1, 1], "{long:?}");
    assert!(long.iter().all(|b| b.1 <= OUTPUT_BUDGET), "{long:?}");
    sizes.lock().unwrap().clear();
    rec.recognize(&img, &rows(12, 150.0)).unwrap();
    let short = sizes.lock().unwrap().clone();
    assert_eq!(short.iter().map(|b| b.0).collect::<Vec<_>>(), [MAX_BATCH, 4], "{short:?}");
    assert!(short.iter().all(|b| b.1 <= OUTPUT_BUDGET), "{short:?}");
}

#[test]
fn batch_outputs_are_checked() {
    let px = page(400, 100, 50.0, 0.0, &[(50, 150), (200, 300)]);
    let img = Gray::new(&px, 400, 100).unwrap();
    let lines = [vec![detected(50, 150, 50.0, 0.0)], vec![detected(200, 300, 50.0, 0.0)]];
    // Probed with one line, the model passes; given two, it still answers for one.
    let rec = Recognizer::with_model(Box::new(Fixed([1, 40, 4])), charset()).unwrap();
    assert!(rec.recognize(&img, &lines).unwrap_err().contains("for 2 lines"));
    // The wrong class count, no steps, more steps than pixels, or a short output fail the probe.
    for shape in [[1, 40, 5], [1, 0, 4], [1, 400, 4], [2, 40, 4]] {
        assert!(Recognizer::with_model(Box::new(Fixed(shape)), charset()).is_err(), "{shape:?}");
    }
    struct Short;
    impl LineModel for Short {
        fn run(&self, _: NdTensor<f32, 4>) -> Result<Output, String> {
            Ok(Output { shape: [1, 40, 4], data: vec![0.0; 10] })
        }
    }
    let rec = Recognizer { model: Box::new(Short), charset: charset() };
    assert!(rec.recognize(&img, &lines[..1]).is_err());
    // No lines, no crops, no run.
    assert!(rec.recognize(&img, &[]).unwrap().is_empty());
}

/// A minimal ONNX model, encoded here: input `x` and output `y` declared with the given
/// dimensions (a number is fixed, a name is free), `y` being a constant of zeros shaped
/// `constant` (rten infers the output's shape from it).
mod onnx {
    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }
    fn int(field: u64, v: u64, out: &mut Vec<u8>) {
        varint(field << 3, out);
        varint(v, out);
    }
    fn bytes(field: u64, b: &[u8], out: &mut Vec<u8>) {
        varint(field << 3 | 2, out);
        varint(b.len() as u64, out);
        out.extend_from_slice(b);
    }
    fn value_info(name: &str, dims: &[&str]) -> Vec<u8> {
        let mut shape = Vec::new();
        for d in dims {
            let mut dim = Vec::new();
            match d.parse::<u64>() {
                Ok(n) => int(1, n, &mut dim),
                Err(_) => bytes(2, d.as_bytes(), &mut dim),
            }
            bytes(1, &dim, &mut shape);
        }
        let mut tensor = Vec::new();
        int(1, 1, &mut tensor); // FLOAT
        bytes(2, &shape, &mut tensor);
        let mut ty = Vec::new();
        bytes(1, &tensor, &mut ty);
        let mut vi = Vec::new();
        bytes(1, name.as_bytes(), &mut vi);
        bytes(2, &ty, &mut vi);
        vi
    }
    /// A ModelProto whose graph holds a node whose attribute holds a graph, `depth` times over:
    /// a few bytes a level, so 20 000 levels make a 234 KB file.
    pub fn nested(depth: usize) -> Vec<u8> {
        let mut graph = Vec::new();
        for _ in 0..depth {
            let mut attribute = Vec::new();
            bytes(6, &graph, &mut attribute);
            let mut node = Vec::new();
            bytes(5, &attribute, &mut node);
            graph.clear();
            bytes(1, &node, &mut graph);
        }
        let mut m = Vec::new();
        int(1, 6, &mut m);
        bytes(7, &graph, &mut m);
        m
    }

    pub fn model(input: &[&str], output: &[&str], constant: &[u64]) -> Vec<u8> {
        let mut init = Vec::new();
        for d in constant {
            int(1, *d, &mut init);
        }
        int(2, 1, &mut init); // FLOAT
        bytes(8, b"c", &mut init);
        bytes(9, &vec![0u8; 4 * constant.iter().product::<u64>() as usize], &mut init);
        let mut node = Vec::new();
        bytes(1, b"c", &mut node);
        bytes(2, b"y", &mut node);
        bytes(4, b"Identity", &mut node);
        let mut graph = Vec::new();
        bytes(1, &node, &mut graph);
        bytes(2, b"g", &mut graph);
        bytes(5, &init, &mut graph);
        bytes(11, &value_info("x", input), &mut graph);
        bytes(12, &value_info("y", output), &mut graph);
        let mut opset = Vec::new();
        bytes(1, b"", &mut opset);
        int(2, 11, &mut opset);
        let mut m = Vec::new();
        int(1, 7, &mut m);
        bytes(7, &graph, &mut m);
        bytes(8, &opset, &mut m);
        m
    }
}

/// A model that doesn't fit the contract or its dictionary is refused when it is loaded, with
/// a message that says why, rather than failing (or misreading) on every page.
#[test]
fn models_are_checked_against_the_contract_and_the_dictionary() {
    let dir = std::env::temp_dir().join(format!("pdfcraft-ocr-models-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dict = dir.join("dict.txt");
    std::fs::write(&dict, "x\ny\n").unwrap();
    let load = |name: &str, bytes: &[u8]| {
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        Recognizer::load(&p, &dict, &sha256_hex(bytes)).err()
    };
    let good = ["N", "3", "48", "W"];
    let out = ["N", "T", "C"];
    assert_eq!(load("good.onnx", &onnx::model(&good, &out, &[1, 40, 4])), None);
    for (name, model, why) in [
        ("grey.onnx", onnx::model(&["N", "1", "48", "W"], &out, &[1, 40, 4]), "1 colour channels"),
        ("tall.onnx", onnx::model(&["N", "3", "32", "W"], &out, &[1, 40, 4]), "32 pixels high"),
        ("flat.onnx", onnx::model(&["N", "3", "W"], &out, &[1, 40, 4]), "input has 3 dimensions"),
        ("seq.onnx", onnx::model(&good, &["T", "C"], &[40, 4]), "output has 2 dimensions"),
        ("classes.onnx", onnx::model(&good, &out, &[1, 40, 7]), "7 classes"),
        ("junk.onnx", b"not a model at all".to_vec(), "junk.onnx"),
        ("empty.onnx", Vec::new(), "empty.onnx"),
    ] {
        let err = load(name, &model).unwrap_or_else(|| panic!("{name} loaded"));
        assert!(err.contains(why) && err.contains(name), "{name}: {err}");
    }
    // A dictionary that isn't one is refused before the model is read.
    std::fs::write(&dict, "xy\n").unwrap();
    assert!(load("good2.onnx", &onnx::model(&good, &out, &[1, 40, 4])).unwrap().contains("dict.txt"));
    assert!(Recognizer::load(&dir.join("missing.onnx"), &dir.join("missing.txt"), "").is_err());
    // Files over the cap are refused unread.
    let p = dir.join("big.bin");
    std::fs::write(&p, [0u8; 11]).unwrap();
    assert!(read_capped(&p, 10).unwrap_err().contains("larger"));
    assert_eq!(read_capped(&p, 11).unwrap().len(), 11);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The reviewer's crash: rten's ONNX reader decodes nested graphs by plain recursion, so a small
/// crafted file named like the model overflowed the stack, which aborts the process (no panic to
/// catch) as soon as a language that needs the model was read. A file that isn't the one PdfCraft
/// ships, byte for byte, is refused before rten sees it; so is a good model under another hash.
/// Before the check, this test aborted the test binary ("has overflowed its stack").
#[test]
fn a_model_that_is_not_the_shipped_one_is_refused_before_it_is_parsed() {
    let dir = std::env::temp_dir().join(format!("pdfcraft-ocr-pinned-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (model, dict) = (dir.join(crate::LATIN_MODEL), dir.join("d.txt"));
    std::fs::write(&dict, "x\ny\n").unwrap();
    // In a thread with a small stack, as the UI's OCR worker has, which overflows soonest.
    let load = |bytes: &[u8]| {
        std::fs::write(&model, bytes).unwrap();
        let (model, dict) = (model.clone(), dict.clone());
        std::thread::Builder::new()
            .stack_size(256 << 10)
            .spawn(move || Recognizer::load(&model, &dict, crate::LATIN_MODEL_SHA256).err())
            .unwrap()
            .join()
            .unwrap()
    };
    let deep = onnx::nested(20_000);
    assert!(deep.len() < 300_000, "{}", deep.len());
    let err = load(&deep).unwrap_or_else(|| panic!("a crafted model loaded"));
    assert!(err.contains("not the model PdfCraft ships") && err.contains(&sha256_hex(&deep)), "{err}");
    let good = onnx::model(&["N", "3", "48", "W"], &["N", "T", "C"], &[1, 40, 4]);
    assert!(load(&good).is_some_and(|e| e.contains("SHA-256")), "a working model under another hash");
    assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A recognizer loaded from a real (tiny) ONNX file runs through rten end to end.
#[test]
fn a_loaded_model_reads_a_page() {
    let dir = std::env::temp_dir().join(format!("pdfcraft-ocr-run-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (model, dict) = (dir.join("m.onnx"), dir.join("d.txt"));
    std::fs::write(&model, onnx::model(&["N", "3", "48", "W"], &["N", "T", "C"], &[1, 40, 4])).unwrap();
    // "ł" is not in WinAnsi, so the text layer couldn't hold it: it is never emitted.
    std::fs::write(&dict, "x\nł\n").unwrap();
    let rec = Recognizer::load(&model, &dict, &sha256_hex(&std::fs::read(&model).unwrap())).unwrap();
    assert_eq!(rec.allowed_chars().collect::<String>(), "x ");
    let px = page(400, 100, 50.0, 0.0, &[(50, 150)]);
    let img = Gray::new(&px, 400, 100).unwrap();
    // The constant model reads blanks everywhere: nothing.
    assert!(rec.recognize(&img, &[vec![detected(50, 150, 50.0, 0.0)]]).unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}
