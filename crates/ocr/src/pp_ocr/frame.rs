//! Line geometry for the PP-OCR recognizer: which way each line reads, the frame it is cut out
//! along, and the upright crop the model reads.
//!
//! ocrs finds word boxes and groups them into lines. A line is cut out of the page upright,
//! rotated by its skew, so the model reads level text and the word boxes it gives back are
//! skewed like the scan.

use rten_imageproc::{RotatedRect, Vec2};

/// Input height of PaddleOCR recognition models.
pub(crate) const REC_HEIGHT: usize = 48;
/// The widest crop read in one go; a longer line is squeezed to fit. PaddleOCR has no cap and
/// ocrs caps at 2400: 4000 px is 500 time steps, a line of about 250 characters.
pub(crate) const MAX_WIDTH: usize = 4000;
/// The narrowest crop.
const MIN_CROP_WIDTH: usize = 8;

/// A long line follows the page's skew when its own angle is within this many degrees of it
/// (one line's estimate is good to about 0.4°): a stamp, which leans its own way, keeps its own.
const LINE_SKEW_TOLERANCE: f32 = 1.0;
/// A page's skew is kept within this many degrees of level unless most of its lines agree on
/// more ([`page_skew`]): one or two word boxes give an angle that can be ten degrees off, a stamp
/// leans its own way, while scans are rarely skewed more than this.
const MAX_GUESSED_SKEW: f32 = 3.0;

/// A grey image, one byte per pixel, rows top to bottom.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gray<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
}

impl<'a> Gray<'a> {
    /// `None` unless `data` is exactly `width` × `height` pixels, neither zero.
    pub(crate) fn new(data: &'a [u8], width: usize, height: usize) -> Option<Gray<'a>> {
        (width > 0 && height > 0 && width.checked_mul(height) == Some(data.len())).then_some(Gray { data, width, height })
    }

    /// The pixel at (`x`, `y`); white outside the image (paper, not ink).
    fn at(&self, x: f32, y: f32) -> f32 {
        if !(x >= 0.0 && y >= 0.0) {
            return 255.0;
        }
        // Saturating casts: a huge coordinate lands outside the image.
        let (x, y) = (x as usize, y as usize);
        if x >= self.width || y >= self.height {
            return 255.0;
        }
        y.checked_mul(self.width).and_then(|r| r.checked_add(x)).and_then(|i| self.data.get(i)).map_or(255.0, |v| f32::from(*v))
    }

    /// Bilinear sample at a continuous position (pixel centres at `i + 0.5`).
    fn sample(&self, x: f32, y: f32) -> f32 {
        let (x, y) = (x - 0.5, y - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let top = self.at(x0, y0) * (1.0 - fx) + self.at(x0 + 1.0, y0) * fx;
        let bottom = self.at(x0, y0 + 1.0) * (1.0 - fx) + self.at(x0 + 1.0, y0 + 1.0) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

/// How much page is cut out around a line's words, as fractions of the words' height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Margins {
    /// Above and below: ocrs's boxes are tight, and accents sit at their edge.
    pub vertical: f32,
    /// Before the first word and after the last: the detector misses small marks there (a
    /// leading dash, a closing "»" or ":").
    pub horizontal: f32,
}

impl Default for Margins {
    /// Measured on synthetic French pages and a real scan: less vertical margin clips accents,
    /// less horizontal margin loses undetected punctuation.
    fn default() -> Self {
        Margins { vertical: 0.15, horizontal: 1.0 }
    }
}

/// `x > min`, false for NaN: `!above(x, min)` rejects NaN where `x <= min` would not.
fn above(x: f32, min: f32) -> bool {
    x > min
}

fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

/// ocrs's boxes can come out with their axes swapped: make "up" point up the page.
pub(crate) fn oriented(words: &[RotatedRect]) -> Vec<RotatedRect> {
    words.iter().map(|r| r.orient_towards(Vec2::from_yx(-1.0, 0.0))).collect()
}

/// Which way a line reads, as measured from its own words.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Direction {
    /// Unit vector along the line, in reading order, in image pixels (y down).
    pub d: [f32; 2],
    /// Whether the line is long enough for `d` to be trusted over the page's skew.
    pub sure: bool,
}

impl Direction {
    fn angle(&self) -> f32 {
        self.d[1].atan2(self.d[0])
    }
}

fn unit(angle: f32) -> [f32; 2] {
    [angle.cos(), angle.sin()]
}

/// A line's reading direction. `words` must be [`oriented`]. `None` for an empty or degenerate
/// line.
pub(crate) fn direction(words: &[RotatedRect]) -> Option<Direction> {
    // The words' own direction, weighted by their width (the reading direction is the up axis
    // turned clockwise: up (0, -1) reads along (1, 0)).
    let (mut x, mut y, mut total_w, mut total_h) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for r in words {
        let up = r.up_axis();
        let w = r.width().max(0.0);
        x += -up.y * w;
        y += up.x * w;
        total_w += w;
        total_h += r.height().max(0.0) * w;
    }
    let len = x.hypot(y);
    if !above(len, 1e-6) || !above(total_w, 0.0) || !len.is_finite() || !total_w.is_finite() {
        return None;
    }
    let d0 = [x / len, y / len];
    let n0 = [-d0[1], d0[0]];
    if words.len() < 2 {
        return Some(Direction { d: d0, sure: false });
    }
    let mean_h = total_h / total_w;
    // A longer line: the slope of its word centres (least squares, weighted by width) is better
    // than any one word's box, whose angle comes from a few pixels of outline.
    let pts: Vec<(f32, f32, f32)> = words
        .iter()
        .map(|r| {
            let c = r.center();
            (dot([c.x, c.y], d0), dot([c.x, c.y], n0), r.width().max(0.0))
        })
        .collect();
    let mu = pts.iter().map(|p| p.0 * p.2).sum::<f32>() / total_w;
    let mv = pts.iter().map(|p| p.1 * p.2).sum::<f32>() / total_w;
    let suu: f32 = pts.iter().map(|p| p.2 * (p.0 - mu) * (p.0 - mu)).sum();
    let suv: f32 = pts.iter().map(|p| p.2 * (p.0 - mu) * (p.1 - mv)).sum();
    let span = pts.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max) - pts.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    if !above(suu, 0.0) || !above(span, 3.0 * mean_h) {
        return Some(Direction { d: d0, sure: false });
    }
    let slope = suv / suu;
    // Words chained across a column or up a staircase do not make a slope worth following.
    if !above(0.2, slope.abs()) {
        return Some(Direction { d: d0, sure: false });
    }
    let l = (1.0 + slope * slope).sqrt();
    let d = [(d0[0] + slope * n0[0]) / l, (d0[1] + slope * n0[1]) / l];
    Some(Direction { d, sure: words.len() >= 3 && span >= 8.0 * mean_h })
}

fn median(mut v: Vec<f32>) -> Option<f32> {
    v.sort_by(f32::total_cmp);
    v.get(v.len() / 2).copied()
}

/// The lower median: of two middle values, the smaller one's index, so two lines that disagree
/// (a body line and a stamp) give the first of them in angle order rather than the larger.
fn lower_median(mut v: Vec<f32>) -> Option<f32> {
    v.sort_by(f32::total_cmp);
    v.get(v.len().saturating_sub(1) / 2).copied()
}

/// Long lines that must agree on a skew past [`MAX_GUESSED_SKEW`] before the page takes it.
const MIN_AGREEING: usize = 3;

/// The page's skew, in radians, from its lines' directions. The long ("sure") lines within
/// [`MAX_GUESSED_SKEW`] of level measure it best: their lower median. A skew past that is taken
/// only when at least [`MIN_AGREEING`] long lines agree on it and most of the page's lines lean
/// within [`MAX_GUESSED_SKEW`] of it (a page scanned askew), never from one or two long lines
/// that lean their own way (a stamp, a tall logo chained to a line beside it) among short ones.
/// Otherwise the median angle of every line, kept within [`MAX_GUESSED_SKEW`] of level.
fn page_skew(dirs: &[Option<Direction>]) -> Option<f32> {
    let limit = MAX_GUESSED_SKEW.to_radians();
    let all: Vec<f32> = dirs.iter().flatten().map(Direction::angle).filter(|a| a.is_finite()).collect();
    let sure: Vec<f32> = dirs.iter().flatten().filter(|d| d.sure).map(Direction::angle).filter(|a| a.is_finite()).collect();
    if let Some(m) = lower_median(sure.clone()).filter(|m| m.abs() > limit) {
        let agreeing = sure.iter().filter(|a| (*a - m).abs() < LINE_SKEW_TOLERANCE.to_radians()).count();
        let leaning = all.iter().filter(|a| (*a - m).abs() <= limit).count();
        if agreeing >= MIN_AGREEING && leaning.saturating_mul(2) > all.len() {
            return Some(m);
        }
    }
    lower_median(sure.into_iter().filter(|a| a.abs() <= limit).collect()).or_else(|| lower_median(all).map(|a| a.clamp(-limit, limit)))
}

/// The direction each line is read along. A scan is skewed as a whole, so the page's skew
/// ([`page_skew`]) is the best estimate for every line that agrees with it, and for every line
/// too short to be measured at all: a short line's own angle comes from a word box or two (a
/// logo's strokes, a descender) and can be far off, which would tilt its crop and its text
/// layer. A long line that clearly leans another way (a stamp) keeps its own angle.
pub(crate) fn line_directions(dirs: &[Option<Direction>]) -> Vec<Option<[f32; 2]>> {
    let page = page_skew(dirs);
    dirs.iter()
        .map(|dir| {
            let (dir, page) = (dir.as_ref()?, page?);
            let own = dir.angle();
            Some(if dir.sure && (own - page).abs() >= LINE_SKEW_TOLERANCE.to_radians() { dir.d } else { unit(page) })
        })
        .collect()
}

/// A line's own coordinate frame: image point = `origin + d·u + n·v`, with `u` along the line
/// (reading direction) and `v` across it (down), in image pixels. The crop covers
/// `u ∈ [0, length]`, `v ∈ [0, band]`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LineFrame {
    origin: [f32; 2],
    d: [f32; 2],
    n: [f32; 2],
    pub length: f32,
    pub band: f32,
    /// The words' detection boxes as `[u0, u1, v0, v1]` in this frame.
    pub words: Vec<[f32; 4]>,
    /// The `v` range the detection boxes span (the ink, without the margins).
    pub ink: [f32; 2],
}

impl LineFrame {
    /// The frame of `words` ([`oriented`]) read along the unit vector `d`, with `margins`
    /// around them. `None` for no words, or words without extent.
    pub(crate) fn along(words: &[RotatedRect], d: [f32; 2], margins: Margins) -> Option<LineFrame> {
        let n = [-d[1], d[0]];
        let (mut u0, mut u1, mut v0, mut v1) = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        let mut boxes = Vec::with_capacity(words.len());
        for r in words {
            let (mut a0, mut a1, mut b0, mut b1) = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
            for c in r.corners() {
                let (u, v) = (dot([c.x, c.y], d), dot([c.x, c.y], n));
                a0 = a0.min(u);
                a1 = a1.max(u);
                b0 = b0.min(v);
                b1 = b1.max(v);
            }
            boxes.push([a0, a1, b0, b1]);
            u0 = u0.min(a0);
            u1 = u1.max(a1);
            v0 = v0.min(b0);
            v1 = v1.max(b1);
        }
        let (ink_h, ink_w) = (v1 - v0, u1 - u0);
        if !above(ink_h, 1.0) || !above(ink_w, 1.0) || !ink_h.is_finite() || !ink_w.is_finite() || !u0.is_finite() || !v0.is_finite() {
            return None;
        }
        let (mv, mu) = (margins.vertical.clamp(0.0, 1.0) * ink_h, margins.horizontal.clamp(0.0, 3.0) * ink_h);
        let (cu0, cv0) = (u0 - mu, v0 - mv);
        let origin = [d[0] * cu0 + n[0] * cv0, d[1] * cu0 + n[1] * cv0];
        let words = boxes.iter().map(|[a0, a1, b0, b1]| [a0 - cu0, a1 - cu0, b0 - cv0, b1 - cv0]).collect();
        Some(LineFrame { origin, d, n, length: ink_w + 2.0 * mu, band: ink_h + 2.0 * mv, words, ink: [mv, mv + ink_h] })
    }

    /// The frame of one ocrs line, read along its own direction.
    pub(crate) fn of_line(words: &[RotatedRect], margins: Margins) -> Option<LineFrame> {
        let words = oriented(words);
        Self::along(&words, direction(&words)?.d, margins)
    }

    /// The image point at frame position (`u`, `v`).
    pub(crate) fn point(&self, u: f32, v: f32) -> [f32; 2] {
        [self.origin[0] + self.d[0] * u + self.n[0] * v, self.origin[1] + self.d[1] * u + self.n[1] * v]
    }

    /// The frame position `[u, v]` of image point `p`, the inverse of [`point`](Self::point)
    /// (`d` and `n` are orthonormal).
    pub(crate) fn locate(&self, p: [f32; 2]) -> [f32; 2] {
        let r = [p[0] - self.origin[0], p[1] - self.origin[1]];
        [dot(r, self.d), dot(r, self.n)]
    }

    /// The line's angle from level, in degrees (positive: going down to the right).
    #[cfg(test)]
    pub(crate) fn angle_degrees(&self) -> f32 {
        self.d[1].atan2(self.d[0]).to_degrees()
    }
}

/// Split one ocrs line (words left to right) where two neighbouring words are further apart
/// than `factor` times the words' median height, along the line: a tab stop, or two columns
/// ocrs chained into one line. The model reads a wide blank as no space at all, and two
/// columns at different heights make the crop too tall.
pub(crate) fn split_at_gaps(line: &[RotatedRect], factor: f32) -> Vec<Vec<RotatedRect>> {
    let Some(frame) = LineFrame::of_line(line, Margins { vertical: 0.0, horizontal: 0.0 }) else {
        return vec![line.to_vec()];
    };
    let Some(median) = median(frame.words.iter().map(|b| b[3] - b[2]).filter(|h| h.is_finite()).collect()) else {
        return vec![line.to_vec()];
    };
    let mut out: Vec<Vec<RotatedRect>> = Vec::new();
    let mut current: Vec<RotatedRect> = Vec::new();
    let mut reach = f32::NEG_INFINITY;
    for (r, b) in line.iter().zip(&frame.words) {
        if !current.is_empty() && b[0] - reach > factor * median {
            out.push(std::mem::take(&mut current));
            reach = f32::NEG_INFINITY;
        }
        current.push(*r);
        reach = reach.max(b[1]);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// A line cut out of the page: [`REC_HEIGHT`] rows of `width` values in `[-1, 1]` (white is
/// +1), as PaddleOCR normalises them.
#[derive(Clone, Debug)]
pub(crate) struct Crop {
    pub pixels: Vec<f32>,
    pub width: usize,
    /// Crop pixels per frame unit along the line.
    pub sx: f32,
    pub frame: LineFrame,
}

/// How wide `frame`'s crop is, in pixels: its length at the scale that makes it [`REC_HEIGHT`]
/// rows high, from [`MIN_CROP_WIDTH`] to [`MAX_WIDTH`]. `None` for a frame without extent,
/// which [`crop`] can't cut out either.
pub(crate) fn crop_width(frame: &LineFrame) -> Option<usize> {
    let sy = REC_HEIGHT as f32 / frame.band;
    let w = (frame.length * sy).round();
    if !w.is_finite() || !sy.is_finite() || !above(sy, 0.0) || w < 1.0 {
        return None;
    }
    // A saturating cast, then clamped.
    let width = (w as usize).clamp(MIN_CROP_WIDTH, MAX_WIDTH);
    let sx = width as f32 / frame.length;
    (sx.is_finite() && above(sx, 0.0)).then_some(width)
}

/// Cut `frame` out of `image`, rotated level and scaled to [`REC_HEIGHT`] rows, [`crop_width`]
/// wide. When shrinking, each output pixel averages up to 4×4 bilinear samples, so thin strokes
/// and accents survive the downscale. `None` for a frame without extent.
pub(crate) fn crop(image: &Gray, frame: LineFrame) -> Option<Crop> {
    let width = crop_width(&frame)?;
    let (sx, sy) = (width as f32 / frame.length, REC_HEIGHT as f32 / frame.band);
    let k = |s: f32| ((1.0 / s).ceil() as usize).clamp(1, 4);
    let (kx, ky) = (k(sx), k(sy));
    let norm = 1.0 / (kx.saturating_mul(ky) as f32);
    let mut pixels = Vec::with_capacity(width.saturating_mul(REC_HEIGHT));
    for oy in 0..REC_HEIGHT {
        for ox in 0..width {
            let mut acc = 0.0;
            for j in 0..ky {
                let v = (oy as f32 + (j as f32 + 0.5) / ky as f32) / sy;
                for i in 0..kx {
                    let u = (ox as f32 + (i as f32 + 0.5) / kx as f32) / sx;
                    let [x, y] = frame.point(u, v);
                    acc += image.sample(x, y);
                }
            }
            pixels.push(acc * norm / 127.5 - 1.0);
        }
    }
    Some(Crop { pixels, width, sx, frame })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::prelude::*;
    use rten_imageproc::PointF;

    /// A word box centred at (`x`, `y`) on a line going down `deg` degrees to the right.
    pub(crate) fn word(x: f32, y: f32, w: f32, h: f32, deg: f32) -> RotatedRect {
        let a = deg.to_radians();
        RotatedRect::new(PointF::from_yx(y, x), Vec2::from_xy(a.sin(), -a.cos()), w, h)
    }

    /// Words along a line through (`x0`, `y0`) at `deg` degrees, at the given distances along it.
    pub(crate) fn line_of(x0: f32, y0: f32, deg: f32, at: &[f32], w: f32, h: f32) -> Vec<RotatedRect> {
        let a = deg.to_radians();
        at.iter().map(|s| word(x0 + s * a.cos(), y0 + s * a.sin(), w, h, deg)).collect()
    }

    #[test]
    fn frames_follow_the_skew() {
        let f = LineFrame::of_line(&line_of(100.0, 100.0, 2.0, &[0.0, 300.0], 80.0, 30.0), Margins { vertical: 0.0, horizontal: 0.0 }).unwrap();
        assert!((f.angle_degrees() - 2.0).abs() < 0.01, "{}", f.angle_degrees());
        assert!((f.band - 30.0).abs() < 0.5, "{}", f.band);
        // The boxes' axes swapped, as ocrs sometimes returns them: the same frame.
        let a = 2.0_f32.to_radians();
        let sw = |x: f32| RotatedRect::new(PointF::from_yx(100.0 + x * a.tan(), x), Vec2::from_xy(a.cos(), a.sin()), 30.0, 80.0);
        let g = LineFrame::of_line(&[sw(100.0), sw(400.0)], Margins { vertical: 0.0, horizontal: 0.0 }).unwrap();
        assert!((g.angle_degrees() - 2.0).abs() < 0.01, "{}", g.angle_degrees());
        assert!(LineFrame::of_line(&[], Margins::default()).is_none());
        // Frame points map back onto the words: the first box's left edge, halfway down.
        let [x, y] = f.point(f.words[0][0], (f.words[0][2] + f.words[0][3]) / 2.0);
        let left = [100.0 - 40.0 * a.cos(), 100.0 - 40.0 * a.sin()];
        assert!((x - left[0]).abs() < 0.5 && (y - left[1]).abs() < 0.5, "{x} {y} vs {left:?}");
        // And image points back onto the frame.
        let [u, v] = f.locate(f.point(12.5, -3.0));
        assert!((u - 12.5).abs() < 1e-3 && (v + 3.0).abs() < 1e-3, "{u} {v}");
    }

    /// The defect the prototype had: a short line's angle came from a logo's strokes (18.5°)
    /// and its words were laid at that angle. A short line takes the page's skew, whatever its
    /// own estimate says; a long line that leans clearly away from the page (a stamp) keeps
    /// its own.
    #[test]
    fn short_lines_take_the_page_skew() {
        let long = |y: f32, deg: f32| direction(&oriented(&line_of(100.0, y, deg, &[0.0, 200.0, 400.0, 600.0, 800.0], 150.0, 30.0))).unwrap();
        let body = [long(100.0, 0.7), long(200.0, 0.72), long(300.0, 0.68)];
        assert!(body.iter().all(|d| d.sure));
        let logo = direction(&oriented(&[word(100.0, 50.0, 60.0, 30.0, 18.5)])).unwrap();
        assert!(!logo.sure && (logo.angle().to_degrees() - 18.5).abs() < 0.1);
        let stamp = long(400.0, -4.0);
        let dirs: Vec<Option<Direction>> = body.iter().copied().chain([logo, stamp]).map(Some).chain([None]).collect();
        let got = line_directions(&dirs);
        let deg = |d: &Option<[f32; 2]>| d.map(|d| d[1].atan2(d[0]).to_degrees());
        assert!(got.iter().take(3).all(|d| (deg(d).unwrap() - 0.7).abs() < 0.01), "{got:?}");
        assert!((deg(&got[3]).unwrap() - 0.7).abs() < 0.01, "the logo's line: {:?}", deg(&got[3]));
        assert!((deg(&got[4]).unwrap() + 4.0).abs() < 0.01, "the stamp: {:?}", deg(&got[4]));
        assert_eq!(got[5], None);
        // Short lines only: their median angle, kept near level.
        let short = |deg: f32| Some(direction(&oriented(&[word(100.0, 50.0, 60.0, 30.0, deg)])).unwrap());
        let got = line_directions(&[short(1.0), short(1.5), short(12.0)]);
        assert!(got.iter().all(|d| (deg(d).unwrap() - 1.5).abs() < 0.01), "{got:?}");
        let got = line_directions(&[short(18.5)]);
        assert!((deg(&got[0]).unwrap() - MAX_GUESSED_SKEW).abs() < 0.01, "{got:?}");
        assert!(line_directions(&[]).is_empty());
    }

    /// The reviewer's defect: one or two long lines that lean their own way (a stamp, a tall
    /// logo chained to a line beside it) set the page's skew, and every short line of a form
    /// was read tilted with them. A page's skew goes past [`MAX_GUESSED_SKEW`] only when most
    /// of its lines lean that way.
    #[test]
    fn a_leaning_stamp_does_not_set_the_page_skew() {
        let deg = |d: &Option<[f32; 2]>| d.map_or(f32::NAN, |d| d[1].atan2(d[0]).to_degrees());
        let degs = |got: &[Option<[f32; 2]>]| got.iter().map(deg).collect::<Vec<_>>();
        let long = |y: f32, deg: f32| direction(&oriented(&line_of(100.0, y, deg, &[0.0, 200.0, 400.0, 600.0, 800.0], 150.0, 30.0)));
        let short = |y: f32, deg: f32| direction(&oriented(&line_of(100.0, y, deg, &[0.0, 120.0], 100.0, 30.0)));
        assert!(long(0.0, 20.0).unwrap().sure && !short(0.0, 0.0).unwrap().sure);
        // Level labels and one stamp at 20°: the labels stay level, the stamp keeps its own.
        let got = line_directions(&[short(100.0, 0.0), short(200.0, 0.0), short(300.0, 0.0), long(400.0, 20.0)]);
        assert!(got.iter().take(3).all(|d| deg(d).abs() < 0.01) && (deg(&got[3]) - 20.0).abs() < 0.01, "{:?}", degs(&got));
        // One body line at 0.5° and a stamp at 20°: the body line's skew, not the stamp's.
        let got = line_directions(&[long(100.0, 0.5), short(200.0, 0.5), short(300.0, 0.5), long(600.0, 20.0)]);
        assert!(got.iter().take(3).all(|d| (deg(d) - 0.5).abs() < 0.01), "{:?}", degs(&got));
        // A stamp of three lines among level labels: still level.
        let mut dirs: Vec<Option<Direction>> = (0..8).map(|i| short(100.0 + 40.0 * i as f32, 0.0)).collect();
        dirs.extend([long(600.0, 20.0), long(640.0, 20.0), long(680.0, 20.0)]);
        let got = line_directions(&dirs);
        assert!(got.iter().take(8).all(|d| deg(d).abs() < 0.01), "{:?}", degs(&got));
        // A page that really is skewed 6°, its short lines too: they follow it.
        let mut dirs: Vec<Option<Direction>> = (0..3).map(|i| long(100.0 + 50.0 * i as f32, 6.0)).collect();
        dirs.extend((0..4).map(|i| short(300.0 + 50.0 * i as f32, 6.3)));
        dirs.push(None);
        let got = line_directions(&dirs);
        assert!(got.iter().take(7).all(|d| (deg(d) - 6.0).abs() < 0.01), "{:?}", degs(&got));
        assert_eq!(got[7], None);
    }

    #[test]
    fn wide_gaps_split_lines() {
        let w = |x: f32, y: f32| word(x, y, 60.0, 20.0, 0.0);
        // Word gaps of 10 px, then a 100 px gap to a second column 8 px lower.
        let line = [w(30.0, 100.0), w(100.0, 100.0), w(230.0, 108.0), w(300.0, 108.0)];
        assert_eq!(split_at_gaps(&line, 2.0).iter().map(Vec::len).collect::<Vec<_>>(), [2, 2]);
        assert_eq!(split_at_gaps(&line, 6.0).len(), 1);
        assert_eq!(split_at_gaps(&[], 2.0).iter().map(Vec::len).collect::<Vec<_>>(), [0]);
    }

    #[test]
    fn crops_are_level_normalised_and_capped() {
        // A black bar 4 px tall on white, going down 3° to the right: cut out along its own
        // frame it is level, so the crop's middle row is dark all along and its top row white.
        let (w, h) = (400usize, 120usize);
        let a = 3.0_f32.to_radians();
        let mut px = vec![255u8; w * h];
        for x in 50..350 {
            let yc = 40.0 + (x as f32 - 50.0) * a.tan();
            for y in 0..h {
                if (y as f32 + 0.5 - yc).abs() < 2.0 {
                    px[y * w + x] = 0;
                }
            }
        }
        let img = Gray::new(&px, w, h).unwrap();
        let frame =
            LineFrame::along(&[word(200.0, 40.0 + 150.0 * a.tan(), 300.0, 4.0, 3.0)], unit(a), Margins { vertical: 1.0, horizontal: 0.0 }).unwrap();
        let c = crop(&img, frame).unwrap();
        let row = |r: usize| &c.pixels[r * c.width..(r + 1) * c.width];
        let mid = row(REC_HEIGHT / 2);
        assert!(mid[c.width / 8..c.width * 7 / 8].iter().all(|v| *v < -0.5), "{mid:?}");
        assert!(row(0).iter().all(|v| (*v - 1.0).abs() < 1e-3), "white is +1");
        // Degenerate and enormous input never panics; the width is capped.
        let one = [0u8; 4];
        let tiny = Gray::new(&one, 2, 2).unwrap();
        assert!(Gray::new(&one, 3, 2).is_none() && Gray::new(&[], 0, 0).is_none() && Gray::new(&one, usize::MAX, 2).is_none());
        let huge = LineFrame::of_line(&[word(1.0, 1.0, 1e9, 2.0, 0.0)], Margins::default()).unwrap();
        assert_eq!(crop(&tiny, huge).unwrap().width, MAX_WIDTH);
    }

    /// Mostly page-sized numbers, sometimes anything (NaN, infinite, huge, negative).
    fn coord() -> impl Strategy<Value = f32> {
        prop_oneof![4 => -2000.0f32..2000.0, 1 => prop::num::f32::ANY]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        /// Any boxes ocrs could return (degenerate, huge, off the page, NaN) give a frame and a
        /// crop or nothing, never a panic, and a crop's size is always within bounds.
        #[test]
        fn line_geometry_is_total(
            boxes in prop::collection::vec((coord(), coord(), coord(), coord(), -90.0f32..90.0), 0..5),
            gap in coord(),
        ) {
            let px = [128u8; 64];
            let img = Gray::new(&px, 8, 8).unwrap();
            let line: Vec<RotatedRect> = boxes.iter().map(|(x, y, w, h, deg)| word(*x, *y, *w, *h, *deg)).collect();
            let parts = split_at_gaps(&line, gap);
            prop_assert_eq!(parts.iter().map(Vec::len).sum::<usize>(), line.len());
            let dirs: Vec<Option<Direction>> = parts.iter().map(|p| direction(&oriented(p))).collect();
            for (part, d) in parts.iter().zip(line_directions(&dirs)) {
                let Some(d) = d else { continue };
                if let Some(c) = LineFrame::along(&oriented(part), d, Margins::default()).and_then(|f| crop(&img, f)) {
                    prop_assert!((MIN_CROP_WIDTH..=MAX_WIDTH).contains(&c.width));
                    prop_assert_eq!(c.pixels.len(), c.width * REC_HEIGHT);
                    prop_assert!(c.pixels.iter().all(|v| v.is_finite() && (-1.001..=1.001).contains(v)));
                }
            }
        }
    }
}
