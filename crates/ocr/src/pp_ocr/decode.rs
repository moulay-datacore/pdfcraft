//! From the model's output to words: greedy CTC decoding, then each word's extent along the
//! line, from its time steps and the detection boxes it overlaps.

use super::dict::Charset;

/// One emitted character: its class, the time step it was read at and its probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Step {
    pub class: usize,
    pub t: usize,
    pub p: f32,
}

/// Greedy CTC decoding of one line: `probs` holds rows of `classes` probabilities (or
/// log-probabilities: only their order matters), one row per time step. Only the first `valid`
/// rows are read; the rest fall on the batch's padding.
///
/// At each step the most likely class the charset allows wins; a NaN or infinite value never
/// does. A class read on consecutive steps is one character (classes with the same character
/// count as one), and the blank (class 0) separates characters and is dropped.
pub(crate) fn greedy(probs: &[f32], classes: usize, valid: usize, charset: &Charset) -> Vec<Step> {
    if classes == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut prev = 0;
    for (t, row) in probs.chunks_exact(classes).take(valid).enumerate() {
        let best = row
            .iter()
            .enumerate()
            .filter(|(k, p)| p.is_finite() && charset.allowed(*k))
            .fold(None, |best: Option<(usize, f32)>, (k, p)| if best.is_none_or(|b| *p > b.1) { Some((k, *p)) } else { best });
        let (class, p) = best.unwrap_or((0, 0.0));
        let canonical = charset.canonical(class);
        if class != 0 && canonical != prev {
            out.push(Step { class, t, p });
        }
        prev = canonical;
    }
    out
}

/// The words in `steps`: runs of characters between spaces.
pub(crate) fn words_of<'a>(steps: &'a [Step], charset: &Charset) -> Vec<&'a [Step]> {
    steps.split(|s| charset.is_space(s.class)).filter(|w| !w.is_empty()).collect()
}

/// Each word's extent `[u0, u1]` along the line, in frame units, from its time steps alone.
/// Step `t` covers crop pixels `[t, t + 1) × stride`, and the crop has `sx` pixels per frame
/// unit. A character's step peaks a little after its left edge and before its right one, so a
/// word runs from 1.5 steps before its first character's step to one step after its last,
/// never past halfway to its neighbours' and never outside `[0, length]`.
pub(crate) fn spans(words: &[&[Step]], stride: f32, sx: f32, length: f32) -> Vec<[f32; 2]> {
    let step_u = stride / sx;
    let centre = |s: &Step| (s.t as f32 + 0.5) * step_u;
    words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let (Some(a), Some(z)) = (w.first(), w.last()) else { return [0.0, 0.0] };
            let mut lo = centre(a) - 1.5 * step_u;
            let mut hi = centre(z) + step_u;
            if let Some(prev) = i.checked_sub(1).and_then(|p| words.get(p)).and_then(|p| p.last()) {
                lo = lo.max((centre(prev) + centre(a)) / 2.0);
            }
            if let Some(next) = words.get(i.saturating_add(1)).and_then(|p| p.first()) {
                hi = hi.min((centre(z) + centre(next)) / 2.0);
            }
            let lo = lo.max(0.0);
            [lo, hi.min(length).max(lo)]
        })
        .collect()
}

/// A line with more detection boxes than this is not snapped (its words keep their CTC spans):
/// snapping compares every word with every box.
const MAX_SNAP_BOXES: usize = 512;

/// The root of `i` in the union-find forest `parent`, compressing the path.
fn find(parent: &mut [usize], i: usize) -> usize {
    let mut root = i;
    // Bounded: a path visits each node at most once.
    for _ in 0..parent.len() {
        match parent.get(root) {
            Some(&p) if p != root => root = p,
            _ => break,
        }
    }
    let mut j = i;
    for _ in 0..parent.len() {
        let Some(slot) = parent.get_mut(j) else { break };
        let next = *slot;
        if next == j || next == root {
            break;
        }
        *slot = root;
        j = next;
    }
    root
}

/// A word belongs to a detection box it overlaps by this share of the shorter of the two.
const SNAP_OVERLAP: f32 = 0.3;

/// Snap words to the detection boxes. `spans` are the words' `[u0, u1]` from their time steps,
/// left to right; `boxes` the detection boxes `[u0, u1, v0, v1]`; `step` one time step in frame
/// units. Words and boxes that overlap by at least [`SNAP_OVERLAP`] of the shorter form groups,
/// and a group's words share its boxes' extent, split halfway across the gaps between their
/// spans. A word that overlaps no box that much gets `None` and keeps its span: a word that
/// only grazes a box is not that box's word but, most often, the edge of a neighbouring line's
/// ink read in this crop's margin, and given a share of the box it would escape
/// [`dedupe`](super::dedupe) beside the neighbour's own reading of that ink. The detector
/// misses small marks (a trailing "." or ":", a leading "-"): where a group's first or last
/// character was read clearly outside its boxes, the box is widened to take it in.
pub(crate) fn snap(spans: &[[f32; 2]], boxes: &[[f32; 4]], step: f32) -> Vec<Option<[f32; 4]>> {
    let (nw, nb) = (spans.len(), boxes.len());
    if nb > MAX_SNAP_BOXES {
        return vec![None; nw];
    }
    let overlap = |a: [f32; 2], b: [f32; 4]| (a[1].min(b[1]) - a[0].max(b[0])).max(0.0);
    // Union-find over words (0..nw) and boxes (nw..nw + nb).
    let mut parent: Vec<usize> = (0..nw.saturating_add(nb)).collect();
    let mut linked = vec![false; nw];
    for (i, s) in spans.iter().enumerate() {
        let len_w = (s[1] - s[0]).max(1e-3);
        for (j, b) in boxes.iter().enumerate() {
            let o = overlap(*s, *b);
            let len_b = (b[1] - b[0]).max(1e-3);
            if o > 0.0 && o >= SNAP_OVERLAP * len_w.min(len_b) {
                let (a, z) = (find(&mut parent, i), find(&mut parent, nw.saturating_add(j)));
                if let Some(p) = parent.get_mut(a.max(z)) {
                    *p = a.min(z);
                }
                if let Some(l) = linked.get_mut(i) {
                    *l = true;
                }
            }
        }
    }
    let roots: Vec<usize> = (0..nw.saturating_add(nb)).map(|i| find(&mut parent, i)).collect();
    let mut out = vec![None; nw];
    for (i, s) in spans.iter().enumerate() {
        if !linked.get(i).copied().unwrap_or(false) {
            continue;
        }
        let root = roots.get(i).copied();
        let group_boxes: Vec<&[f32; 4]> =
            boxes.iter().enumerate().filter(|(j, _)| roots.get(nw.saturating_add(*j)).copied() == root).map(|(_, b)| b).collect();
        let group_words: Vec<usize> = (0..nw).filter(|&k| roots.get(k).copied() == root).collect();
        let e0 = group_boxes.iter().map(|b| b[0]).fold(f32::INFINITY, f32::min);
        let e1 = group_boxes.iter().map(|b| b[1]).fold(f32::NEG_INFINITY, f32::max);
        let v0 = group_boxes.iter().map(|b| b[2]).fold(f32::INFINITY, f32::min);
        let v1 = group_boxes.iter().map(|b| b[3]).fold(f32::NEG_INFINITY, f32::max);
        if !(e1 > e0 && v1 > v0) {
            continue;
        }
        let pos = group_words.iter().position(|&k| k == i).unwrap_or(0);
        let span_of = |k: usize| spans.get(k).copied().unwrap_or(*s);
        // Left edge: the group's start for its first word, else halfway across the gap.
        let lo = match pos.checked_sub(1).and_then(|p| group_words.get(p)) {
            // The span starts 1.5 steps before the first character's step: that step is half a
            // step or more before the box.
            None if e0 - s[0] > 2.0 * step => s[0],
            None => e0,
            Some(&p) => ((span_of(p)[1] + s[0]) / 2.0).clamp(e0, e1),
        };
        let hi = match group_words.get(pos.saturating_add(1)) {
            // The span ends a step after the last character's step: that step is half a step or
            // more past the box, which then ends half a step after it.
            None if s[1] - e1 > 1.5 * step => s[1] - 0.5 * step,
            None => e1,
            Some(&q) => ((s[1] + span_of(q)[0]) / 2.0).clamp(e0, e1),
        };
        if let (true, Some(slot)) = (hi > lo, out.get_mut(i)) {
            *slot = Some([lo, hi, v0, v1]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// "a", "b", "c", "é" and the space: classes 1–4, then 5.
    fn charset() -> Charset {
        Charset::from_dict("a\nb\nc\né\n").unwrap()
    }

    /// Rows of probabilities where each step's winner is the class given (a peak of 0.9).
    fn rows(winners: &[usize], classes: usize) -> Vec<f32> {
        winners.iter().flat_map(|&k| (0..classes).map(move |c| if c == k { 0.9 } else { 0.1 / classes as f32 })).collect()
    }

    fn text(steps: &[Step], cs: &Charset) -> String {
        steps.iter().filter_map(|s| cs.char_of(s.class)).collect()
    }

    #[test]
    fn ctc_merges_repeats_and_drops_blanks() {
        let cs = charset();
        // a a _ a b b _ _ c: "aabc" (a blank separates the two a's), at the steps first read.
        let steps = greedy(&rows(&[1, 1, 0, 1, 2, 2, 0, 0, 3], 6), 6, 9, &cs);
        assert_eq!(text(&steps, &cs), "aabc");
        assert_eq!(steps.iter().map(|s| s.t).collect::<Vec<_>>(), [0, 3, 4, 8]);
        // A space between words, read on two steps, is one space.
        let steps = greedy(&rows(&[1, 5, 5, 4, 0, 4], 6), 6, 6, &cs);
        assert_eq!(text(&steps, &cs), "a éé");
        assert_eq!(words_of(&steps, &cs).iter().map(|w| text(w, &cs)).collect::<Vec<_>>(), ["a", "éé"]);
        // Steps over the padding are not read.
        assert_eq!(text(&greedy(&rows(&[1, 0, 2, 3], 6), 6, 2, &cs), &cs), "a");
        assert!(greedy(&[], 6, 10, &cs).is_empty() && greedy(&[1.0; 12], 0, 10, &cs).is_empty());
    }

    /// The text layer is Helvetica in WinAnsiEncoding: a class the charset leaves out is never
    /// emitted, and the step takes the most likely class that is allowed instead.
    #[test]
    fn excluded_classes_are_never_emitted() {
        let mut cs = Charset::from_dict("a\nł\n").unwrap();
        cs.restrict(|c| pdfcraft_fonts::win_ansi_byte(c).is_some());
        // ł is the most likely at step 0, "a" second.
        let probs = [0.05, 0.3, 0.6, 0.05, 0.9, 0.05, 0.05, 0.0];
        let steps = greedy(&probs, 4, 2, &cs);
        assert_eq!(steps.iter().map(|s| (s.class, s.t)).collect::<Vec<_>>(), [(1, 0)]);
        assert_eq!(steps[0].p, 0.3);
    }

    #[test]
    fn non_finite_scores_never_win() {
        let cs = charset();
        let probs = [f32::NAN, f32::INFINITY, 0.2, 0.1, 0.0, 0.0, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN];
        let steps = greedy(&probs, 6, 2, &cs);
        assert_eq!(steps.iter().map(|s| s.class).collect::<Vec<_>>(), [2]);
    }

    #[test]
    fn a_dictionary_listing_a_letter_twice_reads_it_once() {
        let cs = Charset::from_dict("À\nb\nÀ\n").unwrap();
        // Two neighbouring steps read À as its two classes: one letter.
        let steps = greedy(&rows(&[1, 3, 0, 2], 5), 5, 4, &cs);
        assert_eq!(text(&steps, &cs), "Àb");
    }

    /// Time step `t` covers crop pixels `[t, t + 1) × stride`; the crop has `sx` pixels per
    /// frame unit, so a step's centre is `(t + 0.5) × stride / sx` along the line.
    #[test]
    fn steps_map_to_positions_along_the_line() {
        let s = |t| Step { class: 1, t, p: 1.0 };
        let (a, b) = ([s(2), s(3)], [s(10)]);
        // 8 crop pixels a step, 2 crop pixels per frame unit: 4 frame units a step.
        let got = spans(&[&a, &b], 8.0, 2.0, 1000.0);
        assert_eq!(got[0], [(2.5 - 1.5) * 4.0, (3.5 + 1.0) * 4.0]);
        assert_eq!(got[1], [(10.5 - 1.5) * 4.0, (10.5 + 1.0) * 4.0]);
        // Neighbours closer than that meet halfway; nothing runs outside the line.
        let (c, d) = ([s(0)], [s(1)]);
        let got = spans(&[&c, &d], 8.0, 2.0, 5.0);
        assert_eq!(got, [[0.0, 4.0], [4.0, 5.0]]);
    }

    #[test]
    fn snapping_splits_a_shared_box() {
        // One detection box over two words, one box over one word, one word with no box.
        let boxes = [[10.0, 100.0, 0.0, 20.0], [120.0, 160.0, 2.0, 18.0]];
        let spans = [[12.0, 50.0], [60.0, 98.0], [125.0, 150.0], [300.0, 320.0]];
        let s = snap(&spans, &boxes, 5.0);
        assert_eq!(s, [Some([10.0, 55.0, 0.0, 20.0]), Some([55.0, 100.0, 0.0, 20.0]), Some([120.0, 160.0, 2.0, 18.0]), None]);
        assert!(snap(&[], &boxes, 5.0).is_empty());
        assert_eq!(snap(&spans, &[], 5.0), [None; 4]);
        // A last word read on past its box (an undetected full stop) widens it.
        assert_eq!(snap(&[[12.0, 112.0]], &boxes[..1], 5.0), [Some([10.0, 109.5, 0.0, 20.0])]);
        assert_eq!(snap(&[[12.0, 104.0]], &boxes[..1], 5.0), [Some([10.0, 100.0, 0.0, 20.0])]);
        // A word that only grazes a box (a neighbour's ink in the crop's margin) keeps its span
        // and leaves the box to its word.
        assert_eq!(snap(&[[12.0, 50.0], [95.0, 130.0]], &boxes[..1], 5.0), [Some([10.0, 100.0, 0.0, 20.0]), None]);
        // Too many boxes to compare: the spans stand (the same boxes, fewer, snap).
        let many = vec![boxes[0]; MAX_SNAP_BOXES + 1];
        assert_eq!(snap(&spans, &many, 5.0), [None; 4]);
        assert!(snap(&spans, &many[..MAX_SNAP_BOXES], 5.0)[0].is_some());
    }

    #[test]
    fn union_find_compresses_paths() {
        let mut parent = vec![0, 0, 1, 2, 3];
        assert_eq!(find(&mut parent, 4), 0);
        assert_eq!(parent, [0, 0, 0, 0, 0]);
        // A cycle (which union never makes) still ends.
        let mut cycle = vec![1, 0];
        let _ = find(&mut cycle, 0);
        assert_eq!(find(&mut [], 3), 3);
    }

    proptest! {
        /// Whatever the model outputs, decoding never panics, emits only allowed classes at
        /// increasing steps, and no more characters than steps.
        #[test]
        fn decoding_is_total(probs in prop::collection::vec(prop::num::f32::ANY, 0..200), classes in 0usize..8, valid in 0usize..40) {
            let mut cs = charset();
            cs.restrict(|c| c != 'b');
            let steps = greedy(&probs, classes, valid, &cs);
            prop_assert!(steps.len() <= valid);
            prop_assert!(steps.windows(2).all(|w| w[0].t < w[1].t));
            prop_assert!(steps.iter().all(|s| s.class != 0 && s.class != 2 && s.p.is_finite()));
            let words = words_of(&steps, &cs);
            let sp = spans(&words, 8.0, 1.5, 100.0);
            prop_assert!(sp.iter().all(|s| s[0] <= s[1] && s[0] >= 0.0));
        }

        #[test]
        fn snapping_is_total(
            spans in prop::collection::vec((prop::num::f32::ANY, prop::num::f32::ANY), 0..8),
            boxes in prop::collection::vec((prop::num::f32::ANY, prop::num::f32::ANY, prop::num::f32::ANY, prop::num::f32::ANY), 0..8),
            step in prop::num::f32::ANY,
        ) {
            let spans: Vec<[f32; 2]> = spans.into_iter().map(|(a, b)| [a, b]).collect();
            let boxes: Vec<[f32; 4]> = boxes.into_iter().map(|(a, b, c, d)| [a, b, c, d]).collect();
            prop_assert_eq!(snap(&spans, &boxes, step).len(), spans.len());
        }
    }
}
