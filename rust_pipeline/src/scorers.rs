//! Multi-objective candidate scoring.
//!
//! Inspired by x-algorithm/home-mixer/scorers/:
//!   - weighted_scorer.rs: Score = Σ(w_i × P(action_i))
//!   - author_diversity_scorer.rs: exponential decay for repeated authors

use pyo3::prelude::*;
use std::collections::HashMap;

/// Compute weighted multi-objective score for a single candidate.
///
/// Implements x-algorithm's weighted scoring formula:
///     Score = Σ(weight_i × probability_i)
///
/// Default weights (from x-algorithm home-mixer config):
///     click=1.0, like=2.0, share=4.0, follow=3.0, block=-10.0, mute=-5.0
///
/// Args:
///     action_probs: dict mapping action name → predicted probability
///     weights: dict mapping action name → weight
///
/// Returns:
///     Combined score.
///
/// Example:
///     score = weighted_score(
///         {"click": 0.8, "like": 0.3, "share": 0.1},
///         {"click": 1.0, "like": 2.0, "share": 4.0}
///     )
///     # → 0.8*1.0 + 0.3*2.0 + 0.1*4.0 = 1.8
#[pyfunction]
pub fn weighted_score(
    action_probs: HashMap<String, f64>,
    weights: HashMap<String, f64>,
) -> f64 {
    let mut total = 0.0;
    for (action, weight) in &weights {
        if let Some(prob) = action_probs.get(action) {
            total += weight * prob;
        }
    }
    total
}

/// Batch weighted scoring for many candidates.
///
/// ~20x faster than calling weighted_score in a Python loop.
///
/// Args:
///     candidates_probs: list of dicts, each mapping action → probability
///     weights: dict mapping action name → weight (shared across all candidates)
///
/// Returns:
///     list of scores, same length as candidates_probs.
#[pyfunction]
pub fn weighted_score_batch(
    candidates_probs: Vec<HashMap<String, f64>>,
    weights: HashMap<String, f64>,
) -> Vec<f64> {
    candidates_probs
        .iter()
        .map(|probs| {
            let mut total = 0.0;
            for (action, weight) in &weights {
                if let Some(prob) = probs.get(action) {
                    total += weight * prob;
                }
            }
            total
        })
        .collect()
}

/// Author-diversity-aware reranking.
///
/// Inspired by x-algorithm/home-mixer/scorers/author_diversity_scorer.rs
///
/// For each subsequent item from the same author, the score is multiplied by
/// `decay_factor^n` where `n` is the number of previous items from that author
/// already placed. This ensures feed diversity without hard limits.
///
/// Args:
///     item_ids: list of candidate item IDs (already sorted by score desc)
///     scores: corresponding scores (same length)
///     author_ids: corresponding author IDs (same length)
///     decay_factor: decay multiplier (default 0.7, meaning 30% penalty per repeat)
///
/// Returns:
///     (reranked_item_ids, reranked_scores) — re-sorted after diversity adjustment.
#[pyfunction]
#[pyo3(signature = (item_ids, scores, author_ids, decay_factor=0.7))]
pub fn author_diversity_rerank(
    item_ids: Vec<i64>,
    scores: Vec<f64>,
    author_ids: Vec<i64>,
    decay_factor: f64,
) -> (Vec<i64>, Vec<f64>) {
    let mut author_counts: HashMap<i64, u32> = HashMap::new();
    let mut adjusted: Vec<(i64, f64)> = Vec::with_capacity(item_ids.len());

    for i in 0..item_ids.len() {
        let author = author_ids[i];
        let count = author_counts.entry(author).or_insert(0);
        let multiplier = decay_factor.powi(*count as i32);
        adjusted.push((item_ids[i], scores[i] * multiplier));
        *count += 1;
    }

    // Re-sort by adjusted score
    adjusted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let reranked_ids: Vec<i64> = adjusted.iter().map(|(id, _)| *id).collect();
    let reranked_scores: Vec<f64> = adjusted.iter().map(|(_, s)| *s).collect();

    (reranked_ids, reranked_scores)
}
