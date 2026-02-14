//! Feature engineering utilities.
//!
//! Inspired by x-algorithm/phoenix's approach to hash-based embeddings
//! and feature normalization. The key insight from x-algorithm is:
//!     "No hand-engineered features" — raw IDs are hashed into
//!     embedding indices, eliminating the need for a vocabulary.
//!
//! Also provides fast min-max / z-score normalization via rayon.

use ahash::AHasher;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};

/// Hash a feature value into an embedding table index.
///
/// Uses the same hashing trick as x-algorithm/phoenix:
///     index = hash(feature) % num_buckets
///
/// This replaces explicit vocabulary dictionaries, allowing:
///   - Zero-maintenance feature pipeline (no vocab updates)
///   - Constant memory regardless of cardinality
///   - Graceful degradation on collision (slight accuracy loss)
///
/// Args:
///     feature_values: list of raw feature values (strings or ints as strings)
///     num_buckets: size of the embedding table
///
/// Returns:
///     list of indices in [0, num_buckets).
///
/// Example:
///     indices = hash_embedding_indices(["user_42", "item_1337"], 1_000_000)
#[pyfunction]
pub fn hash_embedding_indices(feature_values: Vec<String>, num_buckets: u64) -> Vec<u64> {
    feature_values
        .into_par_iter()
        .map(|val| {
            let mut hasher = AHasher::default();
            val.hash(&mut hasher);
            hasher.finish() % num_buckets
        })
        .collect()
}

/// Batch hash embedding: process multiple feature columns at once.
///
/// Args:
///     feature_columns: list of lists, each inner list is one column's values
///     num_buckets: list of bucket sizes (one per column)
///
/// Returns:
///     list of lists of indices, same shape as input.
///
/// Example:
///     result = hash_embedding_indices_batch(
///         [["user_1", "user_2"], ["item_A", "item_B"]],
///         [1_000_000, 500_000]
///     )
///     # → [[idx1, idx2], [idx3, idx4]]
#[pyfunction]
pub fn hash_embedding_indices_batch(
    feature_columns: Vec<Vec<String>>,
    num_buckets: Vec<u64>,
) -> Vec<Vec<u64>> {
    feature_columns
        .into_iter()
        .zip(num_buckets.into_iter())
        .map(|(col, buckets)| {
            col.into_par_iter()
                .map(|val| {
                    let mut hasher = AHasher::default();
                    val.hash(&mut hasher);
                    hasher.finish() % buckets
                })
                .collect()
        })
        .collect()
}

/// In-place min-max normalization of a feature vector.
///
/// Normalizes values to [0, 1] range: (x - min) / (max - min).
/// If all values are equal, returns all 0.5.
///
/// Args:
///     values: list of float values
///
/// Returns:
///     normalized values in [0, 1].
#[pyfunction]
pub fn normalize_features(values: Vec<f64>) -> Vec<f64> {
    if values.is_empty() {
        return values;
    }

    let min = values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let range = max - min;
    if range.abs() < 1e-12 {
        return vec![0.5; values.len()];
    }

    values
        .into_par_iter()
        .map(|v| (v - min) / range)
        .collect()
}
