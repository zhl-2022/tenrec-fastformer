//! High-performance recommendation pipeline components.
//!
//! Inspired by x-algorithm (Apache 2.0):
//!   - candidate-pipeline: composable pipeline traits
//!   - thunder: DashMap-based in-memory post store
//!   - home-mixer: filters, scorers, hydrators
//!
//! Phase 3: Tenrec-specific optimizations:
//!   - multi_task_scorer: 4-action engagement fusion (click/like/share/follow)
//!   - negative_sampler: high-perf batch negative sampling for 2.3M item pool
//!   - category_filter: video_category diversity controls
//!   - user_profile: user profiling via behavior + demographics
//!   - history_encoder: hist_1~hist_10 encoding and feature extraction
//!
//! Build: `maturin develop --release`
//! Usage in Python: `import fast_pipeline`

mod feature_engine;
mod filters;
mod post_store;
mod scorers;
mod multi_task_scorer;
mod negative_sampler;
mod category_filter;
mod user_profile;
mod history_encoder;

use pyo3::prelude::*;

/// fast_pipeline — Rust-accelerated recommendation components.
///
/// Modules:
///   - PostStore: concurrent in-memory user→item index (DashMap)
///   - Filters: dedup, age, blocked author, self-post, muted keyword, etc.
///   - Scorers: weighted multi-objective, author diversity
///   - FeatureEngine: hash-based embedding lookup, feature normalization
///   - MultiTaskScorer: Tenrec 4-action engagement fusion
///   - NegativeSampler: high-perf sampling for 2.3M item pool
///   - CategoryFilter: video_category diversity controls
///   - UserProfile: behavior-based user profiling
///   - HistoryEncoder: hist_1~hist_10 feature extraction
#[pymodule]
fn fast_pipeline(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ===== Phase 2: Core Components =====

    // PostStore
    m.add_class::<post_store::PostStore>()?;
    m.add_class::<post_store::ItemRecord>()?;

    // Filters
    m.add_class::<filters::FilterResult>()?;
    m.add_function(wrap_pyfunction!(filters::dedup_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters::age_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters::blocked_author_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters::self_item_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters::muted_keyword_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters::seen_items_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters::batch_filter, m)?)?;

    // Scorers
    m.add_function(wrap_pyfunction!(scorers::weighted_score, m)?)?;
    m.add_function(wrap_pyfunction!(scorers::weighted_score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(scorers::author_diversity_rerank, m)?)?;

    // Feature Engine
    m.add_function(wrap_pyfunction!(feature_engine::hash_embedding_indices, m)?)?;
    m.add_function(wrap_pyfunction!(feature_engine::hash_embedding_indices_batch, m)?)?;
    m.add_function(wrap_pyfunction!(feature_engine::normalize_features, m)?)?;

    // ===== Phase 3: Tenrec-Specific Optimizations =====

    // Multi-Task Scorer (click/like/share/follow fusion)
    m.add_function(wrap_pyfunction!(multi_task_scorer::multi_task_engagement_score, m)?)?;
    m.add_function(wrap_pyfunction!(multi_task_scorer::multi_task_engagement_score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(multi_task_scorer::esmm_cvr_correction, m)?)?;
    m.add_function(wrap_pyfunction!(multi_task_scorer::pareto_rerank, m)?)?;

    // Negative Sampler (high-perf for 2.3M items)
    m.add_function(wrap_pyfunction!(negative_sampler::negative_sample, m)?)?;
    m.add_function(wrap_pyfunction!(negative_sampler::batch_negative_sample, m)?)?;
    m.add_function(wrap_pyfunction!(negative_sampler::popularity_negative_sample, m)?)?;

    // Category Filter (video_category diversity)
    m.add_class::<category_filter::CategoryFilterResult>()?;
    m.add_function(wrap_pyfunction!(category_filter::category_diversity_filter, m)?)?;
    m.add_function(wrap_pyfunction!(category_filter::category_interleave, m)?)?;
    m.add_function(wrap_pyfunction!(category_filter::category_diversity_metrics, m)?)?;

    // User Profile (behavior + demographics)
    m.add_function(wrap_pyfunction!(user_profile::build_category_preference, m)?)?;
    m.add_function(wrap_pyfunction!(user_profile::build_category_preference_batch, m)?)?;
    m.add_function(wrap_pyfunction!(user_profile::compute_user_activity_level, m)?)?;
    m.add_function(wrap_pyfunction!(user_profile::compute_user_activity_level_batch, m)?)?;

    // History Encoder (hist_1~hist_10)
    m.add_function(wrap_pyfunction!(history_encoder::encode_history, m)?)?;
    m.add_function(wrap_pyfunction!(history_encoder::encode_history_batch, m)?)?;
    m.add_function(wrap_pyfunction!(history_encoder::position_decay_weights, m)?)?;
    m.add_function(wrap_pyfunction!(history_encoder::history_stats, m)?)?;
    m.add_function(wrap_pyfunction!(history_encoder::history_stats_batch, m)?)?;
    m.add_function(wrap_pyfunction!(history_encoder::history_overlap_check, m)?)?;

    Ok(())
}
