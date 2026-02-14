use pyo3::prelude::*;
use ahash::AHasher;
use rayon::prelude::*;
use std::hash::Hasher;

/// 历史序列编码 — 针对 Tenrec ctr_data_1M 的 hist_1~hist_10 字段
///
/// Tenrec 的 ctr_data_1M 场景提供了预构建的用户历史 (hist_1~hist_10)。
/// 本模块提供：
/// 1. 历史序列的 hash 编码（映射到 embedding table）
/// 2. 位置权重衰减（近期交互权重更高）
/// 3. 历史序列的统计特征提取

/// 将 hist_1~hist_10 编码为 embedding 索引。
///
/// 使用 hash 函数将 item_id 映射到固定大小的 embedding table，
/// 避免维护巨大的 item 词表。
///
/// Args:
///     hist_items: 历史 item ID 序列 (e.g., [hist_1, hist_2, ..., hist_10])
///     num_buckets: embedding table 大小
///
/// Returns:
///     hash 后的索引序列
#[pyfunction]
#[pyo3(signature = (hist_items, num_buckets=50000))]
pub fn encode_history(
    hist_items: Vec<i64>,
    num_buckets: u64,
) -> Vec<u64> {
    hist_items
        .iter()
        .map(|&item_id| {
            if item_id <= 0 {
                0 // padding
            } else {
                let mut hasher = AHasher::default();
                hasher.write_i64(item_id);
                (hasher.finish() % num_buckets) + 1 // 0 留给 padding
            }
        })
        .collect()
}

/// 批量编码历史序列。
///
/// Args:
///     batch_hist: N x seq_len 的二维数组
///     num_buckets: embedding table 大小
///
/// Returns:
///     N x seq_len 的 hash 索引
#[pyfunction]
#[pyo3(signature = (batch_hist, num_buckets=50000))]
pub fn encode_history_batch(
    batch_hist: Vec<Vec<i64>>,
    num_buckets: u64,
) -> Vec<Vec<u64>> {
    batch_hist
        .into_par_iter()
        .map(|hist| {
            hist.iter()
                .map(|&item_id| {
                    if item_id <= 0 {
                        0
                    } else {
                        let mut hasher = AHasher::default();
                        hasher.write_i64(item_id);
                        (hasher.finish() % num_buckets) + 1
                    }
                })
                .collect()
        })
        .collect()
}

/// 为历史序列生成位置权重（指数衰减）。
///
/// 越近期的交互权重越高：w_i = decay^i，i=0 是最近的。
/// 用于 weighted pooling / attention bias。
///
/// Args:
///     seq_len: 序列长度 (e.g., 10 for hist_1~hist_10)
///     decay: 衰减因子 (默认 0.9)
///
/// Returns:
///     长度为 seq_len 的权重向量 [1.0, 0.9, 0.81, ...]
#[pyfunction]
#[pyo3(signature = (seq_len, decay=0.9))]
pub fn position_decay_weights(
    seq_len: usize,
    decay: f64,
) -> Vec<f64> {
    (0..seq_len)
        .map(|i| decay.powi(i as i32))
        .collect()
}

/// 从历史序列提取统计特征。
///
/// 对 hist_1~hist_10 提取简单统计量作为模型辅助特征：
/// - 有效历史长度（非零 item 个数）
/// - item ID 的唯一度（去重比例）
/// - 跨度（最大 - 最小 item_id，粗略反映兴趣宽度）
///
/// Args:
///     hist_items: 历史 item ID 序列
///
/// Returns:
///     (有效长度, 唯一度, 跨度)
#[pyfunction]
pub fn history_stats(
    hist_items: Vec<i64>,
) -> (usize, f64, i64) {
    let valid: Vec<i64> = hist_items.into_iter().filter(|&x| x > 0).collect();
    let valid_len = valid.len();

    if valid_len == 0 {
        return (0, 0.0, 0);
    }

    // 唯一度
    let mut unique = valid.clone();
    unique.sort_unstable();
    unique.dedup();
    let uniqueness = unique.len() as f64 / valid_len as f64;

    // 跨度
    let min_id = *valid.iter().min().unwrap_or(&0);
    let max_id = *valid.iter().max().unwrap_or(&0);
    let span = max_id - min_id;

    (valid_len, uniqueness, span)
}

/// 批量提取历史统计特征。
#[pyfunction]
pub fn history_stats_batch(
    batch_hist: Vec<Vec<i64>>,
) -> Vec<(usize, f64, i64)> {
    batch_hist
        .into_par_iter()
        .map(|hist| {
            let valid: Vec<i64> = hist.into_iter().filter(|&x| x > 0).collect();
            let valid_len = valid.len();
            if valid_len == 0 {
                return (0, 0.0, 0);
            }
            let mut unique = valid.clone();
            unique.sort_unstable();
            unique.dedup();
            let uniqueness = unique.len() as f64 / valid_len as f64;
            let min_id = *valid.iter().min().unwrap_or(&0);
            let max_id = *valid.iter().max().unwrap_or(&0);
            (valid_len, uniqueness, max_id - min_id)
        })
        .collect()
}

/// 历史序列与候选 item 的重叠检测。
///
/// 检查候选 item 是否出现在用户历史中（防止重复推荐）。
/// 比 Python 的 list comprehension + set 快得多。
///
/// Args:
///     candidate_ids: 候选 item ID 列表
///     hist_items: 用户历史 item ID 列表
///
/// Returns:
///     每个候选 item 是否在历史中的布尔列表
#[pyfunction]
pub fn history_overlap_check(
    candidate_ids: Vec<i64>,
    hist_items: Vec<i64>,
) -> Vec<bool> {
    let hist_set: std::collections::HashSet<i64> = hist_items.into_iter().filter(|&x| x > 0).collect();
    candidate_ids
        .iter()
        .map(|id| hist_set.contains(id))
        .collect()
}
