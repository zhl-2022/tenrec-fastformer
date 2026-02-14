use pyo3::prelude::*;
use ahash::AHashMap;
use rayon::prelude::*;

/// 类目多样性过滤 — 针对 Tenrec video_category 字段
///
/// Tenrec 的 video_category 反映视频类型，推荐结果中同类目过多会影响体验。
/// 本模块实现类目级别的多样性约束和混排。

/// 过滤结果：保留和移除的 item 列表
#[pyclass]
#[derive(Debug, Clone)]
pub struct CategoryFilterResult {
    #[pyo3(get)]
    pub kept: Vec<i64>,
    #[pyo3(get)]
    pub removed: Vec<i64>,
    #[pyo3(get)]
    pub category_counts: Vec<(i64, usize)>,  // 每个类目保留了多少
}

/// 限制每个类目最多保留 N 个 item。
///
/// 按原始顺序（通常按分数排序），依次检查每个 item 的类目，
/// 超出类目限制的 item 被移除。
///
/// Args:
///     item_ids: 候选 item ID 列表（按分数降序）
///     categories: 每个 item 对应的 video_category
///     max_per_category: 每个类目最多保留几个 (默认 3)
///
/// Returns:
///     CategoryFilterResult 包含 kept, removed, category_counts
#[pyfunction]
#[pyo3(signature = (item_ids, categories, max_per_category=3))]
pub fn category_diversity_filter(
    item_ids: Vec<i64>,
    categories: Vec<i64>,
    max_per_category: usize,
) -> CategoryFilterResult {
    let mut cat_count: AHashMap<i64, usize> = AHashMap::new();
    let mut kept = Vec::new();
    let mut removed = Vec::new();

    for (item, cat) in item_ids.iter().zip(categories.iter()) {
        let count = cat_count.entry(*cat).or_insert(0);
        if *count < max_per_category {
            kept.push(*item);
            *count += 1;
        } else {
            removed.push(*item);
        }
    }

    let category_counts: Vec<(i64, usize)> = cat_count.into_iter().collect();

    CategoryFilterResult {
        kept,
        removed,
        category_counts,
    }
}

/// 类目交错混排 (Round-Robin Interleaving)。
///
/// 将不同类目的 item 交替插入结果列表，确保相邻 item 尽量不同类。
/// 类似 Twitter/X 的 home timeline diversity 策略。
///
/// Args:
///     item_ids: 候选 item ID 列表（按分数降序）
///     categories: 每个 item 对应的 category
///     scores: 每个 item 的分数
///
/// Returns:
///     (混排后的 item_ids, 混排后的 scores)
#[pyfunction]
pub fn category_interleave(
    item_ids: Vec<i64>,
    categories: Vec<i64>,
    scores: Vec<f64>,
) -> (Vec<i64>, Vec<f64>) {
    if item_ids.is_empty() {
        return (vec![], vec![]);
    }

    // 按类目分组，保持组内分数顺序
    let mut cat_buckets: AHashMap<i64, Vec<(i64, f64)>> = AHashMap::new();
    let mut cat_order: Vec<i64> = Vec::new(); // 类目出现顺序

    for i in 0..item_ids.len() {
        let cat = categories[i];
        let score = if i < scores.len() { scores[i] } else { 0.0 };

        if !cat_buckets.contains_key(&cat) {
            cat_order.push(cat);
        }
        cat_buckets.entry(cat).or_default().push((item_ids[i], score));
    }

    // 按每个桶的最高分排序类目顺序（最好的类目先出）
    cat_order.sort_by(|a, b| {
        let a_max = cat_buckets.get(a).and_then(|v| v.first()).map(|(_, s)| *s).unwrap_or(0.0);
        let b_max = cat_buckets.get(b).and_then(|v| v.first()).map(|(_, s)| *s).unwrap_or(0.0);
        b_max.partial_cmp(&a_max).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Round-robin：轮流从每个类目取一个
    let mut result_ids = Vec::with_capacity(item_ids.len());
    let mut result_scores = Vec::with_capacity(item_ids.len());
    let mut cursors: AHashMap<i64, usize> = AHashMap::new();
    let total = item_ids.len();

    while result_ids.len() < total {
        let mut added_any = false;
        for cat in &cat_order {
            let cursor = cursors.entry(*cat).or_insert(0);
            if let Some(bucket) = cat_buckets.get(cat) {
                if *cursor < bucket.len() {
                    let (id, score) = bucket[*cursor];
                    result_ids.push(id);
                    result_scores.push(score);
                    *cursor += 1;
                    added_any = true;
                }
            }
        }
        if !added_any {
            break;
        }
    }

    (result_ids, result_scores)
}

/// 计算类目覆盖率和多样性指标。
///
/// Args:
///     categories: 推荐结果中每个 item 的类目
///
/// Returns:
///     (类目数, 类目熵, 基尼系数)
#[pyfunction]
pub fn category_diversity_metrics(
    categories: Vec<i64>,
) -> (usize, f64, f64) {
    if categories.is_empty() {
        return (0, 0.0, 0.0);
    }

    let n = categories.len() as f64;
    let mut cat_count: AHashMap<i64, usize> = AHashMap::new();
    for cat in &categories {
        *cat_count.entry(*cat).or_insert(0) += 1;
    }

    let num_categories = cat_count.len();

    // Shannon Entropy
    let entropy: f64 = cat_count.values().map(|&c| {
        let p = c as f64 / n;
        if p > 0.0 { -p * p.ln() } else { 0.0 }
    }).sum();

    // Gini Coefficient
    let mut counts: Vec<f64> = cat_count.values().map(|&c| c as f64).collect();
    counts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = counts.len() as f64;
    let mean = n / k;
    let gini = if mean > 0.0 {
        let sum_abs_diff: f64 = counts.iter().enumerate().map(|(i, &c)| {
            (2.0 * (i as f64 + 1.0) - k - 1.0) * c
        }).sum();
        sum_abs_diff / (k * k * mean)
    } else {
        0.0
    };

    (num_categories, entropy, gini)
}
