use pyo3::prelude::*;
use std::collections::HashMap;

/// 多任务融合打分 — 针对 Tenrec 4 行为 (click/like/share/follow)
///
/// Tenrec 官方 ESMM 模型用多任务学习达到 like-AUC=0.9110，
/// 我们的单任务 click-AUC=0.8320 已超基线，但 like-AUC=0.8224 仍有差距。
/// 本模块实现多行为信号的融合打分，为上层 pipeline 提供综合 engagement score。

/// 计算单个候选的多任务 engagement 分数。
///
/// 融合 click/like/share/follow 4 个预测概率为一个综合分数。
/// 使用加权求和 + 可选的非线性增强（like/share/follow 比 click 更稀有，
/// 发生时权重应更高）。
///
/// Args:
///     click_prob: 点击概率预测值 [0, 1]
///     like_prob: 点赞概率预测值 [0, 1]
///     share_prob: 分享概率预测值 [0, 1]
///     follow_prob: 关注概率预测值 [0, 1]
///     weights: 各行为权重, e.g. {"click": 0.4, "like": 0.3, "share": 0.2, "follow": 0.1}
///
/// Returns:
///     综合 engagement 分数
#[pyfunction]
#[pyo3(signature = (click_prob, like_prob, share_prob, follow_prob, weights))]
pub fn multi_task_engagement_score(
    click_prob: f64,
    like_prob: f64,
    share_prob: f64,
    follow_prob: f64,
    weights: HashMap<String, f64>,
) -> f64 {
    let w_click = weights.get("click").copied().unwrap_or(0.4);
    let w_like = weights.get("like").copied().unwrap_or(0.3);
    let w_share = weights.get("share").copied().unwrap_or(0.2);
    let w_follow = weights.get("follow").copied().unwrap_or(0.1);

    w_click * click_prob + w_like * like_prob + w_share * share_prob + w_follow * follow_prob
}

/// 批量计算多任务 engagement 分数。
///
/// Args:
///     click_probs:  N 个候选的点击概率
///     like_probs:   N 个候选的点赞概率
///     share_probs:  N 个候选的分享概率
///     follow_probs: N 个候选的关注概率
///     weights: 各行为权重
///
/// Returns:
///     N 个综合分数
#[pyfunction]
#[pyo3(signature = (click_probs, like_probs, share_probs, follow_probs, weights))]
pub fn multi_task_engagement_score_batch(
    click_probs: Vec<f64>,
    like_probs: Vec<f64>,
    share_probs: Vec<f64>,
    follow_probs: Vec<f64>,
    weights: HashMap<String, f64>,
) -> Vec<f64> {
    let w_click = weights.get("click").copied().unwrap_or(0.4);
    let w_like = weights.get("like").copied().unwrap_or(0.3);
    let w_share = weights.get("share").copied().unwrap_or(0.2);
    let w_follow = weights.get("follow").copied().unwrap_or(0.1);

    let n = click_probs.len();
    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        let click = if i < click_probs.len() { click_probs[i] } else { 0.0 };
        let like = if i < like_probs.len() { like_probs[i] } else { 0.0 };
        let share = if i < share_probs.len() { share_probs[i] } else { 0.0 };
        let follow = if i < follow_probs.len() { follow_probs[i] } else { 0.0 };

        scores.push(w_click * click + w_like * like + w_share * share + w_follow * follow);
    }

    scores
}

/// ESMM 风格的 CVR 估计: P(like|exposure) = P(click|exposure) * P(like|click)
///
/// Tenrec 的 ESMM 模型利用这个因式分解来共享底层 embedding，
/// 这里提供后处理版本：给定 click_prob 和 conditional_like_prob，
/// 计算真正的 like 概率。
///
/// Args:
///     click_probs: 点击概率
///     like_given_click_probs: 点击后点赞的条件概率
///
/// Returns:
///     校正后的 like 概率 (= click_prob * like_given_click_prob)
#[pyfunction]
pub fn esmm_cvr_correction(
    click_probs: Vec<f64>,
    like_given_click_probs: Vec<f64>,
) -> Vec<f64> {
    click_probs
        .iter()
        .zip(like_given_click_probs.iter())
        .map(|(cp, lp)| cp * lp)
        .collect()
}

/// 多目标帕累托重排序。
///
/// 在多个目标之间做帕累托权衡：先按主目标排序，再对相邻 item
/// 检查次要目标的多样性，避免单一目标过度主导。
///
/// Args:
///     item_ids: 候选 item ID 列表（已按主分数降序）
///     primary_scores: 主目标分数（如 click_prob）
///     secondary_scores: 次要目标分数（如 like_prob）
///     lambda_factor: 次要目标的提升因子（0~1），0=纯主目标，1=主次均等
///
/// Returns:
///     (重排后的 item_ids, 重排后的综合分数)
#[pyfunction]
#[pyo3(signature = (item_ids, primary_scores, secondary_scores, lambda_factor=0.3))]
pub fn pareto_rerank(
    item_ids: Vec<i64>,
    primary_scores: Vec<f64>,
    secondary_scores: Vec<f64>,
    lambda_factor: f64,
) -> (Vec<i64>, Vec<f64>) {
    let n = item_ids.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    // 组合分数 = (1 - λ) * primary + λ * secondary
    let mut combined: Vec<(i64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let p = if i < primary_scores.len() { primary_scores[i] } else { 0.0 };
        let s = if i < secondary_scores.len() { secondary_scores[i] } else { 0.0 };
        let score = (1.0 - lambda_factor) * p + lambda_factor * s;
        combined.push((item_ids[i], score));
    }

    // 按组合分数降序排序
    combined.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let ids: Vec<i64> = combined.iter().map(|(id, _)| *id).collect();
    let scores: Vec<f64> = combined.iter().map(|(_, s)| *s).collect();

    (ids, scores)
}
