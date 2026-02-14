use pyo3::prelude::*;
use ahash::AHashMap;
use rayon::prelude::*;

/// 用户画像构建 — 针对 Tenrec 的 gender/age/video_category 字段
///
/// Tenrec 提供用户人口统计信息和视频类目，结合行为数据可以构建
/// 丰富的用户画像特征，补充到模型 Side Information 中。

/// 从用户的历史交互中构建类目偏好分布。
///
/// 统计用户在各类目上的 click/like/share/follow 行为，
/// 归一化为概率分布作为用户画像特征。
///
/// Args:
///     categories: 用户交互过的 item 类目列表
///     clicks: 每条交互的 click 标签 (0/1)
///     likes: 每条交互的 like 标签 (0/1)
///     num_categories: 类目总数（用于构建固定维度的特征向量）
///
/// Returns:
///     类目偏好分布向量，长度 = num_categories
///     每个位置的值 = 该类目的加权交互概率
#[pyfunction]
#[pyo3(signature = (categories, clicks, likes, num_categories=100))]
pub fn build_category_preference(
    categories: Vec<i64>,
    clicks: Vec<i64>,
    likes: Vec<i64>,
    num_categories: usize,
) -> Vec<f64> {
    let mut cat_score: AHashMap<i64, f64> = AHashMap::new();

    for i in 0..categories.len() {
        let cat = categories[i];
        let click = if i < clicks.len() { clicks[i] as f64 } else { 0.0 };
        let like = if i < likes.len() { likes[i] as f64 } else { 0.0 };
        // like 权重更高（like 是更强的正信号）
        let score = click * 1.0 + like * 2.0;
        *cat_score.entry(cat).or_insert(0.0) += score;
    }

    // 归一化
    let total: f64 = cat_score.values().sum();
    let mut preference = vec![0.0; num_categories];

    if total > 0.0 {
        for (cat, score) in &cat_score {
            let idx = *cat as usize;
            if idx < num_categories {
                preference[idx] = score / total;
            }
        }
    }

    preference
}

/// 批量构建用户类目偏好。
///
/// 对多个用户并行构建类目偏好分布。
///
/// Args:
///     batch_categories: 每个用户的类目列表
///     batch_clicks: 每个用户的 click 列表
///     batch_likes: 每个用户的 like 列表
///     num_categories: 类目总数
///
/// Returns:
///     每个用户的类目偏好向量 (二维数组)
#[pyfunction]
#[pyo3(signature = (batch_categories, batch_clicks, batch_likes, num_categories=100))]
pub fn build_category_preference_batch(
    batch_categories: Vec<Vec<i64>>,
    batch_clicks: Vec<Vec<i64>>,
    batch_likes: Vec<Vec<i64>>,
    num_categories: usize,
) -> Vec<Vec<f64>> {
    batch_categories
        .into_par_iter()
        .zip(batch_clicks.into_par_iter())
        .zip(batch_likes.into_par_iter())
        .map(|((cats, clicks), likes)| {
            let mut cat_score: AHashMap<i64, f64> = AHashMap::new();
            for i in 0..cats.len() {
                let cat = cats[i];
                let c = if i < clicks.len() { clicks[i] as f64 } else { 0.0 };
                let l = if i < likes.len() { likes[i] as f64 } else { 0.0 };
                *cat_score.entry(cat).or_insert(0.0) += c * 1.0 + l * 2.0;
            }
            let total: f64 = cat_score.values().sum();
            let mut pref = vec![0.0; num_categories];
            if total > 0.0 {
                for (cat, score) in &cat_score {
                    let idx = *cat as usize;
                    if idx < num_categories {
                        pref[idx] = score / total;
                    }
                }
            }
            pref
        })
        .collect()
}

/// 用户活跃度分级。
///
/// 根据用户的交互次数、行为多样性、时间跨度等维度，
/// 将用户分为不同活跃等级（冷启动/低活/中活/高活/超活）。
///
/// Args:
///     interaction_count: 总交互次数
///     click_count: 点击次数
///     like_count: 点赞次数
///     share_count: 分享次数
///     follow_count: 关注次数
///     time_span_days: 活跃天数跨度
///
/// Returns:
///     (活跃等级 0-4, 活跃度分数 0.0-1.0)
///     0=冷启动 (<5次), 1=低活, 2=中活, 3=高活, 4=超活
#[pyfunction]
pub fn compute_user_activity_level(
    interaction_count: i64,
    click_count: i64,
    like_count: i64,
    share_count: i64,
    follow_count: i64,
    time_span_days: f64,
) -> (i64, f64) {
    // 行为多样性: 有多少种不同的行为
    let diversity = (click_count > 0) as i64
        + (like_count > 0) as i64
        + (share_count > 0) as i64
        + (follow_count > 0) as i64;

    // 日均互动
    let daily_rate = if time_span_days > 0.0 {
        interaction_count as f64 / time_span_days
    } else {
        interaction_count as f64
    };

    // 综合活跃度分数 (0~1)
    let volume_score = (interaction_count as f64 / 100.0).min(1.0);
    let diversity_score = diversity as f64 / 4.0;
    let frequency_score = (daily_rate / 10.0).min(1.0);

    let activity_score = volume_score * 0.4 + diversity_score * 0.3 + frequency_score * 0.3;

    let level = if interaction_count < 5 {
        0  // 冷启动
    } else if activity_score < 0.2 {
        1  // 低活
    } else if activity_score < 0.5 {
        2  // 中活
    } else if activity_score < 0.8 {
        3  // 高活
    } else {
        4  // 超活
    };

    (level, activity_score)
}

/// 批量计算用户活跃度。
#[pyfunction]
pub fn compute_user_activity_level_batch(
    interaction_counts: Vec<i64>,
    click_counts: Vec<i64>,
    like_counts: Vec<i64>,
    share_counts: Vec<i64>,
    follow_counts: Vec<i64>,
    time_span_days: Vec<f64>,
) -> (Vec<i64>, Vec<f64>) {
    let n = interaction_counts.len();
    let mut levels = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        let interactions = interaction_counts[i];
        let clicks = if i < click_counts.len() { click_counts[i] } else { 0 };
        let likes = if i < like_counts.len() { like_counts[i] } else { 0 };
        let shares = if i < share_counts.len() { share_counts[i] } else { 0 };
        let follows = if i < follow_counts.len() { follow_counts[i] } else { 0 };
        let days = if i < time_span_days.len() { time_span_days[i] } else { 1.0 };

        let (level, score) = compute_user_activity_level_inner(
            interactions, clicks, likes, shares, follows, days,
        );
        levels.push(level);
        scores.push(score);
    }

    (levels, scores)
}

fn compute_user_activity_level_inner(
    interaction_count: i64,
    click_count: i64,
    like_count: i64,
    share_count: i64,
    follow_count: i64,
    time_span_days: f64,
) -> (i64, f64) {
    let diversity = (click_count > 0) as i64
        + (like_count > 0) as i64
        + (share_count > 0) as i64
        + (follow_count > 0) as i64;

    let daily_rate = if time_span_days > 0.0 {
        interaction_count as f64 / time_span_days
    } else {
        interaction_count as f64
    };

    let volume_score = (interaction_count as f64 / 100.0).min(1.0);
    let diversity_score = diversity as f64 / 4.0;
    let frequency_score = (daily_rate / 10.0).min(1.0);
    let activity_score = volume_score * 0.4 + diversity_score * 0.3 + frequency_score * 0.3;

    let level = if interaction_count < 5 {
        0
    } else if activity_score < 0.2 {
        1
    } else if activity_score < 0.5 {
        2
    } else if activity_score < 0.8 {
        3
    } else {
        4
    };

    (level, activity_score)
}
