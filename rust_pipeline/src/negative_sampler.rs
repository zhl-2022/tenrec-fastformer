use pyo3::prelude::*;
use ahash::AHashSet;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};

/// 高效负采样器 — 针对 Tenrec 230 万 item 池的大规模负采样
///
/// 当前 Python 在 230 万 items 中做 127 负样本采样较慢（尤其是要排除已交互 items）。
/// Rust 实现使用 AHashSet 做 O(1) 排除检查，rayon 做 batch 并行，
/// 预期加速 5-10x。

/// 简易伪随机数生成器 (xoshiro256**)
/// 不引入额外依赖，性能远超 Python random
struct FastRng {
    s: [u64; 4],
}

impl FastRng {
    fn new(seed: u64) -> Self {
        // SplitMix64 初始化
        let mut state = seed;
        let mut s = [0u64; 4];
        for i in 0..4 {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            s[i] = z ^ (z >> 31);
        }
        FastRng { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// 生成 [0, bound) 范围内的随机数
    #[inline]
    fn next_bounded(&mut self, bound: u64) -> u64 {
        // Lemire's fast range reduction
        let r = self.next_u64();
        ((r as u128 * bound as u128) >> 64) as u64
    }
}

/// 为单个用户采样负样本。
///
/// 从 item_pool_size 个 items 中随机采样 num_negatives 个
/// 不在 positive_items 中的 item。
///
/// Args:
///     positive_items: 用户已交互的 item ID 列表
///     item_pool_size: item 候选池大小 (e.g., 2,310,087)
///     num_negatives: 需要采样的负样本数量 (e.g., 127)
///     seed: 随机种子
///
/// Returns:
///     负样本 item ID 列表
#[pyfunction]
#[pyo3(signature = (positive_items, item_pool_size, num_negatives, seed=42))]
pub fn negative_sample(
    positive_items: Vec<i64>,
    item_pool_size: i64,
    num_negatives: i64,
    seed: u64,
) -> Vec<i64> {
    let pos_set: AHashSet<i64> = positive_items.into_iter().collect();
    let pool = item_pool_size as u64;
    let n_neg = num_negatives as usize;
    let mut rng = FastRng::new(seed);
    let mut negatives = Vec::with_capacity(n_neg);

    // 最大尝试次数 = 10 * num_negatives，防止无限循环
    let max_attempts = n_neg * 10;
    let mut attempts = 0;

    while negatives.len() < n_neg && attempts < max_attempts {
        let item = rng.next_bounded(pool) as i64;
        if !pos_set.contains(&item) {
            negatives.push(item);
        }
        attempts += 1;
    }

    negatives
}

/// 批量负采样 — 为多个用户并行采样负样本。
///
/// 使用 rayon 并行处理多个用户的负采样请求。
/// 每个用户有各自的 positive_items 集合。
///
/// Args:
///     batch_positive_items: 每个用户已交互的 item ID 列表（二维数组）
///     item_pool_size: item 候选池大小
///     num_negatives: 每个用户需要的负样本数
///     seed: 基础随机种子（每个用户会额外混入用户索引）
///
/// Returns:
///     每个用户的负样本列表（二维数组）
#[pyfunction]
#[pyo3(signature = (batch_positive_items, item_pool_size, num_negatives, seed=42))]
pub fn batch_negative_sample(
    batch_positive_items: Vec<Vec<i64>>,
    item_pool_size: i64,
    num_negatives: i64,
    seed: u64,
) -> Vec<Vec<i64>> {
    let pool = item_pool_size as u64;
    let n_neg = num_negatives as usize;

    batch_positive_items
        .into_par_iter()
        .enumerate()
        .map(|(idx, pos_items)| {
            let pos_set: AHashSet<i64> = pos_items.into_iter().collect();
            // 每个用户用不同的种子，保证独立性
            let user_seed = seed.wrapping_add(idx as u64).wrapping_mul(0x517CC1B727220A95);
            let mut rng = FastRng::new(user_seed);
            let mut negatives = Vec::with_capacity(n_neg);

            let max_attempts = n_neg * 10;
            let mut attempts = 0;

            while negatives.len() < n_neg && attempts < max_attempts {
                let item = rng.next_bounded(pool) as i64;
                if !pos_set.contains(&item) {
                    negatives.push(item);
                }
                attempts += 1;
            }

            negatives
        })
        .collect()
}

/// 带频率偏置的负采样（流行度采样）。
///
/// 热门 item 更容易被采为负样本（模拟真实分布），
/// 等价于 word2vec 中的 unigram^0.75 采样。
///
/// Args:
///     positive_items: 用户已交互 items
///     item_frequencies: 每个 item 的频率 (长度 = item_pool_size)
///     num_negatives: 负样本数
///     power: 频率的指数 (默认 0.75，同 word2vec)
///     seed: 随机种子
///
/// Returns:
///     负样本 item ID 列表
#[pyfunction]
#[pyo3(signature = (positive_items, item_frequencies, num_negatives, power=0.75, seed=42))]
pub fn popularity_negative_sample(
    positive_items: Vec<i64>,
    item_frequencies: Vec<f64>,
    num_negatives: i64,
    power: f64,
    seed: u64,
) -> Vec<i64> {
    let pos_set: AHashSet<i64> = positive_items.into_iter().collect();
    let n_neg = num_negatives as usize;

    // 构建 alias table 的简化版本：前缀和采样
    let powered: Vec<f64> = item_frequencies.iter().map(|f| f.powf(power)).collect();
    let total: f64 = powered.iter().sum();

    if total <= 0.0 {
        return vec![];
    }

    // 构建 CDF
    let mut cdf = Vec::with_capacity(powered.len());
    let mut acc = 0.0;
    for p in &powered {
        acc += p / total;
        cdf.push(acc);
    }

    let mut rng = FastRng::new(seed);
    let mut negatives = Vec::with_capacity(n_neg);
    let max_attempts = n_neg * 10;
    let mut attempts = 0;

    while negatives.len() < n_neg && attempts < max_attempts {
        // 生成 [0, 1) 的随机浮点数
        let r = (rng.next_u64() as f64) / (u64::MAX as f64);
        // 二分查找 CDF
        let item = match cdf.binary_search_by(|v| v.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(i) => i as i64,
            Err(i) => i as i64,
        };

        if item < item_frequencies.len() as i64 && !pos_set.contains(&item) {
            negatives.push(item);
        }
        attempts += 1;
    }

    negatives
}
