"""
Rust-Accelerated Pipeline Components
======================================
Provides a Python API over the `fast_pipeline` Rust module (PyO3).

If `fast_pipeline` is not installed, all functions gracefully fall back
to pure-Python implementations (slower but functionally identical).

Usage:
    from tenrec_adapter.rust_components import post_store, filters, scorers, features

    # PostStore — ~100x faster than Python dict for concurrent access
    store = post_store.create(retention_seconds=86400)
    store.insert(user_id=1, item_id=42, timestamp=1700000000, item_type=0)
    history = store.get_user_item_ids(user_id=1, max_items=20)

    # Filters — ~50x faster via AHashSet
    result = filters.batch_filter(
        item_ids=[1,2,3], timestamps=[...], author_ids=[...],
        user_id=1, blocked_authors=[5], seen_ids=[2],
    )

    # Scorers — ~20x faster batch scoring
    scores = scorers.weighted_score_batch(
        [{"click": 0.8, "like": 0.3}], {"click": 1.0, "like": 2.0}
    )

    # Feature Engine — hash embeddings without vocabulary
    indices = features.hash_embedding_indices(["user_42", "item_1337"], 1_000_000)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Try importing Rust module; fall back to Python if not available
# --------------------------------------------------------------------------- #

try:
    import fast_pipeline as _rust

    RUST_AVAILABLE = True
    logger.info("fast_pipeline (Rust) loaded — using accelerated components")
except ImportError:
    _rust = None  # type: ignore
    RUST_AVAILABLE = False
    logger.info("fast_pipeline (Rust) not found — using Python fallback")


def is_rust_available() -> bool:
    """Check if the Rust module is installed."""
    return RUST_AVAILABLE


# =========================================================================== #
#  PostStore (Thread-safe user→item index)
# =========================================================================== #

class PostStoreWrapper:
    """Unified API for the Rust PostStore with Python fallback."""

    def __init__(self, retention_seconds: int = 172800, max_per_user: int = 500):
        if RUST_AVAILABLE:
            self._store = _rust.PostStore(
                retention_seconds=retention_seconds,
                max_per_user=max_per_user,
            )
            self._is_rust = True
        else:
            from collections import OrderedDict
            self._data: Dict[int, OrderedDict] = {}
            self._max_per_user = max_per_user
            self._is_rust = False

    def insert(self, user_id: int, item_id: int, timestamp: int, item_type: int = 0):
        if self._is_rust:
            self._store.insert(user_id, item_id, timestamp, item_type)
        else:
            if user_id not in self._data:
                from collections import OrderedDict
                self._data[user_id] = OrderedDict()
            items = self._data[user_id]
            if item_id not in items:
                items[item_id] = (timestamp, item_type)
                while len(items) > self._max_per_user:
                    items.popitem(last=False)

    def insert_batch(
        self,
        user_ids: List[int],
        item_ids: List[int],
        timestamps: List[int],
        item_types: List[int],
    ):
        if self._is_rust:
            self._store.insert_batch(user_ids, item_ids, timestamps, item_types)
        else:
            for uid, iid, ts, it in zip(user_ids, item_ids, timestamps, item_types):
                self.insert(uid, iid, ts, it)

    def get_user_item_ids(self, user_id: int, max_items: int = 20) -> List[int]:
        if self._is_rust:
            return self._store.get_user_item_ids(user_id, max_items)
        items = self._data.get(user_id, {})
        return list(items.keys())[-max_items:][::-1]

    def stats(self) -> Tuple[int, int, int]:
        if self._is_rust:
            return self._store.stats()
        total_items = sum(len(v) for v in self._data.values())
        return (len(self._data), total_items, 0)

    def __repr__(self) -> str:
        users, items, deleted = self.stats()
        engine = "Rust" if self._is_rust else "Python"
        return f"PostStore({engine}, users={users}, items={items})"


# =========================================================================== #
#  Filters
# =========================================================================== #

class FiltersAPI:
    """Unified filter API with Rust/Python fallback."""

    @staticmethod
    def dedup_filter(item_ids: List[int]) -> Dict[str, List[int]]:
        if RUST_AVAILABLE:
            r = _rust.dedup_filter(item_ids)
            return {"kept": r.kept, "removed": r.removed}
        seen = set()
        kept, removed = [], []
        for i in item_ids:
            if i in seen:
                removed.append(i)
            else:
                seen.add(i)
                kept.append(i)
        return {"kept": kept, "removed": removed}

    @staticmethod
    def age_filter(
        item_ids: List[int],
        timestamps: List[int],
        max_age_seconds: int,
        current_time: int,
    ) -> Dict[str, List[int]]:
        if RUST_AVAILABLE:
            r = _rust.age_filter(item_ids, timestamps, max_age_seconds, current_time)
            return {"kept": r.kept, "removed": r.removed}
        cutoff = current_time - max_age_seconds
        kept, removed = [], []
        for i, ts in zip(item_ids, timestamps):
            if cutoff <= ts <= current_time:
                kept.append(i)
            else:
                removed.append(i)
        return {"kept": kept, "removed": removed}

    @staticmethod
    def blocked_author_filter(
        item_ids: List[int],
        author_ids: List[int],
        blocked_authors: List[int],
    ) -> Dict[str, List[int]]:
        if RUST_AVAILABLE:
            r = _rust.blocked_author_filter(item_ids, author_ids, blocked_authors)
            return {"kept": r.kept, "removed": r.removed}
        blocked = set(blocked_authors)
        kept, removed = [], []
        for i, a in zip(item_ids, author_ids):
            if a in blocked:
                removed.append(i)
            else:
                kept.append(i)
        return {"kept": kept, "removed": removed}

    @staticmethod
    def seen_items_filter(
        item_ids: List[int], seen_ids: List[int],
    ) -> Dict[str, List[int]]:
        if RUST_AVAILABLE:
            r = _rust.seen_items_filter(item_ids, seen_ids)
            return {"kept": r.kept, "removed": r.removed}
        seen = set(seen_ids)
        kept, removed = [], []
        for i in item_ids:
            if i in seen:
                removed.append(i)
            else:
                kept.append(i)
        return {"kept": kept, "removed": removed}

    @staticmethod
    def batch_filter(
        item_ids: List[int],
        timestamps: List[int],
        author_ids: List[int],
        user_id: int,
        blocked_authors: List[int],
        seen_ids: List[int],
        max_age_seconds: int = 172800,
        current_time: int = 0,
    ) -> Dict[str, List[int]]:
        """Run ALL filters in a single call (most efficient)."""
        if RUST_AVAILABLE:
            r = _rust.batch_filter(
                item_ids, timestamps, author_ids, user_id,
                blocked_authors, seen_ids, max_age_seconds, current_time,
            )
            return {"kept": r.kept, "removed": r.removed}
        # Python fallback: chain filters manually
        import time as _time
        now = current_time or int(_time.time())
        cutoff = now - max_age_seconds
        blocked = set(blocked_authors)
        seen = set(seen_ids)
        dedup_seen = set()
        kept, removed = [], []
        for idx in range(len(item_ids)):
            i, ts, a = item_ids[idx], timestamps[idx], author_ids[idx]
            if i in dedup_seen or ts < cutoff or a in blocked or a == user_id or i in seen:
                removed.append(i)
            else:
                dedup_seen.add(i)
                kept.append(i)
        return {"kept": kept, "removed": removed}


# =========================================================================== #
#  Scorers
# =========================================================================== #

class ScorersAPI:
    """Unified scorer API with Rust/Python fallback."""

    @staticmethod
    def weighted_score_batch(
        candidates_probs: List[Dict[str, float]],
        weights: Dict[str, float],
    ) -> List[float]:
        if RUST_AVAILABLE:
            return _rust.weighted_score_batch(candidates_probs, weights)
        return [
            sum(weights.get(a, 0) * p for a, p in probs.items())
            for probs in candidates_probs
        ]

    @staticmethod
    def author_diversity_rerank(
        item_ids: List[int],
        scores: List[float],
        author_ids: List[int],
        decay_factor: float = 0.7,
    ) -> Tuple[List[int], List[float]]:
        if RUST_AVAILABLE:
            return _rust.author_diversity_rerank(
                item_ids, scores, author_ids, decay_factor,
            )
        counts: Dict[int, int] = {}
        adjusted = []
        for i, (iid, s, a) in enumerate(zip(item_ids, scores, author_ids)):
            n = counts.get(a, 0)
            adjusted.append((iid, s * (decay_factor ** n)))
            counts[a] = n + 1
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in adjusted], [x[1] for x in adjusted]


# =========================================================================== #
#  Feature Engine
# =========================================================================== #

class FeaturesAPI:
    """Unified feature engineering API with Rust/Python fallback."""

    @staticmethod
    def hash_embedding_indices(
        feature_values: List[str], num_buckets: int,
    ) -> List[int]:
        if RUST_AVAILABLE:
            return _rust.hash_embedding_indices(feature_values, num_buckets)
        return [hash(v) % num_buckets for v in feature_values]

    @staticmethod
    def hash_embedding_indices_batch(
        feature_columns: List[List[str]],
        num_buckets: List[int],
    ) -> List[List[int]]:
        if RUST_AVAILABLE:
            return _rust.hash_embedding_indices_batch(feature_columns, num_buckets)
        return [
            [hash(v) % b for v in col]
            for col, b in zip(feature_columns, num_buckets)
        ]

    @staticmethod
    def normalize_features(values: List[float]) -> List[float]:
        if RUST_AVAILABLE:
            return _rust.normalize_features(values)
        if not values:
            return values
        mn, mx = min(values), max(values)
        rng = mx - mn
        if rng < 1e-12:
            return [0.5] * len(values)
        return [(v - mn) / rng for v in values]


# =========================================================================== #
#  Multi-Task Scorer (Tenrec 4-action: click/like/share/follow)
# =========================================================================== #

class MultiTaskScorersAPI:
    """Tenrec multi-action engagement scoring with Rust/Python fallback."""

    @staticmethod
    def multi_task_engagement_score(
        click_prob: float, like_prob: float, share_prob: float, follow_prob: float,
        weights: Dict[str, float],
    ) -> float:
        if RUST_AVAILABLE:
            return _rust.multi_task_engagement_score(
                click_prob, like_prob, share_prob, follow_prob, weights,
            )
        w = weights
        return (w.get("click", 0.4) * click_prob + w.get("like", 0.3) * like_prob
                + w.get("share", 0.2) * share_prob + w.get("follow", 0.1) * follow_prob)

    @staticmethod
    def multi_task_engagement_score_batch(
        click_probs: List[float], like_probs: List[float],
        share_probs: List[float], follow_probs: List[float],
        weights: Dict[str, float],
    ) -> List[float]:
        if RUST_AVAILABLE:
            return _rust.multi_task_engagement_score_batch(
                click_probs, like_probs, share_probs, follow_probs, weights,
            )
        w = weights
        wc, wl, ws, wf = w.get("click", 0.4), w.get("like", 0.3), w.get("share", 0.2), w.get("follow", 0.1)
        return [wc * c + wl * l + ws * s + wf * f
                for c, l, s, f in zip(click_probs, like_probs, share_probs, follow_probs)]

    @staticmethod
    def esmm_cvr_correction(
        click_probs: List[float], like_given_click_probs: List[float],
    ) -> List[float]:
        if RUST_AVAILABLE:
            return _rust.esmm_cvr_correction(click_probs, like_given_click_probs)
        return [c * l for c, l in zip(click_probs, like_given_click_probs)]

    @staticmethod
    def pareto_rerank(
        item_ids: List[int], primary_scores: List[float],
        secondary_scores: List[float], lambda_factor: float = 0.3,
    ) -> Tuple[List[int], List[float]]:
        if RUST_AVAILABLE:
            return _rust.pareto_rerank(item_ids, primary_scores, secondary_scores, lambda_factor)
        combined = [(iid, (1 - lambda_factor) * p + lambda_factor * s)
                     for iid, p, s in zip(item_ids, primary_scores, secondary_scores)]
        combined.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in combined], [x[1] for x in combined]


# =========================================================================== #
#  Negative Sampler (2.3M item pool)
# =========================================================================== #

class NegativeSamplerAPI:
    """High-performance negative sampling with Rust/Python fallback."""

    @staticmethod
    def negative_sample(
        positive_items: List[int], item_pool_size: int,
        num_negatives: int, seed: int = 42,
    ) -> List[int]:
        if RUST_AVAILABLE:
            return _rust.negative_sample(positive_items, item_pool_size, num_negatives, seed)
        import random
        rng = random.Random(seed)
        pos_set = set(positive_items)
        negs = []
        for _ in range(num_negatives * 10):
            item = rng.randint(0, item_pool_size - 1)
            if item not in pos_set:
                negs.append(item)
                if len(negs) >= num_negatives:
                    break
        return negs

    @staticmethod
    def batch_negative_sample(
        batch_positive_items: List[List[int]], item_pool_size: int,
        num_negatives: int, seed: int = 42,
    ) -> List[List[int]]:
        if RUST_AVAILABLE:
            return _rust.batch_negative_sample(
                batch_positive_items, item_pool_size, num_negatives, seed,
            )
        import random
        results = []
        for idx, pos_items in enumerate(batch_positive_items):
            rng = random.Random(seed + idx)
            pos_set = set(pos_items)
            negs = []
            for _ in range(num_negatives * 10):
                item = rng.randint(0, item_pool_size - 1)
                if item not in pos_set:
                    negs.append(item)
                    if len(negs) >= num_negatives:
                        break
            results.append(negs)
        return results


# =========================================================================== #
#  Category Filter (video_category diversity)
# =========================================================================== #

class CategoryFiltersAPI:
    """Tenrec video_category diversity controls with Rust/Python fallback."""

    @staticmethod
    def category_diversity_filter(
        item_ids: List[int], categories: List[int], max_per_category: int = 3,
    ) -> Dict[str, Any]:
        if RUST_AVAILABLE:
            r = _rust.category_diversity_filter(item_ids, categories, max_per_category)
            return {"kept": r.kept, "removed": r.removed, "category_counts": r.category_counts}
        counts: Dict[int, int] = {}
        kept, removed = [], []
        for iid, cat in zip(item_ids, categories):
            c = counts.get(cat, 0)
            if c < max_per_category:
                kept.append(iid)
                counts[cat] = c + 1
            else:
                removed.append(iid)
        return {"kept": kept, "removed": removed,
                "category_counts": list(counts.items())}

    @staticmethod
    def category_interleave(
        item_ids: List[int], categories: List[int], scores: List[float],
    ) -> Tuple[List[int], List[float]]:
        if RUST_AVAILABLE:
            return _rust.category_interleave(item_ids, categories, scores)
        # Simple Python round-robin
        from collections import OrderedDict
        buckets: Dict[int, list] = OrderedDict()
        for iid, cat, s in zip(item_ids, categories, scores):
            buckets.setdefault(cat, []).append((iid, s))
        result_ids, result_scores = [], []
        while any(buckets.values()):
            for cat in list(buckets.keys()):
                if buckets[cat]:
                    iid, s = buckets[cat].pop(0)
                    result_ids.append(iid)
                    result_scores.append(s)
        return result_ids, result_scores

    @staticmethod
    def category_diversity_metrics(categories: List[int]) -> Tuple[int, float, float]:
        if RUST_AVAILABLE:
            return _rust.category_diversity_metrics(categories)
        import math
        if not categories:
            return (0, 0.0, 0.0)
        counts: Dict[int, int] = {}
        for c in categories:
            counts[c] = counts.get(c, 0) + 1
        n = len(categories)
        entropy = -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)
        return (len(counts), entropy, 0.0)


# =========================================================================== #
#  User Profile (behavior + demographics)
# =========================================================================== #

class UserProfileAPI:
    """User profiling for Tenrec demographics with Rust/Python fallback."""

    @staticmethod
    def build_category_preference(
        categories: List[int], clicks: List[int], likes: List[int],
        num_categories: int = 100,
    ) -> List[float]:
        if RUST_AVAILABLE:
            return _rust.build_category_preference(categories, clicks, likes, num_categories)
        scores: Dict[int, float] = {}
        for i, cat in enumerate(categories):
            c = clicks[i] if i < len(clicks) else 0
            l = likes[i] if i < len(likes) else 0
            scores[cat] = scores.get(cat, 0.0) + c * 1.0 + l * 2.0
        total = sum(scores.values())
        pref = [0.0] * num_categories
        if total > 0:
            for cat, v in scores.items():
                if 0 <= cat < num_categories:
                    pref[cat] = v / total
        return pref

    @staticmethod
    def compute_user_activity_level(
        interaction_count: int, click_count: int, like_count: int,
        share_count: int, follow_count: int, time_span_days: float,
    ) -> Tuple[int, float]:
        if RUST_AVAILABLE:
            return _rust.compute_user_activity_level(
                interaction_count, click_count, like_count,
                share_count, follow_count, time_span_days,
            )
        diversity = sum(1 for x in [click_count, like_count, share_count, follow_count] if x > 0)
        daily_rate = interaction_count / max(time_span_days, 1.0)
        score = (min(interaction_count / 100.0, 1.0) * 0.4
                 + diversity / 4.0 * 0.3
                 + min(daily_rate / 10.0, 1.0) * 0.3)
        if interaction_count < 5: level = 0
        elif score < 0.2: level = 1
        elif score < 0.5: level = 2
        elif score < 0.8: level = 3
        else: level = 4
        return (level, score)


# =========================================================================== #
#  History Encoder (hist_1~hist_10)
# =========================================================================== #

class HistoryEncoderAPI:
    """Tenrec hist_1~hist_10 encoding with Rust/Python fallback."""

    @staticmethod
    def encode_history(hist_items: List[int], num_buckets: int = 50000) -> List[int]:
        if RUST_AVAILABLE:
            return _rust.encode_history(hist_items, num_buckets)
        return [0 if x <= 0 else (hash(str(x)) % num_buckets) + 1 for x in hist_items]

    @staticmethod
    def encode_history_batch(
        batch_hist: List[List[int]], num_buckets: int = 50000,
    ) -> List[List[int]]:
        if RUST_AVAILABLE:
            return _rust.encode_history_batch(batch_hist, num_buckets)
        return [[0 if x <= 0 else (hash(str(x)) % num_buckets) + 1 for x in hist]
                for hist in batch_hist]

    @staticmethod
    def position_decay_weights(seq_len: int, decay: float = 0.9) -> List[float]:
        if RUST_AVAILABLE:
            return _rust.position_decay_weights(seq_len, decay)
        return [decay ** i for i in range(seq_len)]

    @staticmethod
    def history_stats(hist_items: List[int]) -> Tuple[int, float, int]:
        if RUST_AVAILABLE:
            return _rust.history_stats(hist_items)
        valid = [x for x in hist_items if x > 0]
        if not valid:
            return (0, 0.0, 0)
        unique = len(set(valid)) / len(valid)
        return (len(valid), unique, max(valid) - min(valid))

    @staticmethod
    def history_overlap_check(
        candidate_ids: List[int], hist_items: List[int],
    ) -> List[bool]:
        if RUST_AVAILABLE:
            return _rust.history_overlap_check(candidate_ids, hist_items)
        hist_set = set(x for x in hist_items if x > 0)
        return [iid in hist_set for iid in candidate_ids]


# =========================================================================== #
#  Convenience singletons
# =========================================================================== #

post_store = PostStoreWrapper  # Call as post_store(retention=...) to instantiate
filters = FiltersAPI()
scorers = ScorersAPI()
features = FeaturesAPI()
# Phase 3: Tenrec-specific
multi_task_scorers = MultiTaskScorersAPI()
negative_sampler = NegativeSamplerAPI()
category_filters = CategoryFiltersAPI()
user_profile = UserProfileAPI()
history_encoder = HistoryEncoderAPI()
