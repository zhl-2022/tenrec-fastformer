"""
Recommendation Pipeline Framework
==================================
Inspired by x-algorithm/candidate-pipeline (Rust) — Apache 2.0 License.

This module provides a composable, stage-based recommendation pipeline
that mirrors the Rust trait-based architecture:

    Source → Hydrator → Filter → Scorer → Selector → SideEffect

Each stage is an abstract base class. Concrete implementations are
plugged in at pipeline construction time. The pipeline executes stages
in order, with Source and Hydrator running in parallel where possible.

Key Design Decisions (borrowed from x-algorithm):
    1. Separation of pipeline execution from business logic
    2. Parallel execution of independent stages
    3. Graceful error handling — one component failure doesn't crash the pipeline
    4. Composable architecture — easy to add/remove stages

Usage:
    pipeline = RecommendationPipeline(
        sources=[ThunderSource(), PhoenixSource()],
        filters=[DedupFilter(), AgeFilter()],
        scorers=[ModelScorer(model), DiversityScorer()],
        selector=TopKSelector(k=50),
    )
    result = await pipeline.execute(query)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Core Types
# --------------------------------------------------------------------------- #

Q = TypeVar("Q")  # Query type
C = TypeVar("C")  # Candidate type


@dataclass
class Query:
    """Base query object carrying user context for the pipeline."""
    request_id: str
    user_id: int
    user_features: Dict[str, Any] = field(default_factory=dict)
    engagement_history: List[int] = field(default_factory=list)


@dataclass
class Candidate:
    """A recommendation candidate flowing through the pipeline."""
    item_id: int
    score: float = 0.0
    source: str = ""
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Result of a filter stage — split into kept and removed."""
    kept: List[Candidate] = field(default_factory=list)
    removed: List[Candidate] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Final output of the recommendation pipeline."""
    query: Query
    retrieved_candidates: List[Candidate]
    filtered_candidates: List[Candidate]
    selected_candidates: List[Candidate]
    latency_ms: float = 0.0
    stage_latencies: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Abstract Stages (≈ Rust traits)
# --------------------------------------------------------------------------- #

class Source(ABC):
    """Fetches candidate items from a data source.

    Equivalent to Rust trait:
        trait Source<Q, C>: Send + Sync {
            async fn get_candidates(&self, query: &Q) -> Result<Vec<C>>;
        }
    """
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def enable(self, query: Query) -> bool:
        """Whether this source is enabled for the given query."""
        return True

    @abstractmethod
    async def get_candidates(self, query: Query) -> List[Candidate]:
        ...


class Hydrator(ABC):
    """Enriches candidates with additional data (metadata, features, etc.).

    Equivalent to Rust trait:
        trait Hydrator<Q, C>: Send + Sync {
            async fn hydrate(&self, query: &Q, candidates: &[C]) -> Result<Vec<H>>;
            fn update_all(&self, candidates: &mut [C], hydrated: Vec<H>);
        }
    """
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def enable(self, query: Query) -> bool:
        return True

    @abstractmethod
    async def hydrate(
        self, query: Query, candidates: List[Candidate],
    ) -> List[Candidate]:
        ...


class Filter(ABC):
    """Removes candidates that don't meet criteria.

    Equivalent to Rust trait:
        trait Filter<Q, C>: Send + Sync {
            async fn filter(&self, query: &Q, candidates: Vec<C>) -> Result<FilterResult>;
        }
    """
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def enable(self, query: Query) -> bool:
        return True

    @abstractmethod
    async def filter(
        self, query: Query, candidates: List[Candidate],
    ) -> FilterResult:
        ...


class Scorer(ABC):
    """Assigns scores to candidates.

    Equivalent to Rust trait:
        trait Scorer<Q, C>: Send + Sync {
            async fn score(&self, query: &Q, candidates: &[C]) -> Result<Vec<f64>>;
        }
    """
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def enable(self, query: Query) -> bool:
        return True

    @abstractmethod
    async def score(
        self, query: Query, candidates: List[Candidate],
    ) -> List[float]:
        ...


class Selector(ABC):
    """Selects final candidates (e.g., Top-K by score).

    Equivalent to Rust trait:
        trait Selector<Q, C>: Send + Sync {
            fn select(&self, query: &Q, candidates: Vec<C>) -> Vec<C>;
        }
    """
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def enable(self, query: Query) -> bool:
        return True

    @abstractmethod
    def select(
        self, query: Query, candidates: List[Candidate],
    ) -> List[Candidate]:
        ...


class SideEffect(ABC):
    """Runs after selection — logging, caching, analytics.

    Fires asynchronously and does NOT block the response.
    Equivalent to Rust trait:
        trait SideEffect<Q, C>: Send + Sync {
            async fn run(&self, input: Arc<SideEffectInput<Q, C>>);
        }
    """
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def enable(self, query: Query) -> bool:
        return True

    @abstractmethod
    async def run(
        self, query: Query, selected: List[Candidate],
    ) -> None:
        ...


# --------------------------------------------------------------------------- #
#  Pipeline Executor
# --------------------------------------------------------------------------- #

class RecommendationPipeline:
    """Composable recommendation pipeline inspired by x-algorithm/candidate-pipeline.

    Executes stages in order:
        1. Sources     (parallel)  → gather candidates
        2. Hydrators   (parallel)  → enrich candidates
        3. Filters     (sequential) → remove bad candidates
        4. Scorers     (sequential) → assign scores
        5. Selector    (single)     → pick Top-K
        6. SideEffects (fire & forget) → logging, caching
    """

    def __init__(
        self,
        sources: Optional[List[Source]] = None,
        hydrators: Optional[List[Hydrator]] = None,
        filters: Optional[List[Filter]] = None,
        scorers: Optional[List[Scorer]] = None,
        selector: Optional[Selector] = None,
        side_effects: Optional[List[SideEffect]] = None,
        result_size: int = 50,
    ):
        self.sources = sources or []
        self.hydrators = hydrators or []
        self.filters = filters or []
        self.scorers = scorers or []
        self.selector = selector
        self.side_effects = side_effects or []
        self.result_size = result_size

    async def execute(self, query: Query) -> PipelineResult:
        """Run the full recommendation pipeline."""
        t_start = time.monotonic()
        stage_latencies: Dict[str, float] = {}
        all_removed: List[Candidate] = []

        # --- Stage 1: Sources (parallel) ---
        t0 = time.monotonic()
        candidates = await self._fetch_candidates(query)
        stage_latencies["sources"] = (time.monotonic() - t0) * 1000
        retrieved = list(candidates)

        # --- Stage 2: Hydrators (parallel) ---
        t0 = time.monotonic()
        candidates = await self._hydrate(query, candidates)
        stage_latencies["hydrators"] = (time.monotonic() - t0) * 1000

        # --- Stage 3: Filters (sequential) ---
        t0 = time.monotonic()
        candidates, removed = await self._filter(query, candidates)
        all_removed.extend(removed)
        stage_latencies["filters"] = (time.monotonic() - t0) * 1000

        # --- Stage 4: Scorers (sequential) ---
        t0 = time.monotonic()
        candidates = await self._score(query, candidates)
        stage_latencies["scorers"] = (time.monotonic() - t0) * 1000

        # --- Stage 5: Selector ---
        t0 = time.monotonic()
        selected = self._select(query, candidates)
        selected = selected[: self.result_size]
        stage_latencies["selector"] = (time.monotonic() - t0) * 1000

        # --- Stage 6: Side Effects (fire & forget) ---
        self._fire_side_effects(query, selected)

        total_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            "request_id=%s total_ms=%.1f candidates=%d selected=%d",
            query.request_id, total_ms, len(retrieved), len(selected),
        )

        return PipelineResult(
            query=query,
            retrieved_candidates=retrieved,
            filtered_candidates=all_removed,
            selected_candidates=selected,
            latency_ms=total_ms,
            stage_latencies=stage_latencies,
        )

    # ---- Internal stage runners ----

    async def _fetch_candidates(self, query: Query) -> List[Candidate]:
        """Run all sources in parallel."""
        enabled = [s for s in self.sources if s.enable(query)]
        tasks = [s.get_candidates(query) for s in enabled]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        collected: List[Candidate] = []
        for source, result in zip(enabled, results):
            if isinstance(result, Exception):
                logger.error(
                    "request_id=%s stage=Source component=%s failed: %s",
                    query.request_id, source.name, result,
                )
            else:
                logger.info(
                    "request_id=%s stage=Source component=%s fetched %d candidates",
                    query.request_id, source.name, len(result),
                )
                collected.extend(result)
        return collected

    async def _hydrate(
        self, query: Query, candidates: List[Candidate],
    ) -> List[Candidate]:
        """Run all hydrators in parallel."""
        enabled = [h for h in self.hydrators if h.enable(query)]
        tasks = [h.hydrate(query, candidates) for h in enabled]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for hydrator, result in zip(enabled, results):
            if isinstance(result, Exception):
                logger.error(
                    "request_id=%s stage=Hydrator component=%s failed: %s",
                    query.request_id, hydrator.name, result,
                )
            elif len(result) == len(candidates):
                candidates = result
            else:
                logger.warning(
                    "request_id=%s stage=Hydrator component=%s "
                    "length_mismatch expected=%d got=%d",
                    query.request_id, hydrator.name,
                    len(candidates), len(result),
                )
        return candidates

    async def _filter(
        self, query: Query, candidates: List[Candidate],
    ) -> tuple[List[Candidate], List[Candidate]]:
        """Run filters sequentially (order matters)."""
        all_removed: List[Candidate] = []
        for f in self.filters:
            if not f.enable(query):
                continue
            backup = list(candidates)
            try:
                result = await f.filter(query, candidates)
                candidates = result.kept
                all_removed.extend(result.removed)
            except Exception as e:
                logger.error(
                    "request_id=%s stage=Filter component=%s failed: %s",
                    query.request_id, f.name, e,
                )
                candidates = backup

        logger.info(
            "request_id=%s stage=Filter kept=%d removed=%d",
            query.request_id, len(candidates), len(all_removed),
        )
        return candidates, all_removed

    async def _score(
        self, query: Query, candidates: List[Candidate],
    ) -> List[Candidate]:
        """Run scorers sequentially."""
        for scorer in self.scorers:
            if not scorer.enable(query):
                continue
            try:
                scores = await scorer.score(query, candidates)
                if len(scores) == len(candidates):
                    for c, s in zip(candidates, scores):
                        c.score = s
                else:
                    logger.warning(
                        "request_id=%s stage=Scorer component=%s "
                        "length_mismatch expected=%d got=%d",
                        query.request_id, scorer.name,
                        len(candidates), len(scores),
                    )
            except Exception as e:
                logger.error(
                    "request_id=%s stage=Scorer component=%s failed: %s",
                    query.request_id, scorer.name, e,
                )
        return candidates

    def _select(
        self, query: Query, candidates: List[Candidate],
    ) -> List[Candidate]:
        """Run the selector."""
        if self.selector and self.selector.enable(query):
            return self.selector.select(query, candidates)
        # Default: sort by score descending
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def _fire_side_effects(
        self, query: Query, selected: List[Candidate],
    ) -> None:
        """Fire side effects without blocking (fire & forget)."""
        enabled = [se for se in self.side_effects if se.enable(query)]
        if not enabled:
            return
        for se in enabled:
            asyncio.ensure_future(self._run_side_effect(se, query, selected))

    @staticmethod
    async def _run_side_effect(
        se: SideEffect, query: Query, selected: List[Candidate],
    ) -> None:
        try:
            await se.run(query, selected)
        except Exception as e:
            logger.error(
                "request_id=%s stage=SideEffect component=%s failed: %s",
                query.request_id, se.name, e,
            )


# --------------------------------------------------------------------------- #
#  Built-in Implementations
# --------------------------------------------------------------------------- #

class TopKSelector(Selector):
    """Select top-K candidates by score (descending)."""

    def __init__(self, k: int = 50):
        self.k = k

    def select(
        self, query: Query, candidates: List[Candidate],
    ) -> List[Candidate]:
        return sorted(candidates, key=lambda c: c.score, reverse=True)[: self.k]


class DedupFilter(Filter):
    """Remove duplicate candidates (by item_id)."""

    async def filter(
        self, query: Query, candidates: List[Candidate],
    ) -> FilterResult:
        seen: set = set()
        kept, removed = [], []
        for c in candidates:
            if c.item_id in seen:
                removed.append(c)
            else:
                seen.add(c.item_id)
                kept.append(c)
        return FilterResult(kept=kept, removed=removed)


class WeightedScorer(Scorer):
    """Combine multiple prediction scores with configurable weights.

    Implements the x-algorithm weighted scoring:
        Final Score = Σ (weight_i × P(action_i))

    Positive actions (like, share) have positive weights.
    Negative actions (block, mute) have negative weights.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "click": 1.0,
            "like": 2.0,
            "share": 4.0,
            "follow": 3.0,
            "block": -10.0,
            "mute": -5.0,
        }

    async def score(
        self, query: Query, candidates: List[Candidate],
    ) -> List[float]:
        scores = []
        for c in candidates:
            total = 0.0
            preds = c.features.get("predictions", {})
            for action, weight in self.weights.items():
                total += weight * preds.get(action, 0.0)
            scores.append(total)
        return scores


class AuthorDiversityScorer(Scorer):
    """Attenuate repeated author scores to ensure feed diversity.

    Borrowed from x-algorithm/home-mixer/scorers/author_diversity_scorer.rs
    Each subsequent post from the same author gets a decayed score.
    """

    def __init__(self, decay_factor: float = 0.7):
        self.decay_factor = decay_factor

    async def score(
        self, query: Query, candidates: List[Candidate],
    ) -> List[float]:
        author_counts: Dict[int, int] = {}
        scores = []
        for c in candidates:
            author = c.features.get("author_id", c.item_id)
            count = author_counts.get(author, 0)
            multiplier = self.decay_factor ** count
            scores.append(c.score * multiplier)
            author_counts[author] = count + 1
        return scores
