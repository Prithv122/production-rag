"""Retrieval metrics.

Written out rather than imported so the definitions are inspectable -- "recall@k"
means several different things in the wild, and a README number is worthless if
the reader has to guess which one. The definitions used here:

``recall@k``
    Fraction of a question's gold chunks that appear in the top *k*. With a
    single gold chunk this collapses to hit-rate@k, which is the common case in
    this corpus and is why hit-rate is reported separately rather than being
    quietly relabelled as recall.

``nDCG@k``
    Binary-gain DCG over the top *k*, divided by the DCG of the best possible
    ordering given how many gold chunks exist. Bounded at 1.0 even when a
    question has more gold chunks than *k*.

``MRR``
    Reciprocal rank of the *first* gold chunk, 0 if none is retrieved.

All three take a ranked list of chunk ids and a set of gold ids, so they are
independent of which arm produced the ranking.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("recall is undefined with no gold chunks")
    hits = sum(1 for chunk_id in ranked[:k] if chunk_id in gold_set)
    return hits / len(gold_set)


def hit_rate_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """1.0 if any gold chunk is in the top k."""
    gold_set = set(gold)
    return 1.0 if any(chunk_id in gold_set for chunk_id in ranked[:k]) else 0.0


def dcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked[:k], start=1)
        if chunk_id in gold_set
    )


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("nDCG is undefined with no gold chunks")
    ideal_hits = min(len(gold_set), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(ranked, gold_set, k) / ideal


def reciprocal_rank(ranked: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class RetrievalScore:
    """Metrics for a single question."""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    hit_rate_at_5: float
    ndcg_at_10: float
    mrr: float

    @classmethod
    def evaluate(cls, ranked: Sequence[str], gold: Iterable[str]) -> RetrievalScore:
        gold_set = set(gold)
        return cls(
            recall_at_1=recall_at_k(ranked, gold_set, 1),
            recall_at_5=recall_at_k(ranked, gold_set, 5),
            recall_at_10=recall_at_k(ranked, gold_set, 10),
            hit_rate_at_5=hit_rate_at_k(ranked, gold_set, 5),
            ndcg_at_10=ndcg_at_k(ranked, gold_set, 10),
            mrr=reciprocal_rank(ranked, gold_set),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "hit_rate@5": self.hit_rate_at_5,
            "ndcg@10": self.ndcg_at_10,
            "mrr": self.mrr,
        }


def aggregate(scores: Sequence[RetrievalScore]) -> dict[str, float]:
    """Mean of each metric. Empty input gives an empty dict, not a crash."""
    if not scores:
        return {}
    keys = scores[0].as_dict().keys()
    rows = [s.as_dict() for s in scores]
    return {key: statistics.fmean(row[key] for row in rows) for key in keys}
