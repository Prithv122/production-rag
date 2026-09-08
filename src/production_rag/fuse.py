"""Combining a lexical ranking with a dense one.

The two arms are not on the same scale and cannot simply be added. BM25 emits
unbounded non-negative scores with a hard floor at zero -- a chunk sharing no
query term scores exactly 0 and is genuinely *absent* from the result. Cosine
similarity emits values in [-1, 1] and ranks the entire corpus every time: there
is no absent, only less similar. Summing them lets one arm's scale dominate for
reasons that have nothing to do with relevance, and lets dense search's opinion
about every irrelevant chunk outvote BM25's silence.

Two fusions are implemented so the choice can be measured:

``rrf``
    Reciprocal rank fusion. Discards scores entirely and uses only rank, which
    makes the scale problem disappear by construction. The usual default.

``score``
    Min-max normalise each arm over the union of its candidates, then take a
    weighted sum. Keeps magnitude information -- a chunk that BM25 scores far
    above its runner-up should arguably beat one that merely placed first by a
    hair -- at the cost of being sensitive to outliers in the pool.

Which wins is an empirical question, so both are evaluation arms rather than a
decision made here.
"""

from __future__ import annotations

from collections.abc import Sequence

Ranking = Sequence[tuple[str, float]]

RRF_K = 60
"""Rank offset from Cormack et al. (2009). Larger flattens the contribution
curve, so top ranks matter less relative to the tail."""


def reciprocal_rank_fusion(
    rankings: Sequence[Ranking],
    *,
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse by rank alone: `sum_i w_i / (k + rank_i)`, ranks starting at 1."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must be the same length")

    totals: dict[str, float] = {}
    for weight, ranking in zip(weights, rankings, strict=True):
        for rank, (chunk_id, _) in enumerate(ranking, start=1):
            totals[chunk_id] = totals.get(chunk_id, 0.0) + weight / (k + rank)
    return _sorted(totals)


def _min_max(ranking: Ranking) -> dict[str, float]:
    """Scale a ranking's scores into [0, 1] over its own candidate pool."""
    if not ranking:
        return {}
    scores = [s for _, s in ranking]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        # Every candidate scored the same; treat them as equally good rather
        # than dividing by zero.
        return {chunk_id: 1.0 for chunk_id, _ in ranking}
    return {chunk_id: (score - lo) / (hi - lo) for chunk_id, score in ranking}


def score_fusion(
    rankings: Sequence[Ranking],
    *,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse by weighted sum of per-arm min-max normalised scores.

    A chunk missing from one arm contributes 0 for that arm rather than being
    dropped, so a strong single-arm hit can still surface -- which is the whole
    point of running two arms.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must be the same length")

    totals: dict[str, float] = {}
    for weight, ranking in zip(weights, rankings, strict=True):
        for chunk_id, score in _min_max(ranking).items():
            totals[chunk_id] = totals.get(chunk_id, 0.0) + weight * score
    return _sorted(totals)


def _sorted(totals: dict[str, float]) -> list[tuple[str, float]]:
    """Descending by score, then by chunk_id so ties are deterministic.

    Determinism matters more than it looks: without it, two runs of the same
    evaluation can report different recall@k purely from dict ordering, and the
    published number stops being reproducible.
    """
    return sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))


FUSIONS = {
    "rrf": reciprocal_rank_fusion,
    "score": score_fusion,
}


def fuse(
    rankings: Sequence[Ranking],
    method: str = "rrf",
    *,
    k: int = 10,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse `rankings` with the named method and truncate to `k`."""
    try:
        fusion = FUSIONS[method]
    except KeyError:
        raise ValueError(f"unknown fusion {method!r}; expected one of {sorted(FUSIONS)}") from None
    fused = fusion(rankings, weights=weights)
    return fused[:k] if k > 0 else []
