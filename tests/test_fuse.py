"""Fusion tests."""

from __future__ import annotations

import pytest

from production_rag.fuse import (
    RRF_K,
    fuse,
    reciprocal_rank_fusion,
    score_fusion,
)

BM25 = [("a", 30.0), ("b", 12.0), ("c", 1.0)]
DENSE = [("b", 0.91), ("d", 0.88), ("a", 0.40)]


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------
def test_rrf_rewards_agreement_between_arms() -> None:
    """`b` is 2nd and 1st; `a` is 1st and 3rd. Agreement near the top wins."""
    assert reciprocal_rank_fusion([BM25, DENSE])[0][0] == "b"


def test_rrf_uses_the_documented_formula() -> None:
    fused = dict(reciprocal_rank_fusion([BM25, DENSE]))
    expected_a = 1 / (RRF_K + 1) + 1 / (RRF_K + 3)
    assert fused["a"] == pytest.approx(expected_a)


def test_rrf_ignores_score_magnitude() -> None:
    """Multiplying one arm's scores must not change a rank-only fusion."""
    inflated = [(cid, score * 1000) for cid, score in BM25]
    assert reciprocal_rank_fusion([BM25, DENSE]) == reciprocal_rank_fusion([inflated, DENSE])


def test_rrf_keeps_single_arm_candidates() -> None:
    """`d` appears only in the dense arm and must still be reachable."""
    assert "d" in dict(reciprocal_rank_fusion([BM25, DENSE]))


def test_rrf_weights_shift_the_outcome() -> None:
    heavy_lexical = reciprocal_rank_fusion([BM25, DENSE], weights=[10.0, 1.0])
    assert heavy_lexical[0][0] == "a"


def test_rrf_with_an_empty_arm_is_the_other_arm() -> None:
    assert reciprocal_rank_fusion([BM25, []]) == reciprocal_rank_fusion([BM25])


def test_rrf_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="same length"):
        reciprocal_rank_fusion([BM25, DENSE], weights=[1.0])


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------
def test_score_fusion_normalises_away_the_scale_difference() -> None:
    """BM25's 30.0 and cosine's 0.91 are both "best in arm" after min-max."""
    fused = dict(score_fusion([BM25, DENSE]))
    assert fused["a"] == pytest.approx(1.0 + 0.0)
    assert fused["b"] == pytest.approx((12.0 - 1.0) / 29.0 + 1.0)


def test_score_fusion_is_invariant_to_a_linear_rescale() -> None:
    rescaled = [(cid, score * 7 + 3) for cid, score in BM25]
    left = dict(score_fusion([BM25, DENSE]))
    right = dict(score_fusion([rescaled, DENSE]))
    for key in left:
        assert left[key] == pytest.approx(right[key])


def test_score_fusion_handles_an_all_equal_arm() -> None:
    """A flat arm must not divide by zero; treat every candidate as equally good."""
    flat = [("a", 5.0), ("b", 5.0)]
    fused = dict(score_fusion([flat]))
    assert fused == {"a": pytest.approx(1.0), "b": pytest.approx(1.0)}


def test_score_fusion_treats_a_missing_chunk_as_zero_not_absent() -> None:
    """`c` is lexical-only; it keeps its (normalised) lexical contribution."""
    assert dict(score_fusion([BM25, DENSE]))["c"] == pytest.approx(0.0)


def test_score_fusion_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="same length"):
        score_fusion([BM25, DENSE], weights=[1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Shared contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["rrf", "score"])
def test_fuse_truncates_to_k(method: str) -> None:
    assert len(fuse([BM25, DENSE], method, k=2)) == 2


@pytest.mark.parametrize("method", ["rrf", "score"])
def test_fuse_is_sorted_descending(method: str) -> None:
    scores = [s for _, s in fuse([BM25, DENSE], method, k=10)]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("method", ["rrf", "score"])
def test_fuse_k_zero(method: str) -> None:
    assert fuse([BM25, DENSE], method, k=0) == []


def test_rrf_ties_break_by_chunk_id() -> None:
    """Mirror-image arms give `x` and `z` identical RRF scores.

    `z` is 1st then 3rd, `x` is 3rd then 1st -- the sums are equal by
    construction. Without an explicit tiebreak the same evaluation can report
    different recall@k on consecutive runs, quietly making the README
    unreproducible.
    """
    forward = [("z", 3.0), ("y", 2.0), ("x", 1.0)]
    backward = [("x", 3.0), ("y", 2.0), ("z", 1.0)]

    fused = fuse([forward, backward], "rrf", k=3)

    assert fused[0][1] == pytest.approx(fused[1][1])
    assert [cid for cid, _ in fused][:2] == ["x", "z"]


def test_score_fusion_ties_break_by_chunk_id() -> None:
    """A flat arm normalises every candidate to 1.0 -- a genuine three-way tie."""
    arm = [("z", 1.0), ("y", 1.0), ("x", 1.0)]
    assert [cid for cid, _ in fuse([arm, arm], "score", k=3)] == ["x", "y", "z"]


@pytest.mark.parametrize("method", ["rrf", "score"])
def test_fusion_is_repeatable(method: str) -> None:
    assert fuse([BM25, DENSE], method, k=5) == fuse([BM25, DENSE], method, k=5)


def test_fuse_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown fusion"):
        fuse([BM25], "learned-to-rank")
