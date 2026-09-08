"""Metric tests, including hand-computed expected values.

A metric implementation that is merely self-consistent is worthless -- if the
formula is wrong the whole README is wrong and nothing else will catch it. So
the nDCG and MRR cases below carry the arithmetic in the assertion.
"""

from __future__ import annotations

import math

import pytest

from production_rag.metrics import (
    RetrievalScore,
    aggregate,
    dcg_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------
def test_recall_counts_gold_chunks_in_the_window() -> None:
    assert recall_at_k(RANKED, {"a", "c", "z"}, 5) == pytest.approx(2 / 3)


def test_recall_at_1() -> None:
    assert recall_at_k(RANKED, {"a"}, 1) == 1.0
    assert recall_at_k(RANKED, {"b"}, 1) == 0.0


def test_recall_is_bounded_by_the_gold_count_not_k() -> None:
    """One gold chunk found in a window of 10 is recall 1.0, not 0.1."""
    assert recall_at_k(RANKED, {"c"}, 10) == 1.0


def test_recall_beyond_the_ranking_length_is_safe() -> None:
    assert recall_at_k(["a"], {"a"}, 50) == 1.0


def test_recall_with_no_gold_is_an_error() -> None:
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k(RANKED, set(), 5)


# ---------------------------------------------------------------------------
# Hit rate
# ---------------------------------------------------------------------------
def test_hit_rate_is_binary() -> None:
    assert hit_rate_at_k(RANKED, {"e", "z"}, 5) == 1.0
    assert hit_rate_at_k(RANKED, {"f"}, 5) == 0.0


def test_hit_rate_and_recall_agree_for_a_single_gold_chunk() -> None:
    """The common case in this corpus -- which is why both are reported."""
    for gold in ("a", "c", "j"):
        assert hit_rate_at_k(RANKED, {gold}, 10) == recall_at_k(RANKED, {gold}, 10)


# ---------------------------------------------------------------------------
# DCG / nDCG
# ---------------------------------------------------------------------------
def test_dcg_uses_log2_of_rank_plus_one() -> None:
    assert dcg_at_k(RANKED, {"a"}, 10) == pytest.approx(1 / math.log2(2))
    assert dcg_at_k(RANKED, {"c"}, 10) == pytest.approx(1 / math.log2(4))


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k(RANKED, {"a", "b"}, 10) == pytest.approx(1.0)


def test_ndcg_hand_computed() -> None:
    """Gold at ranks 1 and 3; ideal places both at ranks 1 and 2."""
    actual = 1 / math.log2(2) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, {"a", "c"}, 10) == pytest.approx(actual / ideal)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k(RANKED, {"zzz"}, 10) == 0.0


def test_ndcg_stays_bounded_when_gold_exceeds_k() -> None:
    """Five gold chunks but k=2: the ideal must also be truncated to k."""
    gold = {"a", "b", "c", "d", "e"}
    assert ndcg_at_k(RANKED, gold, 2) == pytest.approx(1.0)


def test_ndcg_rewards_a_higher_placement() -> None:
    assert ndcg_at_k(RANKED, {"a"}, 10) > ndcg_at_k(RANKED, {"e"}, 10)


def test_ndcg_with_no_gold_is_an_error() -> None:
    with pytest.raises(ValueError, match="undefined"):
        ndcg_at_k(RANKED, set(), 5)


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------
def test_reciprocal_rank_uses_the_first_hit() -> None:
    assert reciprocal_rank(RANKED, {"c", "e"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_absent() -> None:
    assert reciprocal_rank(RANKED, {"zzz"}) == 0.0


def test_reciprocal_rank_is_not_capped_at_ten() -> None:
    """A hit at rank 11 still counts -- MRR has no k."""
    assert reciprocal_rank(RANKED, {"k"}) == pytest.approx(1 / 11)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_retrieval_score_evaluates_every_metric() -> None:
    score = RetrievalScore.evaluate(RANKED, {"a"})
    assert score.recall_at_1 == 1.0
    assert score.mrr == 1.0
    assert score.ndcg_at_10 == pytest.approx(1.0)


def test_as_dict_keys_are_report_ready() -> None:
    keys = set(RetrievalScore.evaluate(RANKED, {"a"}).as_dict())
    assert keys == {"recall@1", "recall@5", "recall@10", "hit_rate@5", "ndcg@10", "mrr"}


def test_aggregate_takes_the_mean() -> None:
    perfect = RetrievalScore.evaluate(RANKED, {"a"})
    missed = RetrievalScore.evaluate(RANKED, {"zzz"})
    assert aggregate([perfect, missed])["recall@1"] == pytest.approx(0.5)


def test_aggregate_of_nothing_is_empty_not_a_crash() -> None:
    assert aggregate([]) == {}
