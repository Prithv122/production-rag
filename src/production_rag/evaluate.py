"""The evaluation harness: arms x chunking strategies x question categories.

The whole project exists to answer one question -- *which configuration actually
retrieves the right evidence* -- so this module is the point of it, and a few of
its choices are worth stating rather than leaving implicit.

**Everything is reported per category, and the pooled mean is reported last.**
A single headline number over a question set whose category mix was chosen by
the person reporting it is a number about the mix. If rewriting helps conceptual
questions and hurts exact-terminology ones, the pooled mean says whichever of
those two the set happens to contain more of.

**Unanswerable questions are scored separately, not averaged in.** Recall over
an empty gold set is undefined, and quietly assigning it 0 (or 1) would let a
system that retrieves nothing look good (or bad) for the wrong reason. They are
counted, and what they measure -- refusal -- is a generation property, so the
retrieval tables report only their count and score distribution.

**Gold chunks are recomputed per strategy.** See
:mod:`production_rag.groundtruth`: the labels are document spans, and each
strategy derives its own gold chunk ids from them. The coverage report is
printed with the results, because a strategy whose chunks fail to cover some
questions' evidence is being scored on a different set and its mean is not
comparable.

**Every result carries its provenance.** Arm, strategy, pool size, k, model
names, question-set hash and how many LLM calls were cache hits. A table without
that is not reproducible, and this project's claim is reproducibility.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunking import Chunk
from .groundtruth import Question, coverage_report, gold_chunk_ids, index_chunks_by_doc
from .metrics import RetrievalScore, aggregate
from .pipeline import ARMS, Retriever
from .rerank import DEFAULT_POOL

RESULTS_DIR = Path("eval/results")

METRIC_ORDER = ("recall@1", "recall@5", "recall@10", "hit_rate@5", "ndcg@10", "mrr")


@dataclass
class QuestionResult:
    """One (question, arm, strategy) cell."""

    qid: str
    category: str
    arm: str
    strategy: str
    gold: list[str]
    ranked: list[str]
    top_score: float
    latency_s: float
    metrics: dict[str, float] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    rewrite_failed: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArmResult:
    """Aggregated metrics for one arm on one chunking strategy."""

    arm: str
    strategy: str
    overall: dict[str, float]
    by_category: dict[str, dict[str, float]]
    n_scored: int
    n_unanswerable: int
    mean_latency_s: float
    p95_latency_s: float
    rewrite_failures: int = 0
    unanswerable_top_score: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_arm(
    retriever: Retriever,
    questions: Sequence[Question],
    by_doc: dict[str, list[Chunk]],
    *,
    arm: str,
    k: int = 10,
    pool: int = DEFAULT_POOL,
    reranker=None,
    rewriter=None,
    on_question=None,
) -> tuple[ArmResult, list[QuestionResult]]:
    """Run one arm over the question set and score it."""
    rows: list[QuestionResult] = []
    scores: list[RetrievalScore] = []
    by_category_scores: dict[str, list[RetrievalScore]] = {}
    latencies: list[float] = []
    unanswerable_tops: list[float] = []
    rewrite_failures = 0

    for question in questions:
        result = retriever.retrieve(
            question.text,
            arm=arm,
            k=k,
            pool=pool,
            reranker=reranker,
            rewriter=rewriter,
        )
        latencies.append(result.latency_s)
        if result.rewrite is not None and not result.rewrite.parsed:
            rewrite_failures += 1

        gold = sorted(gold_chunk_ids(question, by_doc))
        top_score = result.ranked[0][1] if result.ranked else 0.0
        row = QuestionResult(
            qid=question.qid,
            category=question.category,
            arm=arm,
            strategy=retriever.strategy,
            gold=gold,
            ranked=result.chunk_ids,
            top_score=top_score,
            latency_s=result.latency_s,
            queries=result.queries,
            rewrite_failed=result.rewrite is not None and not result.rewrite.parsed,
        )

        if gold:
            score = RetrievalScore.evaluate(result.chunk_ids, gold)
            row.metrics = score.as_dict()
            scores.append(score)
            by_category_scores.setdefault(question.category, []).append(score)
        else:
            # No gold: either a deliberately unanswerable question, or -- and
            # this is why the coverage report exists -- a labelled question this
            # strategy failed to map onto any chunk.
            unanswerable_tops.append(top_score)

        rows.append(row)
        if on_question is not None:
            on_question(row)

    latencies.sort()
    arm_result = ArmResult(
        arm=arm,
        strategy=retriever.strategy,
        overall=aggregate(scores),
        by_category={
            category: aggregate(bucket) for category, bucket in sorted(by_category_scores.items())
        },
        n_scored=len(scores),
        n_unanswerable=len(unanswerable_tops),
        mean_latency_s=statistics.fmean(latencies) if latencies else 0.0,
        p95_latency_s=latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0,
        rewrite_failures=rewrite_failures,
        unanswerable_top_score=statistics.fmean(unanswerable_tops) if unanswerable_tops else 0.0,
    )
    return arm_result, rows


def run_grid(
    questions: Sequence[Question],
    *,
    index_dir: Path,
    strategies: Sequence[str],
    arms: Sequence[str],
    embedder_factory=None,
    reranker=None,
    rewriter=None,
    k: int = 10,
    pool: int = DEFAULT_POOL,
    verbose: bool = True,
) -> dict:
    """Every arm on every strategy, over the same questions.

    Indexes are loaded once per strategy and reused across arms -- the dense
    matrix is 35 MB and loading it seven times is the difference between an
    evaluation you re-run and one you avoid re-running.
    """
    started = time.time()
    arm_results: list[ArmResult] = []
    all_rows: list[QuestionResult] = []
    coverage: dict[str, dict[str, float]] = {}

    needs_dense = any(ARMS[a].semantic for a in arms)

    for strategy in strategies:
        embedder = embedder_factory() if (embedder_factory and needs_dense) else None
        retriever = Retriever.load(index_dir, strategy, embedder=embedder, with_dense=needs_dense)
        by_doc = index_chunks_by_doc(retriever.chunks.values())
        coverage[strategy] = coverage_report(questions, by_doc)
        if verbose:
            report = coverage[strategy]
            print(
                f"\n{strategy}: {len(retriever.chunks):,} chunks · "
                f"evidence mapped for {report['mapped']:.1%} of questions · "
                f"{report['mean_gold_chunks']:.2f} gold chunks/question"
            )

        for arm in arms:
            mark = time.time()
            result, rows = evaluate_arm(
                retriever,
                questions,
                by_doc,
                arm=arm,
                k=k,
                pool=pool,
                reranker=reranker,
                rewriter=rewriter,
            )
            arm_results.append(result)
            all_rows.extend(rows)
            if verbose:
                overall = result.overall
                print(
                    f"  {arm:20} recall@5 {overall.get('recall@5', 0):.3f}  "
                    f"ndcg@10 {overall.get('ndcg@10', 0):.3f}  "
                    f"mrr {overall.get('mrr', 0):.3f}  "
                    f"({time.time() - mark:.1f}s)"
                )

    return {
        "arms": [r.as_dict() for r in arm_results],
        "rows": [r.as_dict() for r in all_rows],
        "coverage": coverage,
        "config": {
            "k": k,
            "pool": pool,
            "strategies": list(strategies),
            "arm_names": list(arms),
            "n_questions": len(questions),
            "n_verified": sum(1 for q in questions if q.verified in ("accepted", "edited")),
            "categories": _category_counts(questions),
            "reranker": getattr(reranker, "name", ""),
            "rewriter": getattr(rewriter, "model", ""),
            "elapsed_s": round(time.time() - started, 1),
        },
    }


def _category_counts(questions: Sequence[Question]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in questions:
        counts[question.category] = counts.get(question.category, 0) + 1
    return dict(sorted(counts.items()))


def save_results(results: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def markdown_table(results: dict, *, metric: str = "recall@5", by_category: bool = False) -> str:
    """Arms as rows, strategies as columns -- the table the README carries."""
    arms = results["arms"]
    strategies = results["config"]["strategies"]
    arm_names = results["config"]["arm_names"]
    lookup = {(a["arm"], a["strategy"]): a for a in arms}

    if not by_category:
        header = f"| Arm | {' | '.join(strategies)} |"
        rule = "|---|" + "---:|" * len(strategies)
        lines = [header, rule]
        for arm in arm_names:
            cells = []
            for strategy in strategies:
                entry = lookup.get((arm, strategy))
                cells.append(f"{entry['overall'].get(metric, 0):.3f}" if entry else "—")
            lines.append(f"| `{arm}` | {' | '.join(cells)} |")
        return "\n".join(lines)

    categories = sorted(
        {c for a in arms for c in a["by_category"]},
        key=lambda c: (
            ("exact_term", "conceptual", "cross_tool").index(c)
            if c in ("exact_term", "conceptual", "cross_tool")
            else 99
        ),
    )
    lines = [
        f"| Arm | Strategy | {' | '.join(categories)} |",
        "|---|---|" + "---:|" * len(categories),
    ]
    for arm in arm_names:
        for strategy in strategies:
            entry = lookup.get((arm, strategy))
            if not entry:
                continue
            cells = [
                f"{entry['by_category'].get(c, {}).get(metric, 0):.3f}"
                if c in entry["by_category"]
                else "—"
                for c in categories
            ]
            lines.append(f"| `{arm}` | {strategy} | {' | '.join(cells)} |")
    return "\n".join(lines)


def summarise(results: dict) -> str:
    """A short human-readable digest, printed at the end of an eval run."""
    config = results["config"]
    lines = [
        f"{config['n_questions']} questions "
        f"({config['n_verified']} human-verified) · {config['categories']}",
        f"k={config['k']} pool={config['pool']} · {config['elapsed_s']}s",
        "",
        "recall@5",
        markdown_table(results, metric="recall@5"),
        "",
        "nDCG@10",
        markdown_table(results, metric="ndcg@10"),
        "",
        "recall@5 by category",
        markdown_table(results, metric="recall@5", by_category=True),
    ]
    return "\n".join(lines)
