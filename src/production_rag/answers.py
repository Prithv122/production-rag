"""Scoring generated answers -- without an LLM judge.

The retrieval grid could be scored against gold spans. Answers cannot, and the
standard move at this point is to appoint a large model as judge and report its
scores. That is not done here, for one reason that is about measurement rather
than taste: an LLM judge's output is another model's opinion with no error bar,
and this project's whole claim is that its numbers are reproducible offline by
someone who does not trust me. A judge would make the headline figure depend on
a model I cannot pin, cannot cache honestly across versions, and cannot let a
reader re-derive. So the four metrics below are all *arithmetic over things that
either happened or did not*:

``refusal_recall``
    Of the questions with no answer anywhere in the corpus, how many were
    refused. Alone, this is worthless -- a model that refuses everything scores
    1.000 -- so it is never reported without the next one.

``false_refusal``
    Of the answerable questions, how many were refused. This is the price of the
    line above, and the pair is the actual result.

``grounded``
    Of the answerable questions that got an answer, how many cited at least one
    chunk that contains a gold evidence span. This is *not* "the answer is
    correct": a model can cite the right passage and still summarise it wrongly.
    It is the strongest claim the existing ground truth supports, and it is
    stated as that claim and no larger.

``citation_validity`` / ``uncited``
    Of every ``[n]`` written, how many pointed at a passage the model was
    actually shown; and how many answers asserted something with no marker at
    all. These measure the model's compliance with the contract, which is a
    different axis from whether the retrieved passage was the right one.

**Retrieval is frozen, so generation is the only variable.** The rankings are
replayed out of `eval/results/retrieval.json` -- the same rows the retrieval
tables were computed from -- rather than re-retrieved per provider. Two things
follow. Every provider arm sees byte-identical context, so a difference between
two rows of the answer table cannot be a retrieval difference. And the answer
eval needs no encoder, no index load and no torch: it is a JSON file, the chunk
texts, and N LLM calls.

**The subset is stated, not smuggled.** Free-tier generation is a median 65 s a
call, so running four arms over all 184 questions is a wall-clock day. The eval
runs a stratified subset: every unanswerable and every cross_tool question (they
are rare and they are the interesting ones), plus a seeded proportional sample
of the rest. The selection is deterministic from the seed and the sizes are
printed with the results, because a subset chosen after seeing the scores is not
a subset, it is a claim.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .chunking import Chunk
from .generate import Answer, generate
from .groundtruth import Question, gold_chunk_ids

DEFAULT_RESULTS = Path("eval/results/retrieval.json")
DEFAULT_OUT = Path("eval/results/answers.json")

#: The retrieval configuration every provider arm is fed. Chosen because it won
#: the retrieval grid on nDCG, not because it flatters generation.
FROZEN_ARM = "hybrid_score_weighted"
FROZEN_STRATEGY = "heading"

DEFAULT_SUBSET = 60
DEFAULT_SEED = 20260908


def load_frozen_retrievals(
    path: Path = DEFAULT_RESULTS,
    *,
    arm: str = FROZEN_ARM,
    strategy: str = FROZEN_STRATEGY,
    k: int = 10,
) -> dict[str, list[str]]:
    """Replay one arm's rankings out of a saved retrieval grid."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rankings = {
        row["qid"]: list(row["ranked"])[:k]
        for row in payload["rows"]
        if row["arm"] == arm and row["strategy"] == strategy
    }
    if not rankings:
        raise ValueError(f"no rows for arm={arm!r} strategy={strategy!r} in {path}")
    return rankings


def select_questions(
    questions: Sequence[Question],
    *,
    n: int = DEFAULT_SUBSET,
    seed: int = DEFAULT_SEED,
) -> list[Question]:
    """Stratified subset: keep every rare category, sample the common ones.

    cross_tool and unanswerable are 10 of 184 questions and they carry the two
    findings the answer eval exists to produce, so sampling them proportionally
    would leave one or two of each and no result at all.
    """
    rare = [q for q in questions if q.category in ("unanswerable", "cross_tool")]
    common = [q for q in questions if q.category not in ("unanswerable", "cross_tool")]
    budget = max(n - len(rare), 0)

    by_category: dict[str, list[Question]] = {}
    for question in common:
        by_category.setdefault(question.category, []).append(question)

    rng = random.Random(seed)
    picked: list[Question] = []
    for category in sorted(by_category):
        pool = sorted(by_category[category], key=lambda q: q.qid)
        share = round(budget * len(pool) / max(len(common), 1))
        picked.extend(rng.sample(pool, min(share, len(pool))))

    chosen = sorted(rare + picked, key=lambda q: q.qid)
    return chosen


@dataclass
class AnswerRow:
    """One (question, provider arm) outcome."""

    qid: str
    category: str
    arm: str
    model: str
    refused: bool
    refusal_reason: str
    should_refuse: bool
    n_citations: int
    n_valid_citations: int
    cited_gold: bool
    uncited: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    text: str = ""
    cited: list[str] = field(default_factory=list)
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_answer(
    question: Question,
    answer: Answer,
    gold: set[str],
    *,
    arm: str,
) -> AnswerRow:
    cited = answer.cited_chunk_ids
    return AnswerRow(
        qid=question.qid,
        category=question.category,
        arm=arm,
        model=answer.model,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        should_refuse=not question.is_answerable,
        n_citations=len(answer.citations),
        n_valid_citations=sum(1 for c in answer.citations if c.valid),
        cited_gold=bool(gold and set(cited) & gold),
        uncited=answer.uncited,
        latency_s=answer.latency_s,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        text=answer.clean_text[:600],
        cited=cited,
        error=answer.error,
    )


@dataclass
class ArmSummary:
    """The row of the answer-quality table for one generation model."""

    arm: str
    model: str
    n: int
    n_answerable: int
    n_unanswerable: int
    refusal_recall: float
    false_refusal: float
    grounded: float
    citation_validity: float
    uncited_rate: float
    parse_failure: float
    provider_error: float
    fell_back: float
    """Share of this arm's rows actually answered by a *different* model.

    `build_provider` puts a fallback chain behind the named model, which is a
    feature everywhere except here: an arm labelled `nemotron-super` whose calls
    silently degraded to the local Ollama would be a comparison between two
    models reported as one. The per-row `model` field records which model
    actually spoke, and this is that disagreement, surfaced next to the metrics
    rather than left in the rows for nobody to find."""

    median_latency_s: float
    completion_tokens: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarise(rows: Sequence[AnswerRow], *, arm: str, model: str) -> ArmSummary:
    rows = [r for r in rows if r.arm == arm]
    answerable = [r for r in rows if not r.should_refuse]
    unanswerable = [r for r in rows if r.should_refuse]
    answered = [r for r in answerable if not r.refused]
    markers = sum(r.n_citations for r in rows)
    valid = sum(r.n_valid_citations for r in rows)
    latencies = [r.latency_s for r in rows if r.latency_s > 0]

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return ArmSummary(
        arm=arm,
        model=model,
        n=len(rows),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        refusal_recall=rate(sum(1 for r in unanswerable if r.refused), len(unanswerable)),
        false_refusal=rate(sum(1 for r in answerable if r.refused), len(answerable)),
        grounded=rate(sum(1 for r in answered if r.cited_gold), len(answered)),
        citation_validity=rate(valid, markers),
        uncited_rate=rate(sum(1 for r in answered if r.uncited), len(answered)),
        parse_failure=rate(sum(1 for r in rows if r.refusal_reason == "unparseable"), len(rows)),
        provider_error=rate(
            sum(1 for r in rows if r.refusal_reason == "provider_error"), len(rows)
        ),
        fell_back=rate(sum(1 for r in rows if r.model and r.model != model), len(rows)),
        median_latency_s=statistics.median(latencies) if latencies else 0.0,
        completion_tokens=statistics.fmean([r.completion_tokens for r in rows]) if rows else 0.0,
    )


def run_arm(
    questions: Sequence[Question],
    rankings: dict[str, list[str]],
    chunks: dict[str, Chunk],
    provider,
    *,
    arm: str,
    by_doc: dict[str, list[Chunk]],
    workers: int = 4,
    k: int = 10,
    progress=None,
) -> list[AnswerRow]:
    """Generate and score one provider arm over the selected questions.

    Threads, not processes: every worker is blocked on a socket for a median of
    65 seconds, so the GIL is irrelevant and the shared provider's cache stays
    a single dictionary rather than N copies that never see each other's hits.
    """

    def one(question: Question) -> AnswerRow:
        chunk_ids = rankings.get(question.qid, [])[:k]
        answer = generate(question.text, chunk_ids, chunks, provider)
        gold = gold_chunk_ids(question, by_doc) if question.is_answerable else set()
        row = score_answer(question, answer, set(gold), arm=arm)
        if progress is not None:
            progress(row)
        return row

    if workers <= 1:
        return [one(q) for q in questions]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, questions))


def run_grid(
    questions: Sequence[Question],
    rankings: dict[str, list[str]],
    chunks: dict[str, Chunk],
    providers: dict[str, Any],
    *,
    by_doc: dict[str, list[Chunk]],
    workers: int = 4,
    k: int = 10,
    progress=None,
) -> tuple[list[AnswerRow], list[ArmSummary], dict[str, Any]]:
    started = time.time()
    rows: list[AnswerRow] = []
    summaries: list[ArmSummary] = []
    for arm, provider in providers.items():
        arm_rows = run_arm(
            questions,
            rankings,
            chunks,
            provider,
            arm=arm,
            by_doc=by_doc,
            workers=workers,
            k=k,
            progress=progress,
        )
        rows.extend(arm_rows)
        summaries.append(summarise(arm_rows, arm=arm, model=provider.model))

    categories: dict[str, int] = {}
    for question in questions:
        categories[question.category] = categories.get(question.category, 0) + 1
    config = {
        "retrieval_arm": FROZEN_ARM,
        "retrieval_strategy": FROZEN_STRATEGY,
        "k": k,
        "n_questions": len(questions),
        "n_verified": sum(1 for q in questions if q.verified in ("accepted", "edited")),
        "categories": categories,
        "arms": list(providers),
        "elapsed_s": round(time.time() - started, 1),
    }
    return rows, summaries, config


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def format_table(summaries: Sequence[ArmSummary]) -> str:
    header = (
        "| Arm | model | refusal recall | false refusal | grounded | "
        "citation validity | uncited | unparseable | fell back | median s |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for s in summaries:
        lines.append(
            f"| `{s.arm}` | {s.model.split('/')[-1]} | "
            f"{s.refusal_recall:.3f} | {s.false_refusal:.3f} | {s.grounded:.3f} | "
            f"{s.citation_validity:.3f} | {s.uncited_rate:.3f} | {s.parse_failure:.3f} | "
            f"{s.fell_back:.3f} | {s.median_latency_s:.1f} |"
        )
    return "\n".join(lines)


def format_by_category(rows: Sequence[AnswerRow]) -> str:
    """Grounding split by question category -- the pooled mean hides it."""
    arms = sorted({r.arm for r in rows}, key=lambda a: [r.arm for r in rows].index(a))
    categories = ["exact_term", "conceptual", "cross_tool"]
    lines = ["| Arm | " + " | ".join(categories) + " |", "|---|" + "---:|" * len(categories)]
    for arm in arms:
        cells = []
        for category in categories:
            answered = [
                r
                for r in rows
                if r.arm == arm and r.category == category and not r.refused and not r.should_refuse
            ]
            if answered:
                grounded = sum(1 for r in answered if r.cited_gold) / len(answered)
                cells.append(f"{grounded:.3f} (n={len(answered)})")
            else:
                cells.append("--")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save(
    path: Path,
    rows: Sequence[AnswerRow],
    summaries: Sequence[ArmSummary],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": config,
                "summaries": [s.as_dict() for s in summaries],
                "rows": [r.as_dict() for r in rows],
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
