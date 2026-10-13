"""The agentic retrieval arm: does a bounded judge-and-requery loop retrieve better than the fixed
pipeline, and what does the iteration cost?

This module is the instrument. It holds no results, and every definition in it implements the
protocol registered in NOTES.md ("Agentic retrieval arm: registered protocol, frozen before any
run", commit 4523723) rather than restating it. Where the protocol left a detail open, the choice
made at build time is named in the docstring of the function that makes it, and in the NOTES
section "Agentic retrieval arm: harness built, not yet run". The served pipeline is unchanged and
never imports this module: the loop is an evaluation arm only.

What it covers, in the order a run uses it:

* **Inputs.** `load_inputs` reads the questions, labels and frozen top-10 rankings of the
  retrieval table, the 60-question answer subset and its published `ollama-qwen-coder` rows.
* **Step one and gate 1.** `prepare_step_one` retrieves every question once with the baseline arm,
  measures the baseline's retrieval latency, and checks each ranking against the stored one. It
  needs no language model, so it runs first and a mismatch stops the run before any call.
* **The loop.** `run_loop` is one question's loop: `render_judge_prompt`, `parse_verdict`,
  `merge_rankings` (equal-weight RRF), the step cap, the refusal rule, the repeated-query rule and
  the fail-open rule for an invalid reply. Pure apart from the injected retriever and provider.
* **Answer quality.** `run_answers` applies the tie rule on the 60-question subset and generates
  only where the evidence changed or the loop refused.
* **Scoring and verdict.** `build_result` computes the metrics, the paired bootstrap, the cost
  and calibration tables, the four validity gates and, only when every gate holds, the mechanical
  verdict (`decide`). `save` refuses to overwrite a finished run.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from .answers import AnswerRow, score_answer, summarise
from .chunking import Chunk
from .fuse import reciprocal_rank_fusion
from .generate import REFUSAL_TEXT, Passage, build_passages, generate
from .groundtruth import Question, gold_chunk_ids, load_questions
from .injection import load_clean_subset
from .metrics import RetrievalScore
from .providers import MODEL_ARMS, LLMProvider, LLMResponse, ProviderError, extract_json

PROTOCOL_COMMIT = "4523723"
OUT_PATH = Path("eval/results/agentic.json")

#: The baseline and the strategy: the best first-stage arm of the retrieval table, on `heading`.
ARM = "hybrid_score_weighted"
STRATEGY = "heading"
MODEL_ARM = "ollama-qwen-coder"
#: Context window of every live call. Recorded, and checked by validity gate 4.
NUM_CTX = 8192

K = 10
POOL = 50
JUDGE_TOP = 5
PASSAGE_CHARS = 1_200
MAX_STEPS = 3
RRF_K = 60
JUDGE_MAX_TOKENS = 200
MAX_NEXT_QUERY_CHARS = 300

BOOTSTRAP_TRIALS = 20_000
BOOTSTRAP_SEED = 20261013
#: The registered bar on the primary outcome, and the one on the refusal guardrail
#: ("5 points of 54, rounded down" is 2 questions).
GAIN_BAR = 0.02
GUARDRAIL_POINTS = 5
INVALID_RATE_BAR = Fraction(5, 100)
PROMPT_TOKEN_LIMIT = NUM_CTX

#: Metrics reported in every table; the first three are the registered ones.
PRIMARY = "ndcg@10"
SECONDARY = ("recall@5", "mrr")
METRICS = ("recall@1", "recall@5", "recall@10", "hit_rate@5", "ndcg@10", "mrr")
CATEGORY_ORDER = ("exact_term", "conceptual", "cross_tool")

#: The judge prompt, exact text from the registered protocol.
JUDGE_SYSTEM = (
    "You decide whether retrieved documentation passages are enough to answer a question. "
    "Reply with one JSON object and nothing else."
)
JUDGE_TEMPLATE = """\
Question: {question}

Passages:
{passages}

Can the question be answered fully and correctly from the text of these passages alone?
If yes, reply {"sufficient": true}.
If no, reply {"sufficient": false, "missing": "<one sentence: what is missing>", "next_query": "<one search query that would find it>"}.
In next_query keep exact identifiers from the question (function names, settings, flags, commands) unchanged.
Do not answer the question."""  # noqa: E501 -- the registered text, one line per sentence

_PLACEHOLDER = re.compile(r"\{(question|passages)\}")

#: How a loop ended.
OUTCOMES = ("sufficient", "cap_refusal", "repeat_refusal", "invalid_reply")
#: Why a judge reply is invalid. The first three are the registered kinds (the third split in
#: two); `provider_error` is a call that returned nothing.
INVALID_KINDS = (
    "no_json_object",
    "sufficient_not_boolean",
    "empty_next_query",
    "next_query_too_long",
    "provider_error",
)

GAIN = "gain"
WORSE = "worse"
NO_GAIN = "no_gain_for_cost"


# ---------------------------------------------------------------------------
# the judge
# ---------------------------------------------------------------------------
def normalise_query(query: str) -> str:
    """The comparison form of a query: lowercased, whitespace collapsed."""
    return " ".join(query.lower().split())


def judge_passages(ranking: Sequence[tuple[str, float]], chunks: dict[str, Chunk]) -> list[Passage]:
    """The top 5 passages of `ranking` as the answer prompt renders them, each cut to 1,200 chars.

    Built through `generate.build_passages` so the header is the answer prompt's. Build-time
    choice: a cut passage carries the same "[... passage truncated ...]" line the answer prompt
    uses, because `Passage.render` is registered as the renderer and that line is part of it.
    """
    ids = [chunk_id for chunk_id, _ in ranking[:JUDGE_TOP]]
    passages, _ = build_passages(ids, chunks, max_passage_chars=PASSAGE_CHARS)
    return passages


def render_judge_prompt(question: str, passages: Sequence[Passage]) -> str:
    """The registered user prompt. `{passages}` is the rendered passages joined by a blank line.

    Substituted in one pass, so a question or passage that itself contains `{passages}` is left
    alone. The registered text ends at "Do not answer the question." with no trailing newline.
    """
    values = {"question": question, "passages": "\n\n".join(p.render() for p in passages)}
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], JUDGE_TEMPLATE)


@dataclass(frozen=True)
class JudgeVerdict:
    """A parsed judge reply. `sufficient` is `None` when the reply is invalid."""

    sufficient: bool | None
    missing: str = ""
    next_query: str = ""
    invalid: str = ""

    @property
    def valid(self) -> bool:
        return self.sufficient is not None


def parse_verdict(text: str) -> JudgeVerdict:
    """Read a judge reply under the registered validity rule.

    Invalid is: no JSON object; `sufficient` missing or not a JSON boolean; or `sufficient` false
    with an empty or over-300-character `next_query`. Parsing goes through the project's
    `extract_json`, so a fenced or prefixed object is read as it is everywhere else. Build-time
    choices: the 300-character limit is measured on the stripped query, and the stripped query is
    what is issued; `missing` is not validated, only recorded; a `next_query` on a sufficient
    reply is ignored.
    """
    try:
        payload = extract_json(text)
    except ValueError:
        return JudgeVerdict(None, invalid="no_json_object")
    if not isinstance(payload, dict):
        return JudgeVerdict(None, invalid="no_json_object")
    sufficient = payload.get("sufficient")
    if not isinstance(sufficient, bool):
        return JudgeVerdict(None, invalid="sufficient_not_boolean")
    missing = payload.get("missing")
    missing = missing.strip() if isinstance(missing, str) else ""
    if sufficient:
        return JudgeVerdict(True, missing=missing)
    next_query = payload.get("next_query")
    next_query = next_query.strip() if isinstance(next_query, str) else ""
    if not next_query:
        return JudgeVerdict(None, missing=missing, invalid="empty_next_query")
    if len(next_query) > MAX_NEXT_QUERY_CHARS:
        return JudgeVerdict(None, missing=missing, invalid="next_query_too_long")
    return JudgeVerdict(False, missing=missing, next_query=next_query)


def judge_call(provider: LLMProvider, prompt: str) -> tuple[LLMResponse | None, str]:
    """One judge call: temperature 0, JSON mode, 200 tokens. `(response, "")` or `(None, error)`.

    Exactly one attempt. There is no retry of any kind: a step is one call, and a retry would
    give the loop a second attempt the cost accounting does not show.
    """
    try:
        response = provider.complete(
            prompt,
            system=JUDGE_SYSTEM,
            temperature=0.0,
            max_tokens=JUDGE_MAX_TOKENS,
            json_object=True,
        )
    except ProviderError as exc:
        return None, str(exc)[:300]
    return response, ""


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
Ranking = list[tuple[str, float]]
Retrieve = Callable[[str], tuple[Ranking, float]]


def merge_rankings(lists: Sequence[Sequence[tuple[str, float]]]) -> Ranking:
    """The merged ranking over the pool-50 lists of every query issued so far.

    Equal-weight reciprocal rank fusion with k = 60, the project's own `fuse` function. One list
    is returned as it is (it is already the baseline pool, and re-scoring it would change nothing
    but the scores). Equal weights are the registered design: with 2:1 a new rank-1 chunk
    (1/61) scores below the original's rank 50 (2/110) and a re-query could never change the
    top 10.
    """
    if len(lists) == 1:
        return list(lists[0])
    return reciprocal_rank_fusion(lists, k=RRF_K)


@dataclass
class Step:
    """One judge call and what led to it."""

    step: int
    query: str
    """The query whose retrieval produced the ranking this step judged (step 1: the question)."""
    top5: list[str]
    retrieval_s: float
    prompt_chars: int
    reply: str = ""
    sufficient: bool | None = None
    missing: str = ""
    next_query: str = ""
    invalid: str = ""
    model: str = ""
    cached: bool = False
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoopResult:
    """One question's loop."""

    qid: str
    question: str
    steps: list[Step]
    queries: list[str]
    final: Ranking
    refused: bool
    outcome: str

    @property
    def n_calls(self) -> int:
        return len(self.steps)

    @property
    def top10(self) -> list[str]:
        return [chunk_id for chunk_id, _ in self.final[:K]]

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @property
    def latency_s(self) -> float:
        """Retrieval plus judge time, as measured: re-queries are timed, judge calls carry the
        latency recorded with the response (the original one, for a cached response)."""
        return sum(s.retrieval_s + s.latency_s for s in self.steps)


def run_loop(
    qid: str,
    question: str,
    step_one: Ranking,
    step_one_s: float,
    *,
    retrieve: Retrieve,
    chunks: dict[str, Chunk],
    provider: LLMProvider,
    max_steps: int = MAX_STEPS,
) -> LoopResult:
    """The registered loop for one question.

    `step_one` is the baseline pool-50 ranking of the question, already retrieved (and checked
    against the stored ranking by gate 1). Each iteration judges the top 5 of the current
    merged ranking:

    * sufficient: stop, the current ranking is final;
    * insufficient at the step cap: stop and refuse;
    * insufficient with a `next_query` already issued (lowercased, whitespace collapsed; the
      question itself counts as issued): stop and refuse, with no retrieval;
    * insufficient otherwise: retrieve `next_query`, merge, judge again;
    * invalid reply (or a failed call): stop, keep the ranking, do not refuse.
    """
    seen = {normalise_query(question)}
    lists: list[Ranking] = [list(step_one)]
    queries = [question]
    ranking = lists[0]
    query, retrieval_s = question, step_one_s
    steps: list[Step] = []
    refused, outcome = False, ""

    for number in range(1, max_steps + 1):
        passages = judge_passages(ranking, chunks)
        prompt = render_judge_prompt(question, passages)
        response, error = judge_call(provider, prompt)
        step = Step(
            step=number,
            query=query,
            top5=[p.chunk_id for p in passages],
            retrieval_s=retrieval_s,
            prompt_chars=len(prompt),
            error=error,
        )
        if response is None:
            verdict = JudgeVerdict(None, invalid="provider_error")
        else:
            verdict = parse_verdict(response.text)
            step.reply = response.text[:2_000]
            step.model = response.model
            step.cached = response.cached
            step.latency_s = response.latency_s
            step.prompt_tokens = response.prompt_tokens
            step.completion_tokens = response.completion_tokens
            step.finish_reason = response.finish_reason
        step.sufficient = verdict.sufficient
        step.missing, step.next_query, step.invalid = (
            verdict.missing,
            verdict.next_query,
            verdict.invalid,
        )
        steps.append(step)

        if not verdict.valid:
            outcome = "invalid_reply"
            break
        if verdict.sufficient:
            outcome = "sufficient"
            break
        if number == max_steps:
            refused, outcome = True, "cap_refusal"
            break
        key = normalise_query(verdict.next_query)
        if key in seen:
            refused, outcome = True, "repeat_refusal"
            break
        seen.add(key)
        new_list, retrieval_s = retrieve(verdict.next_query)
        lists.append(list(new_list))
        queries.append(verdict.next_query)
        ranking = merge_rankings(lists)
        query = verdict.next_query

    return LoopResult(qid, question, steps, queries, ranking, refused, outcome)


class LoopRetriever:
    """The baseline arm as the loop calls it: pool 50, returned whole, serialised by a lock.

    The encoder is shared between worker threads and the measured time is the retriever's own,
    taken inside the lock, so waiting for another worker is not counted as retrieval.
    """

    def __init__(self, retriever) -> None:
        self._retriever = retriever
        self._lock = threading.Lock()

    def __call__(self, query: str) -> tuple[Ranking, float]:
        with self._lock:
            result = self._retriever.retrieve(query, arm=ARM, k=POOL, pool=POOL)
        return list(result.ranked), result.latency_s


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuestionSpec:
    """One question with the labels and the stored baseline ranking of the retrieval table."""

    question: Question
    gold: tuple[str, ...]
    baseline: tuple[str, ...]
    stored_metrics: dict[str, float]
    in_subset: bool

    @property
    def qid(self) -> str:
        return self.question.qid

    @property
    def text(self) -> str:
        return self.question.text

    @property
    def category(self) -> str:
        return self.question.category

    @property
    def scored(self) -> bool:
        """Scored on retrieval exactly as the retrieval table does: it has a gold chunk."""
        return bool(self.gold)


@dataclass(frozen=True)
class Inputs:
    """Everything a run reads, loaded and cross-checked before any call."""

    specs: list[QuestionSpec]
    subset: list[QuestionSpec]
    published: dict[str, AnswerRow]
    chunks: dict[str, Chunk]
    by_doc: dict[str, list[Chunk]]


def load_inputs(
    chunks: dict[str, Chunk],
    by_doc: dict[str, list[Chunk]],
    *,
    questions_path: Path,
    results_path: Path,
    answers_path: Path,
    published_arm: str = MODEL_ARM,
) -> Inputs:
    """Load and validate the question set, the frozen rankings and the published answer rows.

    The labels are the retrieval table's own (`gold` as stored in `retrieval.json` for the
    baseline arm on `heading`), not recomputed. The loop runs on every question of the table (the
    scored ones, and the unanswerable and unmapped ones the answer subset needs); only the scored
    ones enter the retrieval metrics. Raises on anything that is not the registered input.
    """
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    config = payload["config"]
    if (config.get("k"), config.get("pool")) != (K, POOL):
        raise ValueError(f"{results_path}: table was run at k={config.get('k')}, pool=pool")
    rows = [r for r in payload["rows"] if r["arm"] == ARM and r["strategy"] == STRATEGY]
    if not rows:
        raise ValueError(f"no rows for arm={ARM!r} strategy={STRATEGY!r} in {results_path}")
    questions = {q.qid: q for q in load_questions(questions_path)}
    missing = sorted(r["qid"] for r in rows if r["qid"] not in questions)
    if missing:
        raise ValueError(
            f"{len(missing)} table questions are not in {questions_path}: {missing[:3]}"
        )

    subset_questions = load_clean_subset(answers_path, list(questions.values()))
    subset_ids = {q.qid for q in subset_questions}
    answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    published = {r["qid"]: AnswerRow(**r) for r in answers["rows"] if r["arm"] == published_arm}
    if set(published) != subset_ids:
        raise ValueError(f"{answers_path}: no complete {published_arm!r} row set for the subset")

    specs: list[QuestionSpec] = []
    for row in rows:
        unknown = [c for c in [*row["gold"], *row["ranked"]] if c not in chunks]
        if unknown:
            raise ValueError(f"{row['qid']}: {len(unknown)} chunk ids are not in the index")
        specs.append(
            QuestionSpec(
                question=questions[row["qid"]],
                gold=tuple(row["gold"]),
                baseline=tuple(row["ranked"][:K]),
                stored_metrics=dict(row["metrics"]),
                in_subset=row["qid"] in subset_ids,
            )
        )
    by_qid = {s.qid: s for s in specs}
    absent = sorted(subset_ids - set(by_qid))
    if absent:
        raise ValueError(f"{len(absent)} subset questions have no stored ranking: {absent[:3]}")
    subset = [by_qid[q.qid] for q in subset_questions]
    return Inputs(specs, subset, published, chunks, by_doc)


# ---------------------------------------------------------------------------
# step one and validity gate 1
# ---------------------------------------------------------------------------
@dataclass
class StepOne:
    """Every question retrieved once with the baseline arm, before any language-model call."""

    rankings: dict[str, Ranking]
    latency_s: dict[str, float]
    gate: dict[str, Any]


def prepare_step_one(specs: Sequence[QuestionSpec], retrieve: Retrieve) -> StepOne:
    """Retrieve each question sequentially, time it, and check gate 1.

    Gate 1: every step-1 ranking equals the stored `hybrid_score_weighted`/`heading` top 10 of
    that question. Sequential and single-threaded, so the latencies are the baseline's retrieval
    latency; one unrecorded warm-up retrieval comes first so the encoder's first call is not
    counted. These rankings and times are also the loop's own step 1.
    """
    if specs:
        retrieve(specs[0].text)
    rankings: dict[str, Ranking] = {}
    latency: dict[str, float] = {}
    mismatched: list[dict[str, Any]] = []
    for spec in specs:
        ranking, seconds = retrieve(spec.text)
        rankings[spec.qid] = ranking
        latency[spec.qid] = seconds
        got = [chunk_id for chunk_id, _ in ranking[:K]]
        if got != list(spec.baseline):
            first = next(
                (i + 1 for i, (a, b) in enumerate(zip(got, spec.baseline, strict=False)) if a != b),
                min(len(got), len(spec.baseline)) + 1,
            )
            mismatched.append({"qid": spec.qid, "first_difference_rank": first})
    gate = {
        "id": 1,
        "name": "step-1 rankings equal the stored baseline top 10",
        "passed": not mismatched,
        "checked": len(specs),
        "mismatched": len(mismatched),
        "examples": mismatched[:5],
    }
    return StepOne(rankings, latency, gate)


# ---------------------------------------------------------------------------
# running the loops
# ---------------------------------------------------------------------------
def _map(fn, items, workers: int) -> list:
    if workers <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def run_loops(
    specs: Sequence[QuestionSpec],
    step_one: StepOne,
    *,
    retrieve: Retrieve,
    chunks: dict[str, Chunk],
    provider: LLMProvider,
    workers: int = 4,
    progress=None,
) -> dict[str, LoopResult]:
    """The loop for every question, `workers` at a time. Rerunning resumes from the cache."""

    def one(spec: QuestionSpec) -> LoopResult:
        result = run_loop(
            spec.qid,
            spec.text,
            step_one.rankings[spec.qid],
            step_one.latency_s[spec.qid],
            retrieve=retrieve,
            chunks=chunks,
            provider=provider,
        )
        if progress is not None:
            progress(result)
        return result

    return {r.qid: r for r in _map(one, specs, workers)}


# ---------------------------------------------------------------------------
# answer quality on the subset
# ---------------------------------------------------------------------------
TIE = "tie"
CHANGED = "changed"
LOOP_REFUSED = "loop_refused"


def classify(spec: QuestionSpec, loop: LoopResult) -> str:
    """`tie` (same top 10, not refused: no generation), `loop_refused`, or `changed`.

    A loop refusal is checked first: it is scored as a refusal whatever the ranking did.
    """
    if loop.refused:
        return LOOP_REFUSED
    return TIE if loop.top10 == list(spec.baseline) else CHANGED


def loop_refusal_row(question: Question) -> AnswerRow:
    """A refusal made by the loop: no passages shown to a model, no generation, no model."""
    return AnswerRow(
        qid=question.qid,
        category=question.category,
        arm="agentic",
        model="",
        refused=True,
        refusal_reason="loop",
        should_refuse=not question.is_answerable,
        n_citations=0,
        n_valid_citations=0,
        cited_gold=False,
        uncited=False,
        latency_s=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        text=REFUSAL_TEXT,
    )


@dataclass
class AnswerPair:
    """The two arms' answer rows for one subset question, and where each came from."""

    qid: str
    kind: str
    baseline: AnswerRow
    agentic: AnswerRow
    baseline_source: str
    agentic_source: str


def run_answers(
    inputs: Inputs,
    loops: dict[str, LoopResult],
    provider: LLMProvider,
    *,
    workers: int = 4,
    progress=None,
) -> list[AnswerPair]:
    """Answer quality on the 60-question subset, paired.

    A tie reuses the published `ollama-qwen-coder` row for both arms. Otherwise the baseline
    answer is generated live from the stored top 10, and the agentic answer live from the
    loop's top 10 unless the loop refused. Both go through `generate.generate` and
    `answers.score_answer` unedited, with its default budget and a provider that carries the
    8,192-token window, so the pair has identical settings.
    """

    def one(spec: QuestionSpec) -> AnswerPair:
        loop = loops[spec.qid]
        kind = classify(spec, loop)
        question = spec.question
        if kind == TIE:
            published = inputs.published[spec.qid]
            pair = AnswerPair(
                spec.qid,
                kind,
                replace(published, arm="baseline"),
                replace(published, arm="agentic"),
                "published",
                "published",
            )
        else:
            gold = set(gold_chunk_ids(question, inputs.by_doc)) if question.is_answerable else set()
            baseline = generate(spec.text, list(spec.baseline), inputs.chunks, provider)
            baseline_row = score_answer(question, baseline, gold, arm="baseline")
            if kind == LOOP_REFUSED:
                agentic_row, agentic_source = loop_refusal_row(question), "loop_refusal"
            else:
                agentic = generate(spec.text, loop.top10, inputs.chunks, provider)
                agentic_row = score_answer(question, agentic, gold, arm="agentic")
                agentic_source = "live"
            pair = AnswerPair(spec.qid, kind, baseline_row, agentic_row, "live", agentic_source)
        if progress is not None:
            progress(pair)
        return pair

    return _map(one, inputs.subset, workers)


def planned_generations(inputs: Inputs, loops: dict[str, LoopResult]) -> int:
    """How many answer generations `run_answers` will make for these loops."""
    total = 0
    for spec in inputs.subset:
        kind = classify(spec, loops[spec.qid])
        total += {TIE: 0, LOOP_REFUSED: 1, CHANGED: 2}[kind]
    return total


class CountingProvider:
    """Counts the calls a stack makes and how many came from its cache. Changes nothing else."""

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        self.calls = 0
        self.cached = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def offline(self) -> bool:
        return bool(getattr(self.inner, "offline", False))

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        response = self.inner.complete(prompt, **kwargs)
        with self._lock:
            self.calls += 1
            self.cached += bool(response.cached)
        return response


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def paired_bootstrap(
    diffs_by_metric: dict[str, Sequence[float]],
    *,
    trials: int = BOOTSTRAP_TRIALS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap of the mean per-question difference, percentile interval.

    Questions are resampled with replacement, the same resample for every metric, `trials`
    times. The interval is the sorted resampled means at positions `int(alpha / 2 * trials)` and
    `int((1 - alpha / 2) * trials)`, the convention `evaluate.bootstrap_band` uses. Uses the
    standard library generator, whose seeded stream is stable across platforms.
    """
    sizes = {len(v) for v in diffs_by_metric.values()}
    if len(sizes) != 1:
        raise ValueError("every metric needs the same questions")
    n = sizes.pop()
    if n == 0:
        raise ValueError("no questions to bootstrap")
    rng = random.Random(seed)
    population = range(n)
    draws: dict[str, list[float]] = {name: [] for name in diffs_by_metric}
    for _ in range(trials):
        picked = rng.choices(population, k=n)
        for name, diffs in diffs_by_metric.items():
            draws[name].append(math.fsum(diffs[i] for i in picked) / n)
    out: dict[str, dict[str, float]] = {}
    for name, values in draws.items():
        values.sort()
        out[name] = {
            "mean": math.fsum(diffs_by_metric[name]) / n,
            "lo": values[int(alpha / 2 * trials)],
            "hi": values[min(int((1 - alpha / 2) * trials), trials - 1)],
        }
    return out


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank-below percentile, as `evaluate` reports p95 (`sorted[int(q * (n - 1))]`)."""
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))] if ordered else 0.0


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def retrieval_report(
    scored: Sequence[QuestionSpec],
    loops: dict[str, LoopResult],
    *,
    trials: int = BOOTSTRAP_TRIALS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Baseline against agentic on the scored questions: all six metrics overall and by
    category, and the paired bootstrap for the primary and the two secondary metrics."""
    base = {s.qid: RetrievalScore.evaluate(list(s.baseline), s.gold).as_dict() for s in scored}
    agent = {s.qid: RetrievalScore.evaluate(loops[s.qid].top10, s.gold).as_dict() for s in scored}

    def block(specs: Sequence[QuestionSpec]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(specs)}
        for metric in METRICS:
            b = _mean([base[s.qid][metric] for s in specs])
            a = _mean([agent[s.qid][metric] for s in specs])
            out[metric] = {"baseline": b, "agentic": a, "delta": a - b}
        return out

    diffs = {
        metric: [agent[s.qid][metric] - base[s.qid][metric] for s in scored]
        for metric in (PRIMARY, *SECONDARY)
    }
    intervals = paired_bootstrap(diffs, trials=trials, seed=seed)
    categories = sorted({s.category for s in scored}, key=_category_key)
    return {
        "n": len(scored),
        "overall": block(scored),
        "bootstrap": {"trials": trials, "seed": seed, "intervals": intervals},
        "by_category": {c: block([s for s in scored if s.category == c]) for c in categories},
        "baseline_matches_stored": max(
            (abs(base[s.qid][m] - s.stored_metrics[m]) for s in scored for m in METRICS),
            default=0.0,
        )
        < 1e-9,
        "per_question": {"baseline": base, "agentic": agent},
    }


def _category_key(category: str) -> tuple[int, str]:
    return (CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else 99, category)


def steps_report(specs: Sequence[QuestionSpec], loops: dict[str, LoopResult]) -> dict[str, Any]:
    """How many judge calls, how loops ended, and how often the loop refused."""

    def block(members: Sequence[QuestionSpec]) -> dict[str, Any]:
        results = [loops[s.qid] for s in members]
        return {
            "n": len(results),
            "calls": {str(n): sum(1 for r in results if r.n_calls == n) for n in (1, 2, 3)},
            "iterated": sum(1 for r in results if r.n_calls > 1),
            "outcomes": {o: sum(1 for r in results if r.outcome == o) for o in OUTCOMES},
            "changed_top10": sum(1 for s in members if loops[s.qid].top10 != list(s.baseline)),
        }

    answerable = [s for s in specs if s.question.is_answerable]
    unanswerable = [s for s in specs if not s.question.is_answerable]
    return {
        "all": block(specs),
        "scored": block([s for s in specs if s.scored]),
        "refusals": {
            "answerable": {
                "refused": sum(1 for s in answerable if loops[s.qid].refused),
                "n": len(answerable),
            },
            "unanswerable": {
                "refused": sum(1 for s in unanswerable if loops[s.qid].refused),
                "n": len(unanswerable),
            },
        },
    }


def calibration_report(
    scored: Sequence[QuestionSpec], loops: dict[str, LoopResult]
) -> dict[str, Any]:
    """Judge calibration: the step-1 verdict against whether any gold chunk is in the step-1
    top 5. Only the question's own first judge call is read, so a loop that went on to iterate
    does not move it. Invalid first replies are counted apart."""
    cells = {
        verdict: {"gold_in_top5": 0, "no_gold_in_top5": 0}
        for verdict in ("sufficient", "insufficient", "invalid")
    }
    for spec in scored:
        first = loops[spec.qid].steps[0]
        label = (
            "invalid"
            if first.sufficient is None
            else "sufficient"
            if first.sufficient
            else "insufficient"
        )
        has_gold = bool(set(spec.gold) & set(spec.baseline[:JUDGE_TOP]))
        cells[label]["gold_in_top5" if has_gold else "no_gold_in_top5"] += 1
    insufficient = cells["insufficient"]
    n_insufficient = sum(insufficient.values())
    return {
        "n": len(scored),
        "table": cells,
        "insufficient_and_no_gold_share": (
            insufficient["no_gold_in_top5"] / n_insufficient if n_insufficient else None
        ),
    }


def cost_report(
    members: Sequence[QuestionSpec],
    loops: dict[str, LoopResult],
    baseline_latency: dict[str, float],
) -> dict[str, Any]:
    """Cost of the loop per question against the baseline's measured retrieval latency."""

    def stats(values: Sequence[float]) -> dict[str, float]:
        return {
            "mean": _mean(values),
            "p50": statistics.median(values) if values else 0.0,
            "p95": percentile(values, 0.95),
        }

    results = [loops[s.qid] for s in members]
    return {
        "n": len(results),
        "judge_calls": {
            "mean": _mean([r.n_calls for r in results]),
            "total": sum(r.n_calls for r in results),
        },
        "prompt_tokens": {
            "mean": _mean([r.prompt_tokens for r in results]),
            "total": sum(r.prompt_tokens for r in results),
        },
        "completion_tokens": {
            "mean": _mean([r.completion_tokens for r in results]),
            "total": sum(r.completion_tokens for r in results),
        },
        "loop_latency_s": stats([r.latency_s for r in results]),
        "baseline_retrieval_latency_s": stats([baseline_latency[s.qid] for s in members]),
        "baseline_llm_calls": 0,
    }


# ---------------------------------------------------------------------------
# answer-quality report and the refusal guardrail
# ---------------------------------------------------------------------------
def guardrail(baseline: Sequence[AnswerRow], agentic: Sequence[AnswerRow]) -> dict[str, Any]:
    """The registered refusal guardrail, on counts.

    Agentic false refusal on the answerable questions is at most `5 points of n, rounded down`
    questions above the baseline's (2, for 54), and refusal recall on the unanswerable questions
    is not lower than the baseline's.
    """
    answerable_b = [r for r in baseline if not r.should_refuse]
    answerable_a = [r for r in agentic if not r.should_refuse]
    unanswerable_b = [r for r in baseline if r.should_refuse]
    unanswerable_a = [r for r in agentic if r.should_refuse]
    allowed = GUARDRAIL_POINTS * len(answerable_b) // 100
    fr_b = sum(r.refused for r in answerable_b)
    fr_a = sum(r.refused for r in answerable_a)
    rr_b = sum(r.refused for r in unanswerable_b)
    rr_a = sum(r.refused for r in unanswerable_a)
    false_refusal_ok = fr_a - fr_b <= allowed
    recall_ok = rr_a >= rr_b
    return {
        "holds": false_refusal_ok and recall_ok,
        "allowed_false_refusal_rise": allowed,
        "false_refusal": {"baseline": fr_b, "agentic": fr_a, "n": len(answerable_b)},
        "refusal_recall": {"baseline": rr_b, "agentic": rr_a, "n": len(unanswerable_b)},
        "false_refusal_ok": false_refusal_ok,
        "refusal_recall_ok": recall_ok,
    }


def answers_report(pairs: Sequence[AnswerPair], *, model: str) -> dict[str, Any]:
    """The paired answer table: both arms' summaries, the counts of each kind of question, the
    guardrail, and how the answerable questions moved between arms."""
    base_rows = [p.baseline for p in pairs]
    agent_rows = [p.agentic for p in pairs]
    answerable = [p for p in pairs if not p.baseline.should_refuse]
    both_answered = [p for p in answerable if not p.baseline.refused and not p.agentic.refused]
    return {
        "n": len(pairs),
        "kinds": {k: sum(1 for p in pairs if p.kind == k) for k in (TIE, CHANGED, LOOP_REFUSED)},
        "baseline": summarise(base_rows, arm="baseline", model=model).as_dict(),
        "agentic": summarise(agent_rows, arm="agentic", model=model).as_dict(),
        "guardrail": guardrail(base_rows, agent_rows),
        "paired": {
            "newly_refused": sum(
                1 for p in answerable if p.agentic.refused and not p.baseline.refused
            ),
            "newly_answered": sum(
                1 for p in answerable if p.baseline.refused and not p.agentic.refused
            ),
            "answered_by_both": len(both_answered),
            "grounded_in_both": sum(
                1 for p in both_answered if p.baseline.cited_gold and p.agentic.cited_gold
            ),
            "grounded_only_baseline": sum(
                1 for p in both_answered if p.baseline.cited_gold and not p.agentic.cited_gold
            ),
            "grounded_only_agentic": sum(
                1 for p in both_answered if p.agentic.cited_gold and not p.baseline.cited_gold
            ),
        },
    }


# ---------------------------------------------------------------------------
# validity gates and the verdict
# ---------------------------------------------------------------------------
def gate_two(steps: Sequence[Step], rows: Sequence[AnswerRow], *, model: str) -> dict[str, Any]:
    """Every judge and answer row was answered by the registered model; zero fallbacks.

    Build-time reading: a judge call that returned nothing is not an answer from another model
    and counts against gate 3 only; an answer row that is a `provider_error` was answered by no
    model and fails this gate. Rows with no model by construction (a loop refusal, or a refusal
    before any call) are not rows a model answered and are skipped.
    """
    other = [s.model for s in steps if s.model and s.model != model]
    other += [r.model for r in rows if r.model and r.model != model]
    errors = sum(1 for r in rows if r.refusal_reason == "provider_error")
    return {
        "id": 2,
        "name": "every judge and answer row answered by the registered model",
        "passed": not other and not errors,
        "answered_by_another_model": len(other),
        "answer_provider_errors": errors,
    }


def gate_three(steps: Sequence[Step]) -> dict[str, Any]:
    """Invalid judge replies are at most 5% of judge calls (a failed call is an invalid reply)."""
    invalid = sum(1 for s in steps if s.invalid)
    rate = Fraction(invalid, len(steps)) if steps else Fraction(1)
    return {
        "id": 3,
        "name": "invalid judge replies at most 5% of judge calls",
        "passed": bool(steps) and rate <= INVALID_RATE_BAR,
        "invalid": invalid,
        "judge_calls": len(steps),
        "rate": float(rate),
        "by_kind": {k: sum(1 for s in steps if s.invalid == k) for k in INVALID_KINDS},
    }


def gate_four(steps: Sequence[Step], rows: Sequence[AnswerRow]) -> dict[str, Any]:
    """The largest prompt, as the server counted it, is below the 8,192-token window."""
    judge_max = max((s.prompt_tokens for s in steps), default=0)
    answer_max = max((r.prompt_tokens for r in rows), default=0)
    largest = max(judge_max, answer_max)
    return {
        "id": 4,
        "name": "largest prompt below the 8,192-token window",
        "passed": largest < PROMPT_TOKEN_LIMIT,
        "max_judge_prompt_tokens": judge_max,
        "max_answer_prompt_tokens": answer_max,
        "max_judge_prompt_chars": max((s.prompt_chars for s in steps), default=0),
    }


def decide(delta: float, lo: float, hi: float, guardrail_holds: bool) -> str:
    """The registered verdict rule.

    Gain: delta nDCG@10 at least +0.02, the interval's lower bound above 0, and the guardrail
    held. Worse: the interval's upper bound below 0. Anything else, including a gain that fails
    the guardrail, is no gain for its cost.
    """
    if hi < 0:
        return WORSE
    if round(delta, 12) >= GAIN_BAR and lo > 0 and guardrail_holds:
        return GAIN
    return NO_GAIN


def verdict_sentence(
    verdict: str, retrieval: dict[str, Any], cost: dict[str, Any], guard: dict[str, Any]
) -> str:
    """The verdict with its cost, as the protocol requires every verdict sentence to carry."""
    ndcg = retrieval["bootstrap"]["intervals"][PRIMARY]
    c = cost["scored"]
    label = {GAIN: "Gain", WORSE: "Worse", NO_GAIN: "No gain for its cost"}[verdict]
    text = (
        f"{label}: delta nDCG@10 {ndcg['mean']:+.4f} (95% interval {ndcg['lo']:+.4f} to "
        f"{ndcg['hi']:+.4f}, n = {retrieval['n']})"
    )
    if verdict == NO_GAIN and not guard["holds"] and round(ndcg["mean"], 12) >= GAIN_BAR:
        text += "; the refusal guardrail failed"
    text += (
        f". Cost per question: {c['judge_calls']['mean']:.2f} judge calls, "
        f"{c['prompt_tokens']['mean']:.0f} prompt and {c['completion_tokens']['mean']:.0f} "
        f"completion tokens, loop latency p50 {c['loop_latency_s']['p50']:.1f} s and "
        f"p95 {c['loop_latency_s']['p95']:.1f} s (four concurrent requests); the baseline makes 0 "
        f"LLM calls and retrieves in {c['baseline_retrieval_latency_s']['mean'] * 1000:.0f} ms "
        f"on average (p50 {c['baseline_retrieval_latency_s']['p50'] * 1000:.0f} ms, "
        f"p95 {c['baseline_retrieval_latency_s']['p95'] * 1000:.0f} ms)."
    )
    return text


def build_result(
    inputs: Inputs,
    step_one: StepOne,
    loops: dict[str, LoopResult],
    pairs: Sequence[AnswerPair] | None,
    *,
    model: str = MODEL_ARMS[MODEL_ARM]["model"],
    trials: int = BOOTSTRAP_TRIALS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Everything the run reports, the four gates, and the verdict when every gate holds.

    With `pairs=None` (the loop phase only) the answer-quality part and the verdict are absent.
    A run with any gate failed, or with rows missing, carries `verdict: None` and says why.
    """
    specs = inputs.specs
    scored = [s for s in specs if s.scored]
    steps = [step for spec in specs for step in loops[spec.qid].steps]
    answer_rows = [r for p in pairs for r in (p.baseline, p.agentic)] if pairs is not None else []

    retrieval = retrieval_report(scored, loops, trials=trials, seed=seed)
    by_qid_latency = step_one.latency_s
    cost = {
        "scored": cost_report(scored, loops, by_qid_latency),
        "all": cost_report(specs, loops, by_qid_latency),
    }
    gates = [step_one.gate, gate_two(steps, answer_rows, model=model), gate_three(steps),
             gate_four(steps, answer_rows)]  # fmt: skip
    answers = answers_report(pairs, model=model) if pairs is not None else None

    complete = len(loops) == len(specs) and (pairs is None or len(pairs) == len(inputs.subset))
    failed = [f"gate {g['id']}: {g['name']}" for g in gates if not g["passed"]]
    if not complete:
        failed.append("incomplete: rows are missing")
    if pairs is None:
        failed.append("answer phase not run")

    verdict = None
    sentence = "no verdict: " + "; ".join(failed) if failed else ""
    if not failed:
        assert answers is not None
        ndcg = retrieval["bootstrap"]["intervals"][PRIMARY]
        verdict = decide(ndcg["mean"], ndcg["lo"], ndcg["hi"], answers["guardrail"]["holds"])
        sentence = verdict_sentence(verdict, retrieval, cost, answers["guardrail"])

    return {
        "verdict": verdict,
        "verdict_text": sentence,
        "gates": gates,
        "complete": complete,
        "questions": {
            "n": len(specs),
            "scored": len(scored),
            "by_category_all": _counts(specs),
            "by_category_scored": _counts(scored),
            "subset": len(inputs.subset),
        },
        "retrieval": retrieval,
        "steps": steps_report(specs, loops),
        "calibration": calibration_report(scored, loops),
        "cost": cost,
        "answers": answers,
    }


def _counts(specs: Sequence[QuestionSpec]) -> dict[str, int]:
    counts = Counter(s.category for s in specs)
    return {c: counts[c] for c in sorted(counts, key=_category_key)}


def question_records(
    inputs: Inputs, step_one: StepOne, loops: dict[str, LoopResult], retrieval: dict[str, Any]
) -> list[dict[str, Any]]:
    """One record per question: both rankings, every step, and the per-question metrics."""
    base = retrieval["per_question"]["baseline"]
    agent = retrieval["per_question"]["agentic"]
    records = []
    for spec in inputs.specs:
        loop = loops[spec.qid]
        records.append(
            {
                "qid": spec.qid,
                "category": spec.category,
                "answerable": spec.question.is_answerable,
                "scored": spec.scored,
                "in_subset": spec.in_subset,
                "gold": list(spec.gold),
                "baseline_top10": list(spec.baseline),
                "final_top10": loop.top10,
                "changed": loop.top10 != list(spec.baseline),
                "outcome": loop.outcome,
                "refused": loop.refused,
                "n_calls": loop.n_calls,
                "queries": loop.queries,
                "prompt_tokens": loop.prompt_tokens,
                "completion_tokens": loop.completion_tokens,
                "loop_latency_s": loop.latency_s,
                "baseline_latency_s": step_one.latency_s[spec.qid],
                "metrics": {"baseline": base[spec.qid], "agentic": agent[spec.qid]}
                if spec.scored
                else None,
                "steps": [s.as_dict() for s in loop.steps],
            }
        )
    return records


def pair_records(pairs: Sequence[AnswerPair]) -> list[dict[str, Any]]:
    return [
        {
            "qid": p.qid,
            "kind": p.kind,
            "baseline_source": p.baseline_source,
            "agentic_source": p.agentic_source,
            "baseline": p.baseline.as_dict(),
            "agentic": p.agentic.as_dict(),
        }
        for p in pairs
    ]


# ---------------------------------------------------------------------------
# planning, reporting, saving
# ---------------------------------------------------------------------------
def plan_budget(n_questions: int, n_subset: int, *, max_steps: int = MAX_STEPS) -> dict[str, int]:
    """The call budget a run can spend, before it spends any of it.

    Judge calls are 1 to 3 per question. The answer phase generates two answers for a question
    whose evidence changed and one (the baseline's) for a loop refusal, so at most two per
    subset question; a ties-only run generates none. A live retry (a cut or empty reply) can
    add one call per generation.
    """
    return {
        "loop_questions": n_questions,
        "judge_calls_min": n_questions,
        "judge_calls_max": n_questions * max_steps,
        "answer_questions": n_subset,
        "answer_generations_min": 0,
        "answer_generations_max": 2 * n_subset,
        "llm_calls_max": n_questions * max_steps + 2 * n_subset,
    }


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.1f}%"


def format_report(result: dict[str, Any]) -> str:
    """The numbers a reader checks first, as text. The result file has the rest."""
    r = result["retrieval"]
    lines = [
        "| metric | baseline | agentic | delta | 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in (PRIMARY, *SECONDARY):
        row = r["overall"][metric]
        ci = r["bootstrap"]["intervals"][metric]
        lines.append(
            f"| {metric} | {row['baseline']:.4f} | {row['agentic']:.4f} | {row['delta']:+.4f} | "
            f"{ci['lo']:+.4f} to {ci['hi']:+.4f} |"
        )
    lines += ["", "| category | n | nDCG@10 base | agentic | recall@5 base | agentic |"]
    lines.append("|---|---:|---:|---:|---:|---:|")
    for category, block in r["by_category"].items():
        lines.append(
            f"| {category} | {block['n']} | {block['ndcg@10']['baseline']:.3f} | "
            f"{block['ndcg@10']['agentic']:.3f} | {block['recall@5']['baseline']:.3f} | "
            f"{block['recall@5']['agentic']:.3f} |"
        )
    s = result["steps"]["all"]
    lines += [
        "",
        f"judge calls per question (all {s['n']}): "
        + ", ".join(f"{n}: {c}" for n, c in s["calls"].items())
        + f"; iterated {s['iterated']}; outcomes {s['outcomes']}",
    ]
    ref = result["steps"]["refusals"]
    lines.append(
        f"loop refusals: {ref['answerable']['refused']}/{ref['answerable']['n']} answerable, "
        f"{ref['unanswerable']['refused']}/{ref['unanswerable']['n']} unanswerable"
    )
    c = result["calibration"]["table"]
    lines.append(
        "judge calibration (step 1 vs gold in top 5): "
        + "; ".join(
            f"{k}: {v['gold_in_top5']} with gold, {v['no_gold_in_top5']} without"
            for k, v in c.items()
        )
    )
    if result["answers"]:
        a = result["answers"]
        g = a["guardrail"]
        lines += [
            "",
            f"answers ({a['n']} questions; {a['kinds']}): false refusal "
            f"{g['false_refusal']['baseline']} -> {g['false_refusal']['agentic']} of "
            f"{g['false_refusal']['n']} (allowed rise {g['allowed_false_refusal_rise']}), "
            f"refusal recall {g['refusal_recall']['baseline']} -> "
            f"{g['refusal_recall']['agentic']} of {g['refusal_recall']['n']}; guardrail "
            f"{'holds' if g['holds'] else 'FAILS'}",
        ]
    lines.append("")
    for gate in result["gates"]:
        lines.append(f"gate {gate['id']} {'ok  ' if gate['passed'] else 'FAIL'} {gate['name']}")
    lines += ["", result["verdict_text"]]
    return "\n".join(lines)


def save(path: Path, payload: dict[str, Any]) -> None:
    """Write the finished run, once, beside the target and renamed into place, so a crash
    cannot leave a partial file that looks like the result."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; a finished run is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, ensure_ascii=False) + "\n"  # fails before any file exists
    staging = path.with_name(path.name + ".partial")
    with staging.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    staging.rename(path)


def run_config(model: str) -> dict[str, Any]:
    """What a saved run records about its own inputs."""
    return {
        "protocol_commit": PROTOCOL_COMMIT,
        "model_arm": MODEL_ARM,
        "model": model,
        "arm": ARM,
        "strategy": STRATEGY,
        "k": K,
        "pool": POOL,
        "judge_top": JUDGE_TOP,
        "passage_chars": PASSAGE_CHARS,
        "max_steps": MAX_STEPS,
        "rrf_k": RRF_K,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "num_ctx_live": NUM_CTX,
        "bootstrap_trials": BOOTSTRAP_TRIALS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "gain_bar": GAIN_BAR,
        "judge_system": JUDGE_SYSTEM,
        "judge_template": JUDGE_TEMPLATE,
    }
