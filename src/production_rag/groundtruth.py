"""Ground truth: questions, the evidence that answers them, and how it is checked.

This is the part of a RAG project that usually does not exist, and its absence
is why so many of them report no retrieval numbers at all. Two problems have to
be solved before a single metric means anything.

**Problem 1: gold cannot be a chunk id.** The obvious representation -- "question
Q is answered by chunk 47" -- silently assumes one chunking strategy. This
project has three, they produce 14,660 / 22,789 / 24,120 chunks respectively,
and no id is shared between them. Ground truth pinned to chunk ids would make
the cross-strategy comparison impossible, or worse, would quietly force a
separate labelling effort per strategy and then compare numbers built on
different labels.

So gold is a **character span in a document**: ``(doc_id, start, end)``, anchored
to a verbatim quote. Every chunker already records the span each chunk came
from, so the gold *chunks* for a strategy are derived -- a chunk is gold if it
overlaps the evidence span. One labelling effort, three comparable evaluations,
and the labels survive a change to the chunk size. See :func:`gold_chunk_ids`.

**Problem 2: an LLM that writes the questions decides what "relevant" means.**
Asking a model to generate a question from a passage and then measuring whether
retrieval finds that passage is close to circular -- the model writes questions
whose vocabulary it has just read, which flatters lexical retrieval especially.
Three defences, in increasing order of how much they cost:

1. *The quote must exist.* The model's claimed evidence is located in the real
   document by verbatim (whitespace-normalised) match. A hallucinated quote is
   a rejected proposal, not a silent mislabel. This is free and it removes the
   most common failure.
2. *The category is computed, not claimed.* Whether a question is
   "exact-terminology" or "conceptual" is decided by corpus statistics --
   does it share a rare token with its own gold passage? -- not by the label the
   generating model attached. A model asked to write a conceptual question and
   then to say whether it did will say yes. See :func:`categorise`.
3. *A human verifies a sample.* Nothing above establishes that the question is
   sensible, answerable, or that the gold passage is really the best evidence.
   Only a person who knows the tools can say that, which is why the README
   reports the size of the verified subset and the correction rate found in it
   rather than treating the generated set as ground truth.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .bm25 import BM25Index, tokenize
from .chunking import Chunk
from .ingest import Document
from .providers import LLMProvider, ProviderError, extract_json

QUESTIONS_PATH = Path("eval/questions.jsonl")

CATEGORIES = ("exact_term", "conceptual", "cross_tool", "unanswerable")

#: A token appearing in at most this many chunks counts as "rare" for the
#: purpose of categorising a question. 50 out of ~23k chunks is 0.2% of the
#: corpus -- roughly the frequency band that identifiers like ``on_schema_change``
#: and ``read_parquet`` occupy, and well below ordinary English.
RARE_DF = 50

#: Minimum share of an evidence span a chunk must cover to count as gold.
#: Below this, a chunk that merely clips the edge of the answer would be scored
#: as a hit -- inflating every arm equally, but inflating them.
MIN_SPAN_OVERLAP = 0.5

_WS = re.compile(r"\s+")

# Phrases that give away a question written *about a passage* rather than by
# someone with a problem. "In the passage above" is unanswerable by a retrieval
# system that has no passage yet.
_CONTEXT_DEPENDENT = re.compile(
    r"\b(this (passage|document|page|section|text|example|snippet)"
    r"|the (passage|above|following|preceding)|as (described|shown|mentioned) (above|here))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Evidence:
    """A character span in one document that answers a question."""

    doc_id: str
    start: int
    end: int
    quote: str
    """Verbatim text of the span, kept so the label can be re-verified after a
    re-ingest: if the quote no longer matches, the offsets moved and the label
    is stale rather than quietly wrong."""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Question:
    """One evaluation question."""

    qid: str
    text: str
    category: str
    evidence: tuple[Evidence, ...] = ()
    tools: tuple[str, ...] = ()
    source: str = ""
    """`llm:<model>` or `hand`."""

    verified: str = "unverified"
    """`unverified`, `accepted`, `edited` or `rejected`."""

    verifier: str = ""
    notes: str = ""
    proposed_category: str = ""
    """What the generating model claimed. Kept so the disagreement rate between
    the model's self-label and the computed category can be reported."""

    def as_dict(self) -> dict:
        data = asdict(self)
        data["evidence"] = [e.as_dict() for e in self.evidence]
        data["tools"] = list(self.tools)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Question:
        return cls(
            **{
                **data,
                "evidence": tuple(Evidence(**e) for e in data.get("evidence", ())),
                "tools": tuple(data.get("tools", ())),
            }
        )

    @property
    def is_answerable(self) -> bool:
        return bool(self.evidence)


def save_questions(questions: Sequence[Question], path: Path = QUESTIONS_PATH) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            handle.write(json.dumps(question.as_dict(), ensure_ascii=False) + "\n")
    return len(questions)


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    with Path(path).open(encoding="utf-8") as handle:
        return [Question.from_dict(json.loads(line)) for line in handle if line.strip()]


HAND_WRITTEN_PATH = Path("eval/handwritten.jsonl")


def load_hand_written(path: Path, documents: dict[str, Document]) -> list[Question]:
    """Read the human-authored questions and resolve their quotes to spans.

    These are the questions no passage-sampling loop can produce: **cross-tool**
    ones, whose answer lives in two tools' documentation at once and which are
    therefore the case the hybrid retriever is supposed to earn its keep on; and
    **unanswerable** ones, which have no evidence anywhere in the corpus and
    exist to test refusal rather than recall.

    The file stores a quote, not an offset, so it stays editable by hand and
    survives a re-ingest that shifts character positions. A quote that no longer
    locates raises immediately: silently dropping it would shrink the eval set
    without anyone noticing, and the cross-tool bucket is small enough that
    losing one question moves its mean.
    """
    questions: list[Question] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            evidence: list[Evidence] = []
            for item in row.get("evidence", ()):
                document = documents.get(item["doc_id"])
                if document is None:
                    raise ValueError(f"{path}:{line_number}: unknown doc_id {item['doc_id']!r}")
                span = locate_quote(document.text, item["quote"])
                if span is None:
                    raise ValueError(
                        f"{path}:{line_number}: quote not found in {item['doc_id']!r}. "
                        "Either the corpus was re-pinned or the quote was mistyped."
                    )
                evidence.append(
                    Evidence(item["doc_id"], span[0], span[1], document.text[span[0] : span[1]])
                )
            questions.append(
                Question(
                    qid="",
                    text=row["text"].strip(),
                    category="",
                    evidence=tuple(evidence),
                    tools=tuple(row.get("tools", ())),
                    source="hand",
                    notes=row.get("notes", ""),
                )
            )
    return questions


# ---------------------------------------------------------------------------
# Spans -> gold chunks, per strategy
# ---------------------------------------------------------------------------
def index_chunks_by_doc(chunks: Iterable[Chunk]) -> dict[str, list[Chunk]]:
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    return by_doc


def gold_chunk_ids(
    question: Question,
    by_doc: dict[str, list[Chunk]],
    *,
    min_overlap: float = MIN_SPAN_OVERLAP,
) -> set[str]:
    """Chunks of one strategy that carry this question's evidence.

    A chunk qualifies if it covers at least `min_overlap` of the evidence span,
    **or** if it contains the quote verbatim. The second rule is not redundant:
    a strategy whose chunk boundaries fall inside the span can leave every chunk
    below the overlap threshold while one of them still contains the entire
    answer sentence. Falling back to the single best-overlapping chunk when
    neither rule fires keeps a question from silently becoming unscoreable --
    which would drop it from that strategy's mean and make the strategies
    incomparable again.
    """
    gold: set[str] = set()
    for evidence in question.evidence:
        candidates = by_doc.get(evidence.doc_id, [])
        span = max(evidence.end - evidence.start, 1)
        best: tuple[float, str] | None = None
        for chunk in candidates:
            overlap = min(chunk.end, evidence.end) - max(chunk.start, evidence.start)
            share = overlap / span
            if share >= min_overlap or (evidence.quote and evidence.quote in chunk.text):
                gold.add(chunk.chunk_id)
            if best is None or share > best[0]:
                best = (share, chunk.chunk_id)
        if best is not None and best[0] > 0 and not gold & {c.chunk_id for c in candidates}:
            gold.add(best[1])
    return gold


def coverage_report(
    questions: Sequence[Question], by_doc: dict[str, list[Chunk]]
) -> dict[str, float]:
    """How well a strategy's chunks cover the labelled evidence.

    Reported alongside the metrics because a strategy that maps 5% of questions
    to no gold chunk at all is not merely scoring worse -- it is being scored on
    a different question set, and comparing its mean to another strategy's would
    be meaningless.
    """
    answerable = [q for q in questions if q.is_answerable]
    if not answerable:
        return {"questions": 0.0, "mapped": 0.0, "mean_gold_chunks": 0.0}
    counts = [len(gold_chunk_ids(q, by_doc)) for q in answerable]
    return {
        "questions": float(len(answerable)),
        "mapped": sum(1 for c in counts if c) / len(counts),
        "mean_gold_chunks": float(np.mean(counts)),
    }


# ---------------------------------------------------------------------------
# Locating a quote in a document
# ---------------------------------------------------------------------------
def locate_quote(
    text: str, quote: str, *, hint: tuple[int, int] | None = None
) -> tuple[int, int] | None:
    """Find `quote` in `text`, tolerating whitespace differences only.

    Whitespace is normalised on both sides because a model reproducing a quote
    across a line wrap is still quoting; anything beyond that is not. No fuzzy
    matching, no edit distance: the point of this check is to reject invented
    evidence, and a matcher loose enough to accept a paraphrase would accept an
    invention too. When several copies of the quote exist, the one nearest
    `hint` wins -- boilerplate really does repeat across a page.
    """
    quote = quote.strip()
    if len(quote) < 20:
        # Too short to identify a span. "true" appears in a thousand documents.
        return None

    # Build a map from normalised-text offsets back to raw offsets.
    raw_offsets: list[int] = []
    normalised_chars: list[str] = []
    previous_space = True
    for i, char in enumerate(text):
        if char.isspace():
            if previous_space:
                continue
            normalised_chars.append(" ")
            raw_offsets.append(i)
            previous_space = True
        else:
            normalised_chars.append(char)
            raw_offsets.append(i)
            previous_space = False
    normalised = "".join(normalised_chars)
    needle = _WS.sub(" ", quote).strip()
    if not needle:
        return None

    positions: list[int] = []
    cursor = normalised.find(needle)
    while cursor != -1:
        positions.append(cursor)
        cursor = normalised.find(needle, cursor + 1)
    if not positions:
        return None

    if hint is not None:
        centre = (hint[0] + hint[1]) / 2
        positions.sort(key=lambda p: abs(raw_offsets[p] - centre))

    start_norm = positions[0]
    end_norm = start_norm + len(needle) - 1
    return raw_offsets[start_norm], raw_offsets[end_norm] + 1


# ---------------------------------------------------------------------------
# Category, computed from the corpus rather than claimed by the model
# ---------------------------------------------------------------------------
def document_frequencies(index: BM25Index) -> dict[str, int]:
    """Term -> number of chunks containing it, recovered from the weight matrix.

    Every stored weight is strictly positive (the IDF term cannot be zero for
    any term that appears at all), so the nonzero count of a term's column is
    exactly its document frequency. Recomputing it from the corpus would mean
    re-tokenising 12.8 M characters to learn something the index already knows.
    """
    per_column = np.diff(index.weights.tocsc().indptr)
    return {term: int(per_column[column]) for term, column in index.vocabulary.items()}


def rare_shared_terms(
    question: str, gold_text: str, df: dict[str, int], *, rare_df: int = RARE_DF
) -> set[str]:
    """Rare tokens the question shares verbatim with its own gold passage."""
    gold_tokens = set(tokenize(gold_text))
    return {
        token
        for token in tokenize(question)
        if token in gold_tokens and 0 < df.get(token, 0) <= rare_df
    }


def categorise(
    question: str,
    gold_text: str,
    df: dict[str, int],
    *,
    tools: Sequence[str] = (),
    answerable: bool = True,
    rare_df: int = RARE_DF,
) -> str:
    """Assign a question category from evidence, not from a claimed label."""
    if not answerable:
        return "unanswerable"
    if len({t for t in tools if t}) > 1:
        return "cross_tool"
    return (
        "exact_term"
        if rare_shared_terms(question, gold_text, df, rare_df=rare_df)
        else "conceptual"
    )


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------
PROPOSE_SYSTEM = (
    "You write realistic questions that a working data engineer would type into a "
    "documentation search box. You return JSON only."
)

PROPOSE_TEMPLATE = """\
Below is one passage from the {tool} documentation.

Write {n} questions that this passage answers, as a data engineer would actually type them
into a search box -- not as an exam question about the passage.

Write one question of each style:
1. "literal": the user knows the exact name of the thing (a config key, function, flag or
   error string) and types it verbatim.
2. "conceptual": the user has the problem but does not know the terminology, so the
   question must NOT contain any distinctive identifier from the passage -- describe the
   situation in plain words instead.

Hard rules:
- Never refer to "this passage", "the document", "above" or "the following". The reader of
  the question has not seen the passage.
- The question must be answerable from the passage alone.
- For each question, give "quote": a span copied EXACTLY, character for character, from the
  passage, that contains the answer. It must be at least 20 characters and must appear in
  the passage verbatim. Do not paraphrase the quote.

Return JSON: {{"questions": [{{"question": "...", "style": "literal", "quote": "..."}}]}}

PASSAGE (from {title}):
{passage}
"""


@dataclass
class ProposalStats:
    """Why proposals were thrown away. Reported, not hidden."""

    passages: int = 0
    returned: int = 0
    rejected_unparseable: int = 0
    rejected_no_quote: int = 0
    rejected_context_dependent: int = 0
    rejected_too_short: int = 0
    rejected_duplicate: int = 0
    accepted: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["errors"] = self.errors[:10]
        return data


def sample_passages(
    chunks: Sequence[Chunk],
    *,
    n: int,
    seed: int = 20240908,
    min_chars: int = 400,
    max_chars: int = 2500,
) -> list[Chunk]:
    """Pick passages to generate questions from, stratified by tool.

    Stratification is not cosmetic. The corpus is 1,145 dbt documents against
    430 DuckDB ones, so uniform sampling would produce an eval set that is
    mostly dbt and a headline number that mostly measures dbt retrieval.
    """
    eligible = [c for c in chunks if min_chars <= len(c.text) <= max_chars]
    by_tool: dict[str, list[Chunk]] = {}
    for chunk in eligible:
        by_tool.setdefault(chunk.tool, []).append(chunk)

    rng = random.Random(seed)
    picked: list[Chunk] = []
    tools = sorted(by_tool)
    per_tool = max(n // max(len(tools), 1), 1)
    for tool in tools:
        pool = sorted(by_tool[tool], key=lambda c: c.chunk_id)
        rng.shuffle(pool)
        picked.extend(pool[:per_tool])
    rng.shuffle(picked)
    return picked[:n]


def propose_from_passage(
    chunk: Chunk,
    doc_text: str,
    provider: LLMProvider,
    *,
    n: int = 2,
    structured: bool = True,
    max_tokens: int = 700,
) -> tuple[list[dict], str]:
    """Ask the provider for `n` questions about one passage.

    Returns the raw proposals and an error string; screening happens in
    :func:`screen_proposals` so that the network step and the judgement step can
    be tested, cached and re-run independently of each other.
    """
    prompt = PROPOSE_TEMPLATE.format(
        n=n, tool=chunk.tool, title=chunk.breadcrumb, passage=chunk.text
    )
    try:
        response = provider.complete(
            prompt,
            system=PROPOSE_SYSTEM,
            temperature=0.0,
            max_tokens=max_tokens,
            json_object=structured,
        )
    except ProviderError as exc:
        return [], f"{chunk.chunk_id}: provider: {exc}"

    try:
        data = extract_json(response.text)
    except ValueError as exc:
        return [], f"{chunk.chunk_id}: unparseable: {exc}"

    if isinstance(data, dict):
        data = data.get("questions", [])
    if not isinstance(data, list):
        return [], f"{chunk.chunk_id}: expected a list of questions"
    return [item for item in data if isinstance(item, dict)], ""


def screen_proposals(
    proposals: Sequence[dict],
    chunk: Chunk,
    document: Document,
    *,
    model: str,
    seen: set[str],
    stats: ProposalStats,
    min_question_chars: int = 25,
) -> list[Question]:
    """Turn raw model output into labelled questions, or throw it away.

    Everything rejected here is counted. A generation pipeline that reports only
    its survivors is reporting a number about its filter, not about its model.
    """
    accepted: list[Question] = []
    for item in proposals:
        stats.returned += 1
        text = str(item.get("question", "")).strip()
        quote = str(item.get("quote", "")).strip()
        style = str(item.get("style", "")).strip().lower()

        if len(text) < min_question_chars:
            stats.rejected_too_short += 1
            continue
        if _CONTEXT_DEPENDENT.search(text):
            stats.rejected_context_dependent += 1
            continue

        key = _WS.sub(" ", text.lower()).strip("?. ")
        if key in seen:
            stats.rejected_duplicate += 1
            continue

        span = locate_quote(document.text, quote, hint=(chunk.start, chunk.end))
        if span is None:
            stats.rejected_no_quote += 1
            continue

        seen.add(key)
        accepted.append(
            Question(
                qid="",  # assigned once the whole batch is screened
                text=text,
                category="",  # computed after, from corpus statistics
                evidence=(
                    Evidence(
                        doc_id=document.doc_id,
                        start=span[0],
                        end=span[1],
                        quote=document.text[span[0] : span[1]],
                    ),
                ),
                tools=(chunk.tool,),
                source=f"llm:{model}",
                proposed_category={"literal": "exact_term"}.get(style, "conceptual"),
            )
        )
    stats.accepted += len(accepted)
    return accepted


def assign_categories(
    questions: Sequence[Question], documents: dict[str, Document], df: dict[str, int]
) -> list[Question]:
    """Fill in the computed category for a batch of screened questions."""
    out: list[Question] = []
    for question in questions:
        gold_text = " ".join(
            documents[e.doc_id].text[e.start : e.end]
            for e in question.evidence
            if e.doc_id in documents
        )
        out.append(
            replace(
                question,
                category=categorise(
                    question.text,
                    gold_text,
                    df,
                    tools=question.tools,
                    answerable=question.is_answerable,
                ),
            )
        )
    return out


def assign_ids(questions: Sequence[Question], *, prefix: str = "q") -> list[Question]:
    """Stable, readable ids: `q0001-dbt-exact_term`."""
    out: list[Question] = []
    for i, question in enumerate(sorted(questions, key=lambda q: q.text), start=1):
        tool = question.tools[0] if question.tools else "mixed"
        out.append(
            replace(question, qid=f"{prefix}{i:04d}-{tool}-{question.category or 'unknown'}")
        )
    return out
