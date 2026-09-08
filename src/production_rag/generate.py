"""Answering from retrieved context -- with citations that are checked, and a
refusal path that is measured rather than promised.

Everything up to here produced a *ranking*. A ranking is not an answer, and the
two fail in different ways. The retrieval half of this project can be scored
with recall and nDCG against gold spans; the generation half cannot, because
there is no gold answer string and an LLM judge would only move the question of
whose opinion counts. So this module is built around the three properties of a
generated answer that *are* checkable without a human reading every output:

1. **Every citation resolves to a passage the model was actually shown.**
   The prompt hands the model `N` numbered passages and asks for ``[n]`` markers.
   A marker outside ``1..N`` is a fabricated source, and it is detectable by
   arithmetic. This is the cheapest possible grounding check and it catches the
   most embarrassing failure -- an answer that cites `[7]` when six passages
   were supplied looks authoritative and is not.

2. **An answer that claims sufficiency carries at least one citation.**
   "Uncited assertion" is the failure mode that reads best and grounds least.
   It is counted separately from a wrong citation because the fix is different:
   a dangling marker is a parsing/prompting problem, an uncited paragraph is
   the model ignoring the instruction entirely.

3. **The cited passages can be compared to the gold spans.** Retrieval gold
   already exists per question, so "did the answer cite a passage that actually
   contains the evidence" is computable for free over the whole eval set. It is
   a weaker claim than "the answer is correct" and it is stated as the weaker
   claim -- but it is a real number, not a vibe.

**Refusal is two gates, and only one of them turned out to work.**

The appealing design is to refuse *before* paying for a generation: if the top
retrieval score is low, nothing relevant was found, so say so for free. That
gate exists here (``min_top_score``) and it is **off by default**, because the
measurement says it does not work on this corpus. The fused score is min-max
normalised per query -- that is what makes two rankings addable -- and
normalisation is exactly the step that throws away the absolute magnitude the
gate needs. On the session-2 grid the six unanswerable questions have a *higher*
median fused top score (2.80) than the 121 conceptual ones (2.76). The raw,
un-normalised scores do retain some signal (BM25 AUC 0.81, dense 0.84,
separating answerable from unanswerable) but at n=6 unanswerable that is an
observation, not a threshold anyone should ship. So the gate stays wired,
defaulted off, and the README says why rather than quietly omitting the idea.

What is left is the **generation gate**: the model returns ``sufficient: false``
and a refusal sentence. That is the path the answer eval measures, on both the
questions that should be refused and the ones that should not -- a refuser that
refuses everything scores perfectly on unanswerable questions, so the false
refusal rate on answerable ones is reported next to it, always.

**Why the answer is JSON and the citations are inline.** The model returns
``{"sufficient": bool, "answer": "..."}`` with ``[n]`` markers inside the prose,
not a separate citation list. A separate list is easier to parse and is a lie:
it says "these sources support this answer" without saying which sentence each
one supports. Markers in the prose keep the association the user actually needs
when they click through, and they still parse with one regex.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .chunking import Chunk
from .providers import LLMProvider, LLMResponse, ProviderError, extract_json

#: How much retrieved text to put in front of the model. Ten `heading` chunks
#: average well under this; the cap exists so a pathological long chunk cannot
#: push the prompt past a small model's window and truncate the *instructions*,
#: which are at the top. Passages are dropped whole, never mid-sentence, and the
#: count of dropped passages is reported on the answer.
DEFAULT_CONTEXT_CHARS = 12_000

#: Longest single passage kept verbatim. Above this the passage is cut and
#: marked, so one 40k-character API reference cannot evict every other source.
MAX_PASSAGE_CHARS = 3_000

REFUSAL_TEXT = "The documentation provided does not answer this question."

ANSWER_SYSTEM = (
    "You answer questions about DuckDB, dbt and Dagster strictly from the numbered "
    "documentation passages you are given. You return JSON only."
)

# Three things here are load-bearing, and the third was learned the expensive
# way.
#
# "Do not use knowledge from outside the passages" alone is not enough -- a
# model that half-remembers the right answer will write it and attach the
# nearest-looking citation, which is worse than a refusal because it is
# unfalsifiable to a reader. And naming refusal as a *correct* outcome, with a
# fixed sentence, makes not-answering an available move rather than a failure
# the model is trying to avoid.
#
# The third: **the output example must itself be cited, and must contain no
# content worth stealing.** The first version of this prompt asked for `[n]`
# markers in rule 1 and then showed `{"answer": "..."}` as the shape to return.
# Across 60 questions the model wrote a marker **twice**, produced good answers
# from the right passages, and cited none of them -- because an example is a
# stronger instruction than a rule, and the example contained no markers.
#
# The first repair used a realistic example (`on_schema_change` set to
# `append_new_columns`) and immediately produced the opposite failure:
# llama3.2:3b lifted those identifiers verbatim into an answer about *Dagster
# asset dependencies*, where they do not belong. A small model treats a concrete
# example as retrieved context. So the example is now shouty placeholder text --
# it demonstrates marker placement and carries nothing a model could plausibly
# copy as fact.
ANSWER_TEMPLATE = """\
Answer the question using ONLY the numbered passages below.

Rules:
- EVERY sentence that states a fact must end with a square-bracket marker naming the
  passage it came from, like [1] or [2][3]. An answer with no markers is wrong even if
  the facts in it are right.
- Never cite a number that is not in the list of passages below.
- Do not use any knowledge that is not in the passages, even if you are confident it is
  correct. If the passages do not contain the answer, that is a correct outcome, not a
  failure: set "sufficient" to false and make "answer" exactly this sentence:
  {refusal}
- Keep identifiers, config keys, flags and code exactly as they appear in the passages.
- Be brief: at most {sentences} sentences, each one cited.

Return JSON in exactly this shape, including the markers:
{{"sufficient": true, "answer": "FIRST SENTENCE OF THE ANSWER [2]. SECOND SENTENCE [2][5]."}}

or, when the passages do not answer the question:
{{"sufficient": false, "answer": "{refusal}"}}

PASSAGES
{context}

QUESTION
{question}
"""

_MARKER = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


@dataclass(frozen=True)
class Passage:
    """One numbered passage as it was shown to the model."""

    number: int
    chunk_id: str
    tool: str
    breadcrumb: str
    url: str
    text: str
    truncated: bool = False

    def render(self) -> str:
        body = self.text.strip()
        if self.truncated:
            body += "\n[... passage truncated ...]"
        return f"[{self.number}] {self.breadcrumb}\n{self.url}\n{body}"


@dataclass(frozen=True)
class Citation:
    """One ``[n]`` the model wrote, resolved back to what it was shown.

    `chunk_id` is empty when the marker points outside the numbered passages --
    the fabricated-source case. Keeping the invalid ones in the list rather than
    discarding them is deliberate: they are the measurement.
    """

    marker: int
    chunk_id: str = ""
    url: str = ""
    breadcrumb: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.chunk_id)


@dataclass
class Answer:
    """A generated answer plus everything needed to audit it."""

    question: str
    text: str
    refused: bool
    refusal_reason: str = ""
    """`""`, `"no_context"`, `"low_score"`, `"model"`, `"unparseable"` or
    `"provider_error"`. The last two refuse for the user's sake -- a broken
    generation must not be presented as an answer -- and are counted apart from
    a deliberate refusal, because they are bugs and refusals are not."""

    citations: list[Citation] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    raw: str = ""
    model: str = ""
    provider: str = ""
    latency_s: float = 0.0
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    dropped_passages: int = 0
    error: str = ""

    @property
    def context_ids(self) -> list[str]:
        return [p.chunk_id for p in self.passages]

    @property
    def cited_chunk_ids(self) -> list[str]:
        """Chunk ids for the valid citations, in first-mention order."""
        seen: list[str] = []
        for citation in self.citations:
            if citation.valid and citation.chunk_id not in seen:
                seen.append(citation.chunk_id)
        return seen

    @property
    def invalid_citations(self) -> list[Citation]:
        return [c for c in self.citations if not c.valid]

    @property
    def uncited(self) -> bool:
        """Claimed an answer and supported none of it."""
        return not self.refused and not any(c.valid for c in self.citations)

    @property
    def clean_text(self) -> str:
        """The answer with unresolvable markers removed.

        Shown to a user; never used for measurement. A dangling ``[7]`` in the
        UI invites a click that goes nowhere, but deleting it from the object
        would delete the evidence that it happened, so the raw text stays on
        `text` and only the display copy is cleaned.
        """
        invalid = {c.marker for c in self.invalid_citations}
        if not invalid:
            return self.text

        def scrub(match: re.Match[str]) -> str:
            kept = [n for n in _split_marker(match.group(1)) if n not in invalid]
            return "".join(f"[{n}]" for n in kept)

        return re.sub(r"\s+([.,;:])", r"\1", _MARKER.sub(scrub, self.text)).strip()

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "text": self.text,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "citations": [
                {"marker": c.marker, "chunk_id": c.chunk_id, "url": c.url} for c in self.citations
            ],
            "context_ids": self.context_ids,
            "model": self.model,
            "provider": self.provider,
            "latency_s": self.latency_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "dropped_passages": self.dropped_passages,
            "error": self.error,
        }


def _split_marker(group: str) -> list[int]:
    return [int(part) for part in re.split(r"[,;]", group) if part.strip()]


def build_passages(
    chunk_ids: Sequence[str],
    chunks: dict[str, Chunk],
    *,
    max_chars: int = DEFAULT_CONTEXT_CHARS,
    max_passage_chars: int = MAX_PASSAGE_CHARS,
) -> tuple[list[Passage], int]:
    """Number the retrieved chunks and fit them into the context budget.

    Returns the passages kept and how many were dropped. Order is retrieval
    order, so passage `[1]` is rank 1 -- which means a citation number doubles
    as a statement about where in the ranking the evidence was found.
    """
    passages: list[Passage] = []
    used = 0
    dropped = 0
    for chunk_id in chunk_ids:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            dropped += 1
            continue
        text = chunk.text
        truncated = len(text) > max_passage_chars
        if truncated:
            text = text[:max_passage_chars]
        if used + len(text) > max_chars and passages:
            dropped += 1
            continue
        used += len(text)
        passages.append(
            Passage(
                number=len(passages) + 1,
                chunk_id=chunk.chunk_id,
                tool=chunk.tool,
                breadcrumb=chunk.breadcrumb,
                url=chunk.url,
                text=text,
                truncated=truncated,
            )
        )
    return passages, dropped


def parse_citations(text: str, passages: Sequence[Passage]) -> list[Citation]:
    """Resolve every ``[n]`` in the answer against the numbered passages."""
    by_number = {p.number: p for p in passages}
    citations: list[Citation] = []
    for match in _MARKER.finditer(text):
        for number in _split_marker(match.group(1)):
            passage = by_number.get(number)
            citations.append(
                Citation(
                    marker=number,
                    chunk_id=passage.chunk_id if passage else "",
                    url=passage.url if passage else "",
                    breadcrumb=passage.breadcrumb if passage else "",
                )
            )
    return citations


def build_prompt(question: str, passages: Sequence[Passage], *, sentences: int = 4) -> str:
    context = "\n\n".join(passage.render() for passage in passages)
    return ANSWER_TEMPLATE.format(
        refusal=REFUSAL_TEXT, sentences=sentences, context=context, question=question
    )


def generate(
    question: str,
    chunk_ids: Sequence[str],
    chunks: dict[str, Chunk],
    provider: LLMProvider,
    *,
    top_score: float | None = None,
    min_top_score: float = 0.0,
    sentences: int = 4,
    max_tokens: int = 700,
    max_chars: int = DEFAULT_CONTEXT_CHARS,
    json_object: bool = True,
) -> Answer:
    """Answer `question` from the retrieved `chunk_ids`, or refuse.

    `min_top_score` is the pre-generation gate described in the module
    docstring. It defaults to 0.0 -- disabled -- because the measurement says
    the fused score does not separate answerable from unanswerable questions
    here. It is a parameter rather than deleted code so the claim can be
    re-tested on another corpus, where it may well hold.
    """
    passages, dropped = build_passages(chunk_ids, chunks, max_chars=max_chars)

    if not passages:
        return Answer(
            question=question,
            text=REFUSAL_TEXT,
            refused=True,
            refusal_reason="no_context",
            dropped_passages=dropped,
        )
    if min_top_score > 0.0 and top_score is not None and top_score < min_top_score:
        return Answer(
            question=question,
            text=REFUSAL_TEXT,
            refused=True,
            refusal_reason="low_score",
            passages=passages,
            dropped_passages=dropped,
        )

    prompt = build_prompt(question, passages, sentences=sentences)
    started = time.perf_counter()
    try:
        response = provider.complete(
            prompt,
            system=ANSWER_SYSTEM,
            temperature=0.0,
            max_tokens=max_tokens,
            json_object=json_object,
        )
    except ProviderError as exc:
        return Answer(
            question=question,
            text=REFUSAL_TEXT,
            refused=True,
            refusal_reason="provider_error",
            passages=passages,
            dropped_passages=dropped,
            latency_s=time.perf_counter() - started,
            error=str(exc)[:300],
        )

    return _answer_from_response(question, response, passages, dropped)


def _answer_from_response(
    question: str,
    response: LLMResponse,
    passages: list[Passage],
    dropped: int,
) -> Answer:
    common: dict[str, Any] = {
        "question": question,
        "passages": passages,
        "raw": response.text,
        "model": response.model,
        "provider": response.provider,
        "latency_s": response.latency_s,
        "cached": response.cached,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "dropped_passages": dropped,
    }

    try:
        payload = extract_json(response.text)
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        text = str(payload["answer"]).strip()
    except (ValueError, KeyError, TypeError) as exc:
        # A response we cannot parse is not an answer. Presenting the raw model
        # output here would put unvalidated, uncited prose in front of the user
        # under the same UI as a checked answer, which is the whole failure this
        # module exists to prevent -- so it refuses and records why.
        return Answer(
            text=REFUSAL_TEXT,
            refused=True,
            refusal_reason="unparseable",
            error=f"{type(exc).__name__}: {exc}"[:300],
            **common,
        )

    sufficient = bool(payload.get("sufficient", True))
    citations = parse_citations(text, passages)
    if not sufficient and not text:
        text = REFUSAL_TEXT
    return Answer(
        text=text,
        refused=not sufficient,
        refusal_reason="model" if not sufficient else "",
        citations=citations,
        **common,
    )


def format_answer(answer: Answer, *, width: int = 78) -> str:
    """Human-readable rendering for the CLI."""
    lines = [answer.clean_text, ""]
    cited = set(answer.cited_chunk_ids)
    if cited:
        lines.append("Sources")
        for passage in answer.passages:
            if passage.chunk_id in cited:
                lines.append(f"  [{passage.number}] {passage.breadcrumb[:width]}")
                lines.append(f"      {passage.url}")
    elif not answer.refused:
        lines.append("(no citations -- this answer is ungrounded)")
    if answer.invalid_citations:
        markers = sorted({c.marker for c in answer.invalid_citations})
        lines.append(
            f"  ! dropped {len(markers)} citation(s) to passages that were not shown: {markers}"
        )
    tail = f"{answer.provider}/{answer.model}  {answer.latency_s:.1f}s"
    if answer.cached:
        tail += "  (cached)"
    if answer.refusal_reason:
        tail += f"  refused: {answer.refusal_reason}"
    lines.append("")
    lines.append(tail)
    return "\n".join(lines)


def link_citations(answer: Answer) -> str:
    """The answer as markdown, with every valid ``[n]`` linked to its passage.

    Lives here rather than in the demo so that the one piece of real logic in
    `space/app.py` is covered by the test suite. A demo with untested logic in
    it is a second implementation of the system, which is precisely what
    importing the package was supposed to avoid.
    """
    by_number = {p.number: p for p in answer.passages}

    def repl(match: re.Match[str]) -> str:
        numbers = _split_marker(match.group(1))
        rendered = [
            f"[[{number}]]({by_number[number].url})" for number in numbers if number in by_number
        ]
        return "".join(rendered)

    return re.sub(r"\s+([.,;:])", r"\1", _MARKER.sub(repl, answer.text)).strip()


def markdown_sources(answer: Answer) -> str:
    """Bullet list of the sources an answer actually cited."""
    cited = set(answer.cited_chunk_ids)
    lines = [
        f"- **[{p.number}]** [{p.breadcrumb}]({p.url})"
        for p in answer.passages
        if p.chunk_id in cited
    ]
    if answer.invalid_citations:
        markers = sorted({c.marker for c in answer.invalid_citations})
        lines.append(f"- _dropped {len(markers)} citation(s) to passages never shown: {markers}_")
    return "\n".join(lines)
