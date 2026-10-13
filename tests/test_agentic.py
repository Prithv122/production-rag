"""The agentic retrieval arm harness, tested with fakes only.

No model is ever called here and no encoder is loaded. What is checked is the plumbing the
registered protocol depends on: that the judge prompt is the registered text byte for byte, that
equal-weight fusion lets a new rank-1 chunk into the top 10 (and that 2:1 would not), that the loop
stops where the protocol says and refuses only where it says, that an invalid reply fails open,
that the tie rule decides which answers are generated, and that the verdict and the four validity
gates decide boundary cases the way the registered rule does.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from conftest import FakeProvider, make_chunk
from production_rag import agentic as ag
from production_rag.answers import AnswerRow
from production_rag.fuse import reciprocal_rank_fusion
from production_rag.generate import ANSWER_SYSTEM, REFUSAL_TEXT
from production_rag.groundtruth import Evidence, Question, index_chunks_by_doc
from production_rag.metrics import RetrievalScore
from production_rag.providers import MODEL_ARMS, LLMResponse, ProviderError

ROOT = Path(__file__).resolve().parent.parent
MODEL = MODEL_ARMS[ag.MODEL_ARM]["model"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ranking(ids) -> list[tuple[str, float]]:
    """A ranking with strictly descending scores."""
    return [(cid, 1.0 / (i + 1)) for i, cid in enumerate(ids)]


def ids_of(r) -> list[str]:
    return [cid for cid, _ in r]


def chunk_map(n: int = 60, *, text=None) -> dict:
    return {
        f"c{i}#0": make_chunk(f"c{i}#0", text(i) if text else f"passage number {i} text")
        for i in range(n)
    }


class Retrieve:
    """A fake retriever: a query to a ranking, recording what was asked."""

    def __init__(self, table: dict, *, seconds: float = 0.25) -> None:
        self.table = table
        self.seconds = seconds
        self.asked: list[str] = []

    def __call__(self, query: str):
        self.asked.append(query)
        return list(self.table[query]), self.seconds


SUFFICIENT = '{"sufficient": true}'


def insufficient(query: str, missing: str = "m") -> str:
    return json.dumps({"sufficient": False, "missing": missing, "next_query": query})


def loop(
    replies,
    *,
    question: str = "how do I set x?",
    first=None,
    table=None,
    chunks=None,
    provider=None,
    max_steps: int = ag.MAX_STEPS,
):
    chunks = chunks or chunk_map()
    first = first or ranking([f"c{i}#0" for i in range(50)])
    retrieve = Retrieve(table or {})
    provider = provider or FakeProvider(list(replies), model=MODEL)
    result = ag.run_loop(
        "q1",
        question,
        first,
        0.1,
        retrieve=retrieve,
        chunks=chunks,
        provider=provider,
        max_steps=max_steps,
    )
    return result, retrieve, provider


def second_list(prefix: str = "n") -> list[tuple[str, float]]:
    return ranking([f"{prefix}{i}#0" for i in range(50)])


def table_with(*queries: str) -> dict:
    """Distinct pool-50 lists for each query, with their own chunk ids in `chunk_map`-style."""
    return {q: second_list(f"r{i}_") for i, q in enumerate(queries)}


def all_chunks_for(table: dict) -> dict:
    chunks = chunk_map()
    for lst in table.values():
        for cid, _ in lst:
            chunks.setdefault(cid, make_chunk(cid, f"text of {cid}"))
    return chunks


# ---------------------------------------------------------------------------
# the registered judge prompt
# ---------------------------------------------------------------------------
def registered_judge_prompt() -> tuple[str, str]:
    notes = (ROOT / "NOTES.md").read_text("utf-8").replace("\r\n", "\n")
    start = notes.index("## Agentic retrieval arm: registered protocol")
    end = notes.find("\n## ", start + 1)
    section = notes[start : end if end != -1 else len(notes)]
    system = re.search(r"System: `(.*?)` User:", section, re.S)
    user = re.search(r"```text\n(.*?)\n```", section, re.S)
    assert system and user
    return " ".join(system.group(1).split()), user.group(1)


def test_the_judge_prompt_is_the_registered_text_byte_for_byte():
    system, user = registered_judge_prompt()
    assert system == ag.JUDGE_SYSTEM
    assert user == ag.JUDGE_TEMPLATE
    chunks = chunk_map(5)
    passages = ag.judge_passages(ranking(list(chunks)), chunks)
    rendered = ag.render_judge_prompt("what is x?", passages)
    body = "\n\n".join(p.render() for p in passages)
    assert rendered == user.replace("{question}", "what is x?").replace("{passages}", body)
    assert not rendered.endswith("\n") and rendered.endswith("Do not answer the question.")


def test_the_template_braces_survive_rendering_and_braces_in_inputs_are_left_alone():
    chunks = {"c0#0": make_chunk("c0#0", "use {passages} and {question} literally")}
    passages = ag.judge_passages(ranking(["c0#0"]), chunks)
    rendered = ag.render_judge_prompt("what is {passages}?", passages)
    assert 'reply {"sufficient": true}.' in rendered
    assert "Question: what is {passages}?" in rendered
    assert "use {passages} and {question} literally" in rendered


def test_the_judge_sees_the_top_five_passages_of_the_ranking_with_the_answer_prompt_header():
    chunks = chunk_map(12)
    passages = ag.judge_passages(ranking(list(chunks)), chunks)
    assert [p.chunk_id for p in passages] == [f"c{i}#0" for i in range(5)]
    assert [p.number for p in passages] == [1, 2, 3, 4, 5]
    first = chunks["c0#0"]
    assert passages[0].render() == f"[1] {first.breadcrumb}\n{first.url}\n{first.text}"


def test_each_passage_is_cut_to_1200_characters_and_marked():
    long = "A" * 1_200 + "B" * 300
    chunks = {"c0#0": make_chunk("c0#0", long), "c1#0": make_chunk("c1#0", "short text")}
    passages = ag.judge_passages(ranking(["c0#0", "c1#0"]), chunks)
    assert passages[0].text == "A" * 1_200 and passages[0].truncated
    assert not passages[1].truncated
    prompt = ag.render_judge_prompt("q?", passages)
    assert "A" * 1_200 in prompt and "B" not in prompt.replace("Breadcrumb", "")
    assert "[... passage truncated ...]" in prompt


def test_exactly_1200_characters_is_not_cut():
    chunks = {"c0#0": make_chunk("c0#0", "A" * 1_200)}
    (passage,) = ag.judge_passages(ranking(["c0#0"]), chunks)
    assert len(passage.text) == 1_200 and not passage.truncated


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------
def test_equal_weight_fusion_lets_a_new_rank_one_chunk_into_the_top_ten():
    original = ranking([f"a{i}" for i in range(50)])
    requery = ranking([f"b{i}" for i in range(50)])
    merged = ag.merge_rankings([original, requery])
    assert "b0" in ids_of(merged)[:10]
    assert len(merged) == 100
    # With the original counting double, the same chunk could not get in: 1/61 is below 2/110.
    weighted = reciprocal_rank_fusion([original, requery], k=60, weights=(2.0, 1.0))
    assert "b0" not in ids_of(weighted)[:10]


def test_merge_is_rrf_with_k_60_and_a_single_list_is_returned_untouched():
    one = ranking(["x", "y", "z"])
    assert ag.merge_rankings([one]) == one
    merged = ag.merge_rankings([ranking(["x", "y"]), ranking(["y", "z"])])
    scores = dict(merged)
    assert scores["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["x"] == pytest.approx(1 / 61) and scores["z"] == pytest.approx(1 / 62)
    assert ids_of(merged) == ["y", "x", "z"]


def test_merge_over_three_lists_counts_every_list_once():
    lists = [ranking(["x", "p"]), ranking(["q", "x"]), ranking(["r", "x"])]
    scores = dict(ag.merge_rankings(lists))
    assert scores["x"] == pytest.approx(1 / 61 + 2 / 62)


# ---------------------------------------------------------------------------
# reading a reply
# ---------------------------------------------------------------------------
def test_a_sufficient_reply_and_an_insufficient_one_parse():
    assert ag.parse_verdict(SUFFICIENT) == ag.JudgeVerdict(True)
    verdict = ag.parse_verdict(insufficient("  find it  ", "the flag"))
    assert verdict == ag.JudgeVerdict(False, missing="the flag", next_query="find it")
    assert verdict.valid


def test_a_fenced_or_prefixed_object_is_read_as_the_project_reads_one_everywhere():
    assert ag.parse_verdict('Here you go:\n```json\n{"sufficient": true}\n```').sufficient is True
    assert ag.parse_verdict('Sure. {"sufficient": true} Hope that helps.').sufficient is True


def test_a_sufficient_reply_ignores_everything_else():
    verdict = ag.parse_verdict('{"sufficient": true, "next_query": "", "extra": 1}')
    assert verdict.sufficient is True and verdict.next_query == ""


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("", "no_json_object"),
        ("   ", "no_json_object"),
        ("I think the passages are enough.", "no_json_object"),
        ("[1, 2, 3]", "no_json_object"),
        ('"sufficient"', "no_json_object"),
        ("{}", "sufficient_not_boolean"),
        ('{"sufficient": "true"}', "sufficient_not_boolean"),
        ('{"sufficient": 1}', "sufficient_not_boolean"),
        ('{"sufficient": null}', "sufficient_not_boolean"),
        ('{"missing": "x", "next_query": "y"}', "sufficient_not_boolean"),
        ('{"sufficient": false}', "empty_next_query"),
        ('{"sufficient": false, "next_query": ""}', "empty_next_query"),
        ('{"sufficient": false, "next_query": "   \\n "}', "empty_next_query"),
        ('{"sufficient": false, "next_query": 7}', "empty_next_query"),
        ('{"sufficient": false, "next_query": null}', "empty_next_query"),
        (insufficient("q" * 301), "next_query_too_long"),
    ],
)
def test_every_kind_of_invalid_reply_is_recognised(text, kind):
    verdict = ag.parse_verdict(text)
    assert not verdict.valid and verdict.sufficient is None and verdict.invalid == kind


def test_a_next_query_of_exactly_300_characters_is_valid_and_301_is_not():
    assert ag.parse_verdict(insufficient("q" * 300)).valid
    assert ag.parse_verdict(insufficient(" " + "q" * 300 + " ")).valid  # measured stripped
    assert ag.parse_verdict(insufficient("q" * 301)).invalid == "next_query_too_long"


def test_queries_compare_lowercased_with_whitespace_collapsed():
    assert ag.normalise_query("  How  do\tI \n Set X? ") == "how do i set x?"


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def test_one_judge_call_is_made_with_the_registered_arguments():
    _, _, provider = loop([SUFFICIENT])
    assert len(provider.prompts) == 1
    assert provider.kwargs[0] == {
        "system": ag.JUDGE_SYSTEM,
        "temperature": 0.0,
        "max_tokens": 200,
        "json_object": True,
    }


def test_sufficient_at_step_one_leaves_the_baseline_ranking_untouched_and_retrieves_nothing():
    first = ranking([f"c{i}#0" for i in range(50)])
    result, retrieve, provider = loop([SUFFICIENT], first=first)
    assert result.outcome == "sufficient" and not result.refused and result.n_calls == 1
    assert result.final == first and result.top10 == ids_of(first)[:10]
    assert retrieve.asked == [] and len(provider.prompts) == 1
    assert result.queries == ["how do I set x?"]


def test_insufficient_then_sufficient_retrieves_once_and_merges_equal_weight():
    table = table_with("find the flag")
    chunks = all_chunks_for(table)
    first = ranking([f"c{i}#0" for i in range(50)])
    result, retrieve, provider = loop(
        [insufficient("find the flag"), SUFFICIENT], first=first, table=table, chunks=chunks
    )
    assert result.outcome == "sufficient" and not result.refused and result.n_calls == 2
    assert retrieve.asked == ["find the flag"]
    assert result.queries == ["how do I set x?", "find the flag"]
    assert result.final == ag.merge_rankings([first, table["find the flag"]])
    assert "r0_0#0" in result.top10  # the new rank-1 chunk got in
    # The second judge call saw the merged ranking, not the original.
    expected = ag.render_judge_prompt("how do I set x?", ag.judge_passages(result.final, chunks))
    assert provider.prompts[1] == expected and provider.prompts[0] != provider.prompts[1]


def test_the_step_cap_is_three_judge_calls_and_the_loop_refuses_at_the_cap():
    table = table_with("one", "two")
    result, retrieve, provider = loop(
        [insufficient("one"), insufficient("two"), insufficient("three")],
        table=table,
        chunks=all_chunks_for(table),
    )
    assert result.n_calls == 3 and len(provider.prompts) == 3  # never a fourth
    assert result.refused and result.outcome == "cap_refusal"
    assert retrieve.asked == ["one", "two"]  # the third next_query is never retrieved
    assert result.queries == ["how do I set x?", "one", "two"]


def test_sufficient_at_the_third_step_does_not_refuse():
    table = table_with("one", "two")
    result, _, _ = loop(
        [insufficient("one"), insufficient("two"), SUFFICIENT],
        table=table,
        chunks=all_chunks_for(table),
    )
    assert result.n_calls == 3 and not result.refused and result.outcome == "sufficient"


def test_a_repeated_query_ends_the_loop_with_a_refusal_and_no_retrieval():
    result, retrieve, provider = loop([insufficient("  HOW do I   set x? ")])
    assert result.refused and result.outcome == "repeat_refusal" and result.n_calls == 1
    assert retrieve.asked == [] and len(provider.prompts) == 1
    assert result.final == ranking([f"c{i}#0" for i in range(50)])


def test_a_query_repeated_from_an_earlier_step_also_stops_the_loop():
    table = table_with("Alpha")
    result, retrieve, _ = loop(
        [insufficient("Alpha"), insufficient("alpha")], table=table, chunks=all_chunks_for(table)
    )
    assert result.refused and result.outcome == "repeat_refusal" and result.n_calls == 2
    assert retrieve.asked == ["Alpha"]


def test_a_new_query_is_issued_as_the_judge_wrote_it_stripped():
    table = {"Find The_Flag --x": second_list()}
    result, retrieve, _ = loop(
        [insufficient("  Find The_Flag --x\n"), SUFFICIENT],
        table=table,
        chunks=all_chunks_for(table),
    )
    assert retrieve.asked == ["Find The_Flag --x"] and result.queries[1] == "Find The_Flag --x"


@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        ("no json here", "no_json_object"),
        ('{"sufficient": "yes"}', "sufficient_not_boolean"),
        ('{"sufficient": false, "missing": "m"}', "empty_next_query"),
        (insufficient("q" * 301), "next_query_too_long"),
    ],
)
def test_an_invalid_reply_stops_the_loop_keeps_the_ranking_and_does_not_refuse(reply, kind):
    first = ranking([f"c{i}#0" for i in range(50)])
    result, retrieve, provider = loop([reply], first=first)
    assert result.outcome == "invalid_reply" and not result.refused
    assert result.n_calls == 1 and result.final == first and retrieve.asked == []
    assert result.steps[0].invalid == kind and result.steps[0].sufficient is None
    assert len(provider.prompts) == 1


def test_an_invalid_reply_after_a_requery_keeps_the_merged_ranking():
    table = table_with("find it")
    chunks = all_chunks_for(table)
    first = ranking([f"c{i}#0" for i in range(50)])
    result, _, _ = loop(
        [insufficient("find it"), "garbage"], first=first, table=table, chunks=chunks
    )
    assert result.outcome == "invalid_reply" and not result.refused and result.n_calls == 2
    assert result.final == ag.merge_rankings([first, table["find it"]])


def test_a_failed_judge_call_is_an_invalid_reply_that_fails_open():
    first = ranking([f"c{i}#0" for i in range(50)])
    result, retrieve, _ = loop([], first=first, provider=FakeProvider(fail=True, model=MODEL))
    assert result.outcome == "invalid_reply" and not result.refused
    step = result.steps[0]
    assert step.invalid == "provider_error" and "is down" in step.error and step.model == ""
    assert result.final == first and retrieve.asked == []


def test_a_truncated_reply_is_just_a_reply_that_may_be_invalid():
    cut = LLMResponse(text='{"sufficient": fal', model=MODEL, provider="p", finish_reason="length")
    result, _, _ = loop([cut])
    assert result.steps[0].invalid == "no_json_object" and result.steps[0].finish_reason == "length"


def test_cost_accounting_sums_tokens_and_latency_over_the_steps():
    table = table_with("find it")
    first_reply = LLMResponse(
        text=insufficient("find it"),
        model=MODEL,
        provider="p",
        latency_s=2.0,
        prompt_tokens=1_000,
        completion_tokens=30,
    )
    second_reply = LLMResponse(
        text=SUFFICIENT,
        model=MODEL,
        provider="p",
        latency_s=1.5,
        prompt_tokens=1_100,
        completion_tokens=5,
        cached=True,
    )
    result, _, _ = loop([first_reply, second_reply], table=table, chunks=all_chunks_for(table))
    assert result.n_calls == 2
    assert result.prompt_tokens == 2_100 and result.completion_tokens == 35
    # step-1 retrieval 0.1 + judge 2.0 + re-query 0.25 + judge 1.5
    assert result.latency_s == pytest.approx(3.85)
    first, second = result.steps
    assert (first.retrieval_s, second.retrieval_s) == (0.1, 0.25)
    assert (first.cached, second.cached) == (False, True)
    assert first.query == "how do I set x?" and second.query == "find it"
    assert len(first.top5) == 5 and first.prompt_chars > 0


def test_a_step_is_json_serialisable():
    result, _, _ = loop([SUFFICIENT])
    json.dumps([s.as_dict() for s in result.steps])


# ---------------------------------------------------------------------------
# step one and gate 1
# ---------------------------------------------------------------------------
def spec_for(qid: str, baseline_ids, *, gold=("g#0",), category="conceptual", in_subset=False):
    q = Question(
        qid=qid,
        text=f"text of {qid}",
        category=category,
        evidence=(Evidence(doc_id="g", start=0, end=4, quote="gold"),) if gold else (),
    )
    base = tuple(baseline_ids)[: ag.K]
    stored = RetrievalScore.evaluate(list(base), gold).as_dict() if gold else {}
    return ag.QuestionSpec(q, tuple(gold), base, stored, in_subset)


def test_gate_one_passes_when_every_step_one_ranking_equals_the_stored_top_ten():
    base = [f"c{i}#0" for i in range(50)]
    specs = [spec_for("a", base), spec_for("b", base[::-1])]
    table = {specs[0].text: ranking(base), specs[1].text: ranking(base[::-1])}
    retrieve = Retrieve(table)
    step_one = ag.prepare_step_one(specs, retrieve)
    assert step_one.gate["passed"] and step_one.gate["checked"] == 2
    assert step_one.gate["mismatched"] == 0
    assert step_one.latency_s == {"a": 0.25, "b": 0.25}
    # One unrecorded warm-up retrieval comes first.
    assert retrieve.asked == [specs[0].text, specs[0].text, specs[1].text]


def test_gate_one_fails_on_one_swapped_pair_and_names_where():
    base = [f"c{i}#0" for i in range(50)]
    swapped = list(base)
    swapped[6], swapped[7] = swapped[7], swapped[6]
    specs = [spec_for("a", base), spec_for("b", base)]
    retrieve = Retrieve({specs[0].text: ranking(base), specs[1].text: ranking(swapped)})
    gate = ag.prepare_step_one(specs, retrieve).gate
    assert not gate["passed"] and gate["mismatched"] == 1
    assert gate["examples"] == [{"qid": "b", "first_difference_rank": 7}]


def test_gate_one_compares_only_the_top_ten():
    base = [f"c{i}#0" for i in range(50)]
    deeper = base[:10] + base[10:][::-1]
    specs = [spec_for("a", base)]
    gate = ag.prepare_step_one(specs, Retrieve({specs[0].text: ranking(deeper)})).gate
    assert gate["passed"]


# ---------------------------------------------------------------------------
# a whole small run
# ---------------------------------------------------------------------------
def answer_row(
    qid: str,
    *,
    refused=False,
    should_refuse=False,
    cited_gold=True,
    model=MODEL,
    reason="",
    prompt_tokens=100,
) -> AnswerRow:
    return AnswerRow(
        qid=qid,
        category="conceptual",
        arm="published",
        model=model,
        refused=refused,
        refusal_reason=reason or ("model" if refused else ""),
        should_refuse=should_refuse,
        n_citations=1,
        n_valid_citations=1,
        cited_gold=cited_gold and not refused,
        uncited=False,
        latency_s=1.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=10,
    )


class RuleProvider(FakeProvider):
    """Judge and answer replies as a function of the prompt, so threads cannot change them."""

    def __init__(self, judge, answer, *, tokens=None, **kwargs):
        super().__init__(**kwargs)
        self._judge, self._answer, self._tokens = judge, answer, tokens

    def complete(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(dict(kwargs))
        text = (self._judge if kwargs.get("system") == ag.JUDGE_SYSTEM else self._answer)(prompt)
        tokens = self._tokens(prompt) if self._tokens else len(prompt) // 4
        return LLMResponse(
            text=text,
            model=self._model,
            provider="p",
            latency_s=1.0,
            prompt_tokens=tokens,
            completion_tokens=20,
        )

    def judge_prompts(self) -> list[str]:
        return [
            p
            for p, k in zip(self.prompts, self.kwargs, strict=True)
            if k["system"] == ag.JUDGE_SYSTEM
        ]

    def answer_prompts(self) -> list[str]:
        return [
            p
            for p, k in zip(self.prompts, self.kwargs, strict=True)
            if k["system"] == ANSWER_SYSTEM
        ]


class World:
    """Twelve questions whose gold chunk is at rank 8 and a re-query puts it at rank 1, two whose
    gold chunk is already first, and two unanswerable ones. The first four of the first group, both
    easy ones and both unanswerable ones form the answer subset."""

    def __init__(self, *, n_gain=12, n_easy=2, n_unanswerable=2, refuse_always=False):
        self.refuse_always = refuse_always
        self.chunks = {f"f{i}#0": make_chunk(f"f{i}#0", f"filler passage {i}") for i in range(80)}
        self.by_text: dict[str, list] = {}
        specs: list[ag.QuestionSpec] = []
        fillers = [f"f{j}#0" for j in range(80)]
        for i in range(n_gain + n_easy):
            gid = f"g{i}#0"
            text = f"GOLD answer text {i}"
            self.chunks[gid] = make_chunk(gid, text)
            easy = i >= n_gain
            base = ([gid, *fillers[:49]]) if easy else ([*fillers[:7], gid, *fillers[7:49]])
            q = Question(
                qid=f"q{i}",
                text=f"question number {i}?",
                category="conceptual" if i % 2 == 0 else "exact_term",
                evidence=(Evidence(doc_id=f"g{i}", start=0, end=len(text), quote=text),),
            )
            self.by_text[q.text] = ranking(base)
            stored = RetrievalScore.evaluate(base[: ag.K], [gid]).as_dict()
            specs.append(ag.QuestionSpec(q, (gid,), tuple(base[: ag.K]), stored, False))
        for i in range(n_unanswerable):
            q = Question(qid=f"u{i}", text=f"unanswerable ask {i}?", category="unanswerable")
            base = fillers[i : i + 50]
            self.by_text[q.text] = ranking(base)
            specs.append(ag.QuestionSpec(q, (), tuple(base[: ag.K]), {}, False))
        chosen = {s.qid for s in specs if s.qid in {"q0", "q1", "q2", "q3"}}
        chosen |= {s.qid for s in specs if s.qid.startswith("u") or int(s.qid[1:]) >= n_gain}
        specs = [
            ag.QuestionSpec(s.question, s.gold, s.baseline, s.stored_metrics, s.qid in chosen)
            for s in specs
        ]
        subset = [s for s in specs if s.in_subset]
        published = {
            s.qid: answer_row(s.qid, refused=not s.question.is_answerable,
                              should_refuse=not s.question.is_answerable)
            for s in subset
        }  # fmt: skip
        self.inputs = ag.Inputs(
            specs, subset, published, self.chunks, index_chunks_by_doc(self.chunks.values())
        )
        self.retrieved: list[str] = []

    def retrieve(self, query: str):
        self.retrieved.append(query)
        if query in self.by_text:
            return list(self.by_text[query]), 0.01
        match = re.match(r"find gold for question (\d+)", query)
        if match:
            ids = [f"g{match.group(1)}#0"] + [f"f{j}#0" for j in range(50, 80)]
            return ranking(ids), 0.02
        return ranking([f"f{j}#0" for j in reversed(range(30, 80))]), 0.02

    def judge(self, prompt: str) -> str:
        text = re.search(r"Question: (.*)\n", prompt).group(1)
        if "GOLD" in prompt and not self.refuse_always:
            return SUFFICIENT
        match = re.match(r"question number (\d+)\?", text)
        if match:
            return insufficient(f"find gold for question {match.group(1)}")
        return insufficient(f"other {hashlib.sha256(prompt.encode()).hexdigest()[:8]}")

    @staticmethod
    def answer(prompt: str) -> str:
        question = prompt.rsplit("QUESTION\n", 1)[1].strip()
        if question.startswith("unanswerable"):
            return json.dumps({"sufficient": False, "answer": REFUSAL_TEXT})
        return json.dumps({"sufficient": True, "answer": "The gold text says so [1]."})

    def provider(self, **kwargs) -> RuleProvider:
        kwargs.setdefault("model", MODEL)
        return RuleProvider(self.judge, self.answer, **kwargs)


def run_world(world: World, provider=None, *, retrieve=None, workers=2):
    provider = provider or world.provider()
    retrieve = retrieve or world.retrieve
    step_one = ag.prepare_step_one(world.inputs.specs, retrieve)
    loops = ag.run_loops(
        world.inputs.specs,
        step_one,
        retrieve=retrieve,
        chunks=world.chunks,
        provider=provider,
        workers=workers,
    )
    return step_one, loops, provider


def finish(world: World, provider=None, **kwargs):
    step_one, loops, provider = run_world(world, provider)
    pairs = ag.run_answers(world.inputs, loops, provider, workers=2)
    result = ag.build_result(
        world.inputs, step_one, loops, pairs, model=MODEL, trials=300, **kwargs
    )
    return result, step_one, loops, pairs, provider


def test_a_gain_scenario_runs_end_to_end_and_the_verdict_is_a_gain():
    world = World()
    result, _step_one, _loops, pairs, _provider = finish(world)
    assert [g["passed"] for g in result["gates"]] == [True] * 4
    assert result["complete"] and result["verdict"] == ag.GAIN
    assert result["questions"]["n"] == 16 and result["questions"]["scored"] == 14
    assert result["questions"]["by_category_scored"] == {"exact_term": 7, "conceptual": 7}
    ndcg = result["retrieval"]["bootstrap"]["intervals"]["ndcg@10"]
    assert ndcg["mean"] > 0.5 and ndcg["lo"] > 0
    assert result["verdict_text"].startswith("Gain: delta nDCG@10 +")
    assert (
        "judge calls" in result["verdict_text"]
        and "baseline makes 0 LLM calls" in result["verdict_text"]
    )
    json.dumps(result)  # everything in the result is serialisable
    json.dumps(ag.pair_records(pairs))


def test_the_report_prints_the_paired_answer_table_gates_and_the_verdict():
    world = World()
    result, *_ = finish(world)
    report = ag.format_report(result)
    assert "| answers | refusal recall | false refusal | grounded | citation validity |" in report
    assert "| baseline |" in report and "| agentic |" in report and "paired:" in report
    assert report.count("gate ") == 4 and report.rstrip().endswith(result["verdict_text"])


def test_the_loops_and_the_steps_distribution_are_what_the_rules_give():
    world = World()
    _, _, loops, _, _ = finish(world)
    gain = [loops[f"q{i}"] for i in range(12)]
    assert all(r.n_calls == 2 and r.outcome == "sufficient" for r in gain)
    easy = [loops["q12"], loops["q13"]]
    assert all(r.n_calls == 1 and r.outcome == "sufficient" for r in easy)
    unanswerable = [loops["u0"], loops["u1"]]
    assert all(r.refused for r in unanswerable)
    assert all(r.outcome in ("cap_refusal", "repeat_refusal") for r in unanswerable)


def test_sufficient_at_step_one_gives_the_baseline_ranking_in_the_scored_results():
    world = World()
    result, _, loops, _, _ = finish(world)
    for qid in ("q12", "q13"):
        assert loops[qid].top10 == list(
            next(s for s in world.inputs.specs if s.qid == qid).baseline
        )
    per = result["retrieval"]["per_question"]
    assert per["agentic"]["q12"] == per["baseline"]["q12"]


def test_the_report_tables_match_the_scenario():
    world = World()
    result, *_ = finish(world)
    steps = result["steps"]
    assert steps["all"]["calls"] == {"1": 2, "2": 12, "3": 2}  # the unanswerable ones use all 3
    assert steps["all"]["iterated"] == 14
    assert steps["scored"]["calls"] == {"1": 2, "2": 12, "3": 0}
    assert steps["refusals"] == {
        "answerable": {"refused": 0, "n": 14},
        "unanswerable": {"refused": 2, "n": 2},
    }
    calibration = result["calibration"]["table"]
    assert calibration["sufficient"] == {"gold_in_top5": 2, "no_gold_in_top5": 0}
    assert calibration["insufficient"] == {"gold_in_top5": 0, "no_gold_in_top5": 12}
    assert calibration["invalid"] == {"gold_in_top5": 0, "no_gold_in_top5": 0}
    assert result["calibration"]["insufficient_and_no_gold_share"] == 1.0
    cost = result["cost"]["scored"]
    assert cost["n"] == 14 and cost["judge_calls"]["total"] == 12 * 2 + 2
    assert cost["judge_calls"]["mean"] == pytest.approx(26 / 14)
    assert cost["baseline_llm_calls"] == 0 and cost["baseline_retrieval_latency_s"]["mean"] == 0.01
    per_category = result["retrieval"]["by_category"]
    assert set(per_category) == {"exact_term", "conceptual"}
    assert sum(b["n"] for b in per_category.values()) == 14
    assert result["retrieval"]["baseline_matches_stored"] is True


def test_the_answer_phase_generates_only_where_the_evidence_changed_or_the_loop_refused():
    world = World()
    result, _, loops, pairs, provider = finish(world)
    kinds = {p.qid: p.kind for p in pairs}
    assert [k for k in kinds.values()].count(ag.TIE) == 2
    assert {q for q, k in kinds.items() if k == ag.TIE} == {"q12", "q13"}
    assert {q for q, k in kinds.items() if k == ag.CHANGED} == {"q0", "q1", "q2", "q3"}
    assert {q for q, k in kinds.items() if k == ag.LOOP_REFUSED} == {"u0", "u1"}
    assert (
        len(provider.answer_prompts()) == 4 * 2 + 2 == ag.planned_generations(world.inputs, loops)
    )
    assert result["answers"]["kinds"] == {"tie": 2, "changed": 4, "loop_refused": 2}
    by = {p.qid: p for p in pairs}
    # A tie shares the published row; neither arm generated.
    assert by["q12"].baseline_source == by["q12"].agentic_source == "published"
    assert by["q12"].baseline.text == by["q12"].agentic.text == world.inputs.published["q12"].text
    # A changed question has both answers live, from different contexts.
    assert by["q0"].baseline_source == by["q0"].agentic_source == "live"
    # A loop refusal: the baseline answers live, the agentic arm is a refusal with no generation.
    refusal = by["u0"].agentic
    assert by["u0"].baseline_source == "live" and by["u0"].agentic_source == "loop_refusal"
    assert refusal.refused and refusal.refusal_reason == "loop" and refusal.model == ""
    assert refusal.should_refuse and refusal.text == REFUSAL_TEXT and refusal.prompt_tokens == 0


def test_a_changed_question_is_answered_from_the_stored_top_ten_and_the_loop_top_ten():
    world = World()
    _, _, loops, _pairs, provider = finish(world)
    spec = next(s for s in world.inputs.specs if s.qid == "q0")
    from production_rag.generate import build_passages, build_prompt

    def prompt_for(ids):
        passages, _ = build_passages(ids, world.chunks)
        return build_prompt(spec.text, passages)

    prompts = provider.answer_prompts()
    assert prompt_for(list(spec.baseline)) in prompts
    assert prompt_for(loops["q0"].top10) in prompts
    kwargs = [k for k in provider.kwargs if k["system"] == ANSWER_SYSTEM]
    assert all(k["max_tokens"] == 700 and k["json_object"] is True for k in kwargs)


def test_a_tie_makes_no_answer_call():
    world = World(n_gain=0, n_easy=3, n_unanswerable=0)
    _result, _, _, pairs, provider = finish(world)
    assert provider.answer_prompts() == [] and {p.kind for p in pairs} == {ag.TIE}


def test_an_iterating_loop_that_ends_on_the_same_top_ten_is_still_a_tie():
    # The re-query returns the same ranking, so the equal-weight merge keeps the same top ten.
    chunks = chunk_map(60)
    base = [f"c{i}#0" for i in range(50)]
    spec = spec_for("a", base, in_subset=True)
    table = {spec.text: ranking(base), "same again": ranking(base)}
    provider = FakeProvider([insufficient("same again"), SUFFICIENT], model=MODEL)
    step_one = ag.prepare_step_one([spec], Retrieve(table))
    loops = ag.run_loops(
        [spec], step_one, retrieve=Retrieve(table), chunks=chunks, provider=provider, workers=1
    )
    assert loops["a"].n_calls == 2 and loops["a"].top10 == list(spec.baseline)
    assert ag.classify(spec, loops["a"]) == ag.TIE


def test_a_loop_refusal_wins_over_an_unchanged_ranking():
    spec = spec_for("a", [f"c{i}#0" for i in range(50)])
    refused = ag.LoopResult(
        "a", spec.text, [], [], ranking(list(spec.baseline)), True, "cap_refusal"
    )
    assert ag.classify(spec, refused) == ag.LOOP_REFUSED


# ---------------------------------------------------------------------------
# the guardrail
# ---------------------------------------------------------------------------
def rows_for(n_answerable: int, n_unanswerable: int, *, refused_answerable: int, refused_un: int):
    rows = [answer_row(f"a{i}", refused=i < refused_answerable) for i in range(n_answerable)]
    rows += [
        answer_row(f"u{i}", refused=i < refused_un, should_refuse=True)
        for i in range(n_unanswerable)
    ]
    return rows


def test_the_guardrail_allows_two_more_false_refusals_of_fifty_four_and_not_three():
    base = rows_for(54, 6, refused_answerable=8, refused_un=6)
    assert ag.guardrail(base, rows_for(54, 6, refused_answerable=10, refused_un=6))["holds"]
    over = ag.guardrail(base, rows_for(54, 6, refused_answerable=11, refused_un=6))
    assert not over["holds"] and not over["false_refusal_ok"]
    assert over["allowed_false_refusal_rise"] == 2


def test_the_guardrail_allowance_is_five_points_rounded_down():
    # 5% of 54 = 2.7 -> 2; of 60 = 3; of 19 = 0.95 -> 0
    for n, allowed in ((54, 2), (60, 3), (19, 0), (20, 1)):
        base = rows_for(n, 0, refused_answerable=0, refused_un=0)
        assert ag.guardrail(base, base)["allowed_false_refusal_rise"] == allowed


def test_refusal_recall_may_tie_the_baseline_but_not_fall_below_it():
    base = rows_for(54, 6, refused_answerable=8, refused_un=5)
    assert ag.guardrail(base, rows_for(54, 6, refused_answerable=8, refused_un=5))["holds"]
    assert ag.guardrail(base, rows_for(54, 6, refused_answerable=8, refused_un=6))["holds"]
    worse = ag.guardrail(base, rows_for(54, 6, refused_answerable=8, refused_un=4))
    assert not worse["holds"] and not worse["refusal_recall_ok"] and worse["false_refusal_ok"]


def test_fewer_false_refusals_than_the_baseline_is_fine():
    base = rows_for(54, 6, refused_answerable=8, refused_un=6)
    assert ag.guardrail(base, rows_for(54, 6, refused_answerable=0, refused_un=6))["holds"]


# ---------------------------------------------------------------------------
# the verdict rule
# ---------------------------------------------------------------------------
def test_a_gain_needs_the_bar_a_positive_lower_bound_and_the_guardrail():
    assert ag.decide(0.05, 0.01, 0.09, True) == ag.GAIN
    assert ag.decide(0.02, 0.001, 0.04, True) == ag.GAIN  # exactly the bar
    assert ag.decide(0.0199, 0.001, 0.04, True) == ag.NO_GAIN  # just under it
    assert ag.decide(0.05, 0.0, 0.09, True) == ag.NO_GAIN  # lower bound must be above 0
    assert ag.decide(0.05, -0.001, 0.09, True) == ag.NO_GAIN
    assert ag.decide(0.05, 0.01, 0.09, False) == ag.NO_GAIN  # a gain that fails the guardrail


def test_worse_needs_the_upper_bound_strictly_below_zero():
    assert ag.decide(-0.05, -0.09, -0.01, True) == ag.WORSE
    assert ag.decide(-0.05, -0.09, -0.0001, False) == ag.WORSE  # the guardrail does not matter
    assert ag.decide(-0.05, -0.09, 0.0, True) == ag.NO_GAIN
    assert ag.decide(-0.05, -0.09, 0.01, True) == ag.NO_GAIN


def test_no_gain_is_the_default_for_anything_else():
    assert ag.decide(0.0, -0.02, 0.02, True) == ag.NO_GAIN
    assert ag.decide(0.01, 0.002, 0.02, True) == ag.NO_GAIN


def test_a_gain_that_fails_the_guardrail_is_reported_as_no_gain_with_the_reason():
    world = World(refuse_always=True)
    result, *_ = finish(world)
    assert result["answers"]["guardrail"]["holds"] is False
    ndcg = result["retrieval"]["bootstrap"]["intervals"]["ndcg@10"]
    assert ndcg["lo"] > 0 and ndcg["mean"] >= 0.02  # the retrieval gain is there ...
    assert result["verdict"] == ag.NO_GAIN  # ... and the guardrail still decides
    assert "No gain for its cost" in result["verdict_text"]
    assert "refusal guardrail failed" in result["verdict_text"]


def test_a_worse_scenario_is_worse():
    world = World()
    _step_one, loops, _provider = run_world(world)
    scored = [s for s in world.inputs.specs if s.scored]
    # Make every agentic ranking miss the gold chunk entirely.
    broken = {
        qid: ag.LoopResult(
            r.qid, r.question, r.steps, r.queries, ranking(["zzz"] * 1), r.refused, r.outcome
        )
        for qid, r in loops.items()
    }
    report = ag.retrieval_report(scored, broken, trials=300)
    ci = report["bootstrap"]["intervals"]["ndcg@10"]
    assert ci["hi"] < 0 and ag.decide(ci["mean"], ci["lo"], ci["hi"], True) == ag.WORSE


# ---------------------------------------------------------------------------
# validity gates
# ---------------------------------------------------------------------------
def step(**kwargs) -> ag.Step:
    base = {"step": 1, "query": "q", "top5": [], "retrieval_s": 0.0, "prompt_chars": 10}
    return ag.Step(**{**base, **kwargs})


def test_gate_three_allows_five_percent_of_judge_calls_and_not_more():
    steps = [step(invalid="no_json_object")] + [step() for _ in range(19)]
    ok = ag.gate_three(steps)
    assert ok["passed"] and ok["invalid"] == 1 and ok["judge_calls"] == 20 and ok["rate"] == 0.05
    bad = ag.gate_three([step(invalid="no_json_object")] * 2 + [step() for _ in range(18)])
    assert not bad["passed"]
    assert not ag.gate_three([])["passed"]


def test_gate_three_counts_a_failed_call_and_names_the_kinds():
    steps = [step(invalid="provider_error"), step(invalid="empty_next_query"), *[step()] * 8]
    gate = ag.gate_three(steps)
    assert gate["by_kind"]["provider_error"] == 1 and gate["by_kind"]["empty_next_query"] == 1
    assert not gate["passed"]


def test_gate_four_is_strictly_below_the_window():
    assert ag.gate_four([step(prompt_tokens=8_191)], [])["passed"]
    assert not ag.gate_four([step(prompt_tokens=8_192)], [])["passed"]
    assert not ag.gate_four([step(prompt_tokens=10)], [answer_row("a", prompt_tokens=8_192)])[
        "passed"
    ]
    gate = ag.gate_four(
        [step(prompt_tokens=900, prompt_chars=3_000)], [answer_row("a", prompt_tokens=4_000)]
    )
    assert gate["max_judge_prompt_tokens"] == 900 and gate["max_answer_prompt_tokens"] == 4_000
    assert gate["max_judge_prompt_chars"] == 3_000


def test_gate_two_fails_on_a_different_model_or_an_answer_provider_error():
    assert ag.gate_two([step(model=MODEL)], [answer_row("a")], model=MODEL)["passed"]
    assert not ag.gate_two([step(model="other")], [], model=MODEL)["passed"]
    assert not ag.gate_two([], [answer_row("a", model="other")], model=MODEL)["passed"]
    error = answer_row("a", model="", refused=True, reason="provider_error")
    assert not ag.gate_two([], [error], model=MODEL)["passed"]


def test_rows_with_no_model_by_construction_do_not_trip_gate_two():
    refusal = ag.loop_refusal_row(Question(qid="u", text="t", category="unanswerable"))
    failed_call = step(invalid="provider_error", error="down")  # counts against gate 3 only
    gate = ag.gate_two([failed_call], [refusal], model=MODEL)
    assert gate["passed"] and gate["answered_by_another_model"] == 0


def test_a_failed_gate_means_no_verdict_whatever_the_numbers_say():
    world = World()
    bad_judge = RuleProvider(lambda p: "not json", World.answer, model=MODEL)
    result, *_ = finish(world, bad_judge)
    gate3 = result["gates"][2]
    assert not gate3["passed"] and result["verdict"] is None
    assert result["verdict_text"].startswith("no verdict:") and "gate 3" in result["verdict_text"]


def test_the_wrong_model_fails_gate_two_and_leaves_no_verdict():
    world = World()
    result, *_ = finish(world, world.provider(model="some-other-model"))
    assert not result["gates"][1]["passed"] and result["verdict"] is None
    assert "gate 2" in result["verdict_text"]


def test_an_oversized_prompt_fails_gate_four_and_leaves_no_verdict():
    world = World()
    result, *_ = finish(world, world.provider(tokens=lambda prompt: 9_000))
    assert not result["gates"][3]["passed"] and result["verdict"] is None
    assert "gate 4" in result["verdict_text"]


def test_a_step_one_mismatch_fails_gate_one_and_leaves_no_verdict():
    world = World()

    def shifted(query: str):
        ranked, seconds = world.retrieve(query)
        if query == "question number 5?":
            ranked[0], ranked[1] = ranked[1], ranked[0]
        return ranked, seconds

    provider = world.provider()
    step_one, loops, _ = run_world(world, provider, retrieve=shifted)
    assert not step_one.gate["passed"]
    pairs = ag.run_answers(world.inputs, loops, provider, workers=1)
    result = ag.build_result(world.inputs, step_one, loops, pairs, model=MODEL, trials=100)
    assert result["verdict"] is None and "gate 1" in result["verdict_text"]


def test_a_failed_answer_generation_fails_gate_two_not_silently_a_false_refusal():
    world = World()
    provider = world.provider()
    step_one, loops, _ = run_world(world, provider)

    class AnswersDown(RuleProvider):
        def complete(self, prompt, **kwargs):
            if kwargs.get("system") == ANSWER_SYSTEM:
                raise ProviderError("ollama is down")
            return super().complete(prompt, **kwargs)

    down = AnswersDown(world.judge, world.answer, model=MODEL)
    pairs = ag.run_answers(world.inputs, loops, down, workers=1)
    result = ag.build_result(world.inputs, step_one, loops, pairs, model=MODEL, trials=100)
    assert not result["gates"][1]["passed"] and result["gates"][1]["answer_provider_errors"] > 0
    assert result["verdict"] is None


def test_missing_rows_leave_no_verdict():
    world = World()
    step_one, loops, provider = run_world(world)
    pairs = ag.run_answers(world.inputs, loops, provider, workers=1)
    result = ag.build_result(world.inputs, step_one, loops, pairs[:-1], model=MODEL, trials=100)
    assert result["verdict"] is None and not result["complete"]
    assert "incomplete" in result["verdict_text"]


def test_the_loop_phase_alone_has_no_verdict_and_no_answers():
    world = World()
    step_one, loops, _ = run_world(world)
    result = ag.build_result(world.inputs, step_one, loops, None, model=MODEL, trials=100)
    assert result["verdict"] is None and result["answers"] is None
    assert "answer phase not run" in result["verdict_text"]
    assert "judge calls per question" in ag.format_report(result)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def test_the_bootstrap_is_seeded_paired_and_reproducible():
    diffs = {"a": [0.1, -0.2, 0.3, 0.0, 0.5, 0.1], "b": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]}
    first = ag.paired_bootstrap(diffs, trials=2_000, seed=20261013)
    again = ag.paired_bootstrap(diffs, trials=2_000, seed=20261013)
    other = ag.paired_bootstrap(diffs, trials=2_000, seed=1)
    assert first == again and first != other
    assert first["a"]["mean"] == pytest.approx(sum(diffs["a"]) / 6)
    assert first["a"]["lo"] <= first["a"]["mean"] <= first["a"]["hi"]
    assert ag.BOOTSTRAP_TRIALS == 20_000 and ag.BOOTSTRAP_SEED == 20261013


def test_the_bootstrap_of_a_constant_difference_collapses_to_it():
    out = ag.paired_bootstrap({"a": [0.25] * 9}, trials=500)
    assert out["a"] == {"mean": 0.25, "lo": 0.25, "hi": 0.25}


def test_the_bootstrap_interval_uses_the_percentile_positions():
    diffs = {"a": [0.0, 1.0]}
    out = ag.paired_bootstrap(diffs, trials=40, seed=3)
    assert out["a"]["lo"] in (0.0, 0.5, 1.0) and out["a"]["hi"] in (0.0, 0.5, 1.0)


def test_the_bootstrap_rejects_empty_or_ragged_input():
    with pytest.raises(ValueError):
        ag.paired_bootstrap({"a": []})
    with pytest.raises(ValueError):
        ag.paired_bootstrap({"a": [1.0], "b": [1.0, 2.0]})


def test_the_percentile_is_nearest_rank_below_as_the_retrieval_table_reports_p95():
    values = list(range(1, 101))
    assert ag.percentile(values, 0.95) == 95  # sorted[int(0.95 * 99)] = sorted[94]
    assert ag.percentile([3.0, 1.0, 2.0], 0.95) == 2.0
    assert ag.percentile([], 0.95) == 0.0


# ---------------------------------------------------------------------------
# planning and files
# ---------------------------------------------------------------------------
def test_the_call_budget_is_one_to_three_judge_calls_and_at_most_two_answers_per_subset_question():
    plan = ag.plan_budget(184, 60)
    assert plan["judge_calls_min"] == 184 and plan["judge_calls_max"] == 552
    assert plan["answer_generations_min"] == 0 and plan["answer_generations_max"] == 120
    assert plan["llm_calls_max"] == 672


def test_the_counting_provider_counts_calls_and_cache_hits_and_changes_nothing():
    inner = FakeProvider(
        [LLMResponse(text="a", model=MODEL, provider="p", cached=True), "b"], model=MODEL
    )
    counting = ag.CountingProvider(inner)
    assert counting.model == MODEL and counting.offline is False
    assert counting.complete("x", system="s").text == "a"
    assert counting.complete("y", system="s").text == "b"
    assert (counting.calls, counting.cached) == (2, 1)
    assert inner.prompts == ["x", "y"]


def test_saving_is_all_or_nothing_and_never_overwrites(tmp_path):
    path = tmp_path / "agentic.json"
    with pytest.raises(TypeError):
        ag.save(path, {"bad": object()})
    assert not path.exists() and not list(tmp_path.glob("*.partial"))
    ag.save(path, {"ok": 1})
    assert json.loads(path.read_text("utf-8")) == {"ok": 1}
    with pytest.raises(FileExistsError):
        ag.save(path, {"ok": 2})
    assert not list(tmp_path.glob("*.partial"))


def test_the_run_config_records_the_registered_settings():
    config = ag.run_config(MODEL)
    assert config["protocol_commit"] == "4523723" and config["model"] == MODEL
    assert (config["k"], config["pool"], config["judge_top"], config["max_steps"]) == (10, 50, 5, 3)
    assert (config["rrf_k"], config["judge_max_tokens"], config["num_ctx_live"]) == (60, 200, 8192)
    assert config["passage_chars"] == 1_200 and config["gain_bar"] == 0.02
    assert config["judge_template"] == ag.JUDGE_TEMPLATE


def test_the_question_records_carry_both_rankings_and_every_step():
    world = World()
    result, step_one, loops, _pairs, _ = finish(world)
    records = ag.question_records(world.inputs, step_one, loops, result["retrieval"])
    assert len(records) == 16
    by = {r["qid"]: r for r in records}
    assert by["q0"]["changed"] and by["q0"]["n_calls"] == 2 and len(by["q0"]["steps"]) == 2
    assert by["q0"]["metrics"]["agentic"]["ndcg@10"] > by["q0"]["metrics"]["baseline"]["ndcg@10"]
    assert by["u0"]["metrics"] is None and by["u0"]["refused"]
    assert by["q12"]["changed"] is False and by["q12"]["baseline_latency_s"] == 0.01
    json.dumps(records)


def test_no_module_of_the_served_path_imports_the_agentic_arm():
    src = ROOT / "src" / "production_rag"
    importers = [
        p.name
        for p in src.glob("*.py")
        if p.name not in ("agentic.py", "cli.py")
        and re.search(
            r"(?m)^\s*(from \.agentic|from production_rag\.agentic|import production_rag\.agentic)",
            p.read_text("utf-8"),
        )
    ]
    assert importers == []
    assert not (ROOT / "space").exists() or "agentic" not in "".join(
        p.read_text("utf-8") for p in (ROOT / "space").glob("*.py")
    )


# ---------------------------------------------------------------------------
# the real inputs (need the local indexes)
# ---------------------------------------------------------------------------
def real_inputs() -> ag.Inputs:
    from production_rag.cli import load_chunks

    try:
        chunks = {c.chunk_id: c for c in load_chunks(ROOT / "indexes", ag.STRATEGY)}
    except FileNotFoundError:
        pytest.skip("indexes/ is not built; this checks the registered inputs on real chunks")
    return ag.load_inputs(
        chunks,
        index_chunks_by_doc(chunks.values()),
        questions_path=ROOT / "eval" / "questions.jsonl",
        results_path=ROOT / "eval" / "results" / "retrieval.json",
        answers_path=ROOT / "eval" / "results" / "answers.json",
    )


@pytest.mark.slow
def test_the_real_inputs_have_the_registered_shape():
    inputs = real_inputs()
    scored = [s for s in inputs.specs if s.scored]
    assert len(inputs.specs) == 184 and len(scored) == 177 and len(inputs.subset) == 60
    assert sum(1 for s in inputs.subset if s.question.is_answerable) == 54
    assert ag._counts(scored) == {"exact_term": 52, "conceptual": 121, "cross_tool": 4}
    assert set(inputs.published) == {s.qid for s in inputs.subset}
    assert {r.model for r in inputs.published.values()} == {MODEL}
    # every question has a step-1 judge prompt that fits comfortably in the window
    longest = 0
    for spec in inputs.specs:
        passages = ag.judge_passages(ranking(list(spec.baseline)), inputs.chunks)
        assert len(passages) == 5
        longest = max(longest, len(ag.render_judge_prompt(spec.text, passages)))
    assert longest < 4 * ag.NUM_CTX


@pytest.mark.slow
def test_with_the_loop_changing_nothing_the_baseline_reproduces_the_retrieval_table():
    inputs = real_inputs()
    scored = [s for s in inputs.specs if s.scored]
    same = {
        s.qid: ag.LoopResult(
            s.qid, s.text, [], [s.text], ranking(list(s.baseline)), False, "sufficient"
        )
        for s in scored
    }
    report = ag.retrieval_report(scored, same, trials=200)
    assert report["baseline_matches_stored"] is True
    assert report["overall"]["ndcg@10"]["delta"] == 0.0
    table = json.loads((ROOT / "eval" / "results" / "retrieval.json").read_text("utf-8"))
    (published,) = [a for a in table["arms"] if a["arm"] == ag.ARM and a["strategy"] == ag.STRATEGY]
    assert report["n"] == published["n_scored"]
    assert report["overall"]["ndcg@10"]["baseline"] == pytest.approx(
        published["overall"]["ndcg@10"]
    )


@pytest.mark.slow
def test_gate_one_holds_for_every_real_question_with_the_real_encoder():
    pytest.importorskip("sentence_transformers")
    from production_rag.dense import DEFAULT_MODEL, SentenceTransformerEmbedder
    from production_rag.pipeline import Retriever

    inputs = real_inputs()
    retriever = Retriever.load(
        ROOT / "indexes",
        ag.STRATEGY,
        embedder=SentenceTransformerEmbedder(DEFAULT_MODEL),
        with_dense=True,
        chunks=inputs.chunks,
    )
    step_one = ag.prepare_step_one(inputs.specs, ag.LoopRetriever(retriever))
    assert step_one.gate["passed"] and step_one.gate["checked"] == 184
