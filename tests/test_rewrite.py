from __future__ import annotations

import json

import pytest

from conftest import FakeProvider
from production_rag.rewrite import REWRITE_TEMPLATE, RewriteResult, rewrite_query

FENCE = "`" * 3


def test_expand_keeps_the_original_query_first():
    provider = FakeProvider([json.dumps({"queries": ["alt one", "alt two"]})])
    result = rewrite_query("on_schema_change", provider, mode="expand", n=2)
    assert result.queries == ("on_schema_change", "alt one", "alt two")


def test_replace_drops_the_original():
    provider = FakeProvider([json.dumps({"queries": ["alt one", "alt two"]})])
    result = rewrite_query("on_schema_change", provider, mode="replace", n=2)
    assert result.queries == ("alt one",)


def test_n_caps_the_number_of_variants():
    provider = FakeProvider([json.dumps({"queries": ["a", "b", "c", "d"]})])
    assert len(rewrite_query("q", provider, n=2).variants) == 2


def test_a_bare_array_is_accepted():
    provider = FakeProvider([json.dumps(["alt"])])
    assert rewrite_query("q", provider).variants == ("alt",)


def test_a_differently_named_key_is_accepted():
    provider = FakeProvider([json.dumps({"rewrites": ["alt"]})])
    assert rewrite_query("q", provider).variants == ("alt",)


def test_a_fenced_response_is_accepted():
    provider = FakeProvider([f'{FENCE}json\n{{"queries": ["alt"]}}\n{FENCE}'])
    assert rewrite_query("q", provider).variants == ("alt",)


def test_a_named_queries_key_wins_over_any_other_list():
    provider = FakeProvider([json.dumps({"queries": ["a"], "notes": ["b"]})])
    assert rewrite_query("q", provider).variants == ("a",)


def test_two_unnamed_lists_are_refused_rather_than_guessed():
    """With no `queries` key there is no principled choice, so it fails rather
    than retrieving with whichever list happened to come first."""
    provider = FakeProvider([json.dumps({"rewrites": ["a"], "notes": ["b"]})])
    assert rewrite_query("q", provider).parsed is False


def test_unparseable_output_degrades_to_the_original_query():
    """A rewriter outage must cost accuracy, not availability."""
    provider = FakeProvider(["I'm afraid I can't do that."])
    result = rewrite_query("read_parquet options", provider)
    assert result.parsed is False
    assert result.queries == ("read_parquet options",)
    assert result.error


def test_a_provider_error_degrades_the_same_way():
    result = rewrite_query("q", FakeProvider(fail=True))
    assert result.parsed is False
    assert result.queries == ("q",)


def test_empty_query_list_is_a_parse_failure():
    provider = FakeProvider([json.dumps({"queries": ["", "   "]})])
    assert rewrite_query("q", provider).parsed is False


@pytest.mark.parametrize("mode", ["", "rewrite", "expand_all"])
def test_unknown_mode_is_rejected(mode):
    with pytest.raises(ValueError):
        rewrite_query("q", FakeProvider(), mode=mode)


def test_the_prompt_still_forbids_paraphrasing_identifiers():
    """The whole exact-terminology finding depends on this instruction being
    present -- the rewrite arm has to be given its best shot, or a loss says
    nothing about the technique."""
    prompt = REWRITE_TEMPLATE.format(n=2, query="q")
    assert "EXACTLY" in prompt
    assert "on_schema_change" in prompt


def test_result_with_no_variants_is_just_the_original():
    assert RewriteResult("only this").queries == ("only this",)
