from __future__ import annotations

import pytest

from conftest import FakeProvider
from production_rag.cache import JsonCache
from production_rag.providers import (
    CachedProvider,
    FallbackProvider,
    LLMResponse,
    OpenRouterProvider,
    ProviderError,
    extract_json,
    is_replaying,
)

FENCE = "`" * 3


# ---------------------------------------------------------------------------
# JSON repair -- every case here is one a free-tier model actually produced
# ---------------------------------------------------------------------------
def test_plain_json():
    assert extract_json('{"queries": ["a"]}') == {"queries": ["a"]}


def test_fenced_json():
    text = f'Sure!\n{FENCE}json\n{{"queries": ["a"]}}\n{FENCE}\nHope that helps.'
    assert extract_json(text) == {"queries": ["a"]}


def test_leading_prose_without_a_fence():
    assert extract_json('Here is the JSON: {"a": 1}') == {"a": 1}


def test_trailing_explanation_after_the_object():
    assert extract_json('{"a": 1} -- note that this is approximate.') == {"a": 1}


def test_trailing_comma_is_repaired():
    assert extract_json('{"a": 1, "b": [2, 3,],}') == {"a": 1, "b": [2, 3]}


def test_bare_array():
    assert extract_json('["one", "two"]') == ["one", "two"]


def test_braces_inside_strings_do_not_end_the_object():
    """`{% link %}` appears all over the DuckDB docs and lands in quoted text."""
    assert extract_json('prose {"q": "use {% link %} here"} tail') == {"q": "use {% link %} here"}


@pytest.mark.parametrize("text", ["", "   ", "no json at all", "{unclosed: "])
def test_unrecoverable_input_raises(text):
    with pytest.raises(ValueError):
        extract_json(text)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------
def test_fallback_uses_the_second_provider_and_records_it():
    down = FakeProvider(name="primary", fail=True)
    up = FakeProvider(["ok"], name="secondary")
    chain = FallbackProvider([down, up])

    assert chain.complete("hi").text == "ok"
    assert chain.used == ["secondary"]
    assert chain.failures == ["primary"]


def test_fallback_raises_when_every_provider_fails():
    chain = FallbackProvider([FakeProvider(name="a", fail=True), FakeProvider(name="b", fail=True)])
    with pytest.raises(ProviderError, match="all providers failed"):
        chain.complete("hi")


def test_fallback_does_not_swallow_our_own_bugs():
    """Only ProviderError falls through; a real bug must surface, not degrade."""

    class Broken(FakeProvider):
        def complete(self, prompt, **kwargs):
            raise KeyError("choices")

    chain = FallbackProvider([Broken(name="broken"), FakeProvider(["ok"])])
    with pytest.raises(KeyError):
        chain.complete("hi")


def test_fallback_needs_at_least_one_provider():
    with pytest.raises(ValueError):
        FallbackProvider([])


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def test_second_identical_call_is_served_from_disk(tmp_path):
    inner = FakeProvider(["first", "second"])
    provider = CachedProvider(inner, JsonCache(tmp_path))

    assert provider.complete("q").text == "first"
    again = provider.complete("q")
    assert again.text == "first" and again.cached
    assert len(inner.prompts) == 1


def test_cache_key_separates_sampling_parameters(tmp_path):
    inner = FakeProvider(["a", "b"])
    provider = CachedProvider(inner, JsonCache(tmp_path))
    assert provider.complete("q", temperature=0.0).text == "a"
    assert provider.complete("q", temperature=0.7).text == "b"


def test_replay_refuses_to_call_out(tmp_path):
    inner = FakeProvider(["a"])
    provider = CachedProvider(inner, JsonCache(tmp_path), offline=True)
    with pytest.raises(ProviderError, match="replay mode"):
        provider.complete("never seen")
    assert inner.prompts == []


def test_replay_serves_what_was_cached(tmp_path):
    cache = JsonCache(tmp_path)
    CachedProvider(FakeProvider(["cached answer"]), cache).complete("q")
    replayed = CachedProvider(FakeProvider(fail=True), cache, offline=True).complete("q")
    assert replayed.text == "cached answer"


def test_the_api_key_never_reaches_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-do-not-store")
    cache = JsonCache(tmp_path)
    CachedProvider(FakeProvider(["ok"]), cache).complete("q")
    on_disk = "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))
    assert "sk-secret-do-not-store" not in on_disk


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------
def test_missing_key_is_a_provider_error_not_a_crash(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider().complete("hi")


def test_response_round_trips_through_the_cache_shape():
    response = LLMResponse(text="t", model="m", provider="p", prompt_tokens=3)
    assert LLMResponse.from_dict(response.as_dict(), cached=True).prompt_tokens == 3


# ---------------------------------------------------------------------------
# the upstream error that arrives as HTTP 200
#
# OpenRouter reports a failure *behind* it as a 200 carrying an error object and
# no choices: proxying worked, the model did not. Because that is not a
# transport error it never reached the backoff, so a transient overload became a
# hard refusal. A probe against the deployed image measured it on 6 of 13 calls
# to the free nemotron tier, which is not a rate worth failing at.
# ---------------------------------------------------------------------------
OVERLOADED = {
    "id": "gen-x",
    "error": {
        "message": "Upstream error from Nvidia: Service temporarily overloaded",
        "code": 502,
        "metadata": {"error_type": "provider_unavailable"},
    },
}
ANSWERED = {
    "choices": [{"message": {"content": "recovered"}, "finish_reason": "stop"}],
    "model": "m",
    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
}


def _scripted(monkeypatch, responses):
    calls: list[dict] = []

    def fake_post(url, payload, headers, timeout):
        calls.append(payload)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr("production_rag.providers._post_json", fake_post)
    monkeypatch.setattr("production_rag.providers.time.sleep", lambda _: None)
    return calls


def test_a_transient_upstream_error_is_retried_not_refused(monkeypatch):
    calls = _scripted(monkeypatch, [OVERLOADED, ANSWERED])
    response = OpenRouterProvider(api_key="k").complete("hi")
    assert response.text == "recovered"
    assert len(calls) == 2


def test_an_upstream_error_that_never_clears_still_raises(monkeypatch):
    calls = _scripted(monkeypatch, [OVERLOADED])
    with pytest.raises(ProviderError, match="provider_unavailable"):
        OpenRouterProvider(api_key="k", max_retries=2).complete("hi")
    assert len(calls) == 3, "all attempts used before giving up"


def test_finish_reason_is_carried_off_the_wire(monkeypatch):
    # Nothing read this field for six sessions while it held the answer to why
    # production generation was failing. It is part of the contract now.
    truncated = {
        "choices": [{"message": {"content": "{partial"}, "finish_reason": "length"}],
        "model": "m",
    }
    _scripted(monkeypatch, [truncated])
    assert OpenRouterProvider(api_key="k").complete("hi").finish_reason == "length"


def test_is_replaying_is_true_only_for_an_offline_cache(tmp_path):
    cache = JsonCache(tmp_path)
    assert not is_replaying(FakeProvider())
    assert not is_replaying(CachedProvider(FakeProvider(), cache))
    assert is_replaying(CachedProvider(FakeProvider(), cache, offline=True))


def test_a_completed_response_is_still_memoised(tmp_path):
    done = LLMResponse(text="done", model="m", provider="fake", finish_reason="stop")
    cache = JsonCache(tmp_path)
    CachedProvider(FakeProvider([done]), cache).complete("q")
    replayed = CachedProvider(FakeProvider(fail=True), cache, offline=True).complete("q")
    assert replayed.text == "done" and replayed.cached
