"""The generation model as a swappable component.

Every LLM in this project sits behind :class:`LLMProvider`. That is not
architecture for its own sake -- it is what makes three separate claims testable
rather than asserted:

1. **"Model X is better here"** is only meaningful if the models can be swapped
   with everything else held fixed. The eval harness feeds every provider the
   *same frozen retrieved chunks*, so the only variable is generation.
2. **"It degrades gracefully"** is only true if there is a fallback path that has
   actually been exercised. :class:`FallbackProvider` is that path, and it is
   tested against a provider that always fails.
3. **"These numbers reproduce"** is only true offline. :class:`CachedProvider`
   keys on everything that affects the output, and replay mode turns a cache
   miss into an error rather than a network call.

**Why raw HTTP and not the openai package.** The wire format here is one POST of
a JSON body to ``/chat/completions`` and one dictionary lookup in the response.
Ollama's native ``/api/chat`` is a *different* shape, so a client library covers
one of the two providers and the second still needs hand-written code. Against
~40 lines of ``urllib``, the package would add a dependency tree to the offline
retrieval path that exists purely so ``import openai`` succeeds. The endpoint is
OpenAI-compatible; the client does not have to be.

**Structured output is a capability, not an assumption.** OpenRouter models
differ in whether they honour ``response_format``. ``nemotron-3-super`` does;
``nemotron-3-ultra`` does not, which is precisely why it is kept as a comparison
arm -- it exercises :func:`extract_json` on a model that wraps JSON in prose or
markdown fences. A pipeline that only works with models supporting structured
output has not solved JSON parsing, it has outsourced it.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .cache import JsonCache

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct-q3_K_M"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class ProviderError(RuntimeError):
    """A provider could not answer. Triggers the fallback chain."""


@dataclass(frozen=True)
class LLMResponse:
    """One completion, plus everything needed to audit where it came from."""

    text: str
    model: str
    provider: str
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    finish_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "latency_s": self.latency_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, cached: bool = False) -> LLMResponse:
        return cls(**data, cached=cached)


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that turns a prompt into text."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network-dependent
        body = exc.read().decode("utf-8", "replace")[:400]
        raise ProviderError(f"HTTP {exc.code} from {url}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError(f"could not reach {url}: {exc}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - server-dependent
        raise ProviderError(f"non-JSON response from {url}") from exc


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
@dataclass
class OpenRouterProvider:
    """OpenAI-compatible chat completions against OpenRouter."""

    model_name: str = DEFAULT_OPENROUTER_MODEL
    api_key: str | None = None
    timeout: float = 120.0
    max_retries: int = 2
    url: str = OPENROUTER_URL

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY") or ""

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def model(self) -> str:
        return self.model_name

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> LLMResponse:
        if not self.api_key:
            raise ProviderError("OPENROUTER_API_KEY is not set")

        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        data: dict[str, Any] = {}
        for attempt in range(self.max_retries + 1):
            try:
                data = _post_json(
                    self.url,
                    payload,
                    {
                        "Authorization": f"Bearer {self.api_key}",
                        # OpenRouter attributes free-tier usage to a referrer.
                        "HTTP-Referer": "https://github.com/Prithv122/production-rag",
                        "X-Title": "production-rag",
                    },
                    self.timeout,
                )
                break
            except ProviderError:
                if attempt == self.max_retries:
                    raise
                # Free-tier models rate-limit rather than fail permanently, so a
                # short backoff converts most errors into a slower success.
                time.sleep(2.0 * (attempt + 1))

        if "error" in data and not data.get("choices"):
            raise ProviderError(f"openrouter error: {str(data['error'])[:300]}")
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected response shape: {str(data)[:300]}") from exc

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", self.model_name),
            provider=self.name,
            latency_s=time.perf_counter() - started,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=str(choice.get("finish_reason") or ""),
        )


@dataclass
class OllamaProvider:
    """Local Ollama. The offline fallback -- no key, no network egress."""

    model_name: str = DEFAULT_OLLAMA_MODEL
    host: str | None = None
    timeout: float = 300.0

    def __post_init__(self) -> None:
        self.host = (self.host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self.model_name

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_object:
            payload["format"] = "json"

        started = time.perf_counter()
        data = _post_json(f"{self.host}/api/chat", payload, {}, self.timeout)
        try:
            text = data["message"]["content"]
        except KeyError as exc:
            raise ProviderError(f"unexpected ollama response: {str(data)[:300]}") from exc
        return LLMResponse(
            text=text,
            model=data.get("model", self.model_name),
            provider=self.name,
            latency_s=time.perf_counter() - started,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            finish_reason=str(data.get("done_reason") or ""),
        )


@dataclass
class FallbackProvider:
    """Try providers in order; the first that answers wins.

    Deliberately narrow: it catches :class:`ProviderError` and nothing else. A
    ``KeyError`` in our own parsing is a bug, and silently retrying it on the
    fallback model would hide the bug *and* attribute a number to the wrong
    model.

    ``used`` records which provider actually served each call, because "we have
    a fallback" and "the fallback ran and nobody noticed" look identical in the
    output otherwise.
    """

    providers: list[LLMProvider]
    used: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("FallbackProvider needs at least one provider")

    @property
    def name(self) -> str:
        return "+".join(p.name for p in self.providers)

    @property
    def model(self) -> str:
        return self.providers[0].model

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        errors: list[str] = []
        for provider in self.providers:
            try:
                response = provider.complete(prompt, **kwargs)
            except ProviderError as exc:
                errors.append(f"{provider.name}({provider.model}): {exc}")
                self.failures.append(provider.name)
                continue
            self.used.append(provider.name)
            return response
        raise ProviderError("all providers failed:\n  " + "\n  ".join(errors))


@dataclass
class CachedProvider:
    """Memoises completions on disk so published numbers replay offline."""

    inner: LLMProvider
    cache: JsonCache
    offline: bool = False

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> LLMResponse:
        key = {
            # Everything that changes the output, and nothing that does not.
            # The API key is not here: it is a credential, not an input.
            "provider": self.inner.name,
            "model": self.inner.model,
            "system": system,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_object": json_object,
        }
        hit = self.cache.get(key)
        if hit is not None:
            return LLMResponse.from_dict(hit, cached=True)
        if self.offline:
            raise ProviderError(
                "replay mode: this prompt is not in the cache, and replay must not "
                "call out. Re-run without --replay to populate it."
            )
        response = self.inner.complete(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_object=json_object,
        )
        self.cache.put(key, response.as_dict())
        return response


# ---------------------------------------------------------------------------
# Getting JSON out of a model that was only asked nicely
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"`{3}(?:json)?\s*(.*?)`{3}", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def extract_json(text: str) -> Any:
    """Parse JSON out of a model response, repairing the usual damage.

    In order of how often each actually happens with the free-tier models used
    here: a fenced code block; leading prose ("Here is the JSON:"); a trailing
    comma before a closing brace; and a trailing explanation after the object.
    The repairs are deliberately conservative -- each is a *syntactic* fix that
    cannot change which values are parsed. Nothing here guesses at missing
    fields; a response that is not recoverably JSON raises, and the caller
    counts it against that model's parse rate.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    candidates: list[str] = []
    fence = _FENCE.search(text)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        for attempt in (candidate, _balanced_span(candidate)):
            if not attempt:
                continue
            for repaired in (attempt, _TRAILING_COMMA.sub(r"\1", attempt)):
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"no parseable JSON in response: {text[:200]!r}")


def _balanced_span(text: str) -> str:
    """The first balanced ``{...}`` or ``[...]``, ignoring braces inside strings."""
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return ""
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
CACHE_DIR = Path(".cache/llm")

#: The comparison arms. ``structured`` records whether the model honours
#: ``response_format`` -- verified against the live OpenRouter catalogue in
#: session 1, not copied from a blog post.
MODEL_ARMS: dict[str, dict[str, Any]] = {
    "nemotron-super": {
        "provider": "openrouter",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "structured": True,
        "note": "primary; 262k context, honours response_format",
    },
    "nemotron-ultra": {
        "provider": "openrouter",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "structured": False,
        "note": "NEVER RUN. The id here was wrong until session 3 (missing the -a55b "
        "suffix), so every call to this arm was a 400 that the fallback chain swallowed. "
        "Corrected against the live model list; still unrun, because the free tier is "
        "capped account-wide at 50 requests/day",
    },
    "gemini-flash-lite": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash-lite",
        "structured": True,
        "note": "NEVER RUN. Paid arm, $0.10/$0.40 per M tokens, and the account has no "
        "credits -- every call is a 402",
    },
    "ollama-qwen": {
        "provider": "ollama",
        "model": DEFAULT_OLLAMA_MODEL,
        "structured": True,
        "note": "offline fallback; no key, no egress",
    },
    # The three arms below are what the published answer table actually ran on,
    # and the reason is in the README: the hosted arms are unreachable on a free
    # OpenRouter account (429 at 50 requests/day for the `:free` models, 402 for
    # anything paid). Local models are not a consolation prize here -- they need
    # no key, no quota and no trust in me, so the answer table is the *more*
    # reproducible half of this project rather than the less.
    "ollama-qwen-coder": {
        "provider": "ollama",
        "model": "qwen2.5-coder:7b-instruct-q3_K_M",
        "structured": True,
        "note": "same size and quantisation as ollama-qwen, code-tuned -- isolates tuning",
    },
    "ollama-llama-3b": {
        "provider": "ollama",
        "model": "llama3.2:3b",
        "structured": True,
        "note": "different family, less than half the parameters -- isolates scale",
    },
}

#: Arms that run with no API key and no egress. The answer eval defaults to
#: these, because an evaluation nobody else can re-run is not an evaluation.
LOCAL_ARMS = ("ollama-qwen", "ollama-qwen-coder", "ollama-llama-3b")


def build_provider(
    arm: str = "nemotron-super",
    *,
    cache_dir: Path = CACHE_DIR,
    offline: bool = False,
    fallback: bool = True,
) -> CachedProvider:
    """Assemble the provider stack for a named arm.

    Layering is cache -> fallback chain -> concrete provider, and that order is
    load-bearing. Cache outermost means a replay never touches the chain at all,
    so replay mode cannot accidentally reach the network through a fallback.
    """
    try:
        spec = MODEL_ARMS[arm]
    except KeyError:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(MODEL_ARMS)}") from None

    primary: LLMProvider
    if spec["provider"] == "ollama":
        primary = OllamaProvider(spec["model"])
    else:
        primary = OpenRouterProvider(spec["model"])

    chain: LLMProvider = primary
    if fallback and spec["provider"] != "ollama":
        chain = FallbackProvider([primary, OllamaProvider()])
    return CachedProvider(chain, JsonCache(cache_dir), offline=offline)
