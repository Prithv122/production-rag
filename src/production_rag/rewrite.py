"""Query rewriting -- as a measured arm, not an assumed improvement.

The standard pitch is that users write bad queries and an LLM writes better
ones. That is half true, and the half that is false is expensive here. This
corpus is technical documentation, so a large share of real questions contain
the exact token that BM25 needs: ``on_schema_change``, ``read_parquet``,
``@dg.asset``. A paraphrase that "improves clarity" by turning
``on_schema_change`` into "handling schema changes" has destroyed the single
strongest retrieval signal available and replaced it with a weaker one.

So this module implements two modes and the evaluation decides:

``replace``
    Retrieve with the rewritten query only. The naive implementation, and the
    one that should be hurt most by the failure above.

``expand``
    Retrieve with the original *and* each variant, then fuse the rankings with
    RRF. The original query's ranking is always in the fusion, so a variant can
    add a passage but cannot silently remove the literal match. This costs one
    retrieval pass per variant.

**The prediction was recorded before the measurement** (README, session 1):
rewriting should hurt exact-terminology questions and help conceptual and
cross-tool ones, which -- if true -- makes the honest conclusion "rewrite
conditionally", not "rewriting improves retrieval". Reporting a single pooled
mean over a question set whose category mix you chose is a way of deciding the
answer in advance.

**One step, not a loop.** No re-retrieval based on what came back, no critic, no
agent. That is project 43.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .providers import LLMProvider, ProviderError, extract_json

MODES = ("replace", "expand")

REWRITE_SYSTEM = (
    "You rewrite search queries for a documentation search engine covering DuckDB, "
    "dbt and Dagster. You return JSON only."
)

# The instruction to preserve identifiers verbatim is doing real work: without
# it the model helpfully "cleans up" snake_case into English prose, which is the
# exact failure this arm is meant to expose. Keeping it makes the comparison
# fair -- the rewrite arm gets the best prompt I know how to write, so a loss is
# a loss for the technique rather than for a strawman of it.
REWRITE_TEMPLATE = """\
Rewrite the user's question into {n} alternative search queries for a keyword-and-vector
search engine over DuckDB, dbt and Dagster documentation.

Rules:
- Keep every literal identifier, function name, config key, flag and error string EXACTLY
  as written (for example on_schema_change, read_parquet, @dg.asset, --full-refresh).
  Never paraphrase an identifier into prose.
- Each alternative should use different vocabulary from the others: one closer to the
  words the documentation would use, one stating the underlying concept.
- Do not answer the question. Do not add commentary.
- Return JSON of the form {{"queries": ["...", "..."]}} with exactly {n} strings.

Question: {query}
"""


@dataclass(frozen=True)
class RewriteResult:
    """What the rewriter produced, and whether it actually worked."""

    original: str
    variants: tuple[str, ...] = ()
    mode: str = "expand"
    parsed: bool = True
    """False when the model's output could not be turned into queries. The
    pipeline then falls back to the original query -- degrading to the baseline
    arm rather than failing -- and the eval counts the failure."""

    cached: bool = False
    error: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def queries(self) -> tuple[str, ...]:
        """The queries retrieval should actually run.

        ``replace`` drops the original; ``expand`` keeps it first. If nothing
        parsed, both collapse to the original, which is what makes a rewriter
        outage a performance question rather than an availability one.
        """
        if not self.variants:
            return (self.original,)
        if self.mode == "replace":
            return self.variants[:1]
        return (self.original, *self.variants)


def rewrite_query(
    query: str,
    provider: LLMProvider,
    *,
    mode: str = "expand",
    n: int = 2,
    structured: bool = True,
    max_tokens: int = 300,
) -> RewriteResult:
    """Ask the provider for `n` alternative phrasings of `query`."""
    if mode not in MODES:
        raise ValueError(f"unknown rewrite mode {mode!r}; expected one of {list(MODES)}")
    if n < 1:
        raise ValueError("n must be at least 1")

    prompt = REWRITE_TEMPLATE.format(n=n, query=query)
    try:
        response = provider.complete(
            prompt,
            system=REWRITE_SYSTEM,
            temperature=0.0,
            max_tokens=max_tokens,
            json_object=structured,
        )
    except ProviderError as exc:
        return RewriteResult(query, mode=mode, parsed=False, error=str(exc)[:300])

    try:
        variants = _parse_variants(response.text, n)
    except ValueError as exc:
        return RewriteResult(
            query, mode=mode, parsed=False, cached=response.cached, error=str(exc)[:300]
        )

    return RewriteResult(query, tuple(variants), mode=mode, cached=response.cached)


def _parse_variants(text: str, n: int) -> list[str]:
    """Pull the query list out of the model's response.

    Accepts the documented shape and the two near-misses that actually occur: a
    bare JSON array, and an object with the list under some other single key.
    Anything else raises rather than being coerced -- a rewrite arm that quietly
    retrieves with garbage would still produce a number, and the number would be
    wrong in a way nobody could see.
    """
    data = extract_json(text)
    if isinstance(data, dict):
        if "queries" in data:
            data = data["queries"]
        else:
            lists = [v for v in data.values() if isinstance(v, list)]
            if len(lists) != 1:
                raise ValueError(f"no query list in object with keys {sorted(data)}")
            data = lists[0]
    if not isinstance(data, list):
        raise ValueError(f"expected a list of queries, got {type(data).__name__}")

    variants = [q.strip() for q in data if isinstance(q, str) and q.strip()]
    if not variants:
        raise ValueError("model returned no usable queries")
    return variants[:n]
