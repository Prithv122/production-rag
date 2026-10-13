"""What a completion costs, in dollars, from its token counts.

The trace attaches a cost to every LLM span, so a per-query trace can answer
"what did this answer cost?" as well as "where did the time go?". The rule that
matters most is what happens for a model the table has never heard of: its cost
is **omitted and flagged**, never reported as zero. A zero there would read as
"free" in a dashboard and be summed into totals as if it had been measured.

**Every model this project has run is free**, so the table is mostly zeros, and
that is the true answer rather than a placeholder:

* local Ollama models have no per-token charge (the cost is electricity, which a
  token counter cannot see);
* OpenRouter ids ending in ``:free`` are free by definition.

No price is invented for a model the project does not use. The one non-zero entry
is a model the project *lists* but has never run (its account has no credits);
the price is the one recorded next to it in ``providers.MODEL_ARMS``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: USD per one million tokens, as ``(input, output)``, keyed by model id.
PRICES_USD_PER_M: dict[str, tuple[float, float]] = {
    # Local Ollama: no per-token charge.
    "qwen2.5:7b-instruct-q3_K_M": (0.0, 0.0),
    "qwen2.5-coder:7b-instruct-q3_K_M": (0.0, 0.0),
    "llama3.2:3b": (0.0, 0.0),
    # OpenRouter free tier.
    "nvidia/nemotron-3-super-120b-a12b:free": (0.0, 0.0),
    "nvidia/nemotron-3-ultra-550b-a55b:free": (0.0, 0.0),
    # Listed in providers.MODEL_ARMS, never run: "$0.10/$0.40 per M tokens".
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
}

#: Providers that never bill per token.
FREE_PROVIDERS = frozenset({"ollama"})


@dataclass(frozen=True)
class Cost:
    """The cost of one completion, in USD."""

    input: float
    output: float

    @property
    def total(self) -> float:
        return self.input + self.output

    def as_dict(self) -> dict[str, float]:
        return {"input": self.input, "output": self.output, "total": self.total}


def price_for(model: str, provider: str = "") -> tuple[float, float] | None:
    """``(input, output)`` USD per million tokens, or ``None`` if unknown.

    An exact table entry wins. Otherwise a ``:free`` suffix or a free provider
    means zero; anything else is unknown, and unknown is not zero.
    """
    if model in PRICES_USD_PER_M:
        return PRICES_USD_PER_M[model]
    if model.endswith(":free") or provider in FREE_PROVIDERS:
        return (0.0, 0.0)
    return None


def cost_usd(
    model: str, input_tokens: int, output_tokens: int, *, provider: str = ""
) -> Cost | None:
    """Tokens x price, or ``None`` when the model's price is not known."""
    price = price_for(model, provider)
    if price is None:
        return None
    return Cost(
        input=input_tokens * price[0] / 1_000_000,
        output=output_tokens * price[1] / 1_000_000,
    )
