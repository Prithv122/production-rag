"""Gradio demo for the Hugging Face Space.

The demo runs the *same* pipeline the evaluation measured -- it imports
`production_rag` from the repository rather than reimplementing retrieval in a
UI file, which is the usual way a demo ends up quietly better (or quietly
worse) than the numbers in the README.

Three things are deliberate about how it is deployed.

**The indexes are downloaded, not rebuilt.** Building the dense index means
encoding 22,789 chunks, which is not something to do on every restart of a free
CPU box. So `index` runs once locally and the artefacts live in a companion
dataset repo, pulled with `snapshot_download` and cached on disk: a measured
256 s to first query cold (dominated by the 61 MB download) and 43 s warm. The
demo therefore shows exactly the index the README's numbers came from, not a
rebuild that might differ.

**Retrieval works with no secret; generation is the only part that needs one.**
If `OPENROUTER_API_KEY` is not set in the Space's secrets the app still ranks,
still shows the passages, and says plainly that generation is off. That mirrors
the repository -- every retrieval number reproduces without a key -- and it
means a fork of this Space is useful before its owner has done anything.

**The default arm is the one that won the evaluation**, not the most impressive
sounding one. The cross-encoder is offered as a checkbox with its real cost in
the label, because on this hardware it is roughly seven seconds of latency for
+0.004 nDCG, and a demo that hides that is advertising rather than
demonstrating.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from huggingface_hub import snapshot_download

from production_rag.generate import generate, link_citations, markdown_sources
from production_rag.pipeline import Retriever
from production_rag.providers import MODEL_ARMS, build_provider

INDEX_REPO = os.environ.get("INDEX_REPO", "Prithv122/production-rag-index")
STRATEGY = "heading"
DEFAULT_ARM = "hybrid_score_weighted"
ARMS = ["bm25", "dense", "hybrid", "hybrid_score", "hybrid_score_weighted"]
CACHE = Path(os.environ.get("CACHE_DIR", "/tmp/production-rag"))
INDEX_DIR = os.environ.get("INDEX_DIR", "")

#: Which generation arm the deployment uses. Configurable because a deployment
#: is not always the same shape as the author's laptop: someone self-hosting
#: next to their own Ollama needs no key at all, and pinning the arm in the
#: source would force them to fork the demo to say so.
MODEL_ARM = os.environ.get("MODEL_ARM", "nemotron-super")

EXAMPLES = [
    "how do I read a parquet file in duckdb?",
    "what does on_schema_change do in an incremental model?",
    "how do I partition an asset by date in dagster?",
    "what is the difference between a dbt model and a dagster asset?",
    "how do I configure the quantum flux capacitor in dbt?",
]


def _load() -> tuple[Retriever, str]:
    # `INDEX_DIR` is set by the container image, which bakes the pinned dataset
    # in at build time: a cold start that also downloads 61 MB measured 256 s
    # locally, and Cloud Run gives a container 240 s to accept traffic. With the
    # variable unset -- running the app directly -- behaviour is unchanged and
    # the index is fetched from the Hub.
    baked = Path(INDEX_DIR) if INDEX_DIR else None
    if baked is not None and (baked / STRATEGY).is_dir():
        root, source = baked, f"{INDEX_REPO} (baked into the image)"
    else:
        root = Path(snapshot_download(repo_id=INDEX_REPO, repo_type="dataset"))
        source = INDEX_REPO

    from production_rag.cache import CachedEmbedder
    from production_rag.dense import SentenceTransformerEmbedder

    embedder = CachedEmbedder(SentenceTransformerEmbedder(), CACHE / "embeddings")
    retriever = Retriever.load(root, STRATEGY, embedder=embedder, chunks=_chunks(root))
    return retriever, source


def _chunks(root: Path) -> dict:
    import json

    from production_rag.chunking import Chunk

    path = root / STRATEGY / "chunks.jsonl"
    with path.open(encoding="utf-8") as handle:
        return {
            chunk.chunk_id: chunk
            for chunk in (Chunk.from_dict(json.loads(line)) for line in handle if line.strip())
        }


RETRIEVER, INDEX_PATH = _load()

# Whether generation can run at all. A hosted arm needs OPENROUTER_API_KEY; a
# local Ollama arm needs nothing. `fallback=False` for the same reason the
# answer eval disables it: an answer silently produced by a different model than
# the footer names is worse than an honest failure.
NEEDS_KEY = MODEL_ARMS[MODEL_ARM]["provider"] != "ollama"
CAN_GENERATE = (not NEEDS_KEY) or bool(os.environ.get("OPENROUTER_API_KEY"))
PROVIDER = build_provider(MODEL_ARM, cache_dir=CACHE / "llm", fallback=False)


def run(question: str, arm: str, k: int, rerank: bool, answer_it: bool):
    question = (question or "").strip()
    if not question:
        return "Ask something.", "", []

    reranker = None
    if rerank:
        from production_rag.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        arm = "hybrid_rerank"

    result = RETRIEVER.retrieve(question, arm=arm, k=k, reranker=reranker)
    table = []
    for rank, (chunk_id, score) in enumerate(result.ranked, start=1):
        chunk = RETRIEVER.chunks[chunk_id]
        table.append([rank, round(float(score), 4), chunk.tool, chunk.breadcrumb, chunk.url])

    if not answer_it:
        return "_Generation is off — showing retrieval only._", "", table
    if not CAN_GENERATE:
        return (
            f"**Generation is disabled**: arm `{MODEL_ARM}` needs an `OPENROUTER_API_KEY` "
            "and none is set. Retrieval below is unaffected — every retrieval number in "
            "the README reproduces without a key.",
            "",
            table,
        )

    answer = generate(
        question,
        result.chunk_ids,
        RETRIEVER.chunks,
        PROVIDER,
        top_score=result.ranked[0][1] if result.ranked else None,
    )
    body = link_citations(answer)
    if answer.refused:
        body = f"**Refused** (`{answer.refusal_reason}`)\n\n{body}"
    elif answer.uncited:
        body += "\n\n> ⚠️ This answer cites nothing, so it is ungrounded — treat it as unsupported."

    footer = f"\n\n`{answer.provider}/{answer.model}` · {answer.latency_s:.1f}s"
    return body + footer, markdown_sources(answer), table


with gr.Blocks(title="production-rag") as demo:
    gr.Markdown(
        "# production-rag\n"
        "Hybrid BM25 + dense retrieval over the **DuckDB, dbt and Dagster** documentation, "
        "with cited answers and a refusal path. Every arm below was measured against a "
        "184-question evaluation set — the numbers, and the ones that disappointed, are in "
        "[the repository](https://github.com/Prithv122/production-rag).\n\n"
        f"Index: `{INDEX_REPO}` · strategy `{STRATEGY}` · "
        f"{len(RETRIEVER.chunks):,} chunks"
        + (
            f" · generation `{MODEL_ARM}`"
            if CAN_GENERATE
            else " · **generation disabled (no API key set)**"
        )
    )
    with gr.Row():
        question = gr.Textbox(label="Question", scale=4, placeholder=EXAMPLES[0])
        go = gr.Button("Ask", variant="primary", scale=1)
    with gr.Row():
        arm = gr.Dropdown(ARMS, value=DEFAULT_ARM, label="retrieval arm")
        k = gr.Slider(1, 20, value=8, step=1, label="passages (k)")
        rerank = gr.Checkbox(label="cross-encoder rerank (+~7s, +0.004 nDCG)", value=False)
        answer_it = gr.Checkbox(label="generate an answer", value=True)

    answer_box = gr.Markdown(label="Answer")
    sources_box = gr.Markdown(label="Cited sources")
    results = gr.Dataframe(
        headers=["#", "score", "tool", "section", "url"],
        label="Retrieved passages",
        wrap=True,
    )
    gr.Examples(EXAMPLES, inputs=question)

    go.click(run, [question, arm, k, rerank, answer_it], [answer_box, sources_box, results])
    question.submit(run, [question, arm, k, rerank, answer_it], [answer_box, sources_box, results])

if __name__ == "__main__":
    # Cloud Run injects $PORT and requires the process to listen on it, on all
    # interfaces. Unset -- i.e. run directly -- and this is Gradio's own default
    # of 127.0.0.1:7860, so a local run does not silently start binding 0.0.0.0.
    port = os.environ.get("PORT")
    if port:
        demo.launch(server_name="0.0.0.0", server_port=int(port))
    else:
        demo.launch()
