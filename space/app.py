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
from production_rag.providers import build_provider

INDEX_REPO = os.environ.get("INDEX_REPO", "Prithv122/production-rag-index")
STRATEGY = "heading"
DEFAULT_ARM = "hybrid_score_weighted"
ARMS = ["bm25", "dense", "hybrid", "hybrid_score", "hybrid_score_weighted"]
CACHE = Path(os.environ.get("CACHE_DIR", "/tmp/production-rag"))

EXAMPLES = [
    "how do I read a parquet file in duckdb?",
    "what does on_schema_change do in an incremental model?",
    "how do I partition an asset by date in dagster?",
    "what is the difference between a dbt model and a dagster asset?",
    "how do I configure the quantum flux capacitor in dbt?",
]


def _load() -> tuple[Retriever, str]:
    local = snapshot_download(repo_id=INDEX_REPO, repo_type="dataset", allow_patterns=["*"])
    root = Path(local)
    from production_rag.cache import CachedEmbedder
    from production_rag.dense import SentenceTransformerEmbedder

    embedder = CachedEmbedder(SentenceTransformerEmbedder(), CACHE / "embeddings")
    retriever = Retriever.load(root, STRATEGY, embedder=embedder, chunks=_chunks(root))
    return retriever, local


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
HAS_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))
PROVIDER = build_provider("nemotron-super", cache_dir=CACHE / "llm", fallback=False)


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
    if not HAS_KEY:
        return (
            "**Generation is disabled**: this Space has no `OPENROUTER_API_KEY` secret. "
            "Retrieval below is unaffected — every retrieval number in the README "
            "reproduces without a key.",
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
        + ("" if HAS_KEY else " · **generation disabled (no API key set)**")
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
    demo.launch()
