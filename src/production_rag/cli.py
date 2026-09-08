"""Command line entry point.

Subcommands mirror the pipeline stages so each can be run and inspected on its
own::

    production-rag ingest                 # clone at pinned SHAs, normalise
    production-rag index --strategy ...   # chunk, build BM25 + dense indexes
    production-rag search "..."           # query one arm, or the hybrid
    production-rag stats                  # corpus and index shape

`ingest` and `index` need the network and (for `index`) the `embed` extra;
`search` and `stats` run entirely off the built artefacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .bm25 import BM25Index
from .bm25 import build_from_chunks as build_bm25
from .chunking import CHUNKERS, USES_BREADCRUMB, Chunk, chunk_corpus
from .dense import DenseIndex
from .dense import build_from_chunks as build_dense
from .fuse import FUSIONS, fuse
from .ingest import ingest_source, read_jsonl, write_jsonl
from .sources import DUCKDB_VERSION_ARCHIVE, SOURCES

CORPUS_DIR = Path("corpus")
REPOS_DIR = CORPUS_DIR / "repos"
DOCS_PATH = CORPUS_DIR / "normalised" / "docs.jsonl"
INDEX_DIR = Path("indexes")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace) -> int:
    sources = list(SOURCES)
    if args.include_version_archive:
        sources.append(DUCKDB_VERSION_ARCHIVE)

    docs = []
    for source in sources:
        started = time.time()
        fetched = ingest_source(source, REPOS_DIR)
        docs.extend(fetched)
        chars = sum(len(d.text) for d in fetched)
        print(
            f"{source.name:16} {len(fetched):5} docs  {chars / 1e6:6.2f} M chars  "
            f"{source.licence:<11} {source.sha[:10]}  ({time.time() - started:.1f}s)"
        )

    written = write_jsonl(docs, args.out)
    print(f"\nwrote {written:,} documents to {args.out}")
    return 0


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------
def _chunk_path(root: Path, strategy: str) -> Path:
    return root / strategy / "chunks.jsonl"


def _write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(root: Path, strategy: str) -> list[Chunk]:
    with _chunk_path(root, strategy).open(encoding="utf-8") as handle:
        return [Chunk.from_dict(json.loads(line)) for line in handle if line.strip()]


def cmd_index(args: argparse.Namespace) -> int:
    docs = read_jsonl(args.docs)
    strategies = args.strategy or list(CHUNKERS)

    for strategy in strategies:
        target = args.out / strategy
        started = time.time()
        chunks = chunk_corpus(docs, strategy, max_chars=args.max_chars)
        _write_chunks(chunks, _chunk_path(args.out, strategy))
        breadcrumb = USES_BREADCRUMB[strategy]

        bm25 = build_bm25(chunks, with_breadcrumb=breadcrumb)
        bm25.save(target / "bm25")
        lexical_done = time.time()

        note = ""
        if not args.no_dense:
            from .dense import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(args.model)
            dense = build_dense(chunks, embedder, with_breadcrumb=breadcrumb)
            dense.save(target / "dense")
        else:
            note = "  (dense skipped)"

        print(
            f"{strategy:12} {len(chunks):6,} chunks  "
            f"bm25 {lexical_done - started:5.1f}s  "
            f"total {time.time() - started:6.1f}s{note}"
        )
    return 0


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def cmd_search(args: argparse.Namespace) -> int:
    root = args.indexes / args.strategy
    chunks = {c.chunk_id: c for c in load_chunks(args.indexes, args.strategy)}
    bm25 = BM25Index.load(root / "bm25")

    pool = max(args.k, args.pool)
    lexical = bm25.search(args.query, k=pool)

    if args.arm == "bm25":
        ranked = lexical[: args.k]
    else:
        from .dense import SentenceTransformerEmbedder

        dense = DenseIndex.load(root / "dense")
        embedder = SentenceTransformerEmbedder(dense.model_name)
        semantic = dense.search(args.query, embedder, k=pool)
        if args.arm == "dense":
            ranked = semantic[: args.k]
        else:
            ranked = fuse([lexical, semantic], args.fusion, k=args.k)

    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        chunk = chunks[chunk_id]
        print(f"{rank:2}. {score:8.4f}  {chunk.breadcrumb[:78]}")
        print(f"     {chunk.url}")
    if not ranked:
        print("no results")
    return 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def cmd_stats(args: argparse.Namespace) -> int:
    docs = read_jsonl(args.docs)
    by_tool: dict[str, int] = {}
    for doc in docs:
        by_tool[doc.tool] = by_tool.get(doc.tool, 0) + 1

    print(f"documents      {len(docs):,}")
    print(f"characters     {sum(len(d.text) for d in docs):,}")
    for tool, count in sorted(by_tool.items()):
        print(f"  {tool:12} {count:5,}")

    for strategy in CHUNKERS:
        path = _chunk_path(args.indexes, strategy)
        if not path.exists():
            continue
        chunks = load_chunks(args.indexes, strategy)
        lengths = sorted(len(c.text) for c in chunks)
        mean = sum(lengths) / len(lengths)
        print(
            f"{strategy:14} {len(chunks):6,} chunks  "
            f"mean {mean:5.0f}  median {lengths[len(lengths) // 2]:5}  max {lengths[-1]:5}"
        )
    return 0


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="production-rag", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="fetch and normalise the corpus")
    ingest.add_argument("--out", type=Path, default=DOCS_PATH)
    ingest.add_argument(
        "--include-version-archive",
        action="store_true",
        help="also ingest DuckDB's superseded version trees (ablation only)",
    )
    ingest.set_defaults(func=cmd_ingest)

    index = subparsers.add_parser("index", help="chunk and build retrieval indexes")
    index.add_argument("--docs", type=Path, default=DOCS_PATH)
    index.add_argument("--out", type=Path, default=INDEX_DIR)
    index.add_argument("--strategy", action="append", choices=sorted(CHUNKERS))
    index.add_argument("--max-chars", type=int, default=1200)
    index.add_argument("--model", default=None)
    index.add_argument(
        "--no-dense",
        action="store_true",
        help="build only the lexical index (skips the embedding model)",
    )
    index.set_defaults(func=cmd_index)

    search = subparsers.add_parser("search", help="query the built indexes")
    search.add_argument("query")
    search.add_argument("--indexes", type=Path, default=INDEX_DIR)
    search.add_argument("--strategy", choices=sorted(CHUNKERS), default="heading")
    search.add_argument("--arm", choices=["bm25", "dense", "hybrid"], default="hybrid")
    search.add_argument("--fusion", choices=sorted(FUSIONS), default="rrf")
    search.add_argument("-k", type=int, default=10)
    search.add_argument("--pool", type=int, default=50, help="candidates per arm before fusion")
    search.set_defaults(func=cmd_search)

    stats = subparsers.add_parser("stats", help="corpus and index shape")
    stats.add_argument("--docs", type=Path, default=DOCS_PATH)
    stats.add_argument("--indexes", type=Path, default=INDEX_DIR)
    stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "model", None) is None and hasattr(args, "model"):
        from .dense import DEFAULT_MODEL

        args.model = DEFAULT_MODEL
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
