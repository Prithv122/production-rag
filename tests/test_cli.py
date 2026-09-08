"""CLI tests.

Only the argument wiring and the pure-Python subcommands are covered here.
`ingest` needs the network and `index --strategy X` (with dense) needs the real
encoder, so those are exercised by the slow integration test rather than the
fast suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from production_rag.cli import build_parser, cmd_stats, load_chunks, main
from production_rag.ingest import Document, write_jsonl


@pytest.fixture
def docs_file(tmp_path: Path) -> Path:
    path = tmp_path / "docs.jsonl"
    write_jsonl(
        [
            Document(
                doc_id="dbt/a",
                tool="dbt",
                path="website/docs/a.md",
                title="A",
                url="https://docs.getdbt.com/a",
                text="dbt incremental models",
            ),
            Document(
                doc_id="duckdb/b",
                tool="duckdb",
                path="docs/current/b.md",
                title="B",
                url="https://duckdb.org/docs/stable/b",
                text="duckdb read_parquet",
            ),
        ],
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_search_defaults_to_hybrid_rrf() -> None:
    args = build_parser().parse_args(["search", "how do incremental models work"])
    assert args.arm == "hybrid"
    assert args.fusion == "rrf"
    assert args.strategy == "heading"


def test_index_accepts_repeated_strategies() -> None:
    args = build_parser().parse_args(["index", "--strategy", "fixed", "--strategy", "heading"])
    assert args.strategy == ["fixed", "heading"]


def test_index_strategy_defaults_to_none_meaning_all() -> None:
    assert build_parser().parse_args(["index"]).strategy is None


def test_index_rejects_an_unknown_strategy() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["index", "--strategy", "semantic-magic"])


def test_search_rejects_an_unknown_arm() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["search", "q", "--arm", "magic"])


def test_ingest_version_archive_flag_defaults_off() -> None:
    """The ablation corpus must never be the default -- it poisons precision."""
    assert build_parser().parse_args(["ingest"]).include_version_archive is False
    assert (
        build_parser().parse_args(["ingest", "--include-version-archive"]).include_version_archive
    )


def test_index_no_dense_flag() -> None:
    assert build_parser().parse_args(["index", "--no-dense"]).no_dense is True


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def test_stats_reports_documents_per_tool(
    docs_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = build_parser().parse_args(
        ["stats", "--docs", str(docs_file), "--indexes", str(tmp_path / "none")]
    )
    assert cmd_stats(args) == 0

    out = capsys.readouterr().out
    assert "documents      2" in out
    assert "dbt" in out and "duckdb" in out


def test_stats_skips_strategies_with_no_built_index(
    docs_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = build_parser().parse_args(
        ["stats", "--docs", str(docs_file), "--indexes", str(tmp_path / "missing")]
    )
    cmd_stats(args)
    assert "chunks" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Chunk persistence
# ---------------------------------------------------------------------------
def test_load_chunks_round_trips_the_heading_path(tmp_path: Path, sample_chunks: list) -> None:
    path = tmp_path / "heading" / "chunks.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in sample_chunks:
            handle.write(json.dumps(chunk.to_dict()) + "\n")

    loaded = load_chunks(tmp_path, "heading")

    assert loaded == sample_chunks
    assert isinstance(loaded[0].heading_path, tuple)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def test_main_dispatches_to_the_subcommand(docs_file: Path, tmp_path: Path) -> None:
    code = main(["stats", "--docs", str(docs_file), "--indexes", str(tmp_path / "none")])
    assert code == 0
