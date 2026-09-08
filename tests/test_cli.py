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


def test_verify_shuffles_by_default_so_a_partial_pass_is_a_sample() -> None:
    args = build_parser().parse_args(["verify"])
    assert args.shuffle is True
    assert build_parser().parse_args(["verify", "--no-shuffle"]).shuffle is False


def test_ask_defaults_to_the_arm_that_won_the_grid() -> None:
    args = build_parser().parse_args(["ask", "why?"])
    assert args.arm == "hybrid_score_weighted"
    # The pre-generation refusal gate ships off; see generate.py for why.
    assert args.min_top_score == 0.0


def test_answer_eval_runs_every_provider_arm_unless_told_otherwise() -> None:
    args = build_parser().parse_args(["answer-eval"])
    assert args.model_arm_list is None
    assert args.subset == 60
    picked = build_parser().parse_args(["answer-eval", "--model-arm-list", "ollama-qwen"])
    assert picked.model_arm_list == ["ollama-qwen"]


def test_answer_eval_rejects_an_unknown_provider_arm() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["answer-eval", "--model-arm-list", "gpt-9"])


def test_cache_audit_flags_a_bundle_answered_by_a_different_model(tmp_path: Path) -> None:
    from production_rag.cli import _audit_bundle

    bundle = tmp_path / "llm.jsonl"
    bundle.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"key": {"model": "big/model"}, "value": {"model": "big/model"}},
                {"key": {"model": "big/model"}, "value": {"model": "little/local"}},
                {"key": {"model": "big/model"}, "value": {"model": "little/local"}},
            ]
        ),
        encoding="utf-8",
    )
    assert _audit_bundle(bundle) == 2


def test_cache_audit_is_clean_when_every_answer_came_from_the_model_asked(tmp_path: Path) -> None:
    from production_rag.cli import _audit_bundle

    bundle = tmp_path / "llm.jsonl"
    bundle.write_text(
        json.dumps({"key": {"model": "m"}, "value": {"model": "m"}}), encoding="utf-8"
    )
    assert _audit_bundle(bundle) == 0


def test_cache_audit_on_a_missing_bundle_is_not_a_failure(tmp_path: Path) -> None:
    from production_rag.cli import _audit_bundle

    assert _audit_bundle(tmp_path / "nope.jsonl") == 0


def test_the_answer_eval_arms_that_need_no_key_are_the_default() -> None:
    from production_rag.providers import LOCAL_ARMS, MODEL_ARMS

    assert set(LOCAL_ARMS) <= set(MODEL_ARMS)
    assert all(MODEL_ARMS[a]["provider"] == "ollama" for a in LOCAL_ARMS)
