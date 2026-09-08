"""Normalisation tests.

Everything here is a pure string transform, so the whole file runs with no
network and no corpus checkout. The two regression classes at the bottom cover
bugs that shipped silently in the first version of the normaliser and were only
caught by scanning the whole corpus for leftover markup -- eyeballing three
files would have missed both.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from production_rag.ingest import (
    Document,
    build_url,
    first_heading,
    frontmatter_title,
    parse_document,
    read_jsonl,
    split_frontmatter,
    strip_markup,
    write_jsonl,
)
from production_rag.sources import SOURCES, SOURCES_BY_NAME, Source

FENCE = "```"


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------
def test_split_frontmatter_extracts_and_removes() -> None:
    front, body = split_frontmatter("---\ntitle: Incremental models\n---\nBody text.\n")
    assert front == "title: Incremental models"
    assert body == "Body text.\n"


def test_split_frontmatter_absent_is_passthrough() -> None:
    front, body = split_frontmatter("# Heading\n\nNo frontmatter here.\n")
    assert front == ""
    assert body.startswith("# Heading")


def test_split_frontmatter_does_not_eat_a_horizontal_rule() -> None:
    """A `---` rule mid-document must not be mistaken for frontmatter."""
    text = "Intro paragraph.\n\n---\n\nAfter the rule.\n"
    front, body = split_frontmatter(text)
    assert front == ""
    assert body == text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("title: Plain", "Plain"),
        ('title: "Quoted title"', "Quoted title"),
        ("title: 'Single quoted'", "Single quoted"),
        ("id: x\ntitle: Second line\nsidebar: 3", "Second line"),
        ("id: no-title-here", ""),
    ],
)
def test_frontmatter_title(raw: str, expected: str) -> None:
    assert frontmatter_title(raw) == expected


# ---------------------------------------------------------------------------
# Markup stripping
# ---------------------------------------------------------------------------
def test_strip_markup_removes_mdx_scaffolding() -> None:
    body = (
        "import Tabs from '@theme/Tabs';\n"
        "export const x = 1;\n"
        "\n"
        "<Tabs>\n"
        "Real prose survives.\n"
        "</Tabs>\n"
        "{/* an mdx comment */}\n"
    )
    out = strip_markup(body)
    assert out == "Real prose survives."


def test_strip_markup_keeps_directive_titles_but_drops_the_marker() -> None:
    out = strip_markup(":::note Watch out\nThe body of the note.\n:::\n")
    assert "Watch out" in out
    assert "The body of the note." in out
    assert ":::" not in out


def test_strip_markup_preserves_code_fences_verbatim() -> None:
    """The whole reason fences are split out before tag stripping."""
    body = f"Prose.\n\n{FENCE}sql\nSELECT * FROM t WHERE a < b AND c > d;\n{FENCE}\n"
    out = strip_markup(body)
    assert "SELECT * FROM t WHERE a < b AND c > d;" in out


def test_strip_markup_does_not_eat_generics_in_code() -> None:
    body = f"{FENCE}cpp\nstd::vector<int> values;\n{FENCE}\n"
    assert "std::vector<int> values;" in strip_markup(body)


def test_strip_markup_collapses_excess_blank_lines() -> None:
    assert strip_markup("a\n\n\n\n\nb") == "a\n\nb"


# ---------------------------------------------------------------------------
# Regression: single-pass tag stripping left markup behind
# ---------------------------------------------------------------------------
def test_nested_angle_placeholder_is_fully_stripped() -> None:
    """`<File name='models/<filename>.yml'>` needs more than one pass.

    A left-to-right `re.sub` cannot match the outer tag (its attribute contains a
    `<`), removes only the inner placeholder, and leaves `<File name='models/.yml'>`
    behind. This shape occurs in 135 dbt documents.
    """
    out = strip_markup("<File name='models/<filename>.yml'>\n\nSchema docs.\n")
    assert out == "Schema docs."


def test_multiline_tag_is_stripped() -> None:
    body = '<VersionBlock\n  firstVersion="1.6"\n  lastVersion="1.9">\n\nContent.\n'
    assert strip_markup(body) == "Content."


# ---------------------------------------------------------------------------
# Regression: greedy `\s*` in the directive pattern crossed newlines
# ---------------------------------------------------------------------------
def test_bare_directive_close_does_not_swallow_the_next_line() -> None:
    """`\\s` matches newlines; `[ \\t]` does not.

    With a greedy `\\s*`, the bare closing `:::` consumed the blank line after it
    and captured the following paragraph as the directive's title -- relocating
    body text without any visible error.
    """
    out = strip_markup(":::info Title\nInside.\n:::\n\nA following paragraph.\n")
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines == ["Title", "Inside.", "A following paragraph."]


def test_import_inside_a_code_fence_is_kept() -> None:
    """`import dagster as dg` in an example is content, not MDX scaffolding."""
    body = f"{FENCE}python\nimport dagster as dg\n{FENCE}\n"
    assert "import dagster as dg" in strip_markup(body)


# ---------------------------------------------------------------------------
# Headings, URLs, documents
# ---------------------------------------------------------------------------
def test_first_heading_finds_the_first_atx_heading() -> None:
    assert first_heading("intro\n\n## Second level\n\n# Later\n") == "Second level"


def test_first_heading_absent() -> None:
    assert first_heading("just prose") == ""


@pytest.mark.parametrize(
    ("tool", "rel_path", "expected"),
    [
        (
            "duckdb",
            "docs/current/sql/statements/select.md",
            "https://duckdb.org/docs/stable/sql/statements/select",
        ),
        (
            "dbt",
            "website/docs/reference/commands/run.md",
            "https://docs.getdbt.com/reference/commands/run",
        ),
        (
            "dagster",
            "docs/docs/guides/build/assets.mdx",
            "https://docs.dagster.io/guides/build/assets",
        ),
        ("dagster", "docs/docs/guides/index.md", "https://docs.dagster.io/guides"),
    ],
)
def test_build_url(tool: str, rel_path: str, expected: str) -> None:
    assert build_url(SOURCES_BY_NAME[tool], rel_path) == expected


def test_parse_document_end_to_end(tmp_path: Path) -> None:
    source = SOURCES_BY_NAME["dbt"]
    path = tmp_path / "website" / "docs" / "reference" / "commands" / "run.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\ntitle: 'dbt run'\n---\n\nimport X from 'y';\n\n# dbt run\n\nRuns the models.\n",
        encoding="utf-8",
    )

    doc = parse_document(source, tmp_path, path)

    assert doc is not None
    assert doc.doc_id == "dbt/reference/commands/run"
    assert doc.tool == "dbt"
    assert doc.title == "dbt run"
    assert doc.url == "https://docs.getdbt.com/reference/commands/run"
    assert "Runs the models." in doc.text
    assert "import X" not in doc.text


def test_parse_document_returns_none_for_markup_only_file(tmp_path: Path) -> None:
    source = SOURCES_BY_NAME["dbt"]
    path = tmp_path / "website" / "docs" / "stub.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\ntitle: Stub\n---\n\nimport A from 'b';\n<Only />\n", encoding="utf-8")

    assert parse_document(source, tmp_path, path) is None


def test_parse_document_falls_back_to_heading_then_stem(tmp_path: Path) -> None:
    source = SOURCES_BY_NAME["dbt"]
    path = tmp_path / "website" / "docs" / "no-front.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Heading Wins\n\nBody.\n", encoding="utf-8")
    assert parse_document(source, tmp_path, path).title == "Heading Wins"

    other = tmp_path / "website" / "docs" / "bare-stem.md"
    other.write_text("Body with no heading.\n", encoding="utf-8")
    assert parse_document(source, tmp_path, other).title == "bare-stem"


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------
def test_jsonl_round_trip(tmp_path: Path) -> None:
    docs = [
        Document(
            doc_id="duckdb/sql/select",
            tool="duckdb",
            path="docs/current/sql/select.md",
            title="SELECT",
            url="https://duckdb.org/docs/stable/sql/select",
            text='Prose with a unicode em dash — and a quote ".',
        )
    ]
    out = tmp_path / "nested" / "docs.jsonl"

    assert write_jsonl(docs, out) == 1
    assert read_jsonl(out) == docs


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------
def test_sources_are_pinned_to_full_shas() -> None:
    """A branch name here would make every published metric unreproducible."""
    for source in SOURCES:
        assert len(source.sha) == 40, source.name
        assert all(c in "0123456789abcdef" for c in source.sha), source.name


def test_duckdb_source_excludes_the_version_archive() -> None:
    """Guards the corpus decision: `docs/current`, never the whole `docs/` tree."""
    assert SOURCES_BY_NAME["duckdb"].doc_root == "docs/current"


def test_source_clone_url() -> None:
    source = Source(
        name="x",
        repo="owner/name",
        sha="0" * 40,
        doc_root="docs",
        licence="MIT",
        url_template="https://example.test/{path}",
    )
    assert source.clone_url == "https://github.com/owner/name.git"
