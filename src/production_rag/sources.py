"""The corpus definition: which repositories, which commits, which paths.

Pinning a commit SHA rather than a branch is what makes the corpus reproducible.
A branch moves; "recall@10 was 0.71" is meaningless if nobody can rebuild the
index the number came from. Re-pinning is a deliberate act that invalidates the
published numbers, so it belongs in version control, not in a config file the
ingest step silently refreshes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    """One documentation repository, pinned."""

    name: str
    """Short tool name. Becomes the first path segment of every doc_id."""

    repo: str
    """GitHub `owner/name`."""

    sha: str
    """Full commit SHA. Not a branch -- see module docstring."""

    doc_root: str
    """Repo-relative directory that gets sparse-checked-out. Everything else is
    never downloaded, which is why a 1.5 GB monorepo costs us a few MB."""

    licence: str
    """SPDX identifier, verified against the GitHub licence API at pin time."""

    url_template: str
    """`{path}` -> public docs URL, so a citation can be clicked."""

    strip_prefix: str = ""
    """Removed from the repo-relative path before building the public URL."""

    exclude_dirs: tuple[str, ...] = field(default_factory=tuple)
    """Directory names dropped anywhere under `doc_root`."""

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.repo}.git"


# ---------------------------------------------------------------------------
# The corpus.
#
# DuckDB deliberately uses `docs/current` and NOT the whole `docs/` tree. The
# repo also carries docs/0.10, 1.0, 1.1, 1.2, 1.3 and lts -- 2,073 further
# markdown files that are near-duplicates of `current`. Indexing them makes
# "which chunk is the right answer?" ill-defined and measurably wrecks
# precision. `production-rag eval --ablation versions` quantifies exactly that,
# so the exclusion is an argued decision with a number attached rather than a
# convenient omission.
# ---------------------------------------------------------------------------
SOURCES: tuple[Source, ...] = (
    Source(
        name="duckdb",
        repo="duckdb/duckdb-web",
        sha="6f6cd1659f0e2ddd1965b1d3f1833e7fc512e7ac",
        doc_root="docs/current",
        licence="MIT",
        url_template="https://duckdb.org/docs/stable/{path}",
        strip_prefix="docs/current/",
    ),
    Source(
        name="dbt",
        repo="dbt-labs/docs.getdbt.com",
        sha="cd0e5b0e2b77302807f7dd126faac871e728edab",
        doc_root="website/docs",
        licence="Apache-2.0",
        url_template="https://docs.getdbt.com/{path}",
        strip_prefix="website/docs/",
    ),
    Source(
        name="dagster",
        repo="dagster-io/dagster",
        sha="76eed340c6b84517d91a86163c461c022ebef8d8",
        doc_root="docs/docs",
        licence="Apache-2.0",
        url_template="https://docs.dagster.io/{path}",
        strip_prefix="docs/docs/",
        exclude_dirs=("partials", "api"),
    ),
)

# Only used by the `versions` ablation -- never part of the shipped index.
DUCKDB_VERSION_ARCHIVE = Source(
    name="duckdb-archive",
    repo="duckdb/duckdb-web",
    sha="6f6cd1659f0e2ddd1965b1d3f1833e7fc512e7ac",
    doc_root="docs",
    licence="MIT",
    url_template="https://duckdb.org/docs/{path}",
    strip_prefix="docs/",
)

SOURCES_BY_NAME = {s.name: s for s in SOURCES}
