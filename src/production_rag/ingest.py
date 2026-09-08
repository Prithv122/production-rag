"""Fetch the pinned documentation repos and normalise them into flat documents.

Two deliberate choices here.

**Git partial clone, not an HTML crawl.** Markdown source is cleaner than rendered
HTML (no nav chrome, no cookie banners, no duplicated sidebars), the LICENSE file
travels with the repo so the corpus licence is verifiable rather than asserted, and
``--filter=blob:none --sparse`` means the Dagster monorepo costs a few megabytes
instead of gigabytes. Most importantly the result is addressable by commit SHA:
"rebuild from this SHA" reproduces the exact corpus a metric came from, which
"I scraped the site in September" does not.

**Code fences are protected from tag stripping.** These are technical docs; the
answer to a question is frequently inside a code block, and code legitimately
contains ``<``, ``>`` and ``{``. Running an MDX/JSX tag stripper over a SQL snippet
containing ``WHERE a < b`` silently eats it. So the normaliser splits the document
on fences first and only rewrites prose segments -- see :func:`strip_markup`.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from .sources import Source

DOC_SUFFIXES = (".md", ".mdx")

_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_FRONTMATTER_TITLE = re.compile(r"^title:\s*(.+?)\s*$", re.MULTILINE)
_FENCE = re.compile(r"(^```.*?^```|^~~~.*?^~~~)", re.DOTALL | re.MULTILINE)
_MDX_IMPORT = re.compile(r"^(?:import|export)[ \t]+.*$\n?", re.MULTILINE)
_JSX_TAG = re.compile(r"</?[A-Za-z][\w.:-]*(?:\s[^<>]*?)?/?>")
_MDX_COMMENT = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)
# `[ \t]*` and not `\s*`: `\s` matches newlines, so a greedy `\s*` on a bare
# closing `:::` swallowed the blank line after it and then captured the *next*
# line as the directive title, silently relocating body text. Keep it on-line.
_DIRECTIVE = re.compile(r"^:::[a-zA-Z]*[ \t]*(.*?)[ \t]*$", re.MULTILINE)
_MAX_TAG_PASSES = 5
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.MULTILINE)
_BLANKS = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Document:
    """One normalised documentation page."""

    doc_id: str
    """``<tool>/<path-without-suffix>`` -- stable across ingests of the same SHA."""

    tool: str
    path: str
    title: str
    url: str
    text: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch(source: Source, repos_dir: Path, *, quiet: bool = True) -> Path:
    """Sparse-checkout ``source.doc_root`` at the pinned SHA. Idempotent."""
    dest = repos_dir / source.name
    dest.parent.mkdir(parents=True, exist_ok=True)

    def run(*args: str, cwd: Path | None = None) -> None:
        subprocess.run(
            args,
            cwd=cwd,
            check=True,
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.DEVNULL if quiet else None,
        )

    if not (dest / ".git").exists():
        run(
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--quiet",
            source.clone_url,
            str(dest),
        )
        run("git", "sparse-checkout", "init", "--cone", cwd=dest)

    run("git", "sparse-checkout", "set", source.doc_root, cwd=dest)
    # `checkout <sha>` needs the tree; a filtered clone fetches blobs lazily.
    run("git", "checkout", "--quiet", source.sha, cwd=dest)
    return dest


def iter_doc_files(source: Source, repo_dir: Path) -> Iterator[Path]:
    """Markdown files under the source's doc root, in a stable order."""
    root = repo_dir / source.doc_root
    if not root.exists():
        return
    excluded = set(source.exclude_dirs)
    for path in sorted(root.rglob("*")):
        if path.suffix not in DOC_SUFFIXES or not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts[:-1]
        if excluded.intersection(rel_parts):
            continue
        yield path


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------
def split_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(frontmatter, body)``. Frontmatter is empty when absent."""
    match = _FRONTMATTER.match(text)
    if not match:
        return "", text
    return match.group(1), text[match.end() :]


def frontmatter_title(frontmatter: str) -> str:
    match = _FRONTMATTER_TITLE.search(frontmatter)
    if not match:
        return ""
    return match.group(1).strip().strip("\"'")


def strip_markup(body: str) -> str:
    """Remove MDX/JSX scaffolding from prose, leaving code fences untouched.

    Splitting on fences first is the whole point: :data:`_JSX_TAG` would otherwise
    treat ``WHERE a <b> c`` or a C++ ``std::vector<int>`` as markup and delete it.
    Documentation about databases is full of both.
    """
    out: list[str] = []
    for i, segment in enumerate(_FENCE.split(body)):
        if i % 2:  # odd segments are the fences themselves
            out.append(segment)
            continue
        segment = _MDX_COMMENT.sub("", segment)
        segment = _MDX_IMPORT.sub("", segment)
        segment = _DIRECTIVE.sub(r"\1", segment)
        segment = _strip_tags(segment)
        out.append(segment)
    return _BLANKS.sub("\n\n", "".join(out)).strip()


def _strip_tags(segment: str) -> str:
    """Remove JSX tags, repeating until stable.

    One pass is not enough. `re.sub` scans left to right, and the dbt docs contain
    tags whose attributes embed a placeholder in angle brackets, e.g.
    ``<File name='models/<filename>.yml'>``. The outer tag cannot match, because
    ``[^<>]`` will not cross the inner ``<``; the inner placeholder is removed
    instead, leaving a now-matchable ``<File name='models/.yml'>`` behind for the
    next pass. A single pass leaves visible markup in 135 documents.
    """
    for _ in range(_MAX_TAG_PASSES):
        stripped = _JSX_TAG.sub("", segment)
        if stripped == segment:
            return stripped
        segment = stripped
    return segment


def first_heading(body: str) -> str:
    match = _HEADING.search(body)
    return match.group(2).strip() if match else ""


def _strip_suffix(path: str) -> str:
    for suffix in DOC_SUFFIXES:
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def build_url(source: Source, rel_path: str) -> str:
    path = rel_path
    if source.strip_prefix and path.startswith(source.strip_prefix):
        path = path[len(source.strip_prefix) :]
    path = _strip_suffix(path)
    if path.endswith("/index"):
        path = path[: -len("/index")]
    return source.url_template.format(path=path)


def parse_document(source: Source, repo_dir: Path, path: Path) -> Document | None:
    """Read one file into a :class:`Document`, or ``None`` if it has no prose."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    frontmatter, body = split_frontmatter(raw)
    text = strip_markup(body)
    if not text:
        return None

    rel = path.relative_to(repo_dir).as_posix()
    title = frontmatter_title(frontmatter) or first_heading(text) or path.stem

    doc_rel = rel
    if source.strip_prefix and doc_rel.startswith(source.strip_prefix):
        doc_rel = doc_rel[len(source.strip_prefix) :]
    doc_rel = _strip_suffix(doc_rel)

    return Document(
        doc_id=f"{source.name}/{doc_rel}",
        tool=source.name,
        path=rel,
        title=title,
        url=build_url(source, rel),
        text=text,
    )


def ingest_source(source: Source, repos_dir: Path) -> list[Document]:
    """Fetch and normalise one source. Network-bound."""
    repo_dir = fetch(source, repos_dir)
    docs = []
    for path in iter_doc_files(source, repo_dir):
        doc = parse_document(source, repo_dir, path)
        if doc is not None:
            docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def write_jsonl(docs: Iterable[Document], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for doc in docs:
            handle.write(doc.to_json() + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as handle:
        return [Document(**json.loads(line)) for line in handle if line.strip()]
