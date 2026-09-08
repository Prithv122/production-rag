"""Three chunking strategies, so the choice can be measured instead of guessed.

Chunking is the highest-leverage and least-examined decision in most RAG systems:
it fixes the upper bound on retrieval, because a chunk that does not contain the
answer cannot be retrieved no matter how good the ranker is. So this module
implements three strategies with a shared interface and lets the evaluation pick
the winner.

===============  ==========================================================
``fixed``        Fixed character window with overlap. Structure-blind; the
                 baseline everyone actually ships.
``heading``      One chunk per markdown section, packed up to the budget.
``heading_ctx``  ``heading`` plus an overlap tail and a heading breadcrumb
                 prefixed onto the embedded text.
===============  ==========================================================

Two rules hold across all three.

**Code fences are atomic.** These are technical docs; splitting a ``CREATE TABLE``
example across two chunks yields two chunks that each answer nothing. A fence is
only ever split when it alone exceeds the budget, and then on line boundaries.

**Sizing is in characters, not tokens.** A token budget would drag the embedding
model's tokenizer into the chunker, and with it torch -- into every test.
Characters are a stable, dependency-free proxy; the measured chars-per-token
ratio for this corpus is recorded in the README so the budget can be justified
against the encoder's real 512-token window rather than hand-waved.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from .ingest import Document

# Sized against bge-small-en-v1.5's 512-token window. See module docstring.
DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 200

_FENCE_BLOCK = re.compile(r"(^```.*?^```|^~~~.*?^~~~)", re.DOTALL | re.MULTILINE)
_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")

BlockKind = Literal["prose", "code"]


@dataclass(frozen=True)
class Block:
    """An atomic unit of a document: one paragraph, or one whole code fence."""

    kind: BlockKind
    text: str
    start: int
    """Character offset into `Document.text`, so citations can point at a span."""

    end: int
    heading_path: tuple[str, ...]
    """Enclosing headings, outermost first."""


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit."""

    chunk_id: str
    doc_id: str
    tool: str
    title: str
    url: str
    heading_path: tuple[str, ...]
    text: str
    """Raw text of the chunk, as it appears in the document."""

    start: int
    end: int

    @property
    def breadcrumb(self) -> str:
        return " > ".join((self.title, *self.heading_path))

    def embed_text(self, *, with_breadcrumb: bool) -> str:
        """What actually gets indexed.

        Prefixing the breadcrumb gives a chunk deep inside a page the context its
        own prose omits: a section that says only "Set this to `true`" is
        meaningless alone but retrievable as "dbt > Reference > on_schema_change".
        Whether that helps is an empirical question -- it is the difference
        between the ``heading`` and ``heading_ctx`` arms.
        """
        if not with_breadcrumb:
            return self.text
        return f"{self.breadcrumb}\n\n{self.text}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["heading_path"] = list(self.heading_path)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**{**data, "heading_path": tuple(data["heading_path"])})


# ---------------------------------------------------------------------------
# Document -> blocks
# ---------------------------------------------------------------------------
def split_blocks(text: str) -> list[Block]:
    """Split a document into atomic blocks, tracking the enclosing headings."""
    blocks: list[Block] = []
    heading_stack: list[tuple[int, str]] = []
    offset = 0

    for i, segment in enumerate(_FENCE_BLOCK.split(text)):
        if not segment:
            continue
        if i % 2:  # a fenced code block
            blocks.append(
                Block(
                    kind="code",
                    text=segment.strip(),
                    start=offset,
                    end=offset + len(segment),
                    heading_path=tuple(h for _, h in heading_stack),
                )
            )
            offset += len(segment)
            continue

        cursor = offset
        for para in _split_paragraphs(segment):
            raw, para_start = para
            stripped = raw.strip()
            cursor = offset + para_start
            if not stripped:
                continue
            heading = _HEADING_LINE.match(stripped)
            if heading and "\n" not in stripped:
                level = len(heading.group(1))
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading.group(2).strip()))
                continue
            blocks.append(
                Block(
                    kind="prose",
                    text=stripped,
                    start=cursor,
                    end=cursor + len(raw),
                    heading_path=tuple(h for _, h in heading_stack),
                )
            )
        offset += len(segment)

    return blocks


def _split_paragraphs(segment: str) -> list[tuple[str, int]]:
    """Blank-line-separated paragraphs, each with its offset within `segment`."""
    out: list[tuple[str, int]] = []
    pos = 0
    for part in re.split(r"(\n[ \t]*\n)", segment):
        if not part:
            continue
        if not re.fullmatch(r"\n[ \t]*\n", part):
            out.append((part, pos))
        pos += len(part)
    return out


# ---------------------------------------------------------------------------
# Blocks -> chunks
# ---------------------------------------------------------------------------
def _atoms(text: str, max_chars: int) -> list[str]:
    """Smallest units a block may be cut into: lines, or words within long lines.

    Line boundaries alone are not enough. Markdown routinely stores a whole
    paragraph on one physical line, and such a block is then unsplittable -- an
    early version emitted a single 9,999-character chunk against a 400-character
    budget, far past the encoder's window, and silently. So a line longer than
    the budget is cut again on whitespace.
    """
    units: list[str] = []
    for line in text.splitlines(keepends=True):
        if len(line) <= max_chars:
            units.append(line)
            continue
        buf: list[str] = []
        size = 0
        for word in re.split(r"(\s+)", line):
            if size + len(word) > max_chars and buf:
                units.append("".join(buf))
                buf, size = [], 0
            buf.append(word)
            size += len(word)
        if buf:
            units.append("".join(buf))
    return units


def _hard_split(block: Block, max_chars: int) -> list[Block]:
    """Split an oversized block, as a last resort. Never exceeds `max_chars`."""
    if len(block.text) <= max_chars:
        return [block]

    pieces: list[Block] = []
    buf: list[str] = []
    size = 0
    start = block.start

    def emit() -> None:
        nonlocal buf, size, start
        body = "".join(buf)
        pieces.append(
            Block(
                kind=block.kind,
                text=body.strip(),
                start=start,
                end=start + len(body),
                heading_path=block.heading_path,
            )
        )
        start += len(body)
        buf, size = [], 0

    for unit in _atoms(block.text, max_chars):
        if size + len(unit) > max_chars and buf:
            emit()
        buf.append(unit)
        size += len(unit)
    if buf:
        emit()
    return [p for p in pieces if p.text]


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """Last `overlap_chars` of a chunk, trimmed forward to a word boundary.

    A raw slice starts the next chunk mid-word ("n the query results..."), which
    is both unreadable in a citation and mild noise for the encoder. Advancing to
    the next whitespace costs at most one word of context.
    """
    if not overlap_chars:
        return ""
    tail = text[-overlap_chars:]
    if len(text) > overlap_chars:
        _, sep, rest = tail.partition(" ")
        if sep:
            tail = rest
    return tail.strip()


def _emit(doc: Document, group: Sequence[Block], index: int, tail: str = "") -> Chunk | None:
    if not group:
        return None
    body = "\n\n".join(b.text for b in group)
    if tail:
        body = f"{tail}\n\n{body}"
    if not body.strip():
        return None
    return Chunk(
        chunk_id=f"{doc.doc_id}#{index}",
        doc_id=doc.doc_id,
        tool=doc.tool,
        title=doc.title,
        url=doc.url,
        heading_path=group[0].heading_path,
        text=body,
        start=group[0].start,
        end=group[-1].end,
    )


def _pack(
    doc: Document,
    blocks: Iterable[Block],
    *,
    max_chars: int,
    overlap_chars: int,
    break_on_heading_change: bool,
) -> list[Chunk]:
    """Greedily pack blocks into chunks without splitting a block."""
    chunks: list[Chunk] = []
    group: list[Block] = []
    size = 0
    tail = ""

    def flush() -> None:
        nonlocal group, size, tail
        chunk = _emit(doc, group, len(chunks), tail)
        if chunk is not None:
            chunks.append(chunk)
            tail = _overlap_tail(chunk.text, overlap_chars)
        group = []
        # The overlap tail is prepended to the *next* chunk, so it has to be
        # charged against that chunk's budget. Leaving it out silently pushed
        # 49% of `fixed` chunks over `max_chars` and past the encoder's window.
        size = len(tail) + 2 if tail else 0

    for block in blocks:
        for piece in _hard_split(block, max_chars):
            heading_changed = (
                break_on_heading_change and group and piece.heading_path != group[-1].heading_path
            )
            if group and (heading_changed or size + len(piece.text) > max_chars):
                flush()
            group.append(piece)
            size += len(piece.text) + 2
    flush()
    return chunks


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def chunk_fixed(
    doc: Document,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Fixed window with overlap, blind to document structure."""
    return _pack(
        doc,
        split_blocks(doc.text),
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        break_on_heading_change=False,
    )


def chunk_heading(doc: Document, *, max_chars: int = DEFAULT_MAX_CHARS, **_: object) -> list[Chunk]:
    """One chunk per markdown section, packed up to the budget. No overlap."""
    return _pack(
        doc,
        split_blocks(doc.text),
        max_chars=max_chars,
        overlap_chars=0,
        break_on_heading_change=True,
    )


def chunk_heading_ctx(
    doc: Document,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Heading-aware, plus an overlap tail. Breadcrumbs are added at embed time."""
    return _pack(
        doc,
        split_blocks(doc.text),
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        break_on_heading_change=True,
    )


Chunker = Callable[..., list[Chunk]]

CHUNKERS: dict[str, Chunker] = {
    "fixed": chunk_fixed,
    "heading": chunk_heading,
    "heading_ctx": chunk_heading_ctx,
}

# Only `heading_ctx` prefixes the breadcrumb; that is the arm testing whether it
# is worth anything.
USES_BREADCRUMB = {"fixed": False, "heading": False, "heading_ctx": True}


def chunk_corpus(
    docs: Iterable[Document],
    strategy: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    try:
        chunker = CHUNKERS[strategy]
    except KeyError:
        raise ValueError(
            f"unknown chunking strategy {strategy!r}; expected one of {sorted(CHUNKERS)}"
        ) from None
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunker(doc, max_chars=max_chars, overlap_chars=overlap_chars))
    return out
