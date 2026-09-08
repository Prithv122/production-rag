"""Okapi BM25 over a scipy sparse matrix.

Written out rather than pulled from `rank_bm25` for two reasons. The portfolio
rule is that at least one core component of a flagship project has to be legible
as the author's own work, and -- more practically -- `rank_bm25` scores by
looping in Python over every document for every query, which is unusably slow
for the chunking x retrieval-arm grid this project runs. Precomputing the whole
weight matrix turns a query into one sparse column gather plus a row sum.

The scoring function is standard Okapi BM25::

    score(q, d) = sum_t idf(t) * ( f(t,d) * (k1 + 1) )
                             / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )

    idf(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )

Because the document-dependent part of each term's weight does not involve the
query at all, `build` evaluates it once for every (chunk, term) pair and stores
the result. Query time is then a slice, not an evaluation.

**No stopword list, deliberately.** The obvious move is to strip `in`, `as`,
`is`, `all`, `having`, `order`, `by` -- and every one of those is a SQL keyword
that carries real meaning in this corpus. "order by" is a query someone actually
types. IDF already discounts terms that appear everywhere, and it does so from
the corpus rather than from a generic English list, so a hand-written stoplist
here would only destroy signal.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp

K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+(?:[_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase, then split -- keeping identifiers whole *and* in pieces.

    Technical documentation is full of identifiers like ``on_schema_change``,
    ``read_parquet`` and ``dg.asset``. Splitting them destroys exact-match
    recall, which is precisely what BM25 is for; keeping them whole means the
    query "schema change" misses a document that only ever writes
    ``on_schema_change``. So emit both: the identifier and its parts.
    """
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        token = match.group(0)
        tokens.append(token)
        if "_" in token or "." in token:
            tokens.extend(part for part in re.split(r"[_.]", token) if part)
    return tokens


class BM25Index:
    """A prebuilt BM25 weight matrix over a fixed set of chunks."""

    def __init__(
        self,
        chunk_ids: Sequence[str],
        weights: sp.csc_matrix,
        vocabulary: dict[str, int],
    ) -> None:
        self.chunk_ids = list(chunk_ids)
        self.weights = weights
        """`(n_chunks, n_terms)` CSC. Column `t` holds every chunk's BM25 weight
        for term `t`, which is why scoring is a column gather."""

        self.vocabulary = vocabulary

    # -- build ------------------------------------------------------------
    @classmethod
    def build(
        cls,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        *,
        k1: float = K1,
        b: float = B,
    ) -> BM25Index:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        if not chunk_ids:
            raise ValueError("cannot build an index over zero chunks")

        vocabulary: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        counts: list[int] = []
        doc_lengths = np.zeros(len(texts), dtype=np.float64)

        for row, text in enumerate(texts):
            local: dict[int, int] = {}
            length = 0
            for token in tokenize(text):
                term = vocabulary.setdefault(token, len(vocabulary))
                local[term] = local.get(term, 0) + 1
                length += 1
            doc_lengths[row] = length
            for term, count in local.items():
                rows.append(row)
                cols.append(term)
                counts.append(count)

        n_docs = len(texts)
        n_terms = len(vocabulary)
        tf = sp.csr_matrix(
            (np.asarray(counts, dtype=np.float64), (rows, cols)),
            shape=(n_docs, n_terms),
        )

        # Document frequency: how many chunks contain each term at all.
        df = np.diff(tf.tocsc().indptr).astype(np.float64)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        avgdl = doc_lengths.mean() or 1.0
        # Per-row denominator constant: k1 * (1 - b + b * |d| / avgdl)
        norm = k1 * (1.0 - b + b * doc_lengths / avgdl)

        weights = tf.tocoo()
        freqs = weights.data
        weights.data = (freqs * (k1 + 1.0)) / (freqs + norm[weights.row])
        weights.data *= idf[weights.col]

        return cls(chunk_ids, weights.tocsc().astype(np.float32), vocabulary)

    # -- query ------------------------------------------------------------
    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every chunk against `query`. Unknown terms contribute 0."""
        columns = [self.vocabulary[token] for token in tokenize(query) if token in self.vocabulary]
        total = np.zeros(len(self.chunk_ids), dtype=np.float32)
        if not columns:
            return total
        # Summing selected columns of a CSC matrix touches only their nonzeros.
        selected = self.weights[:, columns]
        return np.asarray(selected.sum(axis=1)).ravel()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top `k` `(chunk_id, score)` pairs, best first. Zero scores are dropped."""
        scores = self.scores(query)
        if k <= 0:
            return []
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0.0]

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path / "weights.npz", self.weights)
        (path / "meta.json").write_text(
            json.dumps({"chunk_ids": self.chunk_ids, "vocabulary": self.vocabulary}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        weights = sp.load_npz(path / "weights.npz").tocsc()
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        return cls(meta["chunk_ids"], weights, meta["vocabulary"])

    def __len__(self) -> int:
        return len(self.chunk_ids)


def build_from_chunks(chunks: Iterable, *, with_breadcrumb: bool = False) -> BM25Index:
    """Convenience builder that respects a strategy's `embed_text` choice."""
    materialised = list(chunks)
    return BM25Index.build(
        [c.chunk_id for c in materialised],
        [c.embed_text(with_breadcrumb=with_breadcrumb) for c in materialised],
    )
