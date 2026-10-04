"""A harness that tests the "exact search is enough" claim instead of restating it.

`dense.py` argues that exact numpy search is fast enough at this corpus size.
That is a claim about a ratio -- exact search against everything else a query
pays for -- so this module measures the ratio: exact numpy against FAISS
approximate indexes on recall versus exact, search latency, build time and
serialised size. It builds the instrument and holds no results; every number is
produced by running it, and it writes only the JSON it is told to.

Two measurement traps shaped it. Index latency is timed around `search_vector`
on *pre-encoded* queries, because the query encoder dominates end-to-end
latency and would bury the thing being compared. And FAISS is imported lazily,
so the fast suite and every other subcommand run without it.

`PgVectorIndex` adds pgvector (HNSW and IVFFlat, inner product) behind the same
two methods, measured the same way plus a bare `SELECT 1` round trip, so the cost
of the database hop is visible next to the cost of the search. psycopg is imported
lazily too. Its rule (2026-10-04) is separate from the exact-search rule above.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .dense import DenseIndex, Embedder, l2_normalise

# Pre-registered by the owner on 2026-09-30, before any measurement: exact
# search stops being enough when exact-search p95 > 10% of the end-to-end p95.
EXACT_SHARE_LIMIT = 0.10
RULE = f"exact_p95 <= {EXACT_SHARE_LIMIT:.2f} * e2e_p95"
RULE_DATE = "2026-09-30"
E2E_ARM = "hybrid_score_weighted"
FAISS_KINDS = ("flat", "hnsw", "ivf", "ivfpq")
PGVECTOR_KINDS = ("pgvector-hnsw", "pgvector-ivfflat")
# Pre-registered by the owner on 2026-10-04, before any pgvector measurement: some
# configuration must reach this recall@10 against exact within the latency budget
# (EXACT_SHARE_LIMIT of the end-to-end p95). Exact search stays the default either way.
PGVECTOR_RECALL_FLOOR = 0.98
PGVECTOR_RULE = (
    f"some pgvector config: recall@10 vs exact >= {PGVECTOR_RECALL_FLOOR} "
    f"and p95 <= {EXACT_SHARE_LIMIT:.2f} * e2e_p95"
)
PGVECTOR_RULE_DATE = "2026-10-04"
# The rule's numbers are fixed, not re-derived from whatever a run happens to measure: the
# budget is 10% of the 164.5 ms end-to-end p95 recorded in eval/results/ann.json, over the
# 184 questions at k = 10. A run that drifts from any of these cannot carry the verdict.
PGVECTOR_E2E_P95_MS = 164.5
PGVECTOR_BUDGET_MS = 16.45
PGVECTOR_N_QUESTIONS = 184
PGVECTOR_K = 10
_TABLE_PREFIX = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")
PGVECTOR_DEFAULT_DSN = "postgresql://postgres@127.0.0.1:5433/postgres"
# A dead port can otherwise hang a connect for minutes (seen on Windows).
PGVECTOR_CONNECT_TIMEOUT_S = 10
SEED = 20260930
HNSW_EF_GRID = (16, 32, 64, 128, 256)
NPROBE_GRID = (1, 4, 16, 64)
_MISSING = "faiss is not installed. It lives in the optional `ann` extra: `uv sync --extra ann`."


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ModuleNotFoundError(_MISSING) from exc
    return faiss


_MISSING_PG = (
    "psycopg is not installed. It lives in the optional `pgvector` extra: "
    "`uv sync --extra pgvector`."
)


def _psycopg():
    try:
        import psycopg
    except ImportError as exc:
        raise ModuleNotFoundError(_MISSING_PG) from exc
    return psycopg


@runtime_checkable
class VectorIndex(Protocol):
    """What `Retriever` needs from a dense index: `.search` and a length."""

    def search(self, query: str, embedder: Embedder, k: int = 10) -> list[tuple[str, float]]: ...

    def search_vector(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]: ...

    def __len__(self) -> int: ...


def default_nlist(n: int) -> int:
    """~4*sqrt(n) lists, capped so k-means sees the 39 points per centroid it wants."""
    return max(1, min(round(4 * math.sqrt(n)), n // 39))


class FaissIndex:
    """FAISS index over unit vectors, scored by inner product (= cosine, as in `DenseIndex`)."""

    def __init__(self, chunk_ids: Sequence[str], vectors: np.ndarray, kind: str, **params) -> None:
        if kind not in FAISS_KINDS:
            raise ValueError(f"unknown index kind {kind!r}; expected one of {FAISS_KINDS}")
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors must be the same length")
        faiss = _faiss()
        data = np.ascontiguousarray(l2_normalise(np.asarray(vectors, dtype=np.float32)))
        n, dim = data.shape
        self.chunk_ids = list(chunk_ids)
        self.kind = kind
        self.dim = dim
        self.nlist = 0
        ip = faiss.METRIC_INNER_PRODUCT
        started = time.perf_counter()
        if kind == "flat":
            self.params: dict = {}
            self._index = faiss.IndexFlatIP(dim)
        elif kind == "hnsw":
            self.params = {
                "M": params.get("M", 32),
                "ef_construction": params.get("ef_construction", 200),
                "ef_search": params.get("ef_search", 64),
            }
            self._index = faiss.IndexHNSWFlat(dim, self.params["M"], ip)
            self._index.hnsw.efConstruction = self.params["ef_construction"]
            self._index.hnsw.efSearch = self.params["ef_search"]
        else:
            self.nlist = params.get("nlist") or default_nlist(n)
            self.params = {"nlist": self.nlist, "nprobe": min(params.get("nprobe", 8), self.nlist)}
            self._quantizer = faiss.IndexFlatIP(dim)  # kept alive: the index borrows it
            if kind == "ivf":
                self._index = faiss.IndexIVFFlat(self._quantizer, dim, self.nlist, ip)
            else:
                m, nbits = params.get("m", 48), params.get("nbits", 8)
                if dim % m:
                    raise ValueError(f"dim {dim} is not divisible by m={m}")
                self.params.update(m=m, nbits=nbits)
                self._index = faiss.IndexIVFPQ(self._quantizer, dim, self.nlist, m, nbits, ip)
            self._index.train(data)
            self._index.nprobe = self.params["nprobe"]
        self._index.add(data)
        self.build_s = time.perf_counter() - started

    def set_search_params(self, ef_search: int | None = None, nprobe: int | None = None) -> None:
        """Change query-time knobs without rebuilding."""
        if ef_search is not None:
            if self.kind != "hnsw":
                raise ValueError("ef_search only applies to hnsw")
            self.params["ef_search"] = self._index.hnsw.efSearch = int(ef_search)
        if nprobe is not None:
            if self.kind not in ("ivf", "ivfpq"):
                raise ValueError("nprobe only applies to ivf and ivfpq")
            self.params["nprobe"] = self._index.nprobe = min(int(nprobe), self.nlist)

    def search_vector(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        query = l2_normalise(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        scores, ids = self._index.search(np.ascontiguousarray(query), min(k, len(self)))
        return [
            (self.chunk_ids[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0
        ]

    def search(self, query: str, embedder: Embedder, k: int = 10) -> list[tuple[str, float]]:
        return self.search_vector(embedder.encode_query(query), k)

    def index_bytes(self) -> int:
        return len(_faiss().serialize_index(self._index))

    def __len__(self) -> int:
        return len(self.chunk_ids)


def vector_literal(vector: np.ndarray) -> str:
    """pgvector's text form. `.9g` is enough digits to round-trip a float32 exactly."""
    return "[" + ",".join(f"{x:.9g}" for x in np.asarray(vector, dtype=np.float32).tolist()) + "]"


class PgVectorIndex:
    """pgvector HNSW or IVFFlat over unit vectors, scored by inner product (= cosine).

    One scratch table per kind (named from `table_prefix`), dropped and rebuilt on
    construction, in the database the DSN names. A session advisory lock on that
    table name makes a second run against the same tables fail instead of
    silently rebuilding them underneath the first. Heap load is not part of
    `build_s`: that times `CREATE INDEX` alone, the part comparable to a FAISS
    build.

    The measured query must run the plan that was checked. So prepared statements
    are off (every execution is planned fresh, exactly as `EXPLAIN` plans it), the
    planner is told not to use a sequential scan or an explicit sort (either would
    let it answer exactly from the primary key and skip the index being measured),
    and the plan is re-checked every time a search knob changes, because the
    planner's cost for IVFFlat moves with `probes`. A configuration whose plan does
    not use the index raises instead of producing a row.
    """

    KINDS = ("hnsw", "ivfflat")
    MAINTENANCE_WORK_MEM = "512MB"
    _SEARCH = "SELECT pos, embedding <#> %s::vector AS d FROM {table} ORDER BY d LIMIT %s"

    def __init__(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        kind: str,
        dsn: str = PGVECTOR_DEFAULT_DSN,
        table_prefix: str = "ann_bench",
        **params,
    ) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown pgvector kind {kind!r}; expected one of {self.KINDS}")
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors must be the same length")
        if not _TABLE_PREFIX.match(table_prefix):
            raise ValueError(f"table_prefix {table_prefix!r} must match {_TABLE_PREFIX.pattern}")
        psycopg = _psycopg()
        data = l2_normalise(np.asarray(vectors, dtype=np.float32))
        self.chunk_ids = list(chunk_ids)
        self.kind = kind
        self.dim = data.shape[1]
        self.nlist = 0
        self.table = f"{table_prefix}_{kind}"
        self.index_name = f"{self.table}_idx"
        try:
            self._conn = psycopg.connect(
                dsn,
                autocommit=True,
                connect_timeout=PGVECTOR_CONNECT_TIMEOUT_S,
                prepare_threshold=None,
            )
        except psycopg.OperationalError as exc:
            raise OSError(f"cannot reach Postgres at the given DSN: {exc}") from exc
        self._cur = self._conn.cursor()
        try:
            self._setup(data, params)
        except psycopg.Error as exc:
            self.close()
            raise OSError(f"Postgres refused while building {self.table}: {exc}") from exc
        except BaseException:
            self.close()
            raise

    def _lock_key(self) -> int:
        digest = hashlib.sha1(self.table.encode()).digest()[:8]
        return int.from_bytes(digest, "big", signed=True)

    def _setup(self, data: np.ndarray, params: dict) -> None:
        cur = self._cur
        n, dim = data.shape
        cur.execute("SELECT pg_try_advisory_lock(%s)", (self._lock_key(),))
        if not cur.fetchone()[0]:
            raise RuntimeError(
                f"another run holds the lock on {self.table}; one sweep at a time, "
                "or use a different table_prefix"
            )
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"DROP TABLE IF EXISTS {self.table}")
        cur.execute(f"CREATE TABLE {self.table} (pos int PRIMARY KEY, embedding vector({dim}))")
        started = time.perf_counter()
        with cur.copy(f"COPY {self.table} (pos, embedding) FROM STDIN") as copy:
            for pos, row in enumerate(data):
                copy.write_row((pos, vector_literal(row)))
        self.load_s = time.perf_counter() - started
        # Parallel index builds would be unfair against FAISS's single thread.
        cur.execute("SELECT set_config('max_parallel_maintenance_workers', '0', false)")
        cur.execute(
            "SELECT set_config('maintenance_work_mem', %s, false)", (self.MAINTENANCE_WORK_MEM,)
        )
        if self.kind == "hnsw":
            self.params: dict = {
                "M": int(params.get("M", 32)),
                "ef_construction": int(params.get("ef_construction", 200)),
                "ef_search": int(params.get("ef_search", 64)),
            }
            options = f"m = {self.params['M']}, ef_construction = {self.params['ef_construction']}"
        else:
            self.nlist = int(params.get("nlist") or default_nlist(n))
            self.params = {"nlist": self.nlist, "nprobe": min(params.get("nprobe", 8), self.nlist)}
            options = f"lists = {self.nlist}"
        started = time.perf_counter()
        cur.execute(
            f"CREATE INDEX {self.index_name} ON {self.table} "
            f"USING {self.kind} (embedding vector_ip_ops) WITH ({options})"
        )
        self.build_s = time.perf_counter() - started
        cur.execute(f"ANALYZE {self.table}")
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET enable_sort = off")
        if self.kind == "hnsw":
            self.set_search_params(ef_search=self.params["ef_search"])
        else:
            self.set_search_params(nprobe=self.params["nprobe"])

    def set_search_params(self, ef_search: int | None = None, nprobe: int | None = None) -> None:
        """Change query-time knobs on this session without rebuilding, then re-check the plan."""
        if ef_search is not None:
            if self.kind != "hnsw":
                raise ValueError("ef_search only applies to hnsw")
            self.params["ef_search"] = int(ef_search)
            self._cur.execute(
                "SELECT set_config('hnsw.ef_search', %s, false)", (str(self.params["ef_search"]),)
            )
        if nprobe is not None:
            if self.kind != "ivfflat":
                raise ValueError("nprobe only applies to ivfflat")
            self.params["nprobe"] = min(int(nprobe), self.nlist)
            self._cur.execute(
                "SELECT set_config('ivfflat.probes', %s, false)", (str(self.params["nprobe"]),)
            )
        if not self.plan_uses_index():
            raise RuntimeError(
                f"the planner did not use {self.index_name} at {self.params}; "
                "refusing to time something else"
            )

    def search_vector(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        query = l2_normalise(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        self._cur.execute(
            self._SEARCH.format(table=self.table), (vector_literal(query), min(k, len(self)))
        )
        # `<#>` is the negative inner product, so negate it back into a similarity.
        return [(self.chunk_ids[pos], -float(d)) for pos, d in self._cur.fetchall()]

    def search(self, query: str, embedder: Embedder, k: int = 10) -> list[tuple[str, float]]:
        return self.search_vector(embedder.encode_query(query), k)

    def plan_uses_index(self) -> bool:
        probe = vector_literal(np.eye(self.dim, dtype=np.float32)[0])
        self._cur.execute("EXPLAIN " + self._SEARCH.format(table=self.table), (probe, 10))
        return self.index_name in "\n".join(row[0] for row in self._cur.fetchall())

    def select1_timings(self, count: int) -> list[float]:
        """Per-call ms of a bare `SELECT 1`: the round trip with no search in it."""
        self._cur.execute("SELECT 1")
        timings: list[float] = []
        for _ in range(count):
            start = time.perf_counter()
            self._cur.execute("SELECT 1")
            self._cur.fetchone()
            timings.append((time.perf_counter() - start) * 1000.0)
        return timings

    def _relation_bytes(self, name: str) -> int:
        self._cur.execute("SELECT pg_relation_size(%s::regclass)", (name,))
        return int(self._cur.fetchone()[0])

    def index_bytes(self) -> int:
        return self._relation_bytes(self.index_name)

    def table_bytes(self) -> int:
        return self._relation_bytes(self.table)

    def server_info(self) -> dict:
        """Versions plus every setting that can move recall or latency, read off the server."""
        cur = self._cur
        cur.execute("SHOW server_version")
        postgres = cur.fetchone()[0]
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        extension = cur.fetchone()[0]
        settings = {}
        for name in (
            "shared_buffers",
            "maintenance_work_mem",
            "max_parallel_maintenance_workers",
            "enable_seqscan",
            "enable_sort",
            "hnsw.iterative_scan",
            "ivfflat.iterative_scan",
            "hnsw.ef_search",
            "ivfflat.probes",
        ):
            cur.execute("SELECT current_setting(%s, true)", (name,))
            settings[name] = cur.fetchone()[0]
        return {
            "postgres": postgres,
            "pgvector": extension,
            "settings": settings,
            "prepared_statements": False,
        }

    def close(self) -> None:
        self._conn.close()  # also releases the advisory lock

    def __len__(self) -> int:
        return len(self.chunk_ids)


class MemoEmbedder:
    """In-process query memo, so N index configs encode each question once, not N times."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._memo: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dim(self) -> int:
        return self._inner.dim

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._inner.encode_documents(texts)

    def encode_query(self, text: str) -> np.ndarray:
        if text not in self._memo:
            self._memo[text] = self._inner.encode_query(text)
        return self._memo[text]


# -- pure helpers (no faiss) ------------------------------------------------
def ann_recall(
    approx: Sequence[Sequence[str]], exact: Sequence[Sequence[str]], k: int = 10
) -> float:
    """Mean over queries of |approx[:k] & exact[:k]| / k."""
    if len(approx) != len(exact) or not exact:
        raise ValueError("need the same, non-zero number of rankings on both sides")
    return sum(len(set(a[:k]) & set(e[:k])) / k for a, e in zip(approx, exact, strict=True)) / len(
        exact
    )


def _draw(count: int, dim: int, clusters: int, noise: float, seed: int, stream: int) -> np.ndarray:
    centres = l2_normalise(np.random.default_rng(seed).standard_normal((clusters, dim)))
    rng = np.random.default_rng([seed, stream])
    pick = rng.integers(0, clusters, size=count)
    jitter = rng.standard_normal((count, dim)).astype("f4") * (noise / math.sqrt(dim))
    return l2_normalise(centres[pick] + jitter).astype(np.float32)


def synthetic_vectors(
    n: int, dim: int, *, clusters: int = 64, noise: float = 0.35, seed: int = SEED
) -> np.ndarray:
    """Seeded, clustered, unit-norm float32 -- neighbour structure a real corpus also has."""
    return _draw(n, dim, clusters, noise, seed, stream=0)


def synthetic_queries(
    n_queries: int, dim: int, *, clusters: int = 64, noise: float = 0.35, seed: int = SEED
) -> np.ndarray:
    """Same cluster centres as `synthetic_vectors`, but fresh draws, so no query is a base row."""
    return _draw(n_queries, dim, clusters, noise, seed, stream=1)


def percentile(values: Sequence[float], q: float) -> float:
    """Same definition `evaluate_arm` uses for p95: sorted, index int(q * (n - 1))."""
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def exact_share(exact_p95_ms: float, e2e_p95_ms: float | None) -> tuple[float | None, bool | None]:
    if e2e_p95_ms is None:
        return None, None
    share = exact_p95_ms / e2e_p95_ms
    return share, share <= EXACT_SHARE_LIMIT


def time_searches(index: VectorIndex, query_vectors: np.ndarray, k: int, *, repeats: int = 3):
    """Per-call search latency in ms: one untimed warm-up pass, then `repeats` timed passes."""
    for vector in query_vectors:
        index.search_vector(vector, k)
    timings: list[float] = []
    for _ in range(repeats):
        for vector in query_vectors:
            start = time.perf_counter()
            index.search_vector(vector, k)
            timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def read_e2e_p95(path: Path) -> float:
    value = json.loads(Path(path).read_text(encoding="utf-8")).get("config", {}).get("e2e_p95_ms")
    if value is None:
        raise ValueError(f"{path} has no config.e2e_p95_ms")
    return float(value)


# -- the bench --------------------------------------------------------------
def _thread_env() -> dict:
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    return {name: os.environ.get(name) for name in names}


def _pq_m(dim: int) -> int:
    """48 when it divides `dim` (as at 384); otherwise the largest divisor below it."""
    return next(m for m in range(min(48, dim), 0, -1) if dim % m == 0)


def _configs(index, ef_grid: Sequence[int], nprobe_grid: Sequence[int]):
    if index.kind == "hnsw":
        return [{"ef_search": ef} for ef in ef_grid]
    if index.kind in ("ivf", "ivfpq", "ivfflat"):
        return [{"nprobe": p} for p in sorted({min(p, index.nlist) for p in nprobe_grid})]
    return [{}]


def _variants(exact, ids, vectors, kinds, dim, ef_grid, nprobe_grid, dsn=PGVECTOR_DEFAULT_DSN):
    """Exact first (it is the ground truth), then each index at each sweep setting.

    Lazy on purpose: one index object is re-tuned in place, so the caller must
    measure each yield before asking for the next. The last element is extra row
    fields (pgvector reports its table size and load time there).
    """
    yield "exact", exact, {}, None, exact.vectors.nbytes, {}
    for kind in kinds:
        if kind in PGVECTOR_KINDS:
            index = PgVectorIndex(ids, vectors, kind.removeprefix("pgvector-"), dsn=dsn)
            extra = {"table_bytes": index.table_bytes(), "load_s": index.load_s}
        else:
            index = FaissIndex(ids, vectors, kind, **({"m": _pq_m(dim)} if kind == "ivfpq" else {}))
            extra = {}
        try:
            size = index.index_bytes()
            for setting in _configs(index, ef_grid, nprobe_grid):
                index.set_search_params(**setting)
                yield kind, index, dict(index.params), index.build_s, size, extra
        finally:
            if hasattr(index, "close"):
                index.close()


def _score_e2e(retriever, memo, questions, by_doc, index, k: int) -> dict:
    from .evaluate import evaluate_arm

    dense, embedder = retriever.dense, retriever.embedder
    retriever.dense, retriever.embedder = index, memo
    try:
        result, _ = evaluate_arm(retriever, questions, by_doc, arm="dense", k=k)
    finally:
        retriever.dense, retriever.embedder = dense, embedder
    overall = result.overall
    return {
        "recall@5": overall.get("recall@5"),
        "ndcg@10": overall.get("ndcg@10"),
        "n_scored": result.n_scored,
    }


def _measure_e2e_p95_ms(retriever, questions, by_doc, k: int) -> float:
    """p95 of the live default arm on the exact index with the real, unmemoised encoder."""
    from .evaluate import evaluate_arm

    retriever.retrieve(questions[0].text, arm=E2E_ARM, k=k)  # warm-up
    result, _ = evaluate_arm(retriever, questions, by_doc, arm=E2E_ARM, k=k)
    return result.p95_latency_s * 1000.0


def run_bench(
    *,
    sizes: Sequence[int] = (20000,),
    dim: int = 384,
    n_queries: int = 200,
    kinds: Sequence[str] = FAISS_KINDS,
    k: int = 10,
    repeats: int = 3,
    seed: int = SEED,
    ef_grid: Sequence[int] = HNSW_EF_GRID,
    nprobe_grid: Sequence[int] = NPROBE_GRID,
    e2e_p95_ms: float | None = None,
    e2e_source: str | None = None,
    dsn: str = PGVECTOR_DEFAULT_DSN,
    retriever=None,
    embedder: Embedder | None = None,
    questions: Sequence | None = None,
    by_doc: dict | None = None,
) -> dict:
    """Synthetic by default; passing `retriever` (with the rest) selects real mode."""
    real = retriever is not None
    faiss = _faiss() if any(kind not in PGVECTOR_KINDS for kind in kinds) else None
    if faiss is not None:
        faiss.omp_set_num_threads(1)

    if real:
        exact = retriever.dense
        memo = MemoEmbedder(embedder)
        queries = np.stack([memo.encode_query(q.text) for q in questions])
        dim, n_queries = exact.vectors.shape[1], len(questions)
        datasets = [(exact.chunk_ids, exact.vectors, exact, queries)]
        if e2e_p95_ms is None:
            e2e_p95_ms = _measure_e2e_p95_ms(retriever, questions, by_doc, k)
            e2e_source = f"measured:{E2E_ARM}"
    else:
        queries = synthetic_queries(n_queries, dim, seed=seed)
        datasets = []
        for n in sizes:
            vectors = synthetic_vectors(n, dim, seed=seed)
            ids = [f"c{i}" for i in range(n)]
            datasets.append((ids, vectors, DenseIndex(ids, vectors), queries))

    rows: list[dict] = []
    verdicts: list[dict] = []
    pg_info: dict | None = None
    for ids, vectors, exact, queries in datasets:
        n = len(ids)
        truth = [[c for c, _ in exact.search_vector(q, k)] for q in queries]
        for kind, index, params, build_s, size, extra in _variants(
            exact, ids, vectors, kinds, dim, ef_grid, nprobe_grid, dsn
        ):
            found = [[c for c, _ in index.search_vector(q, k)] for q in queries]
            timings = time_searches(index, queries, k, repeats=repeats)
            row = {
                "n": n,
                "index": kind,
                "params": params,
                "build_s": build_s,  # null for exact: nothing is built
                "index_bytes": size,
                "p50_ms": percentile(timings, 0.50),
                "p95_ms": percentile(timings, 0.95),
                "recall_at_10_vs_exact": ann_recall(found, truth, k),
                **extra,
            }
            if kind in PGVECTOR_KINDS:
                # Same number of samples as the search timings, taken straight after them
                # and before the end-to-end scoring below.
                bare = index.select1_timings(len(timings))
                row["select1_p50_ms"] = percentile(bare, 0.50)
                row["select1_p95_ms"] = percentile(bare, 0.95)
                row["plan_uses_index"] = index.plan_uses_index()
                if not row["plan_uses_index"]:
                    raise RuntimeError(f"the plan changed mid-measurement at {params}")
                pg_info = pg_info or index.server_info()
            row["e2e"] = _score_e2e(retriever, memo, questions, by_doc, index, k) if real else None
            rows.append(row)
        exact_p95 = next(r["p95_ms"] for r in rows if r["n"] == n and r["index"] == "exact")
        share, enough = exact_share(exact_p95, e2e_p95_ms)
        verdicts.append(
            {
                "n": n,
                "exact_p95_ms": exact_p95,
                "e2e_p95_ms": e2e_p95_ms,
                "exact_share": share,
                "exact_enough": enough,
            }
        )

    config = {
        "mode": "real" if real else "synthetic",
        "strategy": retriever.strategy if real else None,
        "dim": dim,
        "k": k,
        "seed": seed,
        "repeats": repeats,
        "n_queries": n_queries,
        "n": [len(d[0]) for d in datasets],
        "indexes": ["exact", *kinds],
        "sweep": {"hnsw_ef_search": list(ef_grid), "nprobe": list(nprobe_grid)},
        "thread_env": _thread_env(),
        "faiss_threads": 1 if faiss is not None else None,
        "versions": {"numpy": np.__version__, "faiss": getattr(faiss, "__version__", None)},
        "e2e_p95_ms": e2e_p95_ms,
        "e2e_source": e2e_source if e2e_p95_ms is not None else None,
        "rule": RULE,
        "rule_pre_registered": RULE_DATE,
    }
    result = {"config": config, "rows": rows, "verdicts": verdicts}
    if pg_info is not None:
        config["pgvector"] = pg_info
        config["pgvector_rule"] = PGVECTOR_RULE
        config["pgvector_rule_pre_registered"] = PGVECTOR_RULE_DATE
        default_grids = tuple(ef_grid) == HNSW_EF_GRID and tuple(nprobe_grid) == NPROBE_GRID
        result["pgvector_verdicts"] = [
            pgvector_verdict(
                rows, n, real=real, k=k, n_queries=n_queries, default_grids=default_grids
            )
            for n in config["n"]
        ]
    return result


def pgvector_verdict(
    rows: Sequence[dict],
    n: int,
    *,
    real: bool,
    k: int,
    n_queries: int,
    default_grids: bool,
    budget_ms: float = PGVECTOR_BUDGET_MS,
) -> dict:
    """Apply the pre-registered pgvector rule to the rows at one corpus size.

    The budget is the pinned 16.45 ms, never re-derived from this run's own
    end-to-end timing. The statement is deliberately narrower than "fits the app":
    the run times one process on one connection, so it says nothing about pooling,
    concurrency, writes or hosted Postgres. It is withheld, with the reasons
    recorded, unless the run matches the registered protocol: the real index, k = 10,
    all 184 questions, the registered sweep grids, and both pgvector kinds measured.
    """
    here = [r for r in rows if r["n"] == n and r["index"] in PGVECTOR_KINDS]
    passing = [
        {"index": r["index"], "params": r["params"]}
        for r in here
        if r["recall_at_10_vs_exact"] >= PGVECTOR_RECALL_FLOOR and r["p95_ms"] <= budget_ms
    ]
    withheld = []
    if not real:
        withheld.append("synthetic vectors, not the real index")
    if k != PGVECTOR_K:
        withheld.append(f"k={k}, registered k={PGVECTOR_K}")
    if n_queries != PGVECTOR_N_QUESTIONS:
        withheld.append(f"{n_queries} queries, registered {PGVECTOR_N_QUESTIONS}")
    if not default_grids:
        withheld.append("sweep grids differ from the registered ones")
    if {r["index"] for r in here} != set(PGVECTOR_KINDS):
        withheld.append("not both pgvector-hnsw and pgvector-ivfflat were measured")
    if budget_ms != PGVECTOR_BUDGET_MS:
        withheld.append(f"budget {budget_ms} ms, registered {PGVECTOR_BUDGET_MS} ms")
    if withheld:
        statement = None
    elif passing:
        statement = (
            "pgvector qualifies as a retrieval backend under the pre-registered "
            "current-app latency and recall budget."
        )
    else:
        statement = (
            "pgvector did not meet the pre-registered latency and recall budget "
            "at the current scale."
        )
    return {
        "n": n,
        "budget_p95_ms": budget_ms,
        "budget_source": f"pinned: {EXACT_SHARE_LIMIT:.2f} x {PGVECTOR_E2E_P95_MS} ms",
        "recall_floor": PGVECTOR_RECALL_FLOOR,
        "qualifies": bool(passing),
        "passing_configs": passing,
        "statement": statement,
        "statement_withheld": withheld,
    }


def format_table(result: dict) -> str:
    """A compact stdout view of `run_bench` output."""
    lines = [
        f"{'n':>7} {'index':6} {'params':34} {'build_s':>8} {'MB':>7} "
        f"{'p50ms':>8} {'p95ms':>8} {'recall':>7}"
    ]
    for r in result["rows"]:
        params = " ".join(f"{key}={value}" for key, value in r["params"].items())
        build = "-" if r["build_s"] is None else f"{r['build_s']:.2f}"
        line = (
            f"{r['n']:>7} {r['index']:6} {params:34} {build:>8} {r['index_bytes'] / 1e6:>7.1f} "
            f"{r['p50_ms']:>8.3f} {r['p95_ms']:>8.3f} {r['recall_at_10_vs_exact']:>7.3f}"
        )
        if r["e2e"]:
            line += f"  r@5 {r['e2e']['recall@5']:.3f} ndcg@10 {r['e2e']['ndcg@10']:.3f}"
        if "select1_p95_ms" in r:
            line += f"  select1 p95 {r['select1_p95_ms']:.3f}"
        lines.append(line)
    for v in result["verdicts"]:
        share = "n/a" if v["exact_share"] is None else f"{v['exact_share']:.3f}"
        lines.append(
            f"n={v['n']}: exact p95 {v['exact_p95_ms']:.3f} ms, "
            f"share {share}, enough {v['exact_enough']}"
        )
    for v in result.get("pgvector_verdicts", []):
        lines.append(
            f"n={v['n']}: pgvector budget p95 {v['budget_p95_ms']:.2f} ms at recall >= "
            f"{v['recall_floor']}, a config qualifies: {v['qualifies']}"
        )
        if v["statement"]:
            lines.append(f"  {v['statement']}")
        else:
            lines.append(f"  no verdict statement: {'; '.join(v['statement_withheld'])}")
    return "\n".join(lines)
