"""Threshold sweep for the semantic cache: hit rate on paraphrases against false hits.

A semantic cache is worth shipping only if some threshold serves real
paraphrases while almost never serving a near-miss question. This module builds
the instrument; it holds no results, and the rule below was fixed before any
similarity was computed. The rule itself and its definitions live in NOTES.md
("Semantic cache: pre-registered rule and frozen near-miss set"); the constants
here implement it and do not restate it.

True pairs come from the committed rewriter output, read straight from the
replay bundle in memory (never through `JsonCache`, never writing `.cache/`).
Similarity is the dot product of unit query vectors on both sides.

**eval/near_miss.jsonl** (frozen, committed), one JSON object per line::

    {"id": "nm-001", "anchor": "<a question whose answer is in the cache, ideally
     one from eval/questions.jsonl>", "near_miss": "<a question that reads almost
     the same but needs a different answer>", "why": "<one line: what differs, e.g.
     different tool, different function, negation, different version>",
     "kind": "<optional free-text tag>"}

``id`` is a unique non-empty string; ``anchor``, ``near_miss`` and ``why`` are
required non-empty strings; anchor and near_miss must differ after
whitespace/case normalisation; ``kind`` is optional.

**eval/paraphrase_checks.jsonl** (the owner's hand-work, not yet written)::

    {"qid": "...", "question": "<original>", "variant": "<rewriter variant>",
     "same_intent": true | false | null, "note": ""}

``same_intent`` false excludes that pair from the true pairs; null means not
checked yet and is ignored. A row whose (question, variant) is not a true pair
raises, so a typo cannot silently do nothing. For a verdict that counts, the file
must hold exactly the ``SPOT_CHECK_N`` rows that
``write_spot_check_sample(pairs, SPOT_CHECK_N, path)`` writes with the default
``SEED``, each decided true or false; any other checks file makes the run a
diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .bm25 import tokenize
from .dense import DEFAULT_MODEL, l2_normalise
from .groundtruth import QUESTIONS_PATH
from .rewrite import REWRITE_SYSTEM, REWRITE_TEMPLATE, _parse_variants
from .semcache import top1

NEAR_MISS_PATH = Path("eval/near_miss.jsonl")
PARAPHRASE_CHECKS_PATH = Path("eval/paraphrase_checks.jsonl")
REWRITE_BUNDLE = Path("eval/cache/llm.jsonl")  # cli.LLM_BUNDLE; a test keeps them equal
REWRITE_N = 2  # Retriever.retrieve's rewrite_n default; a test keeps them equal
THRESHOLDS = tuple(round(0.70 + 0.01 * i, 2) for i in range(30))
SEED = 20261004
SPOT_CHECK_N = 30
NEAR_MISS_SHA256 = "43a5189c68b27d5d3d0de741b17db12f5c1efc3574dbf95f0d266fa03b54fbe7"
EMBEDDER_NAME = DEFAULT_MODEL  # a test keeps this equal to dense.DEFAULT_MODEL

# From the approved brief of 2026-09-30 and fixed before any measurement.
FALSE_HIT_BAR = 0.02
MIN_HIT_RATE = 0.10
RULE = (
    "recommend the lowest threshold with false_hit_rate <= 0.02 and wrong_entry_rate <= 0.02 "
    "and hit_rate >= 0.10; otherwise recommend nothing"
)
# Owner decision 2026-09-30: a hit that serves a different cached question's answer counts as a
# false hit, and each rate is checked on its own denominator so paraphrase volume cannot dilute
# near-miss failures.
RULE_DATE = "2026-09-30"


class TruePair(NamedTuple):
    qid: str
    question: str
    variant: str
    answered_by: str


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson_upper(k: int, n: int, z: float = 1.96) -> float | None:
    """Wilson score upper bound: at n = 40, zero observed is not a zero rate."""
    if n == 0:
        return None
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + margin) / (1 + z * z / n)


# -- loading ------------------------------------------------------------------
def load_true_pairs(bundle: Path, questions: Sequence) -> tuple[list[TruePair], dict]:
    """(original, variant) pairs from the committed rewrites, plus what was skipped."""
    entries: dict[str, dict] = {}
    with Path(bundle).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["key"].get("system") == REWRITE_SYSTEM:
                entries.setdefault(entry["key"]["prompt"], entry)
    pairs: list[TruePair] = []
    stats = {"questions": len(questions), "matched": 0, "unmatched": 0, "parse_failures": 0}
    stats["identical_dropped"] = 0
    answered_by: Counter = Counter()
    for q in questions:
        entry = entries.get(REWRITE_TEMPLATE.format(n=REWRITE_N, query=q.text))
        if entry is None:
            stats["unmatched"] += 1
            continue
        stats["matched"] += 1
        try:
            variants = _parse_variants(entry["value"]["text"], REWRITE_N)
        except (ValueError, KeyError, TypeError):
            stats["parse_failures"] += 1
            continue
        for variant in variants:
            if _norm(variant) == _norm(q.text):  # would hit at 1.0 and inflate the hit rate
                stats["identical_dropped"] += 1
                continue
            model = entry["value"].get("model", "")
            pairs.append(TruePair(q.qid, q.text, variant, model))
            answered_by[model] += 1
    stats["pairs"] = len(pairs)
    stats["answered_by"] = answered_by
    return pairs, stats


def _read_rows(path: Path):
    """Yield (line number, parsed object) for each non-blank JSONL line, or raise by line."""
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: bad JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield number, row


def load_near_misses(path: Path, questions: Sequence) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: the frozen file is missing and must be restored from git, never regenerated"
        )
    texts = {q.text for q in questions}
    corpus = {_norm(t) for t in texts}
    rows: list[dict] = []
    ids: set[str] = set()
    anchors: set[str] = set()
    for number, row in _read_rows(path):
        where = f"{path}:{number}"
        for field in ("id", "anchor", "near_miss", "why"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{where}: missing or empty {field!r}")
        if row["id"] in ids:
            raise ValueError(f"{where}: duplicate id {row['id']!r}")
        if _norm(row["anchor"]) == _norm(row["near_miss"]):
            raise ValueError(f"{where}: anchor and near_miss are the same question")
        if row["anchor"] not in texts:
            raise ValueError(f"{where}: anchor is not a verbatim corpus question")
        if _norm(row["near_miss"]) in corpus:
            raise ValueError(f"{where}: near_miss is a corpus question")
        if row["anchor"] in anchors:
            raise ValueError(f"{where}: duplicate anchor")
        ids.add(row["id"])
        anchors.add(row["anchor"])
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def load_paraphrase_checks(path: Path, pairs: Sequence[TruePair]) -> dict:
    """Parse the owner's checks. `keys` are all rows; `excluded` those judged different intent."""
    known = {(p.question, p.variant) for p in pairs}
    keys: set[tuple[str, str]] = set()
    excluded: set[tuple[str, str]] = set()
    undecided = 0
    for number, row in _read_rows(path):
        where = f"{path}:{number}"
        key = (row.get("question"), row.get("variant"))
        if key not in known:
            raise ValueError(f"{where}: (question, variant) is not a true pair")
        if key in keys:
            raise ValueError(f"{where}: duplicate check row")
        verdict = row.get("same_intent", "missing")
        if verdict not in (True, False, None):
            raise ValueError(f"{where}: same_intent must be true, false or null")
        keys.add(key)
        if verdict is None:
            undecided += 1
        elif verdict is False:
            excluded.add(key)
    return {"keys": keys, "excluded": excluded, "undecided": undecided}


def sample_pairs(pairs: Sequence[TruePair], n: int, seed: int = SEED) -> list[TruePair]:
    if n > len(pairs):
        raise ValueError(f"cannot sample {n} pairs from {len(pairs)}")
    return random.Random(seed).sample(list(pairs), n)


def write_spot_check_sample(pairs: Sequence[TruePair], n: int, path: Path, *, seed: int = SEED):
    """Write the checks template with same_intent null; never overwrites a filled file."""
    sample = sample_pairs(pairs, n, seed)
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        for p in sample:
            row = {
                "qid": p.qid,
                "question": p.question,
                "variant": p.variant,
                "same_intent": None,
                "note": "",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sample


class HashEmbedder:
    """Hashed bag-of-tokens, for `--fake-embedder` smoke runs only. Its numbers mean nothing."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return f"hash-smoke-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def encode_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dim] += 1.0 if digest[4] % 2 else -1.0
        return l2_normalise(vector)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode_query(t) for t in texts])


# -- the sweep ----------------------------------------------------------------
def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def recommend(rows: Sequence[dict]) -> tuple[float | None, str]:
    """Apply RULE to the rows: the lowest qualifying threshold, or nothing and why."""
    failed = Counter()
    for row in rows:
        ok = {
            "false_hit_rate": row["false_hit_rate"] is not None
            and row["false_hit_rate"] <= FALSE_HIT_BAR,
            "wrong_entry_rate": row["wrong_entry_rate"] is not None
            and row["wrong_entry_rate"] <= FALSE_HIT_BAR,
            "hit_rate": row["hit_rate"] is not None and row["hit_rate"] >= MIN_HIT_RATE,
        }
        if all(ok.values()):
            return row["threshold"], "lowest threshold meeting all three conditions"
        failed.update(name for name, passed in ok.items() if not passed)
    detail = ", ".join(
        f"{name} fails at {failed[name]}/{len(rows)}"
        for name in ("false_hit_rate", "wrong_entry_rate", "hit_rate")
    )
    return None, f"no threshold meets all three conditions ({detail})"


def _failed_conditions(embedder, thresholds, smoke, inputs, checks, pairs) -> list[str]:
    paths = (inputs or {}).get("paths", {})
    failed = []
    if smoke:
        failed.append("smoke mode: a hashing embedder is not a measurement")
    near_miss = paths.get("near_miss")
    if near_miss is None or not Path(near_miss).exists() or _sha256(near_miss) != NEAR_MISS_SHA256:
        failed.append("near-miss file is not the frozen file (sha256 differs)")
    if tuple(thresholds) != THRESHOLDS:
        failed.append("threshold grid is not 0.70 to 0.99")
    if embedder.name != EMBEDDER_NAME:
        failed.append(f"embedder is not {EMBEDDER_NAME}")
    defaults = {"questions": QUESTIONS_PATH, "bundle": REWRITE_BUNDLE, "near_miss": NEAR_MISS_PATH}
    for key, default in defaults.items():
        if paths.get(key) != default:
            failed.append(f"--{key.replace('_', '-')} is not the default path")
    expected = (
        {(p.question, p.variant) for p in sample_pairs(pairs, SPOT_CHECK_N)}
        if (len(pairs) >= SPOT_CHECK_N)
        else None
    )
    if checks is None or checks["keys"] != expected or checks["undecided"]:
        failed.append("paraphrase checks are not the seeded sample with every row decided")
    return failed


def sweep(
    embedder,
    pairs: Sequence[TruePair],
    near_misses: Sequence[dict],
    *,
    questions: Sequence,
    thresholds: Sequence[float] = THRESHOLDS,
    smoke: bool = False,
    inputs: dict | None = None,
) -> dict:
    """Classify every pair at every threshold, scoring each item once.

    `pairs` are the true pairs before spot-check exclusion; `inputs` may hold
    `paths` (name -> Path), `pair_stats` and `checks` (from `load_paraphrase_checks`).
    """
    inputs = inputs or {}
    checks = inputs.get("checks")
    used = [p for p in pairs if not checks or (p.question, p.variant) not in checks["excluded"]]

    memo: dict[str, np.ndarray] = {}

    def encode(text: str) -> np.ndarray:
        if text not in memo:
            memo[text] = l2_normalise(np.asarray(embedder.encode_query(text), dtype=np.float32))
        return memo[text]

    entries: dict[str, object] = {}  # unique question text -> first Question, in file order
    for q in questions:
        entries.setdefault(q.text, q)
    texts = list(entries)
    index_of = {t: i for i, t in enumerate(texts)}
    matrix = np.stack([encode(t) for t in texts])
    docs = [{e.doc_id for e in entries[t].evidence} for t in texts]

    scored = []  # (top-1 index, score, own-original index) per used pair
    for p in used:
        best, score = top1(matrix, encode(p.variant))
        scored.append((best, score, index_of[p.question]))
    near = []
    for row in sorted(near_misses, key=lambda r: r["id"]):
        best, score = top1(matrix, encode(row["near_miss"]))
        near.append((row, best, score))

    rows = []
    for t in thresholds:
        hits = [(b, o) for b, s, o in scored if s >= t]
        correct = sum(b == o for b, o in hits)
        wrong = [(b, o) for b, o in hits if b != o]
        shared = sum(bool(docs[b] & docs[o]) for b, o in wrong)
        false_hits = sum(s >= t for _, _, s in near)
        rows.append(
            {
                "threshold": t,
                "n_true": len(used),
                "correct_hits": correct,
                "wrong_entry_hits": len(wrong),
                "hit_rate": _rate(correct, len(used)),
                "wrong_entry_rate": _rate(len(wrong), len(used)),
                "wrong_entry_upper95": wilson_upper(len(wrong), len(used)),
                "wrong_entry_shared_evidence": shared,
                "n_near_miss": len(near),
                "false_hits": false_hits,
                "false_hit_rate": _rate(false_hits, len(near)),
                "false_hit_upper95": wilson_upper(false_hits, len(near)),
            }
        )
    items = [
        {
            "id": row["id"],
            "top1_qid": entries[texts[best]].qid,
            "score": score,
            "top1_is_anchor": texts[best] == row["anchor"],
        }
        for row, best, score in near
    ]

    failed = _failed_conditions(embedder, thresholds, smoke, inputs, checks, pairs)
    if failed:
        recommended, reason = None, "diagnostic run, no recommendation: " + "; ".join(failed)
    else:
        recommended, reason = recommend(rows)
    chosen = next((r for r in rows if r["threshold"] == recommended), None)
    verdicts = [
        {
            "rule": RULE,
            "rule_date": RULE_DATE,
            "false_hit_bar": FALSE_HIT_BAR,
            "min_hit_rate": MIN_HIT_RATE,
            "eligible": not failed,
            "recommended_threshold": recommended,
            "hit_rate": chosen and chosen["hit_rate"],
            "wrong_entry_rate": chosen and chosen["wrong_entry_rate"],
            "false_hit_rate": chosen and chosen["false_hit_rate"],
            "reason": reason,
        }
    ]

    paths = inputs.get("paths", {})
    shas = {k: _sha256(v) for k, v in paths.items() if v is not None and Path(v).exists()}
    config = {
        "mode": "smoke" if smoke else "real",
        "embedder": embedder.name,
        "dim": embedder.dim,
        "thresholds": list(thresholds),
        "rule": RULE,
        "rule_date": RULE_DATE,
        "false_hit_bar": FALSE_HIT_BAR,
        "min_hit_rate": MIN_HIT_RATE,
        "true_pairs": dict(inputs.get("pair_stats", {})),
        "spot_check": {
            "file_used": checks is not None,
            "checked": len(checks["keys"]) - checks["undecided"] if checks else 0,
            "excluded": len(checks["excluded"]) if checks else 0,
        },
        "n_true_before_exclusion": len(pairs),
        "n_near_miss": len(near),
        "n_cache_entries": len(texts),
        "inputs": {k: {"path": str(v), "sha256": shas.get(k)} for k, v in paths.items() if v},
        "near_miss_sha256": shas.get("near_miss"),
        "near_miss_sha256_matches_pinned": shas.get("near_miss") == NEAR_MISS_SHA256,
        "numpy": np.__version__,
        "seed": SEED,
    }
    return {"config": config, "rows": rows, "verdicts": verdicts, "items": items}


def format_table(result: dict) -> str:
    """A compact stdout view of `sweep` output."""

    def num(value, spec=".3f"):
        return "-" if value is None else format(value, spec)

    lines = [
        f"{'thr':>5} {'n_true':>6} {'hit':>6} {'wrong':>6} {'wr_up95':>7} {'shrEv':>5} "
        f"{'n_nm':>4} {'false':>5} {'fh_rate':>7} {'fh_up95':>7}"
    ]
    for r in result["rows"]:
        lines.append(
            f"{r['threshold']:>5.2f} {r['n_true']:>6} {num(r['hit_rate']):>6} "
            f"{num(r['wrong_entry_rate']):>6} {num(r['wrong_entry_upper95']):>7} "
            f"{r['wrong_entry_shared_evidence']:>5} {r['n_near_miss']:>4} "
            f"{r['false_hits']:>5} {num(r['false_hit_rate']):>7} "
            f"{num(r['false_hit_upper95']):>7}"
        )
    v = result["verdicts"][0]
    lines.append(
        f"eligible={v['eligible']} recommended={v['recommended_threshold']}  {v['reason']}"
    )
    return "\n".join(lines)
