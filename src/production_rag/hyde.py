"""HyDE: search with an LLM-written hypothetical answer instead of (or beside) the question.

The protocol is registered in NOTES.md ("HyDE arm: registered protocol"); this module implements
it and changes none of it. Two steps, kept apart so the slow one runs once:

1. :func:`write_drafts` asks the local model for one short draft per question (cached on disk).
2. :func:`evaluate` re-runs the two baseline arms and the four HyDE arms on the committed
   ``heading`` indexes and scores them against the stored gold lists.

Only the dense query vector changes between a baseline and its HyDE arms. The lexical leg, fusion,
weights and pool are the live default's, so a difference can only come from the draft.
"""

from __future__ import annotations

import ast
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .agentic import paired_bootstrap, percentile
from .dense import l2_normalise
from .fuse import fuse
from .groundtruth import load_questions
from .metrics import ndcg_at_k, recall_at_k
from .pipeline import ARMS
from .providers import LLMProvider

#: The registered prompt. ``test_hyde.py`` compares it byte for byte with the frozen block in
#: NOTES.md, so editing either one breaks the build.
PROMPT = (
    "Write a short passage from technical documentation that answers the question below. "
    "Write it the way the documentation would, in two to four sentences, and include the exact "
    "names of any commands, functions or settings involved. Do not mention the question.\n"
    "\n"
    "Question: {question}"
)

MODEL_ARM = "ollama-qwen"
CACHE_DIR = Path(".cache/hyde-llm")
MAX_TOKENS = 128
#: Ollama context window. The default (32,768) pushed most of the model onto the CPU at about
#: two minutes a draft; a prompt plus a 128-token draft fits in 2,048 many times over.
NUM_CTX = 2048
STRATEGY = "heading"
K = 10
POOL = 50
BASELINE_HYBRID = "hybrid_score_weighted"
BOOTSTRAP_TRIALS = 20_000
BOOTSTRAP_SEED = 20261024
GAIN_BAR = 0.02
METRICS = ("ndcg@10", "recall@5")

#: (arm, leg, vector) -- vector is "query", "draft" or "avg".
HYDE_ARMS = (
    ("dense", "dense", "query"),
    ("dense_hyde", "dense", "draft"),
    ("dense_avg", "dense", "avg"),
    (BASELINE_HYBRID, "hybrid", "query"),
    ("hsw_hyde", "hybrid", "draft"),
    ("hsw_avg", "hybrid", "avg"),
)
BASELINE_OF = {
    "dense_hyde": "dense",
    "dense_avg": "dense",
    "hsw_hyde": BASELINE_HYBRID,
    "hsw_avg": BASELINE_HYBRID,
}


def render_prompt(question: str) -> str:
    # One-pass fill, so braces inside a question are never read as fields.
    return PROMPT.replace("{question}", question)


@dataclass
class Draft:
    qid: str
    text: str
    failed: bool
    latency_s: float
    completion_tokens: int
    cached: bool


def write_drafts(
    provider: LLMProvider, questions: Sequence[tuple[str, str]], *, log=print
) -> dict[str, Draft]:
    """One draft per (qid, question text). No retry; an empty reply is a failed draft."""
    drafts: dict[str, Draft] = {}
    for n, (qid, text) in enumerate(questions, start=1):
        try:
            r = provider.complete(render_prompt(text), temperature=0.0, max_tokens=MAX_TOKENS)
            body = (r.text or "").strip()
            drafts[qid] = Draft(qid, body, not body, r.latency_s, r.completion_tokens, r.cached)
        except Exception as exc:  # a failed call is a failed draft, counted, never retried
            drafts[qid] = Draft(qid, "", True, 0.0, 0, False)
            log(f"{qid}: draft failed ({exc.__class__.__name__})")
        if n % 10 == 0:
            log(f"drafts {n}/{len(questions)}")
    return drafts


def save_drafts(drafts: dict[str, Draft], path: Path, *, model: str) -> None:
    payload = {
        "model_arm": MODEL_ARM,
        "model": model,
        "max_tokens": MAX_TOKENS,
        "num_ctx": NUM_CTX,
        "prompt": PROMPT,
        "drafts": [asdict(d) for d in drafts.values()],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_drafts(path: Path) -> dict[str, Draft]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {d["qid"]: Draft(**d) for d in data["drafts"]}


# -- evaluation ------------------------------------------------------------


@dataclass(frozen=True)
class Scored:
    qid: str
    category: str
    gold: tuple[str, ...]


def scored_questions(retrieval_json: Path) -> list[Scored]:
    """The questions §5 scores on `heading`, with the gold lists exactly as stored."""
    rows = json.loads(retrieval_json.read_text(encoding="utf-8"))["rows"]
    out = []
    for r in rows:
        if r["arm"] != BASELINE_HYBRID or r["strategy"] != STRATEGY:
            continue
        gold = (
            tuple(ast.literal_eval(r["gold"])) if isinstance(r["gold"], str) else tuple(r["gold"])
        )
        if gold:
            out.append(Scored(r["qid"], r["category"], gold))
    return out


def stored_metrics(retrieval_json: Path, arm: str) -> dict[str, dict[str, float]]:
    rows = json.loads(retrieval_json.read_text(encoding="utf-8"))["rows"]
    out = {}
    for r in rows:
        if r["arm"] == arm and r["strategy"] == STRATEGY:
            m = ast.literal_eval(r["metrics"]) if isinstance(r["metrics"], str) else r["metrics"]
            out[r["qid"]] = m
    return out


def rank(retriever, question: str, vector: np.ndarray, leg: str) -> list[str]:
    dense = retriever.dense.search_vector(vector, k=POOL)
    if leg == "dense":
        return [c for c, _ in dense[:K]]
    spec = ARMS[BASELINE_HYBRID]
    lexical = retriever.bm25.search(question, k=POOL)
    fused = fuse([lexical, dense], spec.fusion, k=POOL, weights=list(spec.weights))
    return [c for c, _ in fused[:K]]


def evaluate(
    retriever, embedder, drafts: dict[str, Draft], scored: Sequence[Scored], texts: dict[str, str]
) -> dict[str, Any]:
    per: dict[str, dict[str, dict[str, float]]] = {a: {} for a, _, _ in HYDE_ARMS}
    ranked: dict[str, dict[str, list[str]]] = {a: {} for a, _, _ in HYDE_ARMS}
    encode_s: list[float] = []
    embedder.encode_query("warm-up")
    for s in scored:
        q = texts[s.qid]
        qv = l2_normalise(np.asarray(embedder.encode_query(q), dtype=np.float32).reshape(1, -1))[0]
        d = drafts.get(s.qid)
        if d is None or d.failed:
            dv = qv
        else:
            mark = time.perf_counter()
            dv = embedder.encode_documents([d.text])[0]
            encode_s.append(time.perf_counter() - mark)
            dv = l2_normalise(np.asarray(dv, dtype=np.float32).reshape(1, -1))[0]
        av = l2_normalise((qv + dv).reshape(1, -1))[0]
        vectors = {"query": qv, "draft": dv, "avg": av}
        for arm, leg, which in HYDE_ARMS:
            top = rank(retriever, q, vectors[which], leg)
            ranked[arm][s.qid] = top
            per[arm][s.qid] = {
                "ndcg@10": ndcg_at_k(top, s.gold, 10),
                "recall@5": recall_at_k(top, s.gold, 5),
            }
    return {"per_question": per, "ranked": ranked, "encode_s": encode_s}


def baseline_gate(per: dict, stored: dict[str, dict[str, dict]], arm: str) -> dict[str, Any]:
    mismatched = [
        qid
        for qid, m in per[arm].items()
        if any(abs(m[k] - stored[arm][qid][k]) > 1e-9 for k in METRICS)
    ]
    return {
        "arm": arm,
        "questions": len(per[arm]),
        "mismatched": mismatched,
        "passes": not mismatched,
    }


def decide(mean: float, lo: float, hi: float) -> str:
    if lo > 0 and mean >= GAIN_BAR:
        return "gain"
    if hi < 0:
        return "worse"
    return "no clear difference"


def compare(per: dict, scored: Sequence[Scored]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    qids = [s.qid for s in scored]
    cats = {s.qid: s.category for s in scored}
    for arm, base in BASELINE_OF.items():
        diffs = {m: [per[arm][q][m] - per[base][q][m] for q in qids] for m in METRICS}
        boot = paired_bootstrap(diffs, trials=BOOTSTRAP_TRIALS, seed=BOOTSTRAP_SEED)
        by_cat: dict[str, dict[str, float]] = {}
        for cat in sorted(set(cats.values())):
            sub = [q for q in qids if cats[q] == cat]
            by_cat[cat] = {"n": len(sub)} | {
                f"{m}_{side}": float(np.mean([per[side_arm][q][m] for q in sub]))
                for m in METRICS
                for side, side_arm in (("base", base), ("arm", arm))
            }
        means = {m: float(np.mean([per[arm][q][m] for q in qids])) for m in METRICS}
        out[arm] = {
            "baseline": base,
            "mean": means,
            "baseline_mean": {m: float(np.mean([per[base][q][m] for q in qids])) for m in METRICS},
            "difference": boot,
            "decision": decide(
                boot["ndcg@10"]["mean"], boot["ndcg@10"]["lo"], boot["ndcg@10"]["hi"]
            ),
            "by_category": by_cat,
        }
    return out


def failures(ranked: dict, scored: Sequence[Scored], drafts: dict[str, Draft]) -> dict[str, Any]:
    """Exploratory: gold chunks a HyDE arm dropped from the top 10 that the baseline had."""
    out: dict[str, Any] = {}
    for arm, base in BASELINE_OF.items():
        lost = []
        for s in scored:
            had = set(ranked[base][s.qid]) & set(s.gold)
            kept = set(ranked[arm][s.qid]) & set(s.gold)
            if had - kept:
                lost.append(
                    {
                        "qid": s.qid,
                        "category": s.category,
                        "draft": drafts[s.qid].text if s.qid in drafts else "",
                    }
                )
        counts: dict[str, int] = {}
        for item in lost:
            counts[item["category"]] = counts.get(item["category"], 0) + 1
        out[arm] = {"questions_losing_gold": len(lost), "by_category": counts, "examples": lost[:5]}
    return out


def cost(drafts: dict[str, Draft], encode_s: Sequence[float], scored_ids: set[str]) -> dict:
    used = [d for q, d in drafts.items() if q in scored_ids]
    ok = [d for d in used if not d.failed]
    lat = sorted(d.latency_s for d in ok)
    toks = sorted(d.completion_tokens for d in ok)
    enc = sorted(encode_s)
    return {
        "llm_calls_per_query": 1,
        "drafts": len(used),
        "failed_drafts": len(used) - len(ok),
        "draft_tokens_median": toks[len(toks) // 2] if toks else 0,
        "draft_latency_s_median": lat[len(lat) // 2] if lat else 0.0,
        "draft_latency_s_p95": percentile(lat, 0.95) if lat else 0.0,
        "draft_encode_ms_median": 1000 * enc[len(enc) // 2] if enc else 0.0,
        "cost_usd": 0.0,
    }


def question_texts() -> dict[str, str]:
    return {q.qid: q.text for q in load_questions()}


def run(
    *,
    indexes: Path,
    retrieval_json: Path,
    drafts_path: Path,
    out: Path,
    provider: LLMProvider,
    commit: str,
    phase: str = "all",
    replay: bool = False,
    log=print,
) -> dict[str, Any] | None:
    """Drafts (from the cache when present), then the registered evaluation, written once."""
    texts = question_texts()
    if replay:
        # A clean clone has no model cache; the committed drafts file is the replay source.
        drafts = load_drafts(drafts_path)
        missing = sorted(set(texts) - set(drafts))
        if missing:
            raise ValueError(f"{drafts_path} has no draft for {len(missing)} question(s)")
    else:
        drafts = write_drafts(provider, list(texts.items()), log=log)
        save_drafts(drafts, drafts_path, model=provider.model)
    if phase == "drafts":
        return None
    from .dense import SentenceTransformerEmbedder
    from .pipeline import Retriever

    embedder = SentenceTransformerEmbedder()
    retriever = Retriever.load(indexes, STRATEGY, embedder=embedder)
    scored = scored_questions(retrieval_json)
    res = evaluate(retriever, embedder, drafts, scored, texts)
    stored = {a: stored_metrics(retrieval_json, a) for a in ("dense", BASELINE_HYBRID)}
    gates = [baseline_gate(res["per_question"], stored, a) for a in stored]
    result = {
        "config": {
            "protocol": "NOTES.md, HyDE arm: registered protocol (2026-10-24)",
            "commit": commit,
            "model_arm": MODEL_ARM,
            "model": provider.model,
            "max_tokens": MAX_TOKENS,
            "num_ctx": NUM_CTX,
            "strategy": STRATEGY,
            "k": K,
            "pool": POOL,
            "bootstrap_trials": BOOTSTRAP_TRIALS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "gain_bar": GAIN_BAR,
            "n_scored": len(scored),
        },
        "baseline_gate": gates,
        "comparisons": compare(res["per_question"], scored)
        if all(g["passes"] for g in gates)
        else None,
        "cost": cost(drafts, res["encode_s"], {s.qid for s in scored}),
        "failures_exploratory": failures(res["ranked"], scored, drafts),
        "per_question": res["per_question"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def format_report(result: dict[str, Any]) -> str:
    lines = [
        "| Arm | vs | nDCG@10 | diff [95% CI] | recall@5 | diff [95% CI] | decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, c in (result["comparisons"] or {}).items():
        d = c["difference"]
        lines.append(
            f"| `{arm}` | `{c['baseline']}` | {c['mean']['ndcg@10']:.3f} "
            f"(base {c['baseline_mean']['ndcg@10']:.3f}) | {d['ndcg@10']['mean']:+.3f} "
            f"[{d['ndcg@10']['lo']:+.3f}, {d['ndcg@10']['hi']:+.3f}] | {c['mean']['recall@5']:.3f} "
            f"(base {c['baseline_mean']['recall@5']:.3f}) | {d['recall@5']['mean']:+.3f} "
            f"[{d['recall@5']['lo']:+.3f}, {d['recall@5']['hi']:+.3f}] | {c['decision']} |"
        )
    return "\n".join(lines)
