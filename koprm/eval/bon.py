"""§7 Best-of-N rescoring evaluator.

Two inputs:
  (a) a cached HF dataset of stored generations, e.g.
      "ENSEONG/ko-ko-math-500-test-EXAONE-4.0-1.2B-bon" (config "default",
      split "train"), columns problem / answer / id / completions (64 strings) /
      scores (the PRM step scores already stored for the 현행 baseline);
  (b) our own jsonl {problem_id, problem_ko, answer, completions}.

Steps are `completion.split("\\n\\n")` (empties dropped), the same convention the
komath harness uses.  Completions are rescored with a StudentScorer (or the
stored `scores` are reused with --use-existing-scores, or a saved score file is read
with --scores-from), aggregated per solution with last / min (§7) or the auxiliary
prod / mean, and BoN metrics are computed for
n = 1, 2, 4, ..., 64 over the *first n* completions - the komath
`subsample_completions` convention, which keeps the comparison at small n valid.

Answers come from koprm.verify (last brace-balanced \\boxed + math_verify with a
timeout); equal answers are grouped per problem against previously seen group
representatives, so weighted/majority voting works on equivalence classes rather
than raw strings. That grouping depends only on the dataset (answers and completions),
so it is cached under data/eval/cache/groups_<sha1>.json and reused by every scorer
evaluated on the same rows (--no-cache bypasses it).

    python -m koprm.eval.bon --dataset ENSEONG/ko-ko-math-500-test-EXAONE-4.0-1.2B-bon \\
        --use-existing-scores --limit 20 --out /tmp/bon.json

--agg both (the default) runs last and min off the one rescoring pass and writes
{"last": {...}, "min": {...}, "source": ..., "scorer": ...}; --agg all adds prod and mean in
that order; a single --agg last|min|prod|mean keeps the flat single-aggregation JSON.
--save-scores writes the per-step probabilities as jsonl ({problem_id, answer, scores}), and
--scores-from reads such a file back, so any aggregation can be recomputed without the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from koprm.data.sources import last_boxed
from koprm.io import load_jsonl, write_jsonl
from koprm.paths import EVAL, SYSTEM_PROMPTS
from koprm.verify import answers_equal

METHODS = ("naive", "weighted", "maj")


# ------------------------------------------------------------------ utilities
def split_steps(completion: str) -> list[str]:
    return [s for s in (x.strip() for x in completion.split("\n\n")) if s]


def aggregate(scores: list[float], agg: str = "last") -> float:
    if not scores:
        return 0.0
    if agg == "last":
        return float(scores[-1])
    if agg == "min":
        return float(min(scores))
    if agg == "prod":
        return float(np.prod(np.asarray(scores, dtype=np.float64)))
    if agg == "mean":
        return float(np.mean(np.asarray(scores, dtype=np.float64)))
    raise ValueError(f"unknown aggregation {agg!r} (use last|min|prod|mean)")


AGG_SETS = {"both": ["last", "min"], "all": ["last", "min", "prod", "mean"]}


def n_grid(n_max: int) -> list[int]:
    ns, k = [], 1
    while k <= n_max:
        ns.append(k)
        k *= 2
    return ns


class AnswerGroups:
    """Equivalence classes of answers for one problem, against the gold answer."""

    def __init__(self, gold: str, timeout: float = 3.0):
        self.gold = gold
        self.timeout = timeout
        self.reps: list[str] = []
        self.correct: list[bool] = []
        self._exact: dict[str, int] = {}

    def group_of(self, ans: str | None) -> int:
        key = (ans or "").strip()
        if key in self._exact:
            return self._exact[key]
        gid = None
        if key:
            for i, rep in enumerate(self.reps):
                if rep and _eq(rep, key, self.timeout):
                    gid = i
                    break
        if gid is None:
            gid = len(self.reps)
            self.reps.append(key)
            self.correct.append(bool(key) and _eq_gold(self.gold, key, self.timeout))
        self._exact[key] = gid
        return gid


_EQ_CACHE: dict[tuple[str, str], bool] = {}


def _eq(a: str, b: str, timeout: float) -> bool:
    key = (a, b)
    if key not in _EQ_CACHE:
        _EQ_CACHE[key] = bool(answers_equal(a, b, timeout=timeout))
    return _EQ_CACHE[key]


def _eq_gold(gold: str, pred: str, timeout: float) -> bool:
    return _eq(gold, pred, timeout)


# ------------------------------------------------------- answer grouping cache
# The grouping depends only on (problem_id, answer, completions), never on the scores, so
# every scorer evaluated on the same dataset can reuse it (it is the slow part: math_verify
# over 64 completions x 500 problems). Cache files are written atomically, and a file that
# cannot be read or does not match the rows is simply recomputed.
def groups_key(rows: list[dict]) -> str:
    h = hashlib.sha1()
    for r in rows:
        h.update(str(r.get("problem_id")).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(r["answer"]).encode("utf-8"))
        h.update(b"\x00")
        for c in r["completions"]:
            h.update(str(c).encode("utf-8"))
            h.update(b"\x01")
        h.update(b"\x02")
    return h.hexdigest()


def compute_groups(rows: list[dict], timeout: float = 3.0) -> list[dict]:
    """Per row: the answer group of each completion and whether each group is correct."""
    out = []
    for r in rows:
        g = AnswerGroups(str(r["answer"]), timeout=timeout)
        gids = [g.group_of(last_boxed(c)) for c in r["completions"]]
        out.append({"gids": gids, "correct": [bool(x) for x in g.correct]})
    return out


def _valid_groups(groups, rows: list[dict]) -> bool:
    if not isinstance(groups, list) or len(groups) != len(rows):
        return False
    for g, r in zip(groups, rows):
        if not isinstance(g, dict):
            return False
        gids, correct = g.get("gids"), g.get("correct")
        if not isinstance(gids, list) or not isinstance(correct, list):
            return False
        if len(gids) != len(r["completions"]) or not correct:
            return False
        if any(not isinstance(i, int) or i < 0 or i >= len(correct) for i in gids):
            return False
    return True


def load_groups(cache_dir: str | Path, key: str, rows: list[dict]) -> list[dict] | None:
    path = Path(cache_dir) / f"groups_{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            groups = json.load(f)
    except (OSError, json.JSONDecodeError):  # half-written or corrupt: recompute
        return None
    return groups if _valid_groups(groups, rows) else None


def save_groups(cache_dir: str | Path, key: str, groups: list[dict]) -> None:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"groups_{key}.json.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(groups, f)
        os.replace(tmp, d / f"groups_{key}.json")  # atomic for concurrent evaluations
    except OSError:
        tmp.unlink(missing_ok=True)


def answer_groups(rows: list[dict], timeout: float = 3.0,
                  cache_dir: str | Path | None = None, verbose: bool = True) -> list[dict]:
    if cache_dir is None:
        return compute_groups(rows, timeout)
    key = groups_key(rows)
    cached = load_groups(cache_dir, key, rows)
    if cached is not None:
        if verbose:
            print(f"[bon] answer groups from cache ({key[:12]})")
        return cached
    groups = compute_groups(rows, timeout)
    save_groups(cache_dir, key, groups)
    if verbose:
        print(f"[bon] answer groups computed and cached ({key[:12]})")
    return groups


# ---------------------------------------------------------------- data access
def load_hf_rows(name: str, config: str = "default", split: str = "train",
                 limit: int | None = None) -> list[dict]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for r in ds:
        rows.append({
            "problem_id": r.get("id") or r.get("problem_id"),
            "problem_ko": r.get("problem_ko") or r["problem"],
            "answer": r["answer"],
            "completions": list(r["completions"]),
            "scores": [list(s) for s in r["scores"]] if r.get("scores") is not None else None,
            "level": r.get("level"),
        })
    return rows


def load_jsonl_rows(path: str, limit: int | None = None) -> list[dict]:
    from koprm.io import read_jsonl

    rows = []
    for r in read_jsonl(path):
        rows.append({
            "problem_id": r.get("problem_id") or r.get("id"),
            "problem_ko": r.get("problem_ko") or r.get("problem"),
            "answer": r["answer"],
            "completions": list(r["completions"]),
            "scores": [list(s) for s in r["scores"]] if r.get("scores") is not None else None,
            "level": r.get("level"),
        })
        if limit and len(rows) >= limit:
            break
    return rows


# -------------------------------------------------------------------- scoring
def rescore(rows: list[dict], scorer, batch_size: int = 8) -> list[list[list[float]]]:
    """Per row, per completion, per step probabilities from a StudentScorer."""
    flat_problems, flat_steps, owners = [], [], []
    for i, r in enumerate(rows):
        for c in r["completions"]:
            flat_problems.append(r["problem_ko"])
            flat_steps.append(split_steps(c))
            owners.append(i)
    flat_scores = scorer.score(flat_problems, flat_steps, batch_size=batch_size)
    out: list[list[list[float]]] = [[] for _ in rows]
    for owner, s in zip(owners, flat_scores):
        out[owner].append(s)
    return out


def save_scores(path: str | Path, rows: list[dict],
                step_scores: list[list[list[float]]]) -> int:
    """One row per problem, so any aggregation can be recomputed without the model."""
    out = ({"problem_id": r.get("problem_id"), "answer": r["answer"],
            "scores": [[float(p) for p in c] for c in s]}
           for r, s in zip(rows, step_scores))
    n = write_jsonl(path, out)
    print(f"[bon] wrote step scores for {n} problems to {path}")
    return n


def load_scores(path: str | Path, rows: list[dict]) -> list[list[list[float]]]:
    """Read a --save-scores file back, checking it belongs to these rows."""
    saved = load_jsonl(path)
    if len(saved) != len(rows):
        raise SystemExit(f"--scores-from {path}: {len(saved)} rows, but the dataset has "
                         f"{len(rows)}")
    out = []
    for i, (r, sv) in enumerate(zip(rows, saved)):
        pid, spid = r.get("problem_id"), sv.get("problem_id")
        if pid is not None and spid is not None and pid != spid:
            raise SystemExit(f"--scores-from {path}: row {i} is {spid!r}, the dataset has "
                             f"{pid!r} (different order or dataset)")
        sc = sv.get("scores")
        if not isinstance(sc, list) or len(sc) != len(r["completions"]):
            raise SystemExit(f"--scores-from {path}: row {i} has "
                             f"{len(sc) if isinstance(sc, list) else 'no'} completions, the "
                             f"dataset has {len(r['completions'])}")
        out.append([[float(p) for p in c] for c in sc])
    return out


def existing_scores(rows: list[dict]) -> list[list[list[float]]]:
    out = []
    for r in rows:
        if r.get("scores") is None:
            raise SystemExit("--use-existing-scores needs a `scores` column")
        out.append([list(s) for s in r["scores"]])
    return out


# -------------------------------------------------------------------- metrics
def bootstrap_ci(per_problem_correct, iters: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """Paired problem-level bootstrap CI for a 0/1 per-problem array (§7)."""
    a = np.asarray(per_problem_correct, dtype=float)
    n = len(a)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = a[idx].mean(axis=1)
    return {
        "mean": float(a.mean()),
        "lo": float(np.percentile(means, 100 * alpha / 2)),
        "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": n,
    }


def evaluate(rows: list[dict], step_scores: list[list[list[float]]], agg: str = "last",
             timeout: float = 3.0, verbose: bool = True,
             cache_dir: str | Path | None = None) -> dict:
    """`cache_dir=None` (the default) touches no disk; `main` passes EVAL/"cache"."""
    n_max = min(len(r["completions"]) for r in rows)
    ns = n_grid(n_max)
    per_problem: dict[str, list[int]] = {f"{m}@{n}": [] for m in METHODS for n in ns}
    for n in ns:
        per_problem[f"pass@{n}"] = []
    pass1_per_problem: list[float] = []
    problem_ids: list = []
    groups = answer_groups(rows, timeout=timeout, cache_dir=cache_dir, verbose=verbose)

    for i, r in enumerate(rows):
        gids, correct = groups[i]["gids"], groups[i]["correct"]
        aggs = [aggregate(s, agg) for s in step_scores[i]]
        corr = [1 if correct[g] else 0 for g in gids]
        pass1_per_problem.append(float(np.mean(corr)))
        problem_ids.append(r["problem_id"])
        for n in ns:
            g_n, s_n, c_n = gids[:n], aggs[:n], corr[:n]
            # NAIVE: highest-scoring completion (first one wins ties)
            best = max(range(len(s_n)), key=lambda j: (s_n[j], -j))
            per_problem[f"naive@{n}"].append(c_n[best])
            # WEIGHTED / MAJ: sum of scores / counts per answer group
            wsum: dict[int, float] = {}
            cnt: dict[int, int] = {}
            for g, s in zip(g_n, s_n):
                wsum[g] = wsum.get(g, 0.0) + s
                cnt[g] = cnt.get(g, 0) + 1
            gw = max(wsum, key=lambda g: (wsum[g], -g_n.index(g)))
            gm = max(cnt, key=lambda g: (cnt[g], -g_n.index(g)))
            per_problem[f"weighted@{n}"].append(1 if correct[gw] else 0)
            per_problem[f"maj@{n}"].append(1 if correct[gm] else 0)
            per_problem[f"pass@{n}"].append(1 if any(c_n) else 0)
        if verbose and (i + 1) % 25 == 0:
            print(f"[bon] {i + 1}/{len(rows)} problems")

    metrics = {
        str(n): {
            "naive": float(np.mean(per_problem[f"naive@{n}"])),
            "weighted": float(np.mean(per_problem[f"weighted@{n}"])),
            "maj": float(np.mean(per_problem[f"maj@{n}"])),
            "pass": float(np.mean(per_problem[f"pass@{n}"])),
        }
        for n in ns
    }
    return {
        "n_problems": len(rows),
        "ns": ns,
        "agg": agg,
        "metrics": metrics,
        "pass@1_mean_correct": float(np.mean(pass1_per_problem)),
        "per_problem": {k: v for k, v in per_problem.items()},
        "per_problem_pass1": pass1_per_problem,
        "problem_ids": problem_ids,
        "ci": {f"{m}@{n}": bootstrap_ci(per_problem[f"{m}@{n}"]) for m in METHODS for n in ns},
    }


# ------------------------------------------------------------------------ CLI
def combine(results: dict[str, dict], source: str, scorer: str) -> dict:
    """Output JSON for one or several aggregations.

    A single aggregation keeps today's flat shape (dev evals already on disk still load);
    several are nested under their aggregation name, since rescoring is the expensive part
    and both §7 aggregations come from the same pass.
    """
    out = dict(next(iter(results.values()))) if len(results) == 1 else dict(results)
    out["source"] = source
    out["scorer"] = scorer
    return out


def print_metrics(res: dict, source: str, scorer: str) -> None:
    print(f"\n[bon] {source}  scorer={scorer}  agg={res['agg']}  problems={res['n_problems']}")
    print(f"{'n':>4} {'NAIVE':>8} {'WEIGHTED':>9} {'MAJ':>8} {'pass@n':>8}")
    for n in res["ns"]:
        m = res["metrics"][str(n)]
        print(f"{n:>4} {m['naive']:>8.3f} {m['weighted']:>9.3f} {m['maj']:>8.3f} "
              f"{m['pass']:>8.3f}")
    print(f"pass@1 (mean correctness) = {res['pass@1_mean_correct']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="cached HF dataset name")
    src.add_argument("--jsonl", help="our own generations jsonl")
    ap.add_argument("--dataset-config", default="default")
    ap.add_argument("--split", default="train")
    ap.add_argument("--scorer", default="existing",
                    help="checkpoint dir, or 'existing' to reuse the stored scores")
    ap.add_argument("--use-existing-scores", action="store_true")
    ap.add_argument("--agg", default="both",
                    choices=["both", "all", "last", "min", "prod", "mean"],
                    help="'both' = last+min (§7), 'all' = last+min+prod+mean; several "
                         "aggregations come out of the one scoring pass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--system-prompt", choices=["ko", "en"], default="ko",
                    help="the student's template; ko is what every student was trained with")
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument("--save-scores", default=None,
                    help="write the per-step probabilities as jsonl (re-aggregate later)")
    ap.add_argument("--scores-from", default=None,
                    help="read a --save-scores file instead of rescoring")
    ap.add_argument("--no-cache", action="store_true",
                    help=f"do not reuse the answer grouping cached under {EVAL / 'cache'}")
    args = ap.parse_args()

    rows = (load_hf_rows(args.dataset, args.dataset_config, args.split, args.limit)
            if args.dataset else load_jsonl_rows(args.jsonl, args.limit))
    print(f"[bon] {len(rows)} problems, {len(rows[0]['completions'])} completions each")

    use_existing = args.use_existing_scores or args.scorer == "existing"
    if args.scores_from:
        if args.scorer != "existing":
            ap.error("--scores-from and --scorer are mutually exclusive")
        scores = load_scores(args.scores_from, rows)
        scorer_name = f"scores-from:{Path(args.scores_from).name}"
    elif use_existing:
        scores = existing_scores(rows)
        scorer_name = "existing"
    else:
        from koprm.eval.scorer import StudentScorer

        scorer = StudentScorer(args.scorer, device=args.device, max_len=args.max_len,
                               system_prompt=SYSTEM_PROMPTS[args.system_prompt])
        scores = rescore(rows, scorer, batch_size=args.batch_size)
        scorer_name = args.scorer

    if args.save_scores:
        save_scores(args.save_scores, rows, scores)

    aggs = AGG_SETS.get(args.agg, [args.agg])
    cache_dir = None if args.no_cache else EVAL / "cache"
    results = {a: evaluate(rows, scores, agg=a, timeout=args.timeout, cache_dir=cache_dir)
               for a in aggs}
    source = args.dataset or args.jsonl
    out = combine(results, source, scorer_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for res in results.values():
        print_metrics(res, source, scorer_name)
    print(f"[bon] wrote {args.out}")


if __name__ == "__main__":
    main()
