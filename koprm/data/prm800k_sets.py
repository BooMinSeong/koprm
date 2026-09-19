"""§4.3 / §4.4 PRM800K sets: the audit held-out and the A pool.

`parse_phase2` gives one row per rated pre-generated solution. Here we

  1. attach the outcome y by verifying `pre_generated_answer` against `answer`
     (no boxed answer at all -> y=0),
  2. drop every row whose problem appears in dev or MATH500 (no leakage into the
     audit or into arm A),
  3. carve out the audit held-out set (§4.4): at most one wrong (finish_reason =
     found_error AND y=0) and one correct (solution AND y=1) solution per problem,
     drawn from problems in a seeded random order until 500 wrong and up to 500
     correct are collected, and
  4. keep every remaining row on a problem that is *not* in the held-out set as the
     A pool (§4.3).

Rows carry the parsed fields plus `outcome` and `problem_id` (a stable hash of the
problem text; phase 2 has no problem id of its own and everything downstream -- the
§4.5 schema, the per-problem caps in arm A -- is keyed by problem).

Note on found_error rows with y=1: the labeler found a wrong step but the solution's
final answer still matches the gold one. Those rows are kept (the human labels are
what arm A trains on); they are simply not eligible as audit "wrong" solutions.

    python -m koprm.data.prm800k_sets --workers 64
"""
from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from koprm.data.prm800k import parse_phase2
from koprm.io import read_jsonl, write_jsonl
from koprm.paths import SPLITS, ensure_dirs
from koprm.verify import verify_many

SEED = 20260920
N_WRONG = 500
N_CORRECT = 500


def problem_id_of(problem_en: str) -> str:
    h = hashlib.sha1(problem_en.strip().encode("utf-8")).hexdigest()[:12]
    return f"prm800k/prob/{h}"


def attach_outcome(rows: list[dict], workers: int = 32, timeout: float = 5.0) -> list[dict]:
    """y = 1 iff the pre-generated final answer verifies against the gold answer."""
    todo = [i for i, r in enumerate(rows) if (r.get("pre_generated_answer") or "").strip()]
    for r in rows:
        r["outcome"] = 0
    if todo:
        res = verify_many(
            [rows[i]["answer"] for i in todo],
            ["\\boxed{" + rows[i]["pre_generated_answer"] + "}" for i in todo],
            workers=workers,
            timeout=timeout,
        )
        for i, (ok, _) in zip(todo, res):
            rows[i]["outcome"] = int(ok)
    return rows


def excluded_problem_texts(paths: list[str | Path]) -> set[str]:
    out: set[str] = set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for r in read_jsonl(p):
            if r.get("problem_en"):
                out.add(r["problem_en"].strip())
    return out


def split_sets(
    rows: list[dict],
    seed: int = SEED,
    n_wrong: int = N_WRONG,
    n_correct: int = N_CORRECT,
) -> tuple[list[dict], list[dict]]:
    """(audit, A pool). Rows must already carry `outcome` and `problem_id`."""
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_problem[r["problem_id"]].append(r)

    problems = sorted(by_problem)
    rng = random.Random(seed)
    rng.shuffle(problems)

    audit: list[dict] = []
    held_out: set[str] = set()
    n_w = n_c = 0
    for pid in problems:
        if n_w >= n_wrong and n_c >= n_correct:
            break
        group = sorted(by_problem[pid], key=lambda r: r["id"])
        picked = []
        if n_w < n_wrong:
            cands = [r for r in group if r["finish_reason"] == "found_error" and r["outcome"] == 0]
            if cands:
                picked.append(rng.choice(cands))
                n_w += 1
        if n_c < n_correct:
            cands = [r for r in group if r["finish_reason"] == "solution" and r["outcome"] == 1]
            if cands:
                picked.append(rng.choice(cands))
                n_c += 1
        if picked:
            held_out.add(pid)
            audit.extend(picked)

    pool = [r for r in rows if r["problem_id"] not in held_out]
    audit.sort(key=lambda r: r["id"])
    return audit, pool


def describe(name: str, rows: list[dict]) -> dict:
    y = Counter(r["outcome"] for r in rows)
    cross = Counter((r["finish_reason"], r["outcome"]) for r in rows)
    stats = {
        "rows": len(rows),
        "problems": len({r["problem_id"] for r in rows}),
        "y1": y[1],
        "y0": y[0],
        "crosstab": {f"{fr}|y={yy}": n for (fr, yy), n in sorted(cross.items())},
    }
    print(f"[{name}] rows={stats['rows']} problems={stats['problems']} "
          f"y=1 {stats['y1']} ({stats['y1'] / max(len(rows), 1):.3f}) y=0 {stats['y0']}")
    print(f"[{name}] finish_reason x y: " + "  ".join(
        f"{k}={v}" for k, v in stats["crosstab"].items()))
    return stats


def build(
    max_rows: int | None = None,
    seed: int = SEED,
    workers: int = 32,
    n_wrong: int = N_WRONG,
    n_correct: int = N_CORRECT,
    exclude_paths: list[str | Path] | None = None,
) -> dict:
    ensure_dirs()
    rows = parse_phase2(max_rows=max_rows)
    print(f"[prm800k] parsed {len(rows)} usable solutions")
    attach_outcome(rows, workers=workers)
    for r in rows:
        r["problem_id"] = problem_id_of(r["problem_en"])

    if exclude_paths is None:
        exclude_paths = [SPLITS / "dev.jsonl", SPLITS / "math500.jsonl"]
    bad = excluded_problem_texts(exclude_paths)
    kept = [r for r in rows if r["problem_en"].strip() not in bad]
    n_prob_before = len({r["problem_id"] for r in rows})
    n_prob_after = len({r["problem_id"] for r in kept})
    print(f"[prm800k] dropped {len(rows) - len(kept)} rows overlapping dev/math500 "
          f"({n_prob_before - n_prob_after} problems)")

    audit, pool = split_sets(kept, seed=seed, n_wrong=n_wrong, n_correct=n_correct)
    a_stats = describe("audit", audit)
    p_stats = describe("A_pool", pool)
    write_jsonl(SPLITS / "prm800k_audit.jsonl", audit)
    write_jsonl(SPLITS / "prm800k_A_pool.jsonl", pool)
    print(f"[prm800k] wrote {SPLITS / 'prm800k_audit.jsonl'} and {SPLITS / 'prm800k_A_pool.jsonl'}")
    return {"audit": a_stats, "A_pool": p_stats, "parsed": len(rows), "after_exclusion": len(kept)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None, help="debug: read only N source lines")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--n-wrong", type=int, default=N_WRONG)
    ap.add_argument("--n-correct", type=int, default=N_CORRECT)
    args = ap.parse_args()
    build(args.max_rows, args.seed, args.workers, args.n_wrong, args.n_correct)


if __name__ == "__main__":
    main()
