"""§4.2 Selection: one correct and one wrong solution per mixed problem.

y is free, translation and the 72B teacher are not, so the cut happens here -- before
the expensive stages.

  * A solution is *eligible* only if it finished (not `truncated`) and produced a boxed
    answer (not `no_boxed`); a problem is "mixed" only counting eligible solutions,
    because an unusable solution cannot be the one we pick.
  * Mixed = at least one eligible correct (outcome=1) and one eligible wrong (outcome=0)
    solution across all generators. Only those problems can change a verifier's ranking.
  * Per mixed problem we take one correct and one wrong at random, balancing generators:
    the correct one comes from whichever generator has supplied the fewest correct
    solutions so far (so the two alternate), and the wrong one is preferred from a
    *different* generator, falling back to the same one when only it has a wrong
    solution.
  * Cap at --max-problems (6,000 = 12,000 solutions), filling MATH problems first and
    GSM8K with what is left, because the evaluation is MATH500.

Output: the chosen generation rows, unchanged, plus `arm: "B"`.

    python -m koprm.select --gen 'data/gen/*.scored.jsonl' \
        --problems data/splits/train_pool.jsonl --out data/gen/selected.jsonl
"""
from __future__ import annotations

import argparse
import glob as globlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from koprm.io import load_jsonl, write_jsonl

SEED = 20260920
MAX_PROBLEMS = 6000
SOURCE_ORDER = ["math", "gsm8k"]  # §4.2: MATH first, then GSM8K


def source_rank(source: str) -> int:
    return SOURCE_ORDER.index(source) if source in SOURCE_ORDER else len(SOURCE_ORDER)


def eligible(row: dict) -> bool:
    return (
        not row.get("truncated", False)
        and not row.get("no_boxed", False)
        and bool(row.get("steps"))
    )


def select_rows(
    gen_rows: list[dict],
    problem_meta: dict[str, dict] | None = None,
    max_problems: int = MAX_PROBLEMS,
    seed: int = SEED,
) -> tuple[list[dict], dict]:
    """Return (chosen rows with `arm`, stats). Pure function: no I/O."""
    problem_meta = problem_meta or {}
    gens = sorted({r["generator"] for r in gen_rows})
    # by_problem[pid][outcome][generator] -> rows
    by_problem: dict[str, dict[int, dict[str, list[dict]]]] = defaultdict(
        lambda: {0: defaultdict(list), 1: defaultdict(list)}
    )
    n_eligible = 0
    for r in gen_rows:
        if not eligible(r):
            continue
        n_eligible += 1
        by_problem[r["problem_id"]][int(r["outcome"])][r["generator"]].append(r)

    mixed = [p for p, d in by_problem.items() if d[0] and d[1]]
    mixed_by_source = Counter(problem_meta.get(p, {}).get("source", "unknown") for p in mixed)

    rng = random.Random(seed)
    order = sorted(mixed)
    rng.shuffle(order)
    order.sort(key=lambda p: source_rank(problem_meta.get(p, {}).get("source", "unknown")))
    chosen_problems = order[:max_problems]

    n_correct = dict.fromkeys(gens, 0)
    n_wrong = dict.fromkeys(gens, 0)
    out: list[dict] = []
    for pid in chosen_problems:
        d = by_problem[pid]
        cg = min(sorted(d[1]), key=lambda g: (n_correct[g], g))
        wrong_gens = sorted(d[0])
        pool = [g for g in wrong_gens if g != cg] or wrong_gens
        wg = min(pool, key=lambda g: (n_wrong[g], g))
        n_correct[cg] += 1
        n_wrong[wg] += 1
        for g, o in ((cg, 1), (wg, 0)):
            r = dict(rng.choice(sorted(d[o][g], key=lambda x: x["id"])))
            r["arm"] = "B"
            out.append(r)

    stats = {
        "n_gen_rows": len(gen_rows),
        "n_eligible": n_eligible,
        "n_problems_with_eligible": len(by_problem),
        "n_mixed": len(mixed),
        "mixed_by_source": dict(sorted(mixed_by_source.items())),
        "n_chosen_problems": len(chosen_problems),
        "chosen_by_source": dict(sorted(Counter(
            problem_meta.get(p, {}).get("source", "unknown") for p in chosen_problems).items())),
        "n_chosen_rows": len(out),
        "correct_by_generator": dict(sorted(n_correct.items())),
        "wrong_by_generator": dict(sorted(n_wrong.items())),
    }
    return out, stats


def print_stats(stats: dict) -> None:
    print(f"[select] generation rows {stats['n_gen_rows']} -> eligible {stats['n_eligible']} "
          f"on {stats['n_problems_with_eligible']} problems")
    print(f"[select] mixed problems {stats['n_mixed']}: " + "  ".join(
        f"{k}={v}" for k, v in stats["mixed_by_source"].items()))
    print(f"[select] chosen problems {stats['n_chosen_problems']}: " + "  ".join(
        f"{k}={v}" for k, v in stats["chosen_by_source"].items()))
    print(f"[select] chosen solutions {stats['n_chosen_rows']}")
    for g in stats["correct_by_generator"]:
        print(f"[select]   {g}: correct={stats['correct_by_generator'][g]} "
              f"wrong={stats['wrong_by_generator'][g]}")


def run(
    gen_globs: list[str],
    problems_paths: list[str],
    out_path: str,
    max_problems: int = MAX_PROBLEMS,
    seed: int = SEED,
) -> dict:
    rows: list[dict] = []
    for g in gen_globs:
        for fp in sorted(globlib.glob(g)):
            rows.extend(load_jsonl(fp))
    meta: dict[str, dict] = {}
    for p in problems_paths:
        for r in load_jsonl(p):
            meta[r["problem_id"]] = {"source": r.get("source"), "level": r.get("level")}
    chosen, stats = select_rows(rows, meta, max_problems=max_problems, seed=seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, chosen)
    print_stats(stats)
    print(f"[select] wrote {out_path}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="append", required=True,
                    help="glob of scored generation jsonl (repeatable)")
    ap.add_argument("--problems", action="append", required=True,
                    help="split jsonl with problem metadata (repeatable)")
    ap.add_argument("--out", default="data/gen/selected.jsonl")
    ap.add_argument("--max-problems", type=int, default=MAX_PROBLEMS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    run(args.gen, args.problems, args.out, args.max_problems, args.seed)


if __name__ == "__main__":
    main()
