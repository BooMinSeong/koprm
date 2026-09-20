"""§4.2 Selection: up to K correct and K wrong solutions per mixed problem (K=1 by default).

y is free, translation and the 72B teacher are not, so the cut happens here -- before
the expensive stages.

  * A solution is *eligible* only if it finished (not `truncated`) and produced a boxed
    answer (not `no_boxed`); a problem is "mixed" only counting eligible solutions,
    because an unusable solution cannot be the one we pick.
  * Mixed = at least one eligible correct (outcome=1) and one eligible wrong (outcome=0)
    solution across all generators. Only those problems can change a verifier's ranking.
  * Per mixed problem we take up to --per-class K correct and K wrong solutions at
    random, balancing generators: every pick comes from whichever generator has supplied
    the fewest solutions of that class so far (ties by name), never the same row twice,
    and the *first* wrong pick prefers a generator other than the first correct pick's,
    falling back to it when no other generator has a wrong solution. A problem with fewer
    than K solutions in a class contributes what it has (it is mixed, so at least one of
    each). K=1 is the original behaviour, row for row.
  * Cap at --max-problems (6,000 problems = 12,000 solutions at K=1), filling MATH first and
    GSM8K with what is left, because the evaluation is MATH500.

Output: the chosen generation rows, unchanged, plus `arm: "B"`.

    python -m koprm.select --gen 'data/gen/*.scored.jsonl' \
        --problems data/splits/train_pool.jsonl --out data/gen/selected.jsonl [--per-class 2]
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


def _take(by_gen: dict[str, list[dict]], counts: dict[str, int], k: int,
          rng: random.Random, first_avoid: str | None = None) -> list[dict]:
    """Up to k rows, each from the generator with the fewest picks so far (ties by name)."""
    remaining = {g: sorted(rows, key=lambda x: x["id"]) for g, rows in by_gen.items() if rows}
    picked: list[dict] = []
    for i in range(k):
        avail = sorted(g for g in remaining if remaining[g])
        if not avail:
            break
        pool = avail
        if i == 0 and first_avoid is not None:
            pool = [g for g in avail if g != first_avoid] or avail
        g = min(pool, key=lambda x: (counts[x], x))
        rows = remaining[g]
        # choice(range(n)) draws exactly like choice(rows); popping keeps rows unique.
        picked.append(rows.pop(rng.choice(range(len(rows)))))
        counts[g] += 1
    return picked


def select_rows(
    gen_rows: list[dict],
    problem_meta: dict[str, dict] | None = None,
    max_problems: int = MAX_PROBLEMS,
    seed: int = SEED,
    per_class: int = 1,
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
    n_short = 0
    for pid in chosen_problems:
        d = by_problem[pid]
        correct = _take(d[1], n_correct, per_class, rng)
        wrong = _take(d[0], n_wrong, per_class, rng, first_avoid=correct[0]["generator"])
        if len(correct) < per_class or len(wrong) < per_class:
            n_short += 1
        for r in correct + wrong:
            r = dict(r)
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
        "per_class": per_class,
        "n_chosen_rows": len(out),
        "n_problems_short_of_per_class": n_short,
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
    print(f"[select] chosen solutions {stats['n_chosen_rows']} "
          f"(per-class {stats['per_class']}, short of it on "
          f"{stats['n_problems_short_of_per_class']} problems)")
    for g in stats["correct_by_generator"]:
        print(f"[select]   {g}: correct={stats['correct_by_generator'][g]} "
              f"wrong={stats['wrong_by_generator'][g]}")


def run(
    gen_globs: list[str],
    problems_paths: list[str],
    out_path: str,
    max_problems: int = MAX_PROBLEMS,
    seed: int = SEED,
    per_class: int = 1,
) -> dict:
    rows: list[dict] = []
    for g in gen_globs:
        for fp in sorted(globlib.glob(g)):
            rows.extend(load_jsonl(fp))
    meta: dict[str, dict] = {}
    for p in problems_paths:
        for r in load_jsonl(p):
            meta[r["problem_id"]] = {"source": r.get("source"), "level": r.get("level")}
    chosen, stats = select_rows(rows, meta, max_problems=max_problems, seed=seed,
                                per_class=per_class)
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
    ap.add_argument("--per-class", type=int, default=1,
                    help="solutions per class per problem (1 = the original 1+1)")
    args = ap.parse_args()
    run(args.gen, args.problems, args.out, args.max_problems, args.seed, args.per_class)


if __name__ == "__main__":
    main()
