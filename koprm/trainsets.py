"""§4.5 Training sets: A, B and A+B at 3k / 6k / 12k solutions, nested.

Each pool (A = PRM800K in Korean with human labels, B = on-policy Korean with kernel
labels) is shuffled once per outcome class with a fixed seed; a set of size n is the
prefix of each class, so 3k is a subset of 6k is a subset of 12k by construction. The
split between the classes is as close to 1:1 correct:wrong as the pool allows -- when
one class runs out the other fills the rest. A+B takes half of each size from each arm,
each half balanced the same way.

    python -m koprm.trainsets --a data/labels/A.jsonl --b data/labels/B.jsonl
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from koprm.io import load_jsonl, write_jsonl
from koprm.paths import LABELS, TRAINSETS

SEED = 20260920
SIZES = {"3k": 3000, "6k": 6000, "12k": 12000}
REQUIRED = ("problem_id", "problem_ko", "solution_steps", "step_labels", "outcome", "arm",
            "generator")


def row_key(r: dict, i: int = 0) -> str:
    return str(r.get("id") or f"{r.get('problem_id')}#{i}")


def order_pool(rows: list[dict], seed: int = SEED) -> dict[int, list[dict]]:
    """One deterministic order per outcome class (sampled separately, §4.5)."""
    out: dict[int, list[dict]] = {}
    for y in (0, 1):
        cls = [r for r in rows if int(r["outcome"]) == y]
        cls.sort(key=lambda r: row_key(r))
        random.Random(seed + y).shuffle(cls)
        out[y] = cls
    return out


def split_counts(n_correct: int, n_wrong: int, n: int, odd_to_correct: bool = True
                 ) -> tuple[int, int]:
    """How many correct / wrong make up a set of n, as close to 1:1 as the pool allows.

    Both counts grow monotonically with n, which is what makes the sizes nest.
    `odd_to_correct` decides who gets the odd solution; A+B gives it to the correct half
    of one arm and to the wrong half of the other, so the mix stays 1:1 overall.
    """
    if odd_to_correct:
        c = min(n_correct, (n + 1) // 2)
        w = min(n_wrong, n - c)
        c = min(n_correct, n - w)
    else:
        w = min(n_wrong, (n + 1) // 2)
        c = min(n_correct, n - w)
        w = min(n_wrong, n - c)
    return c, w


def take(ordered: dict[int, list[dict]], n: int, odd_to_correct: bool = True) -> list[dict]:
    c, w = split_counts(len(ordered[1]), len(ordered[0]), n, odd_to_correct)
    return ordered[1][:c] + ordered[0][:w]


def build_trainsets(
    pools: dict[str, list[dict]],
    sizes: dict[str, int] | None = None,
    seed: int = SEED,
) -> dict[str, list[dict]]:
    """pools: {"A": rows, "B": rows} -> {"A_3k": rows, ..., "AB_12k": rows}."""
    sizes = sizes or SIZES
    ordered = {arm: order_pool(rows, seed) for arm, rows in pools.items() if rows}
    out: dict[str, list[dict]] = {}
    for tag, n in sizes.items():
        for arm, o in ordered.items():
            out[f"{arm}_{tag}"] = take(o, n)
        if {"A", "B"} <= set(ordered):
            half = n // 2
            out[f"AB_{tag}"] = (take(ordered["A"], half)
                                + take(ordered["B"], n - half, odd_to_correct=False))
    return out


def summarize(sets: dict[str, list[dict]], sizes: dict[str, int] | None = None) -> list[dict]:
    sizes = sizes or SIZES
    rows = []
    for name, rs in sets.items():
        n_c = sum(1 for r in rs if int(r["outcome"]) == 1)
        rows.append({
            "set": name,
            "target": sizes.get(name.rsplit("_", 1)[-1], 0),
            "n": len(rs),
            "correct": n_c,
            "wrong": len(rs) - n_c,
            "problems": len({r.get("problem_id") for r in rs}),
            "arms": "+".join(sorted({str(r.get("arm")) for r in rs})),
        })
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"{'set':<10}{'target':>8}{'n':>7}{'correct':>9}{'wrong':>8}{'problems':>10}  arms")
    for r in rows:
        short = "  (short)" if r["n"] < r["target"] else ""
        print(f"{r['set']:<10}{r['target']:>8}{r['n']:>7}{r['correct']:>9}{r['wrong']:>8}"
              f"{r['problems']:>10}  {r['arms']}{short}")


def check_rows(rows: list[dict], name: str) -> None:
    missing = {k for r in rows[:50] for k in REQUIRED if k not in r}
    if missing:
        print(f"[trainsets] WARNING {name}: rows are missing {sorted(missing)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(LABELS / "A.jsonl"))
    ap.add_argument("--b", default=str(LABELS / "B.jsonl"))
    ap.add_argument("--out-dir", default=str(TRAINSETS))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    pools = {}
    for arm, path in (("A", args.a), ("B", args.b)):
        if path and Path(path).exists():
            pools[arm] = load_jsonl(path)
            check_rows(pools[arm], arm)
            n_c = sum(1 for r in pools[arm] if int(r["outcome"]) == 1)
            print(f"[trainsets] pool {arm}: {len(pools[arm])} rows "
                  f"(correct={n_c} wrong={len(pools[arm]) - n_c})")
        else:
            print(f"[trainsets] pool {arm}: missing ({path})")
    if not pools:
        raise SystemExit("no label pools found")

    sets = build_trainsets(pools, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in sets.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    print_table(summarize(sets, SIZES))
    print(f"[trainsets] wrote {len(sets)} files -> {out_dir}")


if __name__ == "__main__":
    main()
