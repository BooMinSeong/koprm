"""DVTS with the student vs the stored komath DVTS with the English PRM (현행).

komath ran DVTS on KO MATH500 / EXAONE-4.0-1.2B with Qwen2.5-Math-PRM-7B scoring the Korean
text directly, at the same setting as koprm.eval.dvts (N=64, M=4, 40 iterations, T=0.8), for
seeds 0/42/64. Both runs are compared on the search-and-learn output pool: per subtree, the
prefix before its last iteration + each of that iteration's M candidates, so N = subtrees x M
completions, stored subtree by subtree (koprm.eval.dvts.expand_pool rebuilds it for the
student run).

Budget 64 is the whole pool (16 subtrees). Budget 16 is 4 subtrees drawn at random out of the
16 with all their candidates; the subtrees are independent, so that is a DVTS run at N=16.
Per problem the correctness is averaged over --draws draws (the same draws for every run),
which removes the luck of one draw without changing the expectation.

For each budget this prints naive/weighted/maj/pass (last-step score, first-in-pool wins
ties) for the student and each komath seed, a paired problem-level bootstrap of student -
English PRM on the first seed, and DVTS - BoN@b for each scorer on the stored BoN samples
(first b of 64, the komath convention): the student's saved BoN scores and the English PRM's
stored ones.

    python scripts/dvts_compare.py --out data/reports/dvts_compare.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from koprm.eval.bon import (METHODS, answer_groups, bootstrap_ci, evaluate,  # noqa: E402
                            existing_scores, load_hf_rows, load_scores)
from koprm.eval.dvts import (DATASET, DATASET_CONFIG, expand_pool,  # noqa: E402
                             merge_shards)
from koprm.paths import EVAL  # noqa: E402

KOMATH_DVTS = "ENSEONG/ko-ko-math-500-test-EXAONE-4.0-1.2B-dvts"
KOMATH_CONFIG = ("ENSEONG_ko-math-500-test--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0"
                 "--seed-{seed}--agg_strategy--last")
STUDENT = "data/eval/dvts/exaone-1.2b_q3-8b_B24k_soft_ep3_n64.shard*of8.jsonl"
STUDENT_BON = "data/eval/math500_7b/qwen3-8b_B_24k_soft_ep3_EXAONE-4.0-1.2B.scores.jsonl"
M = 4
BUDGETS = (16, 64)


def komath_rows(seed: int, gold: dict[str, str]) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(KOMATH_DVTS, KOMATH_CONFIG.format(seed=seed), split="train")
    by_id = {r["id"]: r for r in ds}
    return [{"problem_id": pid, "answer": ans, "completions": list(by_id[pid]["completions"]),
             "scores": [list(s) for s in by_id[pid]["scores"]]} for pid, ans in gold.items()]


def student_rows(pattern: str, gold: dict[str, str]) -> list[dict]:
    by_id = {r["problem_id"]: r for r in merge_shards([pattern])}
    rows = []
    for pid, ans in gold.items():
        comps, scores = expand_pool(by_id[pid])
        rows.append({"problem_id": pid, "answer": ans, "completions": comps, "scores": scores})
    return rows


def pick(gids: list[int], last: list[float], correct: list[bool], idx: list[int]) -> dict:
    """naive / weighted / maj / pass correctness over the completions `idx` (in pool order)."""
    best = max(idx, key=lambda i: (last[i], -i))
    wsum: dict[int, float] = {}
    cnt: dict[int, int] = {}
    first: dict[int, int] = {}
    for i in idx:
        g = gids[i]
        wsum[g] = wsum.get(g, 0.0) + last[i]
        cnt[g] = cnt.get(g, 0) + 1
        first.setdefault(g, i)
    gw = max(wsum, key=lambda g: (wsum[g], -first[g]))
    gm = max(cnt, key=lambda g: (cnt[g], -first[g]))
    return {"naive": correct[gids[best]], "weighted": correct[gw], "maj": correct[gm],
            "pass": any(correct[gids[i]] for i in idx)}


def per_problem(rows: list[dict], draws: list[list[list[int]]]) -> dict[int, dict[str, list]]:
    """Per budget, per method: per-problem correctness (averaged over the draws)."""
    groups = answer_groups(rows, cache_dir=EVAL / "cache", verbose=False)
    out = {b: {m: [] for m in (*METHODS, "pass")} for b in BUDGETS}
    for r, g, dr in zip(rows, groups, draws):
        n_sub = len(r["completions"]) // M
        last = [s[-1] if s else 0.0 for s in r["scores"]]
        for b in BUDGETS:
            k = b // M
            subsets = [list(range(n_sub))] if k == n_sub else [d[:k] for d in dr]
            acc = {m: 0.0 for m in out[b]}
            for trees in subsets:
                idx = [t * M + j for t in sorted(trees) for j in range(M)]
                for m, v in pick(g["gids"], last, g["correct"], idx).items():
                    acc[m] += float(v)
            for m in acc:
                out[b][m].append(acc[m] / len(subsets))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=STUDENT)
    ap.add_argument("--student-bon", default=STUDENT_BON)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 64])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hf = load_hf_rows(DATASET, DATASET_CONFIG)
    gold = {r["problem_id"]: str(r["answer"]) for r in hf}
    runs = {"student": student_rows(args.student, gold)}
    for s in args.seeds:
        runs[f"en_prm_seed{s}"] = komath_rows(s, gold)
    n_sub = {len(r["completions"]) // M for rows in runs.values() for r in rows}
    if n_sub != {max(BUDGETS) // M}:
        raise SystemExit(f"expected {max(BUDGETS) // M} subtrees per problem, got {n_sub}")

    rng = np.random.default_rng(args.rng_seed)
    draws = [[list(rng.permutation(max(BUDGETS) // M)) for _ in range(args.draws)]
             for _ in gold]
    pp = {name: per_problem(rows, draws) for name, rows in runs.items()}
    bon = {}
    for name, sc in (("student", load_scores(args.student_bon, hf)),
                     ("en_prm", existing_scores(hf))):
        res = evaluate(hf, sc, agg="last", cache_dir=EVAL / "cache", verbose=False)
        order = {r["problem_id"]: i for i, r in enumerate(hf)}
        bon[name] = {b: {m: [res["per_problem"][f"{m}@{b}"][order[pid]] for pid in gold]
                         for m in (*METHODS, "pass")} for b in BUDGETS}

    ref = f"en_prm_seed{args.seeds[0]}"
    out = {"setting": "KO MATH500, EXAONE-4.0-1.2B, DVTS N=64 M=4, search-and-learn pool; "
                      f"budget 16 = 4 random subtrees, {args.draws} draws",
           "student": args.student, "reference": ref, "budgets": {}}
    for b in BUDGETS:
        block = {}
        for m in (*METHODS, "pass"):
            row = {name: float(np.mean(p[b][m])) for name, p in pp.items()}
            row["en_prm_seed_mean"] = float(np.mean([row[f"en_prm_seed{s}"] for s in args.seeds]))
            row["diff_vs_" + ref] = bootstrap_ci(
                [x - y for x, y in zip(pp["student"][b][m], pp[ref][b][m])])
            for name, run in (("student", "student"), ("en_prm", ref)):
                row[f"bon_{name}"] = float(np.mean(bon[name][b][m]))
                row[f"dvts_minus_bon_{name}"] = bootstrap_ci(
                    [x - y for x, y in zip(pp[run][b][m], bon[name][b][m])])
            block[m] = row
        out["budgets"][str(b)] = block

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for b, block in out["budgets"].items():
        print(f"\n## budget {b}")
        for m, row in block.items():
            d = row["diff_vs_" + ref]
            ens = "  ".join(f"{row[f'en_prm_seed{s}']:.3f}" for s in args.seeds)
            print(f"  {m:>8}: student {row['student']:.3f}  EN-PRM seeds [{ens}]  "
                  f"diff(vs {ref}) {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
            for name in ("student", "en_prm"):
                e = row[f"dvts_minus_bon_{name}"]
                print(f"            BoN {name} {row[f'bon_{name}']:.3f}  DVTS-BoN "
                      f"{e['mean']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
