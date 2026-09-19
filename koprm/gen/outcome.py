"""Attach y (outcome) to generated solutions with parallel math_verify.

Adds: outcome (0/1), pred_answer, no_boxed (bool), truncated (bool).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from koprm.io import load_jsonl, write_jsonl
from koprm.verify import verify_many


def score_file(gen_path: str, problems_path: str, out_path: str, workers: int = 32) -> dict:
    answers = {p["problem_id"]: p["answer"] for p in load_jsonl(problems_path)}
    rows = []
    for fp in sorted(glob.glob(gen_path)):
        rows.extend(load_jsonl(fp))
    golds = [answers[r["problem_id"]] for r in rows]
    res = verify_many(golds, [r["text"] for r in rows], workers=workers)
    for r, (ok, pred) in zip(rows, res):
        r["outcome"] = int(ok)
        r["pred_answer"] = pred
        r["no_boxed"] = pred is None
        r["truncated"] = r.get("finish_reason") == "length"
    write_jsonl(out_path, rows)
    n = len(rows)
    stats = {
        "n": n,
        "acc": sum(r["outcome"] for r in rows) / max(n, 1),
        "no_boxed": sum(r["no_boxed"] for r in rows) / max(n, 1),
        "truncated": sum(r["truncated"] for r in rows) / max(n, 1),
    }
    print(stats)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="glob of generation jsonl shards")
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    score_file(args.gen, args.problems, args.out, args.workers)


if __name__ == "__main__":
    main()
