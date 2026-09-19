"""§5 Glue: build each stage's input file from the previous stage's outputs.

Pure deterministic jsonl joins, no model. Every subcommand is a file in / file out step of
the pipeline order in Plan §5 and README.

    python -m koprm.prep problems-in --splits data/splits/dev.jsonl \
        --splits data/splits/train_pool.jsonl --out data/trans/problems_in.jsonl
    python -m koprm.prep problems-ko --problems-in data/trans/problems_in.jsonl \
        --problems-ko-trans data/trans/problems_ko.jsonl --out data/gen/problems.jsonl
    python -m koprm.prep prm800k-problems-in --rows data/splits/prm800k_A_pool.jsonl \
        --rows data/splits/prm800k_audit.jsonl --out data/trans/prm800k_problems_in.jsonl
    python -m koprm.prep teacher-in --rows data/splits/prm800k_audit.jsonl \
        --steps-from data/trans/audit.steps_en_rt.jsonl --steps-field steps_en_rt \
        --out data/teacher/audit_rt.in.jsonl
"""
from __future__ import annotations

import argparse

from koprm.io import load_jsonl, write_jsonl


def _index(paths: list[str], key: str = "id") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in paths or []:
        for r in load_jsonl(p):
            out[r[key]] = r
    return out


def problems_in(splits: list[str]) -> list[dict]:
    """dev + train pool -> translation input, one row per problem (id = problem_id)."""
    out: list[dict] = []
    seen: set[str] = set()
    for path in splits:
        for r in load_jsonl(path):
            pid = r["problem_id"]
            if pid in seen:
                continue
            seen.add(pid)
            out.append(
                {
                    "id": pid,
                    "problem_id": pid,
                    "source": r.get("source"),
                    "problem_en": r["problem_en"],
                    "answer": r.get("answer"),
                }
            )
    return out


def problems_ko(rows_in: list[dict], trans: list[dict]) -> tuple[list[dict], int]:
    """Translation output -> generator input. Rows with a failed restoration are dropped."""
    by_id = {r.get("problem_id") or r["id"]: r for r in trans}
    out, dropped = [], 0
    for r in rows_in:
        pid = r.get("problem_id") or r["id"]
        t = by_id.get(pid)
        if t is None or not t.get("mask_restore_ok") or not t.get("problem_ko"):
            dropped += 1
            continue
        out.append(
            {
                "problem_id": pid,
                "problem_ko": t["problem_ko"],
                "answer": r.get("answer"),
                "source": r.get("source"),
            }
        )
    return out, dropped


def prm800k_problems_in(rows: list[dict]) -> list[dict]:
    """PRM800K solution rows -> one translation input row per distinct problem."""
    out: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        pid = r["problem_id"]
        if pid in seen:
            continue
        seen.add(pid)
        out.append({"id": pid, "problem_id": pid, "problem_en": r["problem_en"]})
    return out


def teacher_in(
    rows: list[dict],
    steps_by_id: dict[str, dict] | None,
    steps_field: str,
    problems_by_id: dict[str, dict] | None = None,
) -> tuple[list[dict], int]:
    """{id, problem_en, steps_en} for the teacher; rows without a clean translation drop out."""
    problems_by_id = problems_by_id or {}
    out, dropped = [], 0
    for r in rows:
        if steps_by_id is None:  # "direct" variant: the row's own English steps
            steps = r.get("steps_en")
        else:
            t = steps_by_id.get(r["id"])
            steps = t.get(steps_field) if t and t.get("mask_restore_ok") else None
        problem_en = r.get("problem_en") or (
            problems_by_id.get(r.get("problem_id"), {}).get("problem_en")
        )
        if not steps or not problem_en:
            dropped += 1
            continue
        out.append({"id": r["id"], "problem_en": problem_en, "steps_en": list(steps)})
    return out, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("problems-in", help="splits -> problem translation input")
    p.add_argument("--splits", action="append", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("problems-ko", help="problem translations -> generator input")
    p.add_argument("--problems-in", dest="problems_in", required=True)
    p.add_argument("--problems-ko-trans", dest="trans", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("prm800k-problems-in", help="PRM800K rows -> problem translation input")
    p.add_argument("--rows", action="append", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("teacher-in", help="rows + step translations -> teacher input")
    p.add_argument("--rows", required=True)
    p.add_argument("--steps-from", dest="steps_from", default=None)
    p.add_argument("--steps-field", dest="steps_field", default="steps_en")
    p.add_argument("--problems", action="append", default=[])
    p.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "problems-in":
        rows = problems_in(args.splits)
        dropped = 0
    elif args.cmd == "problems-ko":
        rows, dropped = problems_ko(load_jsonl(args.problems_in), load_jsonl(args.trans))
    elif args.cmd == "prm800k-problems-in":
        rows = prm800k_problems_in([r for p in args.rows for r in load_jsonl(p)])
        dropped = 0
    else:
        steps = _index([args.steps_from]) if args.steps_from else None
        rows, dropped = teacher_in(
            load_jsonl(args.rows), steps, args.steps_field, _index(args.problems, "problem_id")
        )
    n = write_jsonl(args.out, rows)
    print(f"[prep] {args.cmd}: wrote {n} rows to {args.out} (dropped {dropped})")


if __name__ == "__main__":
    main()
