"""§7 확장: distribution-shift evaluations (English MATH500, Korean AIME, ProcessBench).

Three questions, one module of file joins plus one evaluator:

  1. Do the Korean-trained students survive *English* solutions? `math500-en` writes the
     MATH500 problems for `koprm.gen.generate --problem-field problem_en
     --system-prompt en`, and `bon-jsonl` turns the scored generations into the input
     `koprm.eval.bon --jsonl` expects.
  2. Korean AIME 2023/2024: the same `bon-jsonl` path, from translated problems.
  3. Korean ProcessBench: `processbench-rows` joins the English rows with their
     translations, and `first-error` scores them step by step (student checkpoint, or a
     teacher log-odds file) and reports the ProcessBench triple err/corr/F1.

    python -m koprm.shift math500-en --out data/shift/math500_en_problems.jsonl
    python -m koprm.shift bon-jsonl --gen 'data/gen/math500_en*.scored.jsonl' \\
        --problems data/shift/math500_en_problems.jsonl --problem-field problem_en \\
        --out data/shift/math500_en_bon.jsonl
    python -m koprm.shift processbench-rows --pb data/trans/pb_in.jsonl \\
        --ko-problems data/trans/pb_problems_ko.jsonl \\
        --ko-steps data/trans/pb_steps_ko.jsonl --out data/shift/pb_rows.jsonl
    python -m koprm.shift first-error --rows data/shift/pb_rows.jsonl \\
        --scorer data/ckpt/big_B_48k_soft/epoch3 --device cuda:0 \\
        --out data/reports/pb_big_B_48k_soft.json [--save-probs data/shift/pb_probs/<name>.jsonl]
"""
from __future__ import annotations

import argparse
import glob as globlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from koprm.io import load_jsonl, write_jsonl
from koprm.paths import DATA, SPLITS, SYSTEM_PROMPTS
from koprm.report import first_error_metrics, predict_first_error

OK = ("ok", "order", "passthrough")


def _rows(path: str | Path) -> list[dict]:
    p = Path(path)
    return load_jsonl(p) if p.exists() else []


# ------------------------------------------------------------------ 1. English MATH500
def math500_en_rows(rows: list[dict]) -> list[dict]:
    """The MATH500 split as a generation input in English."""
    return [{"problem_id": r["problem_id"], "problem_en": r["problem_en"],
             "answer": r.get("answer")} for r in rows]


# ------------------------------------------------------- 2. scored generations -> bon jsonl
def group_completions(
    gen_rows: list[dict],
    problems_by_id: dict[str, dict],
    problem_field: str = "problem_ko",
    n_completions: int | None = None,
) -> tuple[list[dict], dict]:
    """Per problem, the completions in sample_idx order; short problems are dropped."""
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for r in gen_rows:
        by_problem[r["problem_id"]].append(r)
    counts = Counter(len(v) for v in by_problem.values())
    n = n_completions or (max(counts) if counts else 0)

    out, drops = [], Counter()
    for pid in sorted(by_problem):
        rs = sorted(by_problem[pid], key=lambda r: (r.get("sample_idx", 0), r.get("id", "")))
        if len(rs) < n:
            drops["short"] += 1
            continue
        meta = problems_by_id.get(pid)
        if meta is None or not meta.get(problem_field):
            drops["no_problem_text"] += 1
            continue
        out.append({
            "problem_id": pid,
            "problem_ko": meta[problem_field],  # bon.py reads the query from problem_ko
            "answer": meta.get("answer"),
            "completions": [r.get("text", "") for r in rs[:n]],
        })
    stats = {"n_problems": len(by_problem), "n_completions": n, "n_kept": len(out),
             "dropped": dict(drops), "completion_counts": dict(sorted(counts.items()))}
    return out, stats


# ------------------------------------------------------------------- 3. ProcessBench rows
def processbench_rows(
    pb_in: list[dict],
    ko_problems: dict[str, dict],
    ko_steps: dict[str, dict],
) -> tuple[list[dict], dict]:
    """Join the English ProcessBench rows with their Korean translations."""
    out, drops = [], Counter()
    per_split: Counter = Counter()
    for r in pb_in:
        rid, split = r["id"], r.get("split", "unknown")
        per_split[split] += 1
        tp, ts = ko_problems.get(rid), ko_steps.get(rid)
        if tp is None or ts is None:
            drops[f"{split}/no_translation"] += 1
            continue
        if not (tp.get("mask_restore_ok") and ts.get("mask_restore_ok")):
            drops[f"{split}/mask_restore_fail"] += 1
            continue
        problem_ko, steps_ko = tp.get("problem_ko"), ts.get("steps_ko")
        if not problem_ko or not steps_ko:
            drops[f"{split}/no_translation"] += 1
            continue
        if len(steps_ko) != len(r["steps_en"]):
            drops[f"{split}/len_mismatch"] += 1
            continue
        label = int(r["label"])
        out.append({
            "id": rid,
            "split": split,
            "problem_en": r["problem_en"],
            "problem_ko": problem_ko,
            "steps_en": list(r["steps_en"]),
            "steps_ko": list(steps_ko),
            "label": label,
            "human_first_error": label,
            "outcome": 1 if label == -1 else 0,
            "n_steps": len(steps_ko),
            "mask_restore_ok": True,
        })
    stats = {"n_in": len(pb_in), "n_kept": len(out), "in_by_split": dict(sorted(per_split.items())),
             "kept_by_split": dict(sorted(Counter(r["split"] for r in out).items())),
             "dropped": dict(sorted(drops.items()))}
    return out, stats


# ------------------------------------------------------------------ 4. first-error scoring
def teacher_probs(rows: list[dict], teacher_by_id: dict[str, dict],
                  steps_field: str = "steps_ko") -> tuple[list[list[float] | None], int]:
    """sigmoid(z) per step from a teacher jsonl; None when it cannot be used."""
    probs: list[list[float] | None] = []
    n_skipped = 0
    for r in rows:
        z = (teacher_by_id.get(r["id"]) or {}).get("teacher_logodds")
        if z is None or len(z) != len(r[steps_field]):
            n_skipped += 1
            probs.append(None)
            continue
        probs.append([1.0 / (1.0 + math.exp(-float(v))) for v in z])
    return probs, n_skipped


def student_probs(rows: list[dict], scorer, steps_field: str = "steps_ko",
                  problem_field: str = "problem_ko",
                  batch_size: int = 8) -> tuple[list[list[float] | None], int]:
    probs = scorer.score([r[problem_field] for r in rows],
                         [list(r[steps_field]) for r in rows], batch_size=batch_size)
    n_skipped = sum(1 for p in probs if not p)
    return [p if p else None for p in probs], n_skipped


def probs_rows(rows: list[dict], probs: list[list[float] | None]) -> list[dict]:
    """Per-step probabilities as jsonl rows, so an analysis can re-read them (§15.9f)."""
    return [{"id": r["id"], "split": r.get("split"), "label": r.get("label"),
             "probs": [float(x) for x in p]}
            for r, p in zip(rows, probs) if p]


def pb_metrics(rows: list[dict], preds: list[int]) -> dict:
    """ProcessBench triple from the audit metrics: err / corr accuracy and their F1."""
    m = first_error_metrics(rows, preds)
    err, corr = m["exact_frac"], m["correct_no_prediction_frac"]
    f1 = 0.0 if (err + corr) == 0 else 2 * err * corr / (err + corr)
    return {
        "n_rows": m["n_rows"],
        "n_error_rows": m["n_wrong"],
        "n_correct_rows": m["n_correct"],
        "err_acc": err,
        "corr_acc": corr,
        "f1": f1,
        "within1": m["within1_frac"],
        "err_no_prediction_frac": m["wrong_no_prediction_frac"],
    }


def first_error_report(rows: list[dict], probs: list[list[float] | None],
                       threshold: float = 0.5, n_skipped: int = 0) -> dict:
    """Per split and overall; rows without usable scores are skipped, not guessed."""
    usable = [(r, p) for r, p in zip(rows, probs) if p]
    preds = [predict_first_error(p, threshold) for _, p in usable]
    scored = [r for r, _ in usable]
    out = {
        "threshold": threshold,
        "n_rows": len(rows),
        "n_scored": len(scored),
        "n_skipped": len(rows) - len(usable),
        "overall": pb_metrics(scored, preds),
        "by_split": {},
    }
    for split in sorted({r.get("split", "unknown") for r in scored}):
        idx = [i for i, r in enumerate(scored) if r.get("split", "unknown") == split]
        out["by_split"][split] = pb_metrics([scored[i] for i in idx], [preds[i] for i in idx])
    return out


def print_first_error(rep: dict) -> None:
    o = rep["overall"]
    print(f"[shift] scored {rep['n_scored']}/{rep['n_rows']} rows "
          f"(skipped {rep['n_skipped']}), threshold {rep['threshold']}")
    print(f"{'split':<24}{'err':>8}{'corr':>8}{'F1':>8}{'±1':>8}{'n_err':>8}{'n_corr':>8}")
    for name, m in list(rep["by_split"].items()) + [("overall", o)]:
        print(f"{name:<24}{m['err_acc']:>8.3f}{m['corr_acc']:>8.3f}{m['f1']:>8.3f}"
              f"{m['within1']:>8.3f}{m['n_error_rows']:>8}{m['n_correct_rows']:>8}")


# ------------------------------------------------------------------------------------ CLI
def _index(path: str | Path, key: str = "id") -> dict[str, dict]:
    return {r[key]: r for r in _rows(path)}


def _write(path: str | Path, rows: list[dict], stats: dict, cmd: str) -> None:
    write_jsonl(path, rows)
    print(f"[shift] {cmd}: wrote {len(rows)} rows to {path}")
    print(f"[shift] {json.dumps(stats, ensure_ascii=False)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("math500-en", help="MATH500 problems in English for generation")
    p.add_argument("--splits", default=str(SPLITS / "math500.jsonl"))
    p.add_argument("--out", default=str(DATA / "shift/math500_en_problems.jsonl"))

    p = sub.add_parser("bon-jsonl", help="scored generations -> koprm.eval.bon --jsonl input")
    p.add_argument("--gen", action="append", required=True, help="glob (repeatable)")
    p.add_argument("--problems", required=True)
    p.add_argument("--problem-field", default="problem_ko")
    p.add_argument("--n-completions", type=int, default=None, help="default: the max found")
    p.add_argument("--out", required=True)

    p = sub.add_parser("processbench-rows", help="ProcessBench + translations -> scoring rows")
    p.add_argument("--pb", required=True)
    p.add_argument("--ko-problems", required=True)
    p.add_argument("--ko-steps", required=True)
    p.add_argument("--out", default=str(DATA / "shift/pb_rows.jsonl"))

    p = sub.add_parser("first-error", help="first-error detection (ProcessBench style)")
    p.add_argument("--rows", default=str(DATA / "shift/pb_rows.jsonl"))
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scorer", help="student checkpoint dir")
    src.add_argument("--teacher", help="teacher jsonl with teacher_logodds keyed by id")
    p.add_argument("--steps-field", default="steps_ko")
    p.add_argument("--problem-field", default="problem_ko")
    p.add_argument("--system-prompt", choices=["ko", "en"], default="ko")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save-probs", default=None,
                   help="also write the per-step probabilities as jsonl (id/split/label/probs)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "math500-en":
        rows = math500_en_rows(_rows(args.splits))
        _write(args.out, rows, {"n": len(rows)}, args.cmd)
    elif args.cmd == "bon-jsonl":
        gen = [r for g in args.gen for fp in sorted(globlib.glob(g)) for r in load_jsonl(fp)]
        problems = {r["problem_id"]: r for r in _rows(args.problems)}
        rows, stats = group_completions(gen, problems, args.problem_field, args.n_completions)
        _write(args.out, rows, stats, args.cmd)
    elif args.cmd == "processbench-rows":
        rows, stats = processbench_rows(_rows(args.pb), _index(args.ko_problems),
                                        _index(args.ko_steps))
        _write(args.out, rows, stats, args.cmd)
    else:
        rows = _rows(args.rows)
        if not rows:
            raise SystemExit(f"no rows in {args.rows}")
        if args.teacher:
            probs, n_skipped = teacher_probs(rows, _index(args.teacher), args.steps_field)
            scorer_name = f"teacher:{Path(args.teacher).name}"
        else:
            from koprm.eval.scorer import StudentScorer

            scorer = StudentScorer(args.scorer, device=args.device, max_len=args.max_len,
                                   system_prompt=SYSTEM_PROMPTS[args.system_prompt])
            probs, n_skipped = student_probs(rows, scorer, args.steps_field,
                                             args.problem_field, args.batch_size)
            scorer_name = args.scorer
        if args.save_probs:
            n = write_jsonl(args.save_probs, probs_rows(rows, probs))
            print(f"[shift] wrote per-step probabilities for {n} rows to {args.save_probs}")
        rep = first_error_report(rows, probs, args.threshold, n_skipped)
        rep.update({"scorer": scorer_name, "rows": args.rows, "steps_field": args.steps_field,
                    "problem_field": args.problem_field,
                    "system_prompt": None if args.teacher else args.system_prompt})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print_first_error(rep)
        print(f"[shift] wrote {args.out}")


if __name__ == "__main__":
    main()
