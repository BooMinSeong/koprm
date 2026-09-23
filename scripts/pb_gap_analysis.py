"""Why the 8B student loses most F1 on omnimath (Plan §15.9f).

Reads the per-step probabilities of a student (`koprm.shift first-error --save-probs`)
and of the 72B teacher (`koprm.teacher.score` log-odds), both on `data/shift/pb_rows.jsonl`,
and prints the tables the section is built on: what the student inherits from the teacher,
what is its own, and the hypotheses that were ruled out.

    python scripts/pb_gap_analysis.py \
        --probs-ko data/shift/pb_probs/qwen3-8b_B_48k_soft_ko.jsonl \
        --probs-en data/shift/pb_probs/qwen3-8b_B_48k_soft_en.jsonl \
        --teacher-ko data/teacher/pb_ko_prm72b.jsonl \
        --teacher-en data/teacher/pb_en_prm72b.jsonl \
        --trainset data/trainsets/big/B_48k.jsonl
"""
from __future__ import annotations

import argparse
import math
import statistics as stat
from collections import Counter

from koprm.io import load_jsonl

SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")
LANGS = ("en", "ko")
CERTAIN = 0.9  # a teacher probability at or above this counts as "the teacher is sure"


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(z)))


def first_flag(probs: list[float], threshold: float = 0.5) -> int:
    return next((i for i, v in enumerate(probs) if v < threshold), -1)


def load_probs(path: str) -> dict[str, list[float]]:
    return {r["id"]: list(r["probs"]) for r in load_jsonl(path)}


def load_teacher(path: str) -> dict[str, list[float]]:
    return {r["id"]: [sigmoid(v) for v in r["teacher_logodds"]]
            for r in load_jsonl(path) if r.get("teacher_logodds")}


def _rows_of(rows: dict, split: str, correct: bool) -> list[dict]:
    return [r for r in rows.values()
            if r["split"] == split and ((r["label"] == -1) == correct)]


def med(xs) -> float:
    return stat.median(xs) if xs else float("nan")


def inherited(rows, stu, tea, th):
    """1. How much of the omnimath weakness is already in the teacher."""
    print("\n== 1. teacher (72B) doubt on CORRECT rows: steps below p<0.9, rows flagged")
    for lang in LANGS:
        for split in SPLITS:
            rs = [r for r in _rows_of(rows, split, True) if r["id"] in tea[lang]]
            steps = [p for r in rs for p in tea[lang][r["id"]]]
            flagged = sum(first_flag(tea[lang][r["id"]], th) != -1 for r in rs)
            unsure = sum(p < CERTAIN for p in steps) / max(len(steps), 1)
            print(f"  {lang} {split:14s} rows {len(rs):3d}  steps p<0.9 {unsure:5.1%}  "
                  f"rows flagged {flagged:3d} ({flagged / max(len(rs), 1):5.1%})")

    print("\n== 2. CORRECT rows: who flags them (student vs teacher, same language)")
    for lang in LANGS:
        for split in SPLITS:
            c: Counter = Counter()
            for r in _rows_of(rows, split, True):
                if r["id"] not in stu[lang] or r["id"] not in tea[lang]:
                    continue
                s_flag = first_flag(stu[lang][r["id"]], th) != -1
                t_flag = first_flag(tea[lang][r["id"]], th) != -1
                c["both" if s_flag and t_flag else "student_only" if s_flag
                  else "teacher_only" if t_flag else "neither"] += 1
            print(f"  {lang} {split:14s} " + "  ".join(
                f"{k} {c[k]:3d}" for k in ("both", "student_only", "teacher_only", "neither")))


def student_specific(rows, stu, tea, th):
    """2. What the student adds on its own: false flags where the teacher is sure."""
    print("\n== 3. per-step false positives on steps the teacher is sure about (p>=0.9)")
    for lang in LANGS:
        for split in SPLITS:
            n = bad = 0
            for r in _rows_of(rows, split, True):
                sp, tp = stu[lang].get(r["id"]), tea[lang].get(r["id"])
                if not sp or not tp or len(sp) != len(tp):
                    continue
                for ps, pt in zip(sp, tp):
                    if pt >= CERTAIN:
                        n += 1
                        bad += ps < th
            rate = bad / max(n, 1)
            print(f"  {lang} {split:14s} {bad:4d}/{n:5d} = {rate:5.1%}  "
                  f"(compounded over 8 steps: {(1 - rate) ** 8:.2f})")

    print("\n== 4. attenuation: student probability by teacher-probability bin (correct rows)")
    bins = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01))
    for lang in LANGS:
        for lo, hi in bins:
            ps = [s for r in rows.values() if r["label"] == -1
                  for s, t in zip(stu[lang].get(r["id"], []), tea[lang].get(r["id"], []))
                  if lo <= t < hi]
            if not ps:
                continue
            print(f"  {lang} teacher p in [{lo:.1f},{hi:.1f}): n {len(ps):5d}  "
                  f"student mean {stat.mean(ps):.2f}  flagged {sum(p < th for p in ps) / len(ps):5.1%}")

    print("\n== 5. length of the steps the student flags (characters, correct rows)")
    for lang in LANGS:
        field = f"steps_{lang}"
        flag, keep = [], []
        for r in rows.values():
            if r["label"] != -1 or r["id"] not in stu[lang]:
                continue
            for step, p in zip(r[field], stu[lang][r["id"]]):
                (flag if p < th else keep).append(len(step))
        print(f"  {lang} flagged median {med(flag):6.0f} (n {len(flag)})  "
              f"kept median {med(keep):6.0f} (n {len(keep)})")


def domain_gap(rows, trainset: str | None):
    """2b. Training steps are short MATH-level steps; omnimath steps are dense."""
    print("\n== 6. step density: training data vs the benchmark")
    for lang in LANGS:
        for split in SPLITS:
            lens = [len(s) for r in _rows_of(rows, split, True) + _rows_of(rows, split, False)
                    for s in r[f"steps_{lang}"]]
            over = sum(x > 250 for x in lens) / max(len(lens), 1)
            print(f"  {lang} {split:14s} median {med(lens):6.0f}  >250 chars {over:5.1%}")
    if not trainset:
        return
    lens = [len(s) for r in load_jsonl(trainset) for s in (r.get("solution_steps") or [])]
    over = sum(x > 250 for x in lens) / max(len(lens), 1)
    print(f"  training {trainset}: median {med(lens):.0f}  >250 chars {over:.1%}  "
          f"(problems are MATH/GSM8K level only)")


def localisation(rows, stu, th):
    """3. Error localisation is not the problem: it is as good on omnimath as elsewhere."""
    print("\n== 7. ERROR rows: probability at the labelled step, and exact-match accuracy")
    for lang in LANGS:
        for split in SPLITS:
            at, exact, n = [], 0, 0
            for r in _rows_of(rows, split, False):
                p = stu[lang].get(r["id"])
                if not p or r["label"] >= len(p):
                    continue
                n += 1
                at.append(p[r["label"]])
                exact += first_flag(p, th) == r["label"]
            print(f"  {lang} {split:14s} rows {n:3d}  p@label {stat.mean(at):.2f}  "
                  f"err_acc {exact / max(n, 1):.2f}")


def ruled_out(rows, stu, th):
    """4. Translation artefacts, solution length and answer format do not explain it."""
    print("\n== 8. text statistics per split (translation check)")
    for split in SPLITS:
        rs = [r for r in rows.values() if r["split"] == split]
        ko = " ".join(s for r in rs for s in r["steps_ko"])
        n_ko = sum(len(r["steps_ko"]) for r in rs)
        n_en = sum(len(r["steps_en"]) for r in rs)
        en = " ".join(s for r in rs for s in r["steps_en"])
        hangul = sum("가" <= c <= "힣" for c in ko) / max(len(ko), 1)
        print(f"  {split:14s} hangul {hangul:.2f}  backslash/char {ko.count(chr(92)) / len(ko):.3f}"
              f"  $/step ko {ko.count('$') / max(n_ko, 1):.1f} en {en.count('$') / max(n_en, 1):.1f}")

    print("\n== 9. solution length and the last step (correct rows)")
    for lang in LANGS:
        for split in SPLITS:
            rs = [r for r in _rows_of(rows, split, True) if r["id"] in stu[lang]]
            buckets: dict[str, list[bool]] = {"<=5": [], "6-9": [], ">=10": []}
            last_first = 0
            for r in rs:
                p = stu[lang][r["id"]]
                key = "<=5" if r["n_steps"] <= 5 else "6-9" if r["n_steps"] <= 9 else ">=10"
                buckets[key].append(first_flag(p, th) == -1)
                last_first += first_flag(p, th) == len(p) - 1
            kept = "  ".join(f"{k} {sum(v)}/{len(v)}" for k, v in buckets.items() if v)
            print(f"  {lang} {split:14s} kept by length: {kept}   "
                  f"first flag is the last step: {last_first}/{len(rs)}")


def examples(rows, stu, tea, th, n_examples: int):
    """Student-only false flags on omnimath, with the teacher's probability for contrast."""
    print(f"\n== 10. up to {n_examples} omnimath rows the student flags but the teacher does not")
    shown = 0
    for r in rows.values():
        if r["split"] != "omnimath" or r["label"] != -1 or shown >= n_examples:
            continue
        sp, tp = stu["en"].get(r["id"]), tea["en"].get(r["id"])
        if not sp or not tp:
            continue
        i = first_flag(sp, th)
        if i == -1 or first_flag(tp, th) != -1:
            continue
        shown += 1
        t_at = tp[i] if i < len(tp) else float("nan")
        print(f"  --- {r['id']} step {i}/{r['n_steps']}: student {sp[i]:.2f} teacher {t_at:.2f}")
        print(f"      {r['steps_en'][i][:160].replace(chr(10), ' ')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="data/shift/pb_rows.jsonl")
    ap.add_argument("--probs-ko", required=True)
    ap.add_argument("--probs-en", required=True)
    ap.add_argument("--teacher-ko", default="data/teacher/pb_ko_prm72b.jsonl")
    ap.add_argument("--teacher-en", default="data/teacher/pb_en_prm72b.jsonl")
    ap.add_argument("--trainset", default=None, help="training jsonl, for the step-length check")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    rows = {r["id"]: r for r in load_jsonl(args.rows)}
    stu = {"ko": load_probs(args.probs_ko), "en": load_probs(args.probs_en)}
    tea = {"ko": load_teacher(args.teacher_ko), "en": load_teacher(args.teacher_en)}
    print(f"rows {len(rows)}  student ko/en {len(stu['ko'])}/{len(stu['en'])}  "
          f"teacher ko/en {len(tea['ko'])}/{len(tea['en'])}  threshold {args.threshold}")
    inherited(rows, stu, tea, args.threshold)
    student_specific(rows, stu, tea, args.threshold)
    domain_gap(rows, args.trainset)
    localisation(rows, stu, args.threshold)
    ruled_out(rows, stu, args.threshold)
    if args.examples:
        examples(rows, stu, tea, args.threshold, args.examples)


if __name__ == "__main__":
    main()
