"""§2.3-§2.4 + §4.5 Label building: join translation + teacher, fit D_g, run the kernel.

Arm B (`build_labels`)
    rows           : selected generations {id, problem_id, generator, steps, outcome}
    trans_by_id    : {id: {steps_en, mask_restore_ok}}      (ko -> en, §3)
    teacher_by_id  : {id: {teacher_logodds}}                (§2.2)
    problems_by_id : {problem_id: {source, problem_en, problem_ko}}

    A row survives only with a clean placeholder restoration, a teacher score, and
    matching lengths |steps| = |steps_en| = |z|; the dropped counts are reported, not
    hidden. The reference distribution D_g (§2.3) is fitted per generator on that
    generator's y=1 solutions and saved next to the output as `<out>.refs.json`, so the
    audit (§4.4) and re-runs can reuse or supply one with --ref-from.

Arm A (`build_A_rows`)
    PRM800K A-pool rows plus their en->ko step translations. Labels are the human ones
    (§4.3, already mapped to 1/0 by `parse_phase2`); the teacher fields stay empty. The
    pool is cut to a 1:1 correct:wrong balance (<= --n-per-class each), preferring at
    most 2 solutions per problem.

Both arms emit the §4.5 schema:
    {problem_id, problem_source, problem_ko, problem_en, arm, generator, solution_steps,
     solution_steps_en, teacher_logodds, outcome, ceiling, candidates, tail_probs,
     r_lo, r_hi, step_labels, mask_restore_ok}
(plus `id`, kept for joins.) `step_labels` uses null for masked steps.

    python -m koprm.label.build B --rows data/gen/selected.jsonl \
        --trans data/trans/selected.steps_en.jsonl --teacher data/teacher/selected.jsonl \
        --problems data/splits/train_pool.jsonl --problems-ko data/trans/problems_ko.jsonl \
        --out data/labels/B.jsonl
    python -m koprm.label.build A --rows data/splits/prm800k_A_pool.jsonl \
        --trans data/trans/prm800k_A.steps_ko.jsonl \
        --problems-ko data/trans/prm800k_problems_ko.jsonl --out data/labels/A.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from koprm.io import load_jsonl, write_jsonl
from koprm.label.kernel import HI, LO, RefDist, kernel

SEED = 20260920
A_PER_CLASS = 6000
A_MAX_PER_PROBLEM = 2
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _q(values) -> dict:
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return {}
    return {f"q{int(q * 100)}": float(np.quantile(v, q)) for q in QUANTILES}


def _problem_fields(row: dict, problems_by_id: dict[str, dict]) -> dict:
    pid = row.get("problem_id") or row["id"]
    p = problems_by_id.get(pid) or problems_by_id.get(row["id"]) or {}
    if not p and row.get("problem_en"):
        p = problems_by_id.get(row["problem_en"].strip(), {})
    return {
        "problem_id": pid,
        "problem_source": p.get("source") or row.get("source"),
        "problem_ko": p.get("problem_ko") or row.get("problem_ko"),
        "problem_en": p.get("problem_en") or row.get("problem_en"),
    }


def prepare(
    rows: list[dict],
    trans_by_id: dict[str, dict],
    teacher_by_id: dict[str, dict],
    steps_field: str = "steps",
    trans_field: str = "steps_en",
) -> tuple[list[tuple[dict, list[str], np.ndarray]], Counter]:
    """Join the three sources, dropping rows that cannot be labeled."""
    kept: list[tuple[dict, list[str], np.ndarray]] = []
    drops: Counter = Counter()
    for r in rows:
        steps = r.get(steps_field) or []
        if not steps:
            drops["no_steps"] += 1
            continue
        tr = trans_by_id.get(r["id"])
        if tr is None:
            drops["no_translation"] += 1
            continue
        if not tr.get("mask_restore_ok"):
            drops["mask_restore_fail"] += 1
            continue
        steps_tr = tr.get(trans_field)
        if not steps_tr:
            drops["no_translation"] += 1
            continue
        te = teacher_by_id.get(r["id"]) or {}
        z = te.get("teacher_logodds")
        if z is None:
            drops["no_teacher"] += 1
            continue
        if not (len(steps) == len(steps_tr) == len(z)):
            drops["len_mismatch"] += 1
            continue
        kept.append((r, list(steps_tr), np.asarray(z, dtype=np.float64)))
    return kept, drops


def fit_refs(kept: list[tuple[dict, list[str], np.ndarray]]) -> dict[str, RefDist]:
    """D_g per generator, fitted on that generator's y=1 solutions (§2.3)."""
    by_gen: dict[str, list[np.ndarray]] = defaultdict(list)
    for r, _, z in kept:
        if int(r["outcome"]) == 1:
            by_gen[r.get("generator", "default")].append(z)
    return {g: RefDist.fit(zs) for g, zs in by_gen.items() if zs}


def build_labels(
    rows: list[dict],
    trans_by_id: dict[str, dict],
    teacher_by_id: dict[str, dict],
    problems_by_id: dict[str, dict],
    arm: str = "B",
    ref: dict[str, RefDist] | RefDist | None = None,
    steps_field: str = "steps",
    trans_field: str = "steps_en",
    hi: float = HI,
    lo: float = LO,
) -> tuple[list[dict], dict[str, RefDist], dict]:
    """Join, fit (or reuse) D_g, run the kernel. Returns (rows, refs, stats)."""
    kept, drops = prepare(rows, trans_by_id, teacher_by_id, steps_field, trans_field)
    if ref is None:
        refs = fit_refs(kept)
    elif isinstance(ref, RefDist):
        refs = {r.get("generator", "default"): ref for r, _, _ in kept}
    else:
        refs = dict(ref)

    out: list[dict] = []
    for r, steps_tr, z in kept:
        g = r.get("generator", "default")
        rd = refs.get(g)
        if rd is None and len(refs) == 1:  # a single shared D (e.g. the audit's)
            rd = next(iter(refs.values()))
        if rd is None:
            drops["no_reference"] += 1
            continue
        y = int(r["outcome"])
        k = kernel(z, y, rd, hi=hi, lo=lo)
        out.append(
            {
                "id": r["id"],
                **_problem_fields(r, problems_by_id),
                "arm": arm,
                "generator": g,
                "solution_steps": list(r[steps_field]),
                "solution_steps_en": steps_tr,
                "teacher_logodds": z.tolist(),
                "outcome": y,
                "ceiling": k.ceiling,
                "candidates": k.candidates,
                "tail_probs": k.tail_probs,
                "r_lo": k.r_lo,
                "r_hi": k.r_hi,
                "step_labels": k.labels,
                "mask_restore_ok": True,
            }
        )
    stats = label_stats(out, dict(refs), drops, n_in=len(rows))
    return out, dict(refs), stats


def label_stats(rows: list[dict], refs: dict[str, RefDist], drops: Counter, n_in: int) -> dict:
    wrong = [r for r in rows if r["outcome"] == 0]
    comp: Counter = Counter()
    for r in wrong:
        for l in r["step_labels"]:
            comp["mask" if l is None else str(l)] += 1
    n_steps = max(sum(comp.values()), 1)
    per_gen = {}
    for g, rd in refs.items():
        ceils = [r["ceiling"] for r in rows if r["generator"] == g and r["outcome"] == 1]
        per_gen[g] = {
            "b_min": rd.b_min,
            "n_ref_steps": len(rd.drops),
            "ceiling_q": _q(ceils),
            "drop_q": _q(rd.drops),
        }
    return {
        "n_in": n_in,
        "n_labeled": len(rows),
        "dropped": dict(drops),
        "n_correct": sum(1 for r in rows if r["outcome"] == 1),
        "n_wrong": len(wrong),
        "wrong_label_composition": {k: comp[k] / n_steps for k in ("1", "0", "mask")},
        "wrong_no_candidates_frac": (
            sum(1 for r in wrong if not r["candidates"]) / max(len(wrong), 1)),
        "per_generator": per_gen,
    }


def print_stats(stats: dict) -> None:
    print(f"[label] in={stats['n_in']} labeled={stats['n_labeled']} "
          f"(correct={stats['n_correct']} wrong={stats['n_wrong']}) dropped={stats['dropped']}")
    c = stats["wrong_label_composition"]
    print(f"[label] y=0 step labels: 1={c['1']:.3f} 0={c['0']:.3f} mask={c['mask']:.3f}; "
          f"y=0 solutions with no candidate: {stats['wrong_no_candidates_frac']:.3f}")
    for g, d in stats["per_generator"].items():
        print(f"[label]   {g}: b_min={d['b_min']:.3f} ref_steps={d['n_ref_steps']}")
        print(f"[label]     ceiling {d['ceiling_q']}")
        print(f"[label]     drop    {d['drop_q']}")


# --------------------------------------------------------------------------- arm A


def build_A_rows(
    pool_rows: list[dict],
    trans_by_id: dict[str, dict],
    problems_by_id: dict[str, dict] | None = None,
    n_per_class: int = A_PER_CLASS,
    max_per_problem: int = A_MAX_PER_PROBLEM,
    seed: int = SEED,
    trans_field: str = "steps_ko",
) -> tuple[list[dict], dict]:
    """PRM800K A-pool + en->ko translations -> §4.5 rows with human step labels."""
    problems_by_id = problems_by_id or {}
    drops: Counter = Counter()
    usable: dict[int, list[dict]] = {0: [], 1: []}
    for r in pool_rows:
        tr = trans_by_id.get(r["id"])
        if tr is None:
            drops["no_translation"] += 1
            continue
        if not tr.get("mask_restore_ok"):
            drops["mask_restore_fail"] += 1
            continue
        steps_ko = tr.get(trans_field)
        if not steps_ko:
            drops["no_translation"] += 1
            continue
        if not (len(steps_ko) == len(r["steps_en"]) == len(r["step_labels"])):
            drops["len_mismatch"] += 1
            continue
        usable[int(r["outcome"])].append({**r, "_steps_ko": list(steps_ko)})

    rng = random.Random(seed)
    queues = {}
    for y in (0, 1):
        pool = sorted(usable[y], key=lambda r: r["id"])
        rng.shuffle(pool)
        queues[y] = deque(pool)
    picked: dict[int, list[dict]] = {0: [], 1: []}
    leftover: dict[int, list[dict]] = {0: [], 1: []}
    per_problem: Counter = Counter()
    # Round-robin between the two classes so neither eats the per-problem budget alone.
    while any(queues[y] and len(picked[y]) < n_per_class for y in (0, 1)):
        for y in (1, 0):
            if len(picked[y]) >= n_per_class:
                continue
            while queues[y]:
                r = queues[y].popleft()
                if per_problem[r["problem_id"]] < max_per_problem:
                    per_problem[r["problem_id"]] += 1
                    picked[y].append(r)
                    break
                leftover[y].append(r)
    # Relax the per-problem cap only if a class would otherwise fall short.
    for y in (1, 0):
        need = n_per_class - len(picked[y])
        if need > 0 and leftover[y]:
            for r in leftover[y][:need]:
                per_problem[r["problem_id"]] += 1
                picked[y].append(r)
    n = min(len(picked[0]), len(picked[1]))  # keep 1:1
    chosen = picked[1][:n] + picked[0][:n]

    out = []
    for r in chosen:
        out.append(
            {
                "id": r["id"],
                **_problem_fields(r, problems_by_id),
                "arm": "A",
                "generator": "prm800k",
                "solution_steps": r["_steps_ko"],
                "solution_steps_en": list(r["steps_en"]),
                "teacher_logodds": None,
                "outcome": int(r["outcome"]),
                "ceiling": None,
                "candidates": [],
                "tail_probs": [],
                "r_lo": [],
                "r_hi": [],
                "step_labels": list(r["step_labels"]),
                "mask_restore_ok": True,
                "human_first_error": r.get("human_first_error"),
            }
        )
    for r in out:
        if r["problem_source"] is None:
            r["problem_source"] = "prm800k"
    stats = {
        "n_in": len(pool_rows),
        "dropped": dict(drops),
        "n_usable": {str(y): len(usable[y]) for y in (0, 1)},
        "n_labeled": len(out),
        "n_correct": n,
        "n_wrong": n,
        "solutions_per_problem_max": max(per_problem.values()) if per_problem else 0,
        "n_problems": len({r["problem_id"] for r in out}),
        "n_missing_problem_ko": sum(1 for r in out if not r["problem_ko"]),
    }
    return out, stats


# --------------------------------------------------------------------------- CLI


def _index(path: str | None, key: str = "id") -> dict[str, dict]:
    if not path:
        return {}
    return {r[key]: r for r in load_jsonl(path)}


def load_problems(problems: list[str] | None, problems_ko: list[str] | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in problems or []:
        for r in load_jsonl(p):
            d = out.setdefault(r["problem_id"], {})
            d.update({k: r[k] for k in ("source", "problem_en", "problem_ko") if k in r})
    for p in problems_ko or []:
        for r in load_jsonl(p):
            pid = r.get("problem_id") or r["id"]
            d = out.setdefault(pid, {})
            if r.get("problem_ko"):
                d["problem_ko"] = r["problem_ko"]
            if r.get("problem_en"):
                d.setdefault("problem_en", r["problem_en"])
    return out


def refs_path(out_path: str | Path) -> Path:
    p = Path(out_path)
    return p.with_name(p.stem + ".refs.json")


def save_refs(path: str | Path, refs: dict[str, RefDist]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({g: rd.to_dict() for g, rd in refs.items()}, f)


def load_refs(path: str | Path) -> dict[str, RefDist]:
    with open(path, encoding="utf-8") as f:
        return {g: RefDist.from_dict(d) for g, d in json.load(f).items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="arm", required=True)

    b = sub.add_parser("B", help="on-policy Korean solutions, kernel labels")
    b.add_argument("--rows", required=True)
    b.add_argument("--trans", required=True, help="ko->en step translations")
    b.add_argument("--teacher", required=True)
    b.add_argument("--problems", action="append", default=[])
    b.add_argument("--problems-ko", action="append", default=[])
    b.add_argument("--out", default="data/labels/B.jsonl")
    b.add_argument("--steps-field", default="steps")
    b.add_argument("--trans-field", default="steps_en")
    b.add_argument("--ref-from", default=None, help="reuse D_g from this json instead of fitting")
    b.add_argument("--hi", type=float, default=HI)
    b.add_argument("--lo", type=float, default=LO)

    a = sub.add_parser("A", help="PRM800K translated to Korean, human labels")
    a.add_argument("--rows", required=True, help="data/splits/prm800k_A_pool.jsonl")
    a.add_argument("--trans", required=True, help="en->ko step translations")
    a.add_argument("--problems", action="append", default=[])
    a.add_argument("--problems-ko", action="append", default=[])
    a.add_argument("--out", default="data/labels/A.jsonl")
    a.add_argument("--trans-field", default="steps_ko")
    a.add_argument("--n-per-class", type=int, default=A_PER_CLASS)
    a.add_argument("--max-per-problem", type=int, default=A_MAX_PER_PROBLEM)
    a.add_argument("--seed", type=int, default=SEED)

    args = ap.parse_args()
    rows = load_jsonl(args.rows)
    trans = _index(args.trans)
    problems = load_problems(args.problems, args.problems_ko)

    if args.arm == "B":
        teacher = _index(args.teacher)
        ref = load_refs(args.ref_from) if args.ref_from else None
        out, refs, stats = build_labels(
            rows, trans, teacher, problems, arm="B", ref=ref,
            steps_field=args.steps_field, trans_field=args.trans_field,
            hi=args.hi, lo=args.lo,
        )
        print_stats(stats)
        write_jsonl(args.out, out)
        if refs:
            save_refs(refs_path(args.out), refs)
            print(f"[label] refs -> {refs_path(args.out)}")
    else:
        out, stats = build_A_rows(
            rows, trans, problems, n_per_class=args.n_per_class,
            max_per_problem=args.max_per_problem, seed=args.seed,
            trans_field=args.trans_field,
        )
        print(f"[label:A] {json.dumps(stats, ensure_ascii=False)}")
        write_jsonl(args.out, out)
    print(f"[label] wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
