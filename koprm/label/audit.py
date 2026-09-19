"""§4.4 Audit: how well the kernel's hard labels match the human ones.

The PRM800K held-out solutions go en -> ko -> en (the same pipeline as B, §3) and are
scored by the teacher on the round-tripped English; the kernel then labels the y=0
solutions and we compare against the human first-error position. The reference
distribution D is fitted on the *audit's own* y=1 solutions, separately from B (§4.4).

Input rows (one per held-out solution, joined by id):
    steps_en                 original PRM800K steps
    steps_ko                 en -> ko
    steps_en_rt              ko -> en (round trip)
    teacher_logodds_rt       teacher on steps_en_rt
    teacher_logodds_direct   optional: teacher on the original steps_en ("no bridge")
    step_labels              human 1/0 per step, human_first_error, outcome, finish_reason

Metrics, computed over non-last steps only (the last step is 0 by rule, §2.4, so
scoring it would flatter the kernel):
    precision   unmasked kernel labels that equal the human label (also split 1 / 0)
    coverage    non-last steps that are not masked
    bridge cost the same numbers from `teacher_logodds_direct` minus the round trip's

Reported at 0.9/0.1 and at the tighter 0.95/0.05 (Plan §9).

    python -m koprm.label.audit --rows data/splits/prm800k_audit.jsonl \
        --ko-trans data/trans/audit.steps_ko.jsonl --rt-trans data/trans/audit.steps_en_rt.jsonl \
        --rt-teacher data/teacher/audit_rt.jsonl --direct-teacher data/teacher/audit_direct.jsonl \
        --out data/reports/audit.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from koprm.io import load_jsonl, write_jsonl
from koprm.label.kernel import HI, LO, RefDist, kernel

THRESHOLDS = [(HI, LO), (0.95, 0.05)]


def _valid(rows: list[dict], z_field: str) -> tuple[list[dict], Counter]:
    out, drops = [], Counter()
    for r in rows:
        z = r.get(z_field)
        if z is None:
            drops["no_teacher"] += 1
            continue
        if r.get("mask_restore_ok") is False:
            drops["mask_restore_fail"] += 1
            continue
        labels = r.get("step_labels") or []
        if len(z) != len(labels) or len(labels) < 2:
            drops["len_mismatch"] += 1
            continue
        out.append(r)
    return out, drops


def variant_metrics(rows: list[dict], z_field: str, hi: float, lo: float) -> dict:
    """Fit D on the y=1 solutions, label the y=0 ones, compare with the humans."""
    usable, drops = _valid(rows, z_field)
    correct = [r for r in usable if int(r["outcome"]) == 1]
    wrong = [r for r in usable if int(r["outcome"]) == 0]
    if not correct or not wrong:
        return {"n_correct": len(correct), "n_wrong": len(wrong), "dropped": dict(drops),
                "error": "need both y=1 and y=0 solutions"}
    ref = RefDist.fit([np.asarray(r[z_field], dtype=np.float64) for r in correct])

    n_steps = n_unmasked = n_agree = 0
    pred = Counter()
    agree = Counter()
    n_no_cand = 0
    first_err_exact = 0
    per_row = []
    for r in wrong:
        z = np.asarray(r[z_field], dtype=np.float64)
        k = kernel(z, 0, ref, hi=hi, lo=lo)
        human = r["step_labels"]
        T = len(z)
        if not k.candidates:
            n_no_cand += 1
        row_agree = row_unmasked = 0
        for t in range(T - 1):  # last step is 0 by rule
            n_steps += 1
            l = k.labels[t]
            if l is None:
                continue
            n_unmasked += 1
            row_unmasked += 1
            pred[l] += 1
            if l == human[t]:
                n_agree += 1
                agree[l] += 1
                row_agree += 1
        # the kernel's implied first error = first 0 among the non-last steps
        zeros = [t for t in range(T - 1) if k.labels[t] == 0]
        if zeros and zeros[0] == r.get("human_first_error"):
            first_err_exact += 1
        per_row.append({"id": r.get("id"), "unmasked": row_unmasked, "agree": row_agree,
                        "n_steps": T})
    return {
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "dropped": dict(drops),
        "ref_b_min": ref.b_min,
        "n_ref_steps": len(ref.drops),
        "n_eval_steps": n_steps,
        "n_unmasked": n_unmasked,
        "coverage": n_unmasked / max(n_steps, 1),
        "precision": n_agree / max(n_unmasked, 1),
        "precision_1": agree[1] / max(pred[1], 1),
        "precision_0": agree[0] / max(pred[0], 1),
        "n_pred_1": pred[1],
        "n_pred_0": pred[0],
        "no_candidate_frac": n_no_cand / max(len(wrong), 1),
        "first_error_exact_frac": first_err_exact / max(len(wrong), 1),
        "_per_row": per_row,
    }


def audit_metrics(rows: list[dict], hi: float = HI, lo: float = LO) -> dict:
    """Round-trip metrics, plus the direct (no-bridge) ones and the bridge cost."""
    out: dict = {"hi": hi, "lo": lo, "n_rows": len(rows)}
    out["rt"] = variant_metrics(rows, "teacher_logodds_rt", hi, lo)
    if any(r.get("teacher_logodds_direct") is not None for r in rows):
        out["direct"] = variant_metrics(rows, "teacher_logodds_direct", hi, lo)
        if "error" not in out["direct"] and "error" not in out["rt"]:
            out["bridge_cost"] = {
                k: out["direct"][k] - out["rt"][k]
                for k in ("precision", "precision_1", "precision_0", "coverage",
                          "first_error_exact_frac")
            }
    return out


def report(rows: list[dict], thresholds=THRESHOLDS, keep_per_row: bool = False) -> dict:
    out = {"n_rows": len(rows), "thresholds": {}}
    for hi, lo in thresholds:
        m = audit_metrics(rows, hi, lo)
        if not keep_per_row:
            for v in ("rt", "direct"):
                if isinstance(m.get(v), dict):
                    m[v].pop("_per_row", None)
        out["thresholds"][f"{hi}/{lo}"] = m
    return out


def print_report(rep: dict) -> None:
    for name, m in rep["thresholds"].items():
        print(f"[audit] thresholds {name}  (y=0 solutions: {m['rt'].get('n_wrong')}, "
              f"D from {m['rt'].get('n_correct')} y=1 solutions)")
        for v in ("rt", "direct"):
            d = m.get(v)
            if not isinstance(d, dict) or "error" in d:
                continue
            print(f"[audit]   {v:6s} precision={d['precision']:.3f} "
                  f"(1s={d['precision_1']:.3f} n={d['n_pred_1']}, "
                  f"0s={d['precision_0']:.3f} n={d['n_pred_0']}) "
                  f"coverage={d['coverage']:.3f} "
                  f"no_cand={d['no_candidate_frac']:.3f} "
                  f"first_err_exact={d['first_error_exact_frac']:.3f}")
        if "bridge_cost" in m:
            b = m["bridge_cost"]
            print(f"[audit]   bridge cost (direct - rt): precision {b['precision']:+.3f}, "
                  f"coverage {b['coverage']:+.3f}")


# --------------------------------------------------------------------------- CLI


def _merge(rows: list[dict], path: str | None, dst: str, src_candidates: list[str]) -> list[dict]:
    """Merge a side file keyed by id: the first present source field lands in `dst`."""
    if not path:
        return rows
    side = {r["id"]: r for r in load_jsonl(path)}
    for r in rows:
        s = side.get(r["id"])
        if s is None:
            r.setdefault("mask_restore_ok", None)
            continue
        for src in src_candidates:
            if s.get(src) is not None:
                r[dst] = s[src]
                break
        if s.get("mask_restore_ok") is False:
            r["mask_restore_ok"] = False
        elif r.get("mask_restore_ok") is None:
            r["mask_restore_ok"] = s.get("mask_restore_ok", True)
    return rows


def load_audit_rows(
    rows_path: str,
    ko_trans: str | None = None,
    rt_trans: str | None = None,
    rt_teacher: str | None = None,
    direct_teacher: str | None = None,
) -> list[dict]:
    rows = load_jsonl(rows_path)
    _merge(rows, ko_trans, "steps_ko", ["steps_ko"])
    _merge(rows, rt_trans, "steps_en_rt", ["steps_en_rt", "steps_en"])
    _merge(rows, rt_teacher, "teacher_logodds_rt", ["teacher_logodds_rt", "teacher_logodds"])
    _merge(rows, direct_teacher, "teacher_logodds_direct",
           ["teacher_logodds_direct", "teacher_logodds"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True, help="audit rows (already joined, or the base set)")
    ap.add_argument("--ko-trans", default=None)
    ap.add_argument("--rt-trans", default=None)
    ap.add_argument("--rt-teacher", default=None)
    ap.add_argument("--direct-teacher", default=None)
    ap.add_argument("--out", default="data/reports/audit.json")
    ap.add_argument("--dump-rows", default=None, help="also write the joined rows here")
    args = ap.parse_args()

    rows = load_audit_rows(args.rows, args.ko_trans, args.rt_trans, args.rt_teacher,
                           args.direct_teacher)
    if args.dump_rows:
        write_jsonl(args.dump_rows, rows)
    rep = report(rows)
    print_report(rep)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"[audit] wrote {args.out}")


if __name__ == "__main__":
    main()
