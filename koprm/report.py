"""§7 보고 항목: assemble the report from the files the pipeline already wrote.

Nothing is recomputed that a stage already stored: this module joins
`data/eval/*`, `data/reports/audit.json`, `data/labels/*`, `data/trans/*` and
`data/trainsets/*` into `reports/results.json` (the numbers) and `reports/results.md`
(the tables). A missing input degrades to "(not available)"; nothing crashes.

The second part, "확장 실험 (2026-09-21)", reads the enlarged nested family
(eval/dev_big, eval/math500_big), the 3B student (eval/dev3b, eval/math500_3b), the
soft+y run on the original 12k set and the four aggregations (eval/agg). Those runs carry
no selection.json, so the epoch is chosen here: argmax over epochs of
(naive@16 + weighted@16)/2 on dev, ties to the later epoch.

The third part, "분포 이동 (2026-09-21)", reads data/shift/{en,aime}/<scorer>_<gen>.json
(bon `all` shape) and data/shift/pb/<scorer>_<lang>.json (koprm.shift first-error), and
puts each scorer's KO MATH500 naive@64 beside the shifted number.

    python -m koprm.report results [--data-dir data] [--out-dir data/reports]
    python -m koprm.report teacher-ref --dataset ENSEONG/ko-ko-math-500-test-EXAONE-4.0-1.2B-bon \\
        --dataset-config <cfg> --teacher data/teacher/math500_exaone16.jsonl \\
        --out data/eval/math500/teacher_bridge_EXAONE-4.0-1.2B.json
    python -m koprm.report student-audit --ckpt data/ckpt/B_12k/ep2 \\
        --rows data/labels/audit_rows.jsonl --device cuda:0

The three pre-registered comparisons (§1) are computed on the primary generator with the
problem-level paired bootstrap: B vs A at each size, A+B vs each, and the 3k/6k/12k curve.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from koprm.io import load_jsonl
from koprm.paths import DATA, REPORTS

NA = "(not available)"
SIZES = ("3k", "6k", "12k")
ARMS = ("A", "B", "AB")
RUNS = tuple(f"{a}_{s}" for a in ARMS for s in SIZES) + ("B_12k_soft", "B_12k_outcome")
ABLATION = ("B_12k", "B_12k_soft", "B_12k_outcome")
DEV_GENS = ("exaone-1.2b", "qwen-3b")
M500_GENS = ("EXAONE-4.0-1.2B", "Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct")
PRIMARY_GEN = M500_GENS[0]
LAST_COLS = (("naive", 16), ("weighted", 16), ("maj", 16),
             ("naive", 64), ("weighted", 64), ("maj", 64))
MIN_COLS = (("naive", 16), ("naive", 64))
VERDICT_METRICS = ("naive@16", "weighted@16")
TRANS_FILES = ("problems_ko.jsonl", "selected.steps_en.jsonl", "prm800k_A.steps_ko.jsonl",
               "audit.steps_ko.jsonl", "audit.steps_en_rt.jsonl",
               "math500_exaone16.steps_en.jsonl")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

# --- 확장 실험 (2026-09-21): the enlarged nested family, the 3B backbone, the aggregations
DEV_GEN = "exaone-1.2b"
BIG_SIZES = ("12k", "24k", "48k")
LABEL_FORMS = (("", "하드(커널)"), ("_soft", "소프트"), ("_soft_y", "소프트+y"),
               ("_outcome", "결과 항만"))
BIG_RUNS = tuple(f"B_{s}{sfx}" for s in BIG_SIZES for sfx, _ in LABEL_FORMS)
Q3B_BASE = ("A_12k", "B_12k", "B_12k_soft", "B_12k_soft_y", "B_12k_outcome")
AGG_RUNS = ("existing", "A_12k", "B_12k", "B_12k_soft", "B_12k_outcome", "AB_12k")
AGGS = ("last", "min", "prod", "mean")
AGG_COLS = (("naive", 16), ("naive", 64), ("weighted", 64))
SOFT_Y_RUN = "B_12k_soft_y"

# --- 분포 이동 (2026-09-21): English MATH500, Korean AIME, Korean ProcessBench
SHIFT_SCORERS = ("prm7b", "B_48k_soft", "B_48k_outcome", "B_48k", "B_12k_soft",
                 "B_12k_outcome", "A_12k")
PB_SCORERS = ("prm7b", "prm72b") + SHIFT_SCORERS[1:]
SHIFT_GENS = {"exaone-1.2b": PRIMARY_GEN, "qwen-3b": M500_GENS[1]}
PB_LANGS = ("ko", "en")
PB_SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")
SHIFT_COLS = (("naive", 16), ("weighted", 16), ("naive", 64), ("weighted", 64))
PRM7B_PUBLISHED_F1 = 73.5  # Qwen2.5-Math-PRM-7B on the full (English) ProcessBench
_EP_RE = re.compile(r"_ep(\d+)_")


# ----------------------------------------------------------------- loading / formatting
def load_json(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    return load_jsonl(p) if p.exists() else []


def flat_result(res: dict | None, agg: str = "last") -> dict | None:
    """One flat bon result out of either output shape (see koprm/eval/bon.py::combine)."""
    if not isinstance(res, dict):
        return None
    if "metrics" in res:
        return res if res.get("agg", agg) == agg else None
    sub = res.get(agg)
    return sub if isinstance(sub, dict) and "metrics" in sub else None


def metric(res: dict | None, method: str, n: int) -> float | None:
    if not res:
        return None
    v = (res.get("metrics") or {}).get(str(n), {}).get(method)
    return float(v) if v is not None else None


def ci_half(res: dict | None, method: str, n: int) -> float | None:
    if not res:
        return None
    c = (res.get("ci") or {}).get(f"{method}@{n}")
    if not c or c.get("lo") is None or c.get("hi") is None:
        return None
    if any(math.isnan(float(c[k])) for k in ("lo", "hi")):
        return None
    return (float(c["hi"]) - float(c["lo"])) / 2


def fmt(value: float | None, half: float | None = None, nd: int = 3) -> str:
    if value is None:
        return NA
    s = f"{value:.{nd}f}"
    return f"{s} ±{half:.{nd}f}" if half is not None else s


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _q(values) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {}
    return {f"q{int(q * 100)}": float(np.quantile(v, q)) for q in QUANTILES}


# ------------------------------------------------------------------------ paired bootstrap
def paired_diff(a: dict | None, b: dict | None, key: str, iters: int = 1000,
                seed: int = 0) -> dict | None:
    """mean(a - b) over the problems both results share, with a 95% bootstrap CI (§7)."""
    if not a or not b:
        return None
    pa, pb = (a.get("per_problem") or {}).get(key), (b.get("per_problem") or {}).get(key)
    ida, idb = a.get("problem_ids"), b.get("problem_ids")
    if not pa or not pb or not ida or not idb:
        return None
    da, db = dict(zip(ida, pa)), dict(zip(idb, pb))
    ids = [i for i in ida if i in db]
    if not ids:
        return None
    d = np.array([float(da[i]) - float(db[i]) for i in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(iters, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n_problems": len(ids)}


def fmt_diff(d: dict | None) -> str:
    if not d:
        return NA
    return f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] (n={d['n_problems']})"


# ------------------------------------------------------------------------ pipeline stats
def trans_stats(path: str | Path) -> dict | None:
    """Placeholder restore rate of one translation file, per row and (if steps) per step."""
    rows = load_rows(path)
    if not rows:
        return None
    n_ok = sum(1 for r in rows if r.get("mask_restore_ok"))
    statuses: Counter = Counter()
    n_units = n_units_ok = 0
    for r in rows:
        st = r.get("restore_status")
        st_list = st if isinstance(st, list) else [st]
        for s in st_list:
            statuses[str(s)] += 1
            n_units += 1
            n_units_ok += int(s in ("ok", "order", "passthrough"))
    is_steps = any(isinstance(r.get("restore_status"), list) for r in rows)
    return {
        "file": Path(path).name,
        "rows": len(rows),
        "rows_ok": n_ok,
        "rows_ok_frac": n_ok / len(rows),
        "steps": n_units if is_steps else None,
        "steps_ok_frac": (n_units_ok / n_units) if (is_steps and n_units) else None,
        "status_counts": dict(statuses),
    }


def label_stats(rows: list[dict]) -> dict | None:
    """§7: kernel label composition on y=0, no-candidate fraction, step counts and lengths."""
    if not rows:
        return None
    wrong = [r for r in rows if int(r.get("outcome", 1)) == 0]
    comp: Counter = Counter()
    for r in wrong:
        for lab in r.get("step_labels") or []:
            comp["mask" if lab is None else str(lab)] += 1
    total = max(sum(comp.values()), 1)
    per_gen: dict[str, dict] = {}
    for g in sorted({r.get("generator", "default") for r in rows}):
        rs = [r for r in rows if r.get("generator", "default") == g]
        per_gen[g] = {
            "n": len(rs),
            "n_correct": sum(1 for r in rs if int(r.get("outcome", 0)) == 1),
            "ceiling_q": _q([r.get("ceiling") for r in rs if int(r.get("outcome", 0)) == 1]),
        }
    ko_steps = [len(r.get("solution_steps") or []) for r in rows]
    en_steps = [len(r.get("solution_steps_en") or []) for r in rows]
    ko_len = [len(s) for r in rows for s in (r.get("solution_steps") or [])]
    en_len = [len(s) for r in rows for s in (r.get("solution_steps_en") or [])]
    return {
        "n": len(rows),
        "n_correct": sum(1 for r in rows if int(r.get("outcome", 0)) == 1),
        "n_wrong": len(wrong),
        "wrong_label_composition": {k: comp[k] / total for k in ("1", "0", "mask")},
        "wrong_no_candidates_frac": (
            sum(1 for r in wrong if not r.get("candidates")) / max(len(wrong), 1)),
        "per_generator": per_gen,
        "steps_per_solution": {"ko": _q(ko_steps), "en": _q(en_steps)},
        "chars_per_step": {"ko": _q(ko_len), "en": _q(en_len)},
    }


def ref_stats(path: str | Path) -> dict | None:
    """b_min and the drop distribution per generator (RefDist.to_dict, §2.3)."""
    refs = load_json(path)
    if not refs:
        return None
    out = {}
    for g, d in refs.items():
        drops = d.get("drops") or []
        out[g] = {"b_min": d.get("b_min"), "n_drop_steps": len(drops), "drop_q": _q(drops)}
    return out


def audit_summary(rep: dict | None) -> dict | None:
    """Precision / coverage / bridge cost per threshold (koprm.label.audit.report)."""
    if not rep or "thresholds" not in rep:
        return None
    out = {}
    for name, m in rep["thresholds"].items():
        entry: dict = {}
        for variant in ("rt", "direct"):
            d = m.get(variant)
            if isinstance(d, dict) and "error" not in d:
                entry[variant] = {k: d.get(k) for k in
                                  ("precision", "precision_1", "precision_0", "coverage",
                                   "no_candidate_frac", "first_error_exact_frac")}
        if isinstance(m.get("bridge_cost"), dict):
            entry["bridge_cost"] = m["bridge_cost"]
        out[name] = entry
    return out


def trainset_sizes(dirpath: str | Path) -> dict | None:
    d = Path(dirpath)
    if not d.exists():
        return None
    return {p.stem: len(load_rows(p)) for p in sorted(d.glob("*.jsonl"))}


# ----------------------------------------------------------------------------- collection
def _run_epoch(selection: dict | None, run: str) -> int | None:
    entry = (selection or {}).get(run)
    return entry.get("epoch") if isinstance(entry, dict) else None


def collect_dev(data: Path, selection: dict | None) -> dict:
    out: dict[str, dict] = {}
    for run in RUNS:
        ep = _run_epoch(selection, run)
        entry: dict = {"epoch": ep}
        for gen in DEV_GENS:
            res = None
            if ep is not None:
                res = flat_result(load_json(data / "eval/dev" / f"{run}_ep{ep}_{gen}.json"))
            entry[gen] = {
                "naive@16": metric(res, "naive", 16),
                "weighted@16": metric(res, "weighted", 16),
            }
        sel = (selection or {}).get(run)
        if isinstance(sel, dict):
            entry["selection_score"] = sel.get("score")
        out[run] = entry
    return out


def _cells(res_last: dict | None, res_min: dict | None) -> dict:
    return {
        "last": {f"{m}@{n}": metric(res_last, m, n) for m, n in LAST_COLS},
        "last_ci": {f"{m}@{n}": ci_half(res_last, m, n) for m, n in LAST_COLS},
        "min": {f"{m}@{n}": metric(res_min, m, n) for m, n in MIN_COLS},
    }


def collect_math500(data: Path, selection: dict | None) -> tuple[dict, dict]:
    """Per generator: the two baselines and every run at its chosen epoch."""
    tables: dict[str, dict] = {}
    raw: dict[str, dict] = {}  # gen -> row name -> flat result (agg last), for the bootstrap
    for gen in M500_GENS:
        d = data / "eval/math500"
        base_min = flat_result(load_json(d / f"existing_{gen}_min.json"), "min")
        base_last = flat_result(load_json(d / f"existing_{gen}_last.json"), "last")
        rows: dict[str, dict] = {
            "현행 (existing)": _cells(base_last, base_min),
        }
        raw[gen] = {"현행 (existing)": base_last}
        for run in RUNS:
            ep = _run_epoch(selection, run)
            res = load_json(d / f"{run}_ep{ep}_{gen}.json") if ep is not None else None
            last, mn = flat_result(res, "last"), flat_result(res, "min")
            cell = _cells(last, mn)
            cell["epoch"] = ep
            rows[run] = cell
            raw[gen][run] = last
        tables[gen] = rows
    return tables, raw


def collect_verdicts(raw_primary: dict[str, dict | None]) -> list[dict]:
    """§1: B vs A, A+B vs both, and the learning curve, on the primary generator."""
    pairs: list[tuple[str, str]] = []
    for s in SIZES:
        pairs += [(f"B_{s}", f"A_{s}"), (f"AB_{s}", f"A_{s}"), (f"AB_{s}", f"B_{s}")]
    for arm in ARMS:
        pairs += [(f"{arm}_12k", f"{arm}_6k"), (f"{arm}_6k", f"{arm}_3k")]
    out = []
    for a, b in pairs:
        for key in VERDICT_METRICS:
            out.append({
                "comparison": f"{a} - {b}",
                "metric": key,
                "result": paired_diff(raw_primary.get(a), raw_primary.get(b), key),
            })
    return out


def collect_pipeline(data: Path) -> dict:
    trans = [t for t in (trans_stats(data / "trans" / f) for f in TRANS_FILES) if t]
    return {
        "translation": trans,
        "labels_B": label_stats(load_rows(data / "labels/B.jsonl")),
        "labels_A": label_stats(load_rows(data / "labels/A.jsonl")),
        "reference_dist": ref_stats(data / "labels/B.refs.json"),
        "audit": audit_summary(load_json(data / "reports/audit.json")),
        "trainsets": trainset_sizes(data / "trainsets"),
    }


def collect_reference_lines(data: Path) -> dict:
    d = data / "eval/math500"
    out: dict[str, dict] = {}
    for gen in M500_GENS:
        entry = {
            "existing_min": metric(
                flat_result(load_json(d / f"existing_{gen}_min.json"), "min"), "naive", 16),
            "existing_last": metric(
                flat_result(load_json(d / f"existing_{gen}_last.json"), "last"), "naive", 16),
        }
        tb = flat_result(load_json(d / f"teacher_bridge_{gen}.json"), "min")
        entry["teacher_bridge_min"] = metric(tb, "naive", 16)
        entry["teacher_bridge_weighted"] = metric(tb, "weighted", 16)
        entry["teacher_bridge_n_missing"] = (tb or {}).get("n_missing")
        out[gen] = entry
    return out


# ------------------------------------------------- 확장 실험: file readers and epoch choice
def _epoch_of(name: str) -> int | None:
    m = _EP_RE.search(name)
    return int(m.group(1)) if m else None


def dev_epoch(dev_dir: str | Path, run: str, gen: str = DEV_GEN) -> tuple[int | None, dict | None]:
    """The epoch with the best (naive@16 + weighted@16)/2 on dev; ties go to the later one."""
    best: tuple[float, int, dict] | None = None
    for path in sorted(Path(dev_dir).glob(f"{run}_ep*_{gen}.json")):
        ep = _epoch_of(path.name)
        res = flat_result(load_json(path))
        n, w = metric(res, "naive", 16), metric(res, "weighted", 16)
        if ep is None or n is None or w is None:
            continue
        score = (n + w) / 2
        if best is None or score > best[0] or (score == best[0] and ep >= best[1]):
            best = (score, ep, res)
    return (None, None) if best is None else (best[1], best[2])


def run_file(dirpath: str | Path, run: str, gen: str, epoch: int | None = None) -> dict | None:
    """The raw json of <run>_ep<epoch>_<gen>.json; falls back to the only file if unique."""
    d = Path(dirpath)
    if epoch is not None:
        path = d / f"{run}_ep{epoch}_{gen}.json"
        if path.exists():
            return load_json(path)
    cands = sorted(d.glob(f"{run}_ep*_{gen}.json"))
    return load_json(cands[0]) if len(cands) == 1 else None


def _gen_cells(raw: dict | None, agg: str = "last") -> dict:
    res = flat_result(raw, agg)
    return {"naive@16": metric(res, "naive", 16), "naive@64": metric(res, "naive", 64),
            "weighted@64": metric(res, "weighted", 64)}


def collect_agg_table(data: Path) -> dict:
    """(g) label form x aggregation, EXAONE only: eval/agg/*.json plus the soft+y run."""
    out: dict[str, dict] = {}
    for name in AGG_RUNS:
        raw = load_json(data / "eval/agg" / f"{name}.json")
        out[name] = {a: _gen_cells(raw, a) for a in AGGS}
    ep, _ = dev_epoch(data / "eval/dev", SOFT_Y_RUN)
    raw = run_file(data / "eval/math500", SOFT_Y_RUN, PRIMARY_GEN, ep)
    out[SOFT_Y_RUN] = {a: _gen_cells(raw, a) for a in AGGS}
    return out


def collect_big(data: Path) -> tuple[dict, list[dict]]:
    """(h) the nested 12k/24k/48k family (1.2B): dev, MATH500 and the size deltas."""
    dev_dir, m500 = data / "eval/dev_big", data / "eval/math500_big"
    rows: dict[str, dict] = {}
    primary: dict[str, dict | None] = {}
    for run in BIG_RUNS:
        ep, dev = dev_epoch(dev_dir, run)
        entry: dict = {
            "epoch": ep,
            "dev": {"naive@16": metric(dev, "naive", 16),
                    "weighted@16": metric(dev, "weighted", 16)},
        }
        for gen in (PRIMARY_GEN, M500_GENS[1]):
            raw = run_file(m500, run, gen, ep)
            entry[gen] = _gen_cells(raw, "last")
            if gen == PRIMARY_GEN:
                primary[run] = flat_result(raw, "last")
        rows[run] = entry

    deltas = []
    for sfx, label in LABEL_FORMS:
        for a, b in (("24k", "12k"), ("48k", "24k")):
            for key in ("naive@16", "naive@64"):
                deltas.append({
                    "label_form": label,
                    "comparison": f"B_{a}{sfx} - B_{b}{sfx}",
                    "metric": key,
                    "result": paired_diff(primary.get(f"B_{a}{sfx}"),
                                          primary.get(f"B_{b}{sfx}"), key),
                })
    return rows, deltas


def collect_backbone(data: Path, selection: dict | None) -> dict:
    """(i) 3B vs 1.2B student on the original 12k sets."""
    out: dict[str, dict] = {}
    for run in Q3B_BASE:
        ep12 = _run_epoch(selection, run)
        if ep12 is None:
            ep12, _ = dev_epoch(data / "eval/dev", run)
        ep3b, _ = dev_epoch(data / "eval/dev3b", f"q3b_{run}")
        entry: dict = {"epoch_1.2b": ep12, "epoch_3b": ep3b}
        for tag, (d, name, ep) in {
            "1.2b": (data / "eval/math500", run, ep12),
            "3b": (data / "eval/math500_3b", f"q3b_{run}", ep3b),
        }.items():
            entry[tag] = {
                PRIMARY_GEN: _gen_cells(run_file(d, name, PRIMARY_GEN, ep), "last"),
                M500_GENS[1]: _gen_cells(run_file(d, name, M500_GENS[1], ep), "last"),
            }
        out[run] = entry
    return out


def collect_soft_y(data: Path, selection: dict | None) -> dict:
    """(j) soft+y vs soft vs outcome on the original 12k set."""
    out: dict[str, dict] = {}
    for run in (SOFT_Y_RUN, "B_12k_soft", "B_12k_outcome"):
        ep = _run_epoch(selection, run)
        if ep is None:
            ep, _ = dev_epoch(data / "eval/dev", run)
        entry: dict = {"epoch": ep}
        for gen in (PRIMARY_GEN, M500_GENS[1]):
            entry[gen] = _gen_cells(run_file(data / "eval/math500", run, gen, ep), "last")
        out[run] = entry
    return out


# ------------------------------------------------------------------- 분포 이동: readers
def ko_naive64(data: Path, scorer: str, gen: str, selection: dict | None) -> float | None:
    """The same scorer's KO MATH500 naive@64 (last), for the side-by-side column."""
    if scorer == "prm7b":  # the 현행 baseline: the English PRM on the Korean solutions
        return metric(flat_result(
            load_json(data / "eval/math500" / f"existing_{gen}_last.json"), "last"), "naive", 64)
    if scorer.startswith("B_48k"):
        ep, _ = dev_epoch(data / "eval/dev_big", scorer)
        return metric(flat_result(run_file(data / "eval/math500_big", scorer, gen, ep), "last"),
                      "naive", 64)
    ep = _run_epoch(selection, scorer)
    if ep is None:
        ep, _ = dev_epoch(data / "eval/dev", scorer)
    return metric(flat_result(run_file(data / "eval/math500", scorer, gen, ep), "last"),
                  "naive", 64)


def collect_bon_shift(data: Path, subdir: str, selection: dict | None,
                      with_pass: bool = False) -> dict:
    """(k)/(l) one table per generator: every scorer on the shifted benchmark."""
    out: dict[str, dict] = {}
    for gen, gen_full in SHIFT_GENS.items():
        rows: dict[str, dict] = {}
        for scorer in SHIFT_SCORERS:
            raw = load_json(data / "shift" / subdir / f"{scorer}_{gen}.json")
            last, mean = flat_result(raw, "last"), flat_result(raw, "mean")
            cell = {f"{m}@{n}": metric(last, m, n) for m, n in SHIFT_COLS}
            cell["mean naive@64"] = metric(mean, "naive", 64)
            cell["KO naive@64"] = ko_naive64(data, scorer, gen_full, selection)
            if with_pass:
                cell["pass@1"] = metric(last, "pass", 1)
                cell["pass@64"] = metric(last, "pass", 64)
            rows[scorer] = cell
        out[gen] = rows
    return out


def _pb_parts(raw: dict | None) -> tuple[dict, dict]:
    """(overall, per split) from a koprm.shift first-error json, whatever the split key is."""
    if not isinstance(raw, dict):
        return {}, {}
    for key in ("by_split", "splits", "per_split"):
        if isinstance(raw.get(key), dict):
            return raw.get("overall") or {}, raw[key]
    return raw.get("overall") or {}, {}


def collect_processbench(data: Path) -> dict:
    """(m) first-error detection on Korean and English ProcessBench."""
    counts: Counter = Counter()
    for r in load_rows(data / "shift/pb_rows.jsonl"):
        counts[r.get("split", "unknown")] += 1
    splits = [s for s in PB_SPLITS if s in counts] or sorted(counts)
    scorers: dict[str, dict] = {}
    for scorer in PB_SCORERS:
        per_lang: dict[str, dict] = {}
        for lang in PB_LANGS:
            raw = load_json(data / "shift/pb" / f"{scorer}_{lang}.json")
            overall, by_split = _pb_parts(raw)
            per_lang[lang] = {
                "err_acc": overall.get("err_acc"),
                "corr_acc": overall.get("corr_acc"),
                "f1": overall.get("f1"),
                "within1": overall.get("within1"),
                "n_error_rows": overall.get("n_error_rows"),
                "n_correct_rows": overall.get("n_correct_rows"),
                "n_scored": (raw or {}).get("n_scored"),
                "split_f1": {s: (by_split.get(s) or {}).get("f1") for s in splits},
            }
        scorers[scorer] = per_lang
    return {"rows_by_split": dict(counts), "splits": splits, "scorers": scorers,
            "prm7b_published_f1": PRM7B_PUBLISHED_F1}


def collect_shift(data: Path, selection: dict | None) -> dict:
    return {
        "en_math500": collect_bon_shift(data, "en", selection),
        "aime": collect_bon_shift(data, "aime", selection, with_pass=True),
        "processbench": collect_processbench(data),
    }


def collect_extended(data: Path, selection: dict | None) -> dict:
    big, deltas = collect_big(data)
    return {
        "agg_table": collect_agg_table(data),
        "big": big,
        "big_deltas": deltas,
        "backbone": collect_backbone(data, selection),
        "soft_y": collect_soft_y(data, selection),
    }


def collect(data: Path) -> dict:
    selection = load_json(data / "eval/selection.json")
    tables, raw = collect_math500(data, selection)
    return {
        "dev": collect_dev(data, selection),
        "math500": tables,
        "verdicts": collect_verdicts(raw.get(PRIMARY_GEN, {})),
        "pipeline": collect_pipeline(data),
        "reference_lines": collect_reference_lines(data),
        "primary_generator": PRIMARY_GEN,
        "extended": collect_extended(data, selection),
        "shift": collect_shift(data, selection),
    }


# ------------------------------------------------------------------------------ rendering
def _dev_md(dev: dict) -> str:
    header = ["run", "epoch", "naive@16 (EXAONE)", "weighted@16 (EXAONE)",
              "naive@16 (qwen-3b)", "weighted@16 (qwen-3b)"]
    rows = []
    for run in RUNS:
        e = dev.get(run, {})
        ex, qw = e.get("exaone-1.2b", {}), e.get("qwen-3b", {})
        rows.append([run, str(e.get("epoch") if e.get("epoch") is not None else NA),
                     fmt(ex.get("naive@16")), fmt(ex.get("weighted@16")),
                     fmt(qw.get("naive@16")), fmt(qw.get("weighted@16"))])
    return md_table(header, rows)


def _m500_md(table: dict) -> str:
    header = (["run", "epoch"] + [f"{m}@{n}" for m, n in LAST_COLS]
              + [f"min {m}@{n}" for m, n in MIN_COLS])
    rows = []
    for name, cell in table.items():
        ep = cell.get("epoch")
        r = [name, str(ep) if ep is not None else "-"]
        r += [fmt(cell["last"][f"{m}@{n}"], cell["last_ci"].get(f"{m}@{n}"))
              for m, n in LAST_COLS]
        r += [fmt(cell["min"][f"{m}@{n}"]) for m, n in MIN_COLS]
        rows.append(r)
    return md_table(header, rows)


def _curve_md(res: dict) -> str:
    table = res["math500"].get(PRIMARY_GEN, {})
    dev = res["dev"]
    header = ["arm", "size", "MATH500 naive@16", "MATH500 weighted@16",
              "dev naive@16", "dev weighted@16"]
    rows = []
    for arm in ARMS:
        for size in SIZES:
            run = f"{arm}_{size}"
            cell = table.get(run, {})
            d = dev.get(run, {}).get("exaone-1.2b", {})
            rows.append([arm, size,
                         fmt((cell.get("last") or {}).get("naive@16")),
                         fmt((cell.get("last") or {}).get("weighted@16")),
                         fmt(d.get("naive@16")), fmt(d.get("weighted@16"))])
    return md_table(header, rows)


def _ablation_md(res: dict) -> str:
    table = res["math500"].get(PRIMARY_GEN, {})
    header = ["run"] + [f"{m}@{n}" for m, n in LAST_COLS] + ["min naive@16"]
    rows = []
    for run in ABLATION:
        cell = table.get(run, {})
        last, mn = cell.get("last") or {}, cell.get("min") or {}
        rows.append([run] + [fmt(last.get(f"{m}@{n}")) for m, n in LAST_COLS]
                    + [fmt(mn.get("naive@16"))])
    return md_table(header, rows)


def _pipeline_md(p: dict) -> list[str]:
    out: list[str] = ["### e. 파이프라인 통계", "", "**자리표시자 복원**", ""]
    trans = p.get("translation") or []
    if trans:
        rows = [[t["file"], str(t["rows"]), fmt(t["rows_ok_frac"]),
                 str(t["steps"]) if t["steps"] is not None else "-",
                 fmt(t["steps_ok_frac"]) if t["steps_ok_frac"] is not None else "-",
                 ", ".join(f"{k}={v}" for k, v in sorted(t["status_counts"].items()))]
                for t in trans]
        out.append(md_table(["file", "rows", "rows ok", "steps", "steps ok", "statuses"], rows))
    else:
        out.append(NA)

    for arm, key in (("B", "labels_B"), ("A", "labels_A")):
        st = p.get(key)
        out += ["", f"**{arm} 라벨**", ""]
        if not st:
            out.append(NA)
            continue
        comp = st["wrong_label_composition"]
        out.append(f"- 풀이 {st['n']} (정답 {st['n_correct']}, 오답 {st['n_wrong']})")
        out.append(f"- y=0 스텝 라벨 구성비: 1={comp['1']:.3f} 0={comp['0']:.3f} "
                   f"mask={comp['mask']:.3f}")
        out.append(f"- 후보 없는 오답 풀이 비율: {st['wrong_no_candidates_frac']:.3f}")
        sps, cps = st["steps_per_solution"], st["chars_per_step"]
        out.append(f"- 스텝 수(중앙값) 한국어 {fmt(sps['ko'].get('q50'), nd=1)} / "
                   f"영어 {fmt(sps['en'].get('q50'), nd=1)}; "
                   f"스텝 길이(중앙값, 자) 한국어 {fmt(cps['ko'].get('q50'), nd=1)} / "
                   f"영어 {fmt(cps['en'].get('q50'), nd=1)}")
        for g, d in st["per_generator"].items():
            out.append(f"- {g}: 풀이 {d['n']}, 천장 분포 "
                       f"{ {k: round(v, 3) for k, v in d['ceiling_q'].items()} }")

    refs = p.get("reference_dist")
    out += ["", "**기준 분포 D_g**", ""]
    if refs:
        rows = [[g, fmt(d.get("b_min")), str(d.get("n_drop_steps")),
                 ", ".join(f"{k}={v:.3f}" for k, v in d.get("drop_q", {}).items())]
                for g, d in refs.items()]
        out.append(md_table(["generator", "b_min", "drop steps", "하강폭 분위수"], rows))
    else:
        out.append(NA)

    aud = p.get("audit")
    out += ["", "**감사 세트**", ""]
    if aud:
        rows = []
        for name, m in aud.items():
            rt, direct = m.get("rt") or {}, m.get("direct") or {}
            bridge = m.get("bridge_cost") or {}
            rows.append([name, fmt(rt.get("precision")), fmt(rt.get("coverage")),
                         fmt(direct.get("precision")), fmt(direct.get("coverage")),
                         fmt(bridge.get("precision")), fmt(rt.get("first_error_exact_frac"))])
        out.append(md_table(["임계", "rt 정밀도", "rt 커버리지", "direct 정밀도",
                             "direct 커버리지", "다리 비용(정밀도)", "첫 오류 정확 일치"], rows))
    else:
        out.append(NA)

    ts = p.get("trainsets")
    out += ["", "**학습 세트 크기**", ""]
    out.append(md_table(["trainset", "rows"], [[k, str(v)] for k, v in ts.items()])
               if ts else NA)
    return out


def _reference_md(ref: dict) -> str:
    header = ["generator", "현행 min naive@16", "현행 last naive@16",
              "교사 다리 min naive@16", "교사 다리 min weighted@16", "교사 결측"]
    rows = [[gen, fmt(e.get("existing_min")), fmt(e.get("existing_last")),
             fmt(e.get("teacher_bridge_min")), fmt(e.get("teacher_bridge_weighted")),
             str(e.get("teacher_bridge_n_missing")) if e.get("teacher_bridge_n_missing")
             is not None else NA]
            for gen, e in ref.items()]
    return md_table(header, rows)


def _agg_md(table: dict) -> str:
    header = ["scorer"] + [f"{a} {m}@{n}" for a in AGGS for m, n in AGG_COLS]
    rows = []
    for name, per_agg in table.items():
        r = [name]
        for a in AGGS:
            cell = per_agg.get(a) or {}
            r += [fmt(cell.get(f"{m}@{n}")) for m, n in AGG_COLS]
        rows.append(r)
    return md_table(header, rows)


def _big_md(big: dict) -> str:
    header = ["라벨 형식", "크기", "epoch", "dev naive@16", "dev weighted@16",
              "EXAONE naive@16", "EXAONE naive@64", "EXAONE weighted@64",
              "Qwen-3B naive@64", "Qwen-3B weighted@64"]
    rows = []
    for sfx, label in LABEL_FORMS:
        for size in BIG_SIZES:
            e = big.get(f"B_{size}{sfx}", {})
            dev = e.get("dev") or {}
            ex, qw = e.get(PRIMARY_GEN) or {}, e.get(M500_GENS[1]) or {}
            rows.append([label, size,
                         str(e.get("epoch")) if e.get("epoch") is not None else NA,
                         fmt(dev.get("naive@16")), fmt(dev.get("weighted@16")),
                         fmt(ex.get("naive@16")), fmt(ex.get("naive@64")),
                         fmt(ex.get("weighted@64")),
                         fmt(qw.get("naive@64")), fmt(qw.get("weighted@64"))])
    return md_table(header, rows)


def _backbone_md(backbone: dict) -> str:
    header = ["라벨 형식(12k)", "학생", "epoch", "EXAONE naive@16", "EXAONE naive@64",
              "EXAONE weighted@64", "Qwen-3B naive@64"]
    rows = []
    for run, e in backbone.items():
        for tag, size in (("1.2b", "EXAONE-4.0-1.2B"), ("3b", "Qwen2.5-3B-Instruct")):
            cells = e.get(tag) or {}
            ex, qw = cells.get(PRIMARY_GEN) or {}, cells.get(M500_GENS[1]) or {}
            ep = e.get(f"epoch_{tag}")
            rows.append([run, size, str(ep) if ep is not None else NA,
                         fmt(ex.get("naive@16")), fmt(ex.get("naive@64")),
                         fmt(ex.get("weighted@64")), fmt(qw.get("naive@64"))])
    return md_table(header, rows)


def _soft_y_md(soft_y: dict) -> str:
    header = ["run", "epoch", "EXAONE naive@16", "EXAONE naive@64", "EXAONE weighted@64",
              "Qwen-3B naive@64", "Qwen-3B weighted@64"]
    rows = []
    for run, e in soft_y.items():
        ex, qw = e.get(PRIMARY_GEN) or {}, e.get(M500_GENS[1]) or {}
        rows.append([run, str(e.get("epoch")) if e.get("epoch") is not None else NA,
                     fmt(ex.get("naive@16")), fmt(ex.get("naive@64")),
                     fmt(ex.get("weighted@64")),
                     fmt(qw.get("naive@64")), fmt(qw.get("weighted@64"))])
    return md_table(header, rows)


def render_extended_md(ext: dict) -> list[str]:
    out = ["## 확장 실험 (2026-09-21)", "",
           "에폭은 dev의 (naive@16 + weighted@16)/2가 가장 큰 것을 쓴다(동점이면 나중 에폭).",
           "", "### g. 라벨 형식 × 집계 (EXAONE)", "", _agg_md(ext["agg_table"]), "",
           "### h. 학습 곡선 12k/24k/48k (늘린 B 풀, 학생 1.2B)", "", _big_md(ext["big"]), "",
           "크기 간 차이(EXAONE, 문제 단위 대응 부트스트랩 95% CI):", ""]
    out.append(md_table(["라벨 형식", "비교", "지표", "차이 [CI]"],
                        [[d["label_form"], d["comparison"], d["metric"], fmt_diff(d["result"])]
                         for d in ext["big_deltas"]]))
    out += ["", "### i. 학생 백본 비교 (원래 12k 세트)", "", _backbone_md(ext["backbone"]), "",
            "### j. 소프트+y 대 소프트 대 결과 항만 (원래 12k 세트)", "",
            _soft_y_md(ext["soft_y"]), ""]
    return out


def _shift_bon_md(table: dict, with_pass: bool = False) -> list[str]:
    cols = [f"{m}@{n}" for m, n in SHIFT_COLS] + ["mean naive@64", "KO naive@64"]
    if with_pass:
        cols += ["pass@1", "pass@64"]
    out: list[str] = []
    for gen, rows in table.items():
        out += [f"**생성기 {gen}**", ""]
        out.append(md_table(["scorer"] + cols,
                            [[name] + [fmt((cell or {}).get(c)) for c in cols]
                             for name, cell in rows.items()]))
        out.append("")
    return out


def _pb_md(pb: dict) -> list[str]:
    splits = pb["splits"]
    header = ["scorer", "언어", "err", "corr", "F1", "±1"] + [f"F1 {s}" for s in splits]
    rows = []
    for scorer, per_lang in pb["scorers"].items():
        for lang in PB_LANGS:
            m = per_lang.get(lang) or {}
            rows.append([scorer, lang, fmt(m.get("err_acc")), fmt(m.get("corr_acc")),
                         fmt(m.get("f1")), fmt(m.get("within1"))]
                        + [fmt((m.get("split_f1") or {}).get(s)) for s in splits])
    counts = pb["rows_by_split"]
    n = sum(counts.values())
    out = [md_table(header, rows), "",
           f"행 수: {n}" + (" (" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                            + ")" if counts else ""), ""]
    ref = (pb["scorers"].get("prm7b") or {}).get("en", {}).get("f1")
    out.append(f"참고: Qwen2.5-Math-PRM-7B의 공개 ProcessBench F1은 전체 벤치마크에서 "
               f"약 {pb['prm7b_published_f1']}이다. 같은 모델이 이 영어 표본에서 내는 값은 "
               f"{fmt(ref)}(F1 비율)이므로, 표본과 채점 방식의 차이를 감안해 읽는다.")
    return out


def render_shift_md(shift: dict) -> list[str]:
    out = ["## 분포 이동 (2026-09-21)", "",
           ("한국어로 학습한 학생을 영어 풀이, 한국어 AIME, 한국어 ProcessBench에 그대로 "
            "얹었다. 현행(prm7b)은 Qwen2.5-Math-PRM-7B를 풀이에 직접 적용한 값이다."),
           "", "### k. 영어 MATH500 BoN (한국어 템플릿으로 채점)", ""]
    out += _shift_bon_md(shift["en_math500"])
    out += ["### l. 한국어 AIME 2023/2024 (56문제 × 64샘플)", ""]
    out += _shift_bon_md(shift["aime"], with_pass=True)
    out += ["### m. 한국어·영어 ProcessBench (첫 오류 식별)", ""]
    out += _pb_md(shift["processbench"])
    return out


def render_md(res: dict) -> str:
    out = ["# 결과 (Plan §7 보고 항목)", "",
           f"주 평가 생성기: {res['primary_generator']}. 값이 없는 칸은 {NA}다.", "",
           "## a. dev 선택", "", _dev_md(res["dev"]), ""]
    out += ["## b. KO MATH500", ""]
    for gen in M500_GENS:
        out += [f"### {gen}", "", _m500_md(res["math500"].get(gen, {})), ""]
    out += ["### 사전 등록된 비교 (§1, 문제 단위 대응 부트스트랩 95% CI)", ""]
    out.append(md_table(["비교", "지표", "차이 [CI]"],
                        [[v["comparison"], v["metric"], fmt_diff(v["result"])]
                         for v in res["verdicts"]]))
    out += ["", "## c. 학습 곡선", "", _curve_md(res), ""]
    out += ["## d. 절제", "", _ablation_md(res), ""]
    out += _pipeline_md(res["pipeline"])
    out += ["", "### f. 참조선", "", _reference_md(res["reference_lines"]), ""]
    if res.get("extended"):
        out += render_extended_md(res["extended"])
    if res.get("shift"):
        out += render_shift_md(res["shift"])
    return "\n".join(out)


def write_results(data: Path, out_dir: Path) -> dict:
    res = collect(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(render_md(res), encoding="utf-8")
    print(f"[report] wrote {out_dir / 'results.md'} and {out_dir / 'results.json'}")
    return res


# -------------------------------------------------------------- teacher reference line
def sigmoid(z) -> list[float]:
    return [float(1.0 / (1.0 + math.exp(-float(v)))) for v in z]


def teacher_ref_scores(rows: list[dict], teacher_by_id: dict[str, dict],
                       n_completions: int = 16, generator: str = "exaone",
                       ) -> tuple[list[list[list[float]]], int]:
    """Per-step sigmoid(z) for the first `n_completions` completions; 0.5 where z is unusable."""
    from koprm.eval.bon import split_steps

    scores: list[list[list[float]]] = []
    n_missing = 0
    for r in rows:
        r["completions"] = list(r["completions"])[:n_completions]
        per_row = []
        for k, c in enumerate(r["completions"]):
            n_steps = len(split_steps(c))
            pid = r["problem_id"]
            t = (teacher_by_id.get(f"{pid}#{generator}#{k}")
                 or teacher_by_id.get(f"math500/{pid}#{generator}#{k}"))
            z = (t or {}).get("teacher_logodds")
            if z is None or len(z) != n_steps:
                n_missing += 1
                per_row.append([0.5] * n_steps)
            else:
                per_row.append(sigmoid(z))
        scores.append(per_row)
    return scores, n_missing


def teacher_ref(dataset: str, config: str, split: str, teacher_path: str, out_path: str,
                limit: int | None = None, n_completions: int = 16,
                timeout: float = 3.0) -> dict:
    import os

    os.environ["HF_HUB_OFFLINE"] = "0"  # bon.load_hf_rows setdefaults it to 1
    from koprm.eval import bon

    rows = bon.load_hf_rows(dataset, config, split, limit)
    teacher_by_id = {r["id"]: r for r in load_rows(teacher_path)}
    scores, n_missing = teacher_ref_scores(rows, teacher_by_id, n_completions)
    res = bon.evaluate(rows, scores, agg="min", timeout=timeout)  # §7: teacher uses min
    res.update({"source": dataset, "scorer": "teacher-bridge", "n_missing": n_missing,
                "n_completions": n_completions})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    bon.print_metrics(res, dataset, "teacher-bridge")
    print(f"[report] teacher steps replaced by 0.5: {n_missing}; wrote {out_path}")
    return res


# ------------------------------------------------------- student first-error audit (§7)
def predict_first_error(probs: list[float], threshold: float = 0.5) -> int:
    """Index of the first step scored below `threshold`; -1 when the student flags none."""
    for i, p in enumerate(probs):
        if float(p) < threshold:
            return i
    return -1


def first_error_metrics(rows: list[dict], preds: list[int]) -> dict:
    wrong = [(r, p) for r, p in zip(rows, preds) if int(r.get("outcome", 1)) == 0]
    correct = [(r, p) for r, p in zip(rows, preds) if int(r.get("outcome", 0)) == 1]
    exact = within1 = none_pred = 0
    for r, p in wrong:
        human = r.get("human_first_error")
        if human is None or human < 0 or p < 0:
            none_pred += int(p < 0)
            continue
        exact += int(p == human)
        within1 += int(abs(p - human) <= 1)
    n_wrong = max(len(wrong), 1)
    return {
        "n_rows": len(rows),
        "n_wrong": len(wrong),
        "n_correct": len(correct),
        "exact_frac": exact / n_wrong,
        "within1_frac": within1 / n_wrong,
        "wrong_no_prediction_frac": none_pred / n_wrong,
        "correct_no_prediction_frac": (
            sum(1 for _, p in correct if p < 0) / max(len(correct), 1)),
    }


def student_audit(ckpt: str, rows_path: str, out_path: str, device: str = "cpu",
                  batch_size: int = 8, max_len: int = 4096, threshold: float = 0.5,
                  steps_field: str = "steps_ko") -> dict:
    from koprm.eval.scorer import StudentScorer

    rows = [r for r in load_rows(rows_path) if r.get(steps_field) and r.get("step_labels")]
    scorer = StudentScorer(ckpt, device=device, max_len=max_len)
    # The audit rows carry no problem_ko (they are PRM800K problems, §4.4), so the English
    # problem text is used as the question; the Korean solution steps are the scored part.
    probs = scorer.score([r["problem_en"] for r in rows],
                         [list(r[steps_field]) for r in rows], batch_size=batch_size)
    preds = [predict_first_error(p, threshold) for p in probs]
    res = first_error_metrics(rows, preds)
    res.update({"ckpt": ckpt, "rows": rows_path, "threshold": threshold,
                "steps_field": steps_field, "n_truncated": scorer.n_truncated,
                "question_text": "problem_en (the audit rows have no Korean problem text)"})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] first error exact={res['exact_frac']:.3f} "
          f"±1={res['within1_frac']:.3f} (y=0 풀이 {res['n_wrong']}개); wrote {out_path}")
    return res


# ------------------------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("results", help="assemble reports/results.{md,json}")
    p.add_argument("--data-dir", default=str(DATA))
    p.add_argument("--out-dir", default=str(REPORTS))

    p = sub.add_parser("teacher-ref", help="teacher-bridge reference line (§7)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-config", default="default")
    p.add_argument("--split", default="train")
    p.add_argument("--teacher", required=True, help="teacher jsonl for the first completions")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-completions", type=int, default=16)

    p = sub.add_parser("student-audit", help="first-error accuracy on the audit set (§7)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--rows", default="data/labels/audit_rows.jsonl")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--steps-field", default="steps_ko")

    args = ap.parse_args()
    if args.cmd == "results":
        write_results(Path(args.data_dir), Path(args.out_dir))
    elif args.cmd == "teacher-ref":
        teacher_ref(args.dataset, args.dataset_config, args.split, args.teacher, args.out,
                    limit=args.limit, n_completions=args.n_completions)
    else:
        out = args.out or str(REPORTS / f"student_audit_{Path(args.ckpt).name}.json")
        student_audit(args.ckpt, args.rows, out, device=args.device,
                      batch_size=args.batch_size, max_len=args.max_len,
                      threshold=args.threshold, steps_field=args.steps_field)


if __name__ == "__main__":
    main()
