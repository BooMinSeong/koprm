"""§7 report assembly on tiny synthetic files: shapes, joins and graceful degradation."""
import json

import pytest

from koprm.io import write_jsonl
from koprm.report import (
    NA,
    ci_half,
    collect,
    first_error_metrics,
    flat_result,
    fmt,
    label_stats,
    metric,
    paired_diff,
    predict_first_error,
    ref_stats,
    render_md,
    teacher_ref_scores,
    trans_stats,
    write_results,
)


def _bon(naive16, weighted16=0.5, ids=("A", "B", "C", "D"), per_naive=(1, 1, 0, 0),
         agg="last"):
    """A minimal bon result: enough of the shape for the report's readers."""
    return {
        "n_problems": len(ids),
        "ns": [1, 16, 64],
        "agg": agg,
        "metrics": {n: {"naive": naive16, "weighted": weighted16, "maj": 0.4, "pass": 0.9}
                    for n in ("1", "16", "64")},
        "per_problem": {"naive@16": list(per_naive), "weighted@16": list(per_naive)},
        "problem_ids": list(ids),
        "ci": {"naive@16": {"mean": naive16, "lo": naive16 - 0.05, "hi": naive16 + 0.05},
               "weighted@16": {"mean": weighted16, "lo": weighted16 - 0.04,
                               "hi": weighted16 + 0.04}},
    }


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- small helpers


def test_flat_result_handles_both_shapes():
    flat = _bon(0.6)
    assert flat_result(flat, "last") is flat
    assert flat_result(flat, "min") is None          # a flat 'last' file has no min
    both = {"last": _bon(0.6), "min": _bon(0.5, agg="min"), "source": "s", "scorer": "x"}
    assert flat_result(both, "last")["metrics"]["16"]["naive"] == 0.6
    assert flat_result(both, "min")["metrics"]["16"]["naive"] == 0.5
    assert flat_result(None, "last") is None


def test_metric_ci_and_formatting():
    res = _bon(0.62, 0.71)
    assert metric(res, "naive", 16) == 0.62
    assert metric(res, "naive", 8) is None
    assert ci_half(res, "naive", 16) == pytest.approx(0.05)
    assert ci_half(res, "naive", 8) is None
    assert fmt(0.6256) == "0.626"
    assert fmt(0.6, 0.05) == "0.600 ±0.050"
    assert fmt(None) == NA


def test_paired_diff_joins_on_problem_ids():
    a = _bon(0.75, per_naive=(1, 1, 1, 0))
    b = _bon(0.50, per_naive=(1, 0, 1, 0), ids=("A", "B", "C", "Z"))
    d = paired_diff(a, b, "naive@16")
    assert d["n_problems"] == 3                      # D/Z are not shared
    assert d["diff"] > 0 and d["lo"] <= d["diff"] <= d["hi"]
    assert paired_diff(a, None, "naive@16") is None
    assert paired_diff(a, b, "maj@16") is None       # key not stored per problem


def test_trans_stats_counts_rows_and_steps(tmp_path):
    p = tmp_path / "steps.jsonl"
    write_jsonl(p, [
        {"id": "1", "steps_en": ["a", "b"], "mask_restore_ok": True,
         "restore_status": ["ok", "passthrough"]},
        {"id": "2", "steps_en": None, "mask_restore_ok": False,
         "restore_status": ["ok", "missing"]},
    ])
    st = trans_stats(p)
    assert st["rows"] == 2 and st["rows_ok"] == 1 and st["rows_ok_frac"] == 0.5
    assert st["steps"] == 4 and st["steps_ok_frac"] == 0.75
    assert st["status_counts"] == {"ok": 2, "passthrough": 1, "missing": 1}

    q = tmp_path / "problems.jsonl"
    write_jsonl(q, [{"id": "1", "problem_ko": "가", "mask_restore_ok": True,
                     "restore_status": "ok"}])
    st = trans_stats(q)
    assert st["steps"] is None and st["steps_ok_frac"] is None and st["rows_ok_frac"] == 1.0
    assert trans_stats(tmp_path / "missing.jsonl") is None


def test_label_stats_composition():
    rows = [
        {"outcome": 1, "generator": "g1", "ceiling": 4.0, "candidates": [],
         "step_labels": [1, 1], "solution_steps": ["가나", "다"],
         "solution_steps_en": ["ab", "c"]},
        {"outcome": 0, "generator": "g1", "ceiling": 3.0, "candidates": [1],
         "step_labels": [1, None, 0], "solution_steps": ["가", "나", "다"],
         "solution_steps_en": ["a", "b", "c"]},
        {"outcome": 0, "generator": "g1", "ceiling": 3.0, "candidates": [],
         "step_labels": [0, 0], "solution_steps": ["가", "나"],
         "solution_steps_en": ["a", "b"]},
    ]
    st = label_stats(rows)
    assert st["n"] == 3 and st["n_wrong"] == 2 and st["n_correct"] == 1
    comp = st["wrong_label_composition"]
    assert comp["1"] == 0.2 and comp["0"] == 0.6 and comp["mask"] == 0.2
    assert st["wrong_no_candidates_frac"] == 0.5
    assert st["per_generator"]["g1"]["n"] == 3
    assert st["steps_per_solution"]["ko"]["q50"] == 2.0
    assert label_stats([]) is None


def test_ref_stats(tmp_path):
    p = tmp_path / "B.refs.json"
    _write_json(p, {"exaone-1.2b": {"b_min": 1.5, "drops": [0.0, 1.0, 2.0, 3.0]}})
    st = ref_stats(p)
    assert st["exaone-1.2b"]["b_min"] == 1.5
    assert st["exaone-1.2b"]["n_drop_steps"] == 4
    assert st["exaone-1.2b"]["drop_q"]["q50"] == 1.5
    assert ref_stats(tmp_path / "nope.json") is None


# ------------------------------------------------------------------- teacher / student bits


def _completion(ans, n_steps=2):
    body = "\n\n".join(f"## 단계 {i + 1}: 설명" for i in range(n_steps - 1))
    return f"{body}\n\n따라서 최종 답은: $\\boxed{{{ans}}}$입니다."


def test_teacher_ref_scores_fills_missing_with_half():
    rows = [{"problem_id": "math500/7", "problem_ko": "p", "answer": "4",
             "completions": [_completion("4"), _completion("5"), _completion("6")]}]
    teacher = {"math500/7#exaone#0": {"teacher_logodds": [0.0, 2.0]},
               "math500/7#exaone#1": {"teacher_logodds": [0.0]}}  # wrong step count
    scores, n_missing = teacher_ref_scores(rows, teacher, n_completions=2)
    assert len(rows[0]["completions"]) == 2          # cut to n_completions
    assert n_missing == 1                            # only completion 1 is unusable
    assert scores[0][0][0] == 0.5 and scores[0][0][1] > 0.8
    assert scores[0][1] == [0.5, 0.5]


def test_predict_first_error_and_metrics():
    assert predict_first_error([0.9, 0.8, 0.2, 0.7]) == 2
    assert predict_first_error([0.9, 0.8]) == -1
    rows = [{"outcome": 0, "human_first_error": 2}, {"outcome": 0, "human_first_error": 2},
            {"outcome": 0, "human_first_error": 2}, {"outcome": 1, "human_first_error": -1}]
    m = first_error_metrics(rows, [2, 3, -1, -1])
    assert m["n_wrong"] == 3 and m["n_correct"] == 1
    assert m["exact_frac"] == 1 / 3
    assert m["within1_frac"] == 2 / 3
    assert m["wrong_no_prediction_frac"] == 1 / 3
    assert m["correct_no_prediction_frac"] == 1.0


# ------------------------------------------------------------------------ full assembly


def _tiny_tree(tmp_path):
    """dev + MATH500 for two runs of one size, plus one translation file and B labels."""
    data = tmp_path / "data"
    _write_json(data / "eval/selection.json",
                {"A_3k": {"epoch": 1, "naive16": 0.4, "weighted16": 0.4, "score": 0.4},
                 "B_3k": {"epoch": 2, "naive16": 0.5, "weighted16": 0.5, "score": 0.5}})
    for run, ep, v in (("A_3k", 1, 0.40), ("B_3k", 2, 0.55)):
        for gen in ("exaone-1.2b", "qwen-3b"):
            _write_json(data / "eval/dev" / f"{run}_ep{ep}_{gen}.json", _bon(v))
        _write_json(data / "eval/math500" / f"{run}_ep{ep}_EXAONE-4.0-1.2B.json",
                    {"last": _bon(v, per_naive=(1, 1, 0, 0) if run == "A_3k" else (1, 1, 1, 0)),
                     "min": _bon(v - 0.02, agg="min"), "source": "s", "scorer": run})
    _write_json(data / "eval/math500/existing_EXAONE-4.0-1.2B_min.json",
                _bon(0.60, agg="min"))
    write_jsonl(data / "trans/problems_ko.jsonl",
                [{"id": "p1", "problem_ko": "가", "mask_restore_ok": True,
                  "restore_status": "ok"}])
    write_jsonl(data / "labels/B.jsonl",
                [{"outcome": 0, "generator": "g1", "ceiling": 3.0, "candidates": [1],
                  "step_labels": [1, 0], "solution_steps": ["가", "나"],
                  "solution_steps_en": ["a", "b"]}])
    write_jsonl(data / "trainsets/B_3k.jsonl", [{"id": i} for i in range(3)])
    return data


def test_collect_and_render(tmp_path):
    data = _tiny_tree(tmp_path)
    res = collect(data)

    assert res["dev"]["B_3k"]["epoch"] == 2
    assert res["dev"]["B_3k"]["exaone-1.2b"]["naive@16"] == 0.55
    assert res["dev"]["A_12k"]["epoch"] is None      # no selection entry -> no numbers
    assert res["dev"]["A_12k"]["exaone-1.2b"]["naive@16"] is None

    ex = res["math500"]["EXAONE-4.0-1.2B"]
    assert ex["B_3k"]["last"]["naive@16"] == 0.55 and ex["B_3k"]["epoch"] == 2
    assert ex["B_3k"]["min"]["naive@16"] == pytest.approx(0.53)
    assert ex["B_3k"]["last_ci"]["naive@16"] == pytest.approx(0.05)
    assert ex["현행 (existing)"]["min"]["naive@16"] == 0.60
    assert ex["현행 (existing)"]["last"]["naive@16"] is None   # only the min file exists
    assert ex["A_12k"]["last"]["naive@16"] is None

    verdict = next(v for v in res["verdicts"]
                   if v["comparison"] == "B_3k - A_3k" and v["metric"] == "naive@16")
    assert verdict["result"]["n_problems"] == 4
    assert verdict["result"]["diff"] == 0.25         # one problem flips
    missing = next(v for v in res["verdicts"] if v["comparison"] == "B_12k - A_12k")
    assert missing["result"] is None

    pipe = res["pipeline"]
    assert pipe["translation"][0]["file"] == "problems_ko.jsonl"
    assert pipe["labels_B"]["n_wrong"] == 1
    assert pipe["labels_A"] is None and pipe["audit"] is None
    assert pipe["trainsets"] == {"B_3k": 3}

    md = render_md(res)
    assert "# 결과 (Plan §7 보고 항목)" in md
    assert "B_3k - A_3k" in md
    assert NA in md                                   # missing runs degrade, never crash
    assert md.count("| run |") >= 1


def test_write_results_creates_both_files(tmp_path):
    data = _tiny_tree(tmp_path)
    out = tmp_path / "reports"
    write_results(data, out)
    assert (out / "results.md").exists() and (out / "results.json").exists()
    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert saved["primary_generator"] == "EXAONE-4.0-1.2B"


def test_collect_on_an_empty_tree(tmp_path):
    res = collect(tmp_path / "nothing")
    assert all(v["result"] is None for v in res["verdicts"])
    assert res["pipeline"]["labels_B"] is None
    md = render_md(res)
    assert NA in md and "## b. KO MATH500" in md
