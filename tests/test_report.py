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


# ------------------------------------------------------------- 확장 실험 (2026-09-21)


def _all_shape(v, per_naive=(1, 1, 0, 0)):
    """An --agg all file: last/min/prod/mean, each a flat result."""
    return {a: _bon(v - i * 0.01, per_naive=per_naive, agg=a)
            for i, a in enumerate(("last", "min", "prod", "mean"))}


def test_dev_epoch_picks_the_best_mean_and_breaks_ties_late(tmp_path):
    from koprm.report import dev_epoch

    for ep, (n, w) in enumerate([(0.40, 0.60), (0.55, 0.60), (0.50, 0.65)], start=1):
        _write_json(tmp_path / f"B_24k_ep{ep}_exaone-1.2b.json", _bon(n, w))
    ep, res = dev_epoch(tmp_path, "B_24k")
    assert ep == 3                               # (0.50+0.65)/2 ties ep2, later epoch wins
    assert res["metrics"]["16"]["weighted"] == 0.65
    assert dev_epoch(tmp_path, "B_48k") == (None, None)


def test_run_file_prefers_the_selected_epoch(tmp_path):
    from koprm.report import run_file

    _write_json(tmp_path / "B_24k_ep2_EXAONE-4.0-1.2B.json", _all_shape(0.50))
    _write_json(tmp_path / "B_24k_ep3_EXAONE-4.0-1.2B.json", _all_shape(0.70))
    got = run_file(tmp_path, "B_24k", "EXAONE-4.0-1.2B", 3)
    assert got["last"]["metrics"]["16"]["naive"] == 0.70
    # no epoch match and more than one candidate -> nothing rather than a guess
    assert run_file(tmp_path, "B_24k", "EXAONE-4.0-1.2B", 9) is None
    # a single file is used whatever the epoch asked for
    _write_json(tmp_path / "B_48k_ep1_EXAONE-4.0-1.2B.json", _all_shape(0.60))
    assert run_file(tmp_path, "B_48k", "EXAONE-4.0-1.2B", 3)["last"]["agg"] == "last"


def _extended_tree(tmp_path):
    data = tmp_path / "data"
    # (g) the four aggregations for two scorers
    for name, v in (("existing", 0.66), ("B_12k", 0.52)):
        _write_json(data / "eval/agg" / f"{name}.json", _all_shape(v))
    # the soft+y run on the original 12k set: dev picks the epoch, math500 carries it
    _write_json(data / "eval/dev/B_12k_soft_y_ep1_exaone-1.2b.json", _bon(0.60, 0.68))
    _write_json(data / "eval/dev/B_12k_soft_y_ep2_exaone-1.2b.json", _bon(0.64, 0.68))
    for gen in ("EXAONE-4.0-1.2B", "Qwen2.5-3B-Instruct"):
        _write_json(data / "eval/math500" / f"B_12k_soft_y_ep2_{gen}.json", _all_shape(0.71))
    # (h) two sizes of one label form
    for size, v, per in (("12k", 0.60, (1, 1, 0, 0)), ("24k", 0.70, (1, 1, 1, 0))):
        _write_json(data / "eval/dev_big" / f"B_{size}_ep3_exaone-1.2b.json", _bon(v))
        for gen in ("EXAONE-4.0-1.2B", "Qwen2.5-3B-Instruct"):
            _write_json(data / "eval/math500_big" / f"B_{size}_ep3_{gen}.json",
                        _all_shape(v, per_naive=per))
    # (i) the 3B student on the original B_12k set
    _write_json(data / "eval/dev3b/q3b_B_12k_ep2_exaone-1.2b.json", _bon(0.66, 0.70))
    _write_json(data / "eval/math500_3b/q3b_B_12k_ep2_EXAONE-4.0-1.2B.json", _all_shape(0.74))
    _write_json(data / "eval/math500/B_12k_ep3_EXAONE-4.0-1.2B.json",
                {"last": _bon(0.61), "min": _bon(0.62, agg="min")})
    _write_json(data / "eval/selection.json", {"B_12k": {"epoch": 3}})
    return data


def test_collect_extended_reads_every_part(tmp_path):
    from koprm.report import collect_extended, load_json

    data = _extended_tree(tmp_path)
    ext = collect_extended(data, load_json(data / "eval/selection.json"))

    agg = ext["agg_table"]
    assert agg["existing"]["last"]["naive@16"] == 0.66
    assert agg["existing"]["mean"]["naive@16"] == pytest.approx(0.63)
    assert agg["A_12k"]["last"]["naive@16"] is None          # file absent
    assert agg["B_12k_soft_y"]["last"]["naive@16"] == 0.71   # epoch 2 chosen on dev

    big = ext["big"]
    assert big["B_12k"]["epoch"] == 3 and big["B_24k"]["dev"]["naive@16"] == 0.70
    assert big["B_24k"]["EXAONE-4.0-1.2B"]["naive@64"] == 0.70
    assert big["B_24k"]["Qwen2.5-3B-Instruct"]["weighted@64"] == 0.5
    assert big["B_48k"]["epoch"] is None                     # no files for that size
    assert big["B_12k_soft"]["EXAONE-4.0-1.2B"]["naive@16"] is None

    d = next(x for x in ext["big_deltas"]
             if x["comparison"] == "B_24k - B_12k" and x["metric"] == "naive@16")
    assert d["label_form"] == "하드(커널)" and d["result"]["diff"] == 0.25
    assert next(x for x in ext["big_deltas"]
                if x["comparison"] == "B_48k - B_24k")["result"] is None

    bb = ext["backbone"]["B_12k"]
    assert bb["epoch_1.2b"] == 3 and bb["epoch_3b"] == 2
    assert bb["1.2b"][PRIMARY := "EXAONE-4.0-1.2B"]["naive@16"] == 0.61
    assert bb["3b"][PRIMARY]["naive@16"] == 0.74
    assert ext["backbone"]["A_12k"]["1.2b"][PRIMARY]["naive@16"] is None

    sy = ext["soft_y"]["B_12k_soft_y"]
    assert sy["epoch"] == 2 and sy["EXAONE-4.0-1.2B"]["weighted@64"] == 0.5
    assert ext["soft_y"]["B_12k_outcome"]["EXAONE-4.0-1.2B"]["naive@16"] is None


def test_extended_section_is_rendered_and_degrades(tmp_path):
    data = _extended_tree(tmp_path)
    md = render_md(collect(data))
    assert "## 확장 실험 (2026-09-21)" in md
    for head in ("### g. 라벨 형식 × 집계", "### h. 학습 곡선 12k/24k/48k",
                 "### i. 학생 백본 비교", "### j. 소프트+y"):
        assert head in md
    assert "B_24k - B_12k" in md and NA in md

    empty = render_md(collect(tmp_path / "nothing"))
    assert "## 확장 실험 (2026-09-21)" in empty and NA in empty
