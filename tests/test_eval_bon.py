import json

import pytest

from koprm.eval.bon import (
    AGG_SETS,
    AnswerGroups,
    aggregate,
    bootstrap_ci,
    combine,
    evaluate,
    n_grid,
    split_steps,
)


def _completion(ans):
    return (f"## 단계 1: 설명\n계산을 합니다.\n\n## 단계 2: 계산\n값을 구합니다.\n\n"
            f"따라서 최종 답은: $\\boxed{{{ans}}}$입니다.")


def _steps(vals):
    """Per-step score list whose `last` is vals and whose `min` is min(vals)."""
    return [0.5, float(vals)]


def test_split_steps_and_grid():
    assert split_steps("a\n\nb\n\n\n\nc\n\n") == ["a", "b", "c"]
    assert n_grid(64) == [1, 2, 4, 8, 16, 32, 64]
    assert n_grid(4) == [1, 2, 4]


def test_aggregate():
    assert aggregate([0.9, 0.2, 0.7], "last") == pytest.approx(0.7)
    assert aggregate([0.9, 0.2, 0.7], "min") == pytest.approx(0.2)
    assert aggregate([0.9, 0.2, 0.7], "prod") == pytest.approx(0.9 * 0.2 * 0.7)
    assert aggregate([0.9, 0.2, 0.7], "mean") == pytest.approx(1.8 / 3)
    assert aggregate([0.5], "prod") == pytest.approx(0.5)
    for agg in ("last", "min", "prod", "mean"):
        assert aggregate([], agg) == 0.0
    with pytest.raises(ValueError):
        aggregate([0.1], "sum")


def test_agg_sets():
    assert AGG_SETS["both"] == ["last", "min"]
    assert AGG_SETS["all"] == ["last", "min", "prod", "mean"]


def test_answer_grouping_uses_math_equivalence():
    g = AnswerGroups("\\frac{1}{2}")
    a = g.group_of("0.5")
    b = g.group_of("\\frac{1}{2}")
    c = g.group_of("\\dfrac{1}{2}")
    d = g.group_of("3")
    e = g.group_of(None)
    assert a == b == c
    assert d != a and e != a and e != d
    assert g.correct[a] is True
    assert g.correct[d] is False
    assert g.correct[e] is False


def test_evaluate_naive_weighted_maj():
    rows = [
        {"problem_id": "A", "problem_ko": "p", "answer": "4",
         "completions": [_completion(x) for x in ["4", "5", "5", "4"]]},
        {"problem_id": "B", "problem_ko": "p", "answer": "7",
         "completions": [_completion(x) for x in ["3", "3", "7", "7"]]},
    ]
    scores = [
        [_steps(v) for v in (0.9, 0.8, 0.7, 0.2)],
        [_steps(v) for v in (0.9, 0.8, 0.95, 0.1)],
    ]
    res = evaluate(rows, scores, agg="last", verbose=False)
    assert res["ns"] == [1, 2, 4]
    m = res["metrics"]
    # n=1: A correct, B wrong for every method
    assert m["1"] == {"naive": 0.5, "weighted": 0.5, "maj": 0.5, "pass": 0.5}
    # n=2: same picks
    assert m["2"]["naive"] == 0.5 and m["2"]["weighted"] == 0.5 and m["2"]["maj"] == 0.5
    # n=4: naive finds both (A: 0.9 -> "4", B: 0.95 -> "7")
    assert m["4"]["naive"] == 1.0
    # weighted sums: A 4->1.1 vs 5->1.5 (wrong); B 3->1.7 vs 7->1.05 (wrong)
    assert m["4"]["weighted"] == 0.0
    # majority ties (2 vs 2) go to the first occurring group: A "4" (right), B "3" (wrong)
    assert m["4"]["maj"] == 0.5
    assert m["4"]["pass"] == 1.0
    assert res["pass@1_mean_correct"] == pytest.approx(0.5)
    assert res["per_problem"]["naive@4"] == [1, 1]
    assert res["per_problem"]["weighted@4"] == [0, 0]
    assert res["problem_ids"] == ["A", "B"]


def test_evaluate_min_aggregation_changes_the_pick():
    rows = [{"problem_id": "A", "problem_ko": "p", "answer": "4",
             "completions": [_completion("4"), _completion("5")]}]
    # completion 0 ends high but dips; completion 1 is flat
    scores = [[[0.1, 0.9], [0.6, 0.6]]]
    assert evaluate(rows, scores, agg="last", verbose=False)["metrics"]["2"]["naive"] == 1.0
    assert evaluate(rows, scores, agg="min", verbose=False)["metrics"]["2"]["naive"] == 0.0


def test_combine_both_aggregations_and_single():
    """--agg both nests the two results; --agg last|min keeps the old flat shape."""
    rows = [{"problem_id": "A", "problem_ko": "p", "answer": "4",
             "completions": [_completion("4"), _completion("5")]}]
    scores = [[[0.1, 0.9], [0.6, 0.6]]]
    results = {a: evaluate(rows, scores, agg=a, verbose=False) for a in ("last", "min")}

    both = combine(results, "src.jsonl", "ckpt/x")
    assert set(both) == {"last", "min", "source", "scorer"}
    assert both["source"] == "src.jsonl" and both["scorer"] == "ckpt/x"
    assert both["last"]["agg"] == "last" and both["min"]["agg"] == "min"
    assert both["last"]["metrics"]["2"]["naive"] == 1.0
    assert both["min"]["metrics"]["2"]["naive"] == 0.0
    assert both["last"]["problem_ids"] == ["A"]

    flat = combine({"last": results["last"]}, "src.jsonl", "ckpt/x")
    assert flat["agg"] == "last" and flat["metrics"]["2"]["naive"] == 1.0
    assert flat["source"] == "src.jsonl" and flat["scorer"] == "ckpt/x"
    assert "last" not in flat  # unchanged single-aggregation layout


def test_bootstrap_ci():
    assert bootstrap_ci([1, 1, 1, 1])["mean"] == 1.0
    assert bootstrap_ci([1, 1, 1, 1])["lo"] == 1.0
    ci = bootstrap_ci([1, 0] * 50, iters=500, seed=1)
    assert ci["mean"] == pytest.approx(0.5)
    assert ci["lo"] < 0.5 < ci["hi"]
    assert ci["n"] == 100
    empty = bootstrap_ci([])
    assert empty["n"] == 0


def test_evaluate_uses_and_validates_the_group_cache(tmp_path, monkeypatch):
    """The grouping is dataset-only, so it is cached; a corrupt file is recomputed."""
    from koprm.eval import bon

    rows = [
        {"problem_id": "A", "problem_ko": "p", "answer": "4",
         "completions": [_completion(x) for x in ["4", "5", "5", "4"]]},
        {"problem_id": "B", "problem_ko": "p", "answer": "7",
         "completions": [_completion(x) for x in ["3", "3", "7", "7"]]},
    ]
    scores = [
        [_steps(v) for v in (0.9, 0.8, 0.7, 0.2)],
        [_steps(v) for v in (0.9, 0.8, 0.95, 0.1)],
    ]
    first = evaluate(rows, scores, agg="last", verbose=False, cache_dir=tmp_path)
    cached = list(tmp_path.glob("groups_*.json"))
    assert len(cached) == 1
    assert not list(tmp_path.glob("*.tmp"))                    # the tmp file was renamed

    def boom(*a, **kw):
        raise AssertionError("AnswerGroups must not run on a cache hit")

    monkeypatch.setattr(bon, "AnswerGroups", boom)
    assert evaluate(rows, scores, agg="last", verbose=False, cache_dir=tmp_path) == first
    monkeypatch.undo()

    # a half-written / corrupt file, and a file that does not match the rows, are ignored
    for bad in ("{not json", json.dumps([{"gids": [0], "correct": [True]}])):
        cached[0].write_text(bad, encoding="utf-8")
        assert evaluate(rows, scores, agg="last", verbose=False, cache_dir=tmp_path) == first

    # no cache dir -> nothing is written, same numbers
    assert evaluate(rows, scores, agg="last", verbose=False) == first


def test_groups_key_depends_on_the_dataset_only():
    from koprm.eval.bon import groups_key

    rows = [{"problem_id": "A", "answer": "4", "completions": [_completion("4")]}]
    same = [{"problem_id": "A", "answer": "4", "completions": [_completion("4")],
             "problem_ko": "다른 텍스트"}]
    other = [{"problem_id": "A", "answer": "5", "completions": [_completion("4")]}]
    assert groups_key(rows) == groups_key(same)
    assert groups_key(rows) != groups_key(other)


def test_prod_and_mean_change_the_pick():
    """prod punishes a long dip, mean averages it: the two pick different completions."""
    rows = [{"problem_id": "A", "problem_ko": "p", "answer": "4",
             "completions": [_completion("4"), _completion("5")]}]
    # completion 0: one high step after a dip; completion 1: flat and middling
    scores = [[[0.2, 0.99], [0.6, 0.6]]]
    assert evaluate(rows, scores, agg="last", verbose=False)["metrics"]["2"]["naive"] == 1.0
    assert evaluate(rows, scores, agg="prod", verbose=False)["metrics"]["2"]["naive"] == 0.0
    assert evaluate(rows, scores, agg="mean", verbose=False)["metrics"]["2"]["naive"] == 0.0
    # a completion that is better everywhere wins under every aggregation
    scores = [[[0.9, 0.9], [0.2, 0.2]]]
    for agg in ("last", "min", "prod", "mean"):
        assert evaluate(rows, scores, agg=agg, verbose=False)["metrics"]["2"]["naive"] == 1.0


def test_save_and_load_scores_round_trip(tmp_path):
    from koprm.eval.bon import load_scores, save_scores

    rows = [
        {"problem_id": "A", "problem_ko": "p", "answer": "4",
         "completions": [_completion("4"), _completion("5")]},
        {"problem_id": "B", "problem_ko": "p", "answer": "7",
         "completions": [_completion("7")]},
    ]
    scores = [[[0.1, 0.9], [0.6, 0.6]], [[0.4]]]
    path = tmp_path / "scores.jsonl"
    assert save_scores(path, rows, scores) == 2
    assert load_scores(path, rows) == scores

    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert saved[0]["problem_id"] == "A" and saved[0]["answer"] == "4"
    assert saved[1]["scores"] == [[0.4]]

    # the same numbers come out of a reloaded file
    assert (evaluate(rows, load_scores(path, rows), agg="min", verbose=False)
            == evaluate(rows, scores, agg="min", verbose=False))


def test_load_scores_rejects_a_mismatched_file(tmp_path):
    import pytest as _pytest

    from koprm.eval.bon import load_scores, save_scores

    rows = [{"problem_id": "A", "problem_ko": "p", "answer": "4",
             "completions": [_completion("4"), _completion("5")]}]
    path = tmp_path / "scores.jsonl"
    save_scores(path, rows, [[[0.1, 0.9], [0.6, 0.6]]])

    two_rows = rows + [{"problem_id": "B", "answer": "7", "completions": [_completion("7")]}]
    with _pytest.raises(SystemExit, match="1 rows, but the dataset has 2"):
        load_scores(path, two_rows)

    renamed = [{**rows[0], "problem_id": "Z"}]
    with _pytest.raises(SystemExit, match="different order or dataset"):
        load_scores(path, renamed)

    three = [{**rows[0], "completions": rows[0]["completions"] + [_completion("6")]}]
    with _pytest.raises(SystemExit, match="the dataset has 3"):
        load_scores(path, three)
