import pytest

from koprm.eval.bon import (
    AnswerGroups,
    aggregate,
    bootstrap_ci,
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
    assert aggregate([], "last") == 0.0
    with pytest.raises(ValueError):
        aggregate([0.1], "prod")


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


def test_bootstrap_ci():
    assert bootstrap_ci([1, 1, 1, 1])["mean"] == 1.0
    assert bootstrap_ci([1, 1, 1, 1])["lo"] == 1.0
    ci = bootstrap_ci([1, 0] * 50, iters=500, seed=1)
    assert ci["mean"] == pytest.approx(0.5)
    assert ci["lo"] < 0.5 < ci["hi"]
    assert ci["n"] == 100
    empty = bootstrap_ci([])
    assert empty["n"] == 0
