"""§4.2 selection on a synthetic generation set."""
from koprm.select import select_rows


def grow(pid, gen, idx, outcome, **kw):
    r = {
        "id": f"{pid}#{gen}#{idx}",
        "problem_id": pid,
        "generator": gen,
        "sample_idx": idx,
        "text": "풀이",
        "steps": ["단계 1", "단계 2"],
        "finish_reason": "stop",
        "n_tokens": 42,
        "outcome": outcome,
        "pred_answer": "1",
        "no_boxed": False,
        "truncated": False,
    }
    r.update(kw)
    return r


def mixed_problem(pid, gens=("a", "b")):
    rows = []
    for g in gens:
        rows.append(grow(pid, g, 0, 1))
        rows.append(grow(pid, g, 1, 0))
    return rows


def meta(pids, source="math"):
    return {p: {"source": source, "level": "Level 3"} for p in pids}


def test_only_mixed_problems_and_one_of_each():
    rows = mixed_problem("p1") + [grow("p2", "a", 0, 1), grow("p2", "b", 0, 1)]
    rows += [grow("p3", "a", 0, 0)]
    chosen, stats = select_rows(rows, meta(["p1", "p2", "p3"]))
    assert stats["n_mixed"] == 1
    assert {r["problem_id"] for r in chosen} == {"p1"}
    assert sorted(r["outcome"] for r in chosen) == [0, 1]
    assert all(r["arm"] == "B" for r in chosen)
    # the generation row is otherwise untouched
    assert chosen[0]["steps"] == ["단계 1", "단계 2"]


def test_truncated_and_no_boxed_are_not_eligible():
    rows = [
        grow("p1", "a", 0, 1, truncated=True),   # only correct solution is truncated
        grow("p1", "a", 1, 0),
        grow("p2", "a", 0, 1, no_boxed=True),    # a "correct" row without a boxed answer
        grow("p2", "a", 1, 0),
        grow("p3", "a", 0, 1),
        grow("p3", "a", 1, 0),
    ]
    chosen, stats = select_rows(rows, meta(["p1", "p2", "p3"]))
    assert stats["n_mixed"] == 1
    assert {r["problem_id"] for r in chosen} == {"p3"}


def test_generator_balance_and_cross_generator_pairs():
    rows = []
    for i in range(10):
        rows += mixed_problem(f"p{i}")
    chosen, stats = select_rows(rows, meta([f"p{i}" for i in range(10)]))
    assert stats["n_chosen_problems"] == 10
    assert stats["correct_by_generator"] == {"a": 5, "b": 5}
    assert stats["wrong_by_generator"] == {"a": 5, "b": 5}
    by_problem = {}
    for r in chosen:
        by_problem.setdefault(r["problem_id"], {})[r["outcome"]] = r["generator"]
    # when both generators have both outcomes, the wrong one comes from the other model
    for pid, d in by_problem.items():
        assert d[0] != d[1], pid


def test_wrong_falls_back_to_the_same_generator():
    rows = [
        grow("p1", "a", 0, 1),
        grow("p1", "b", 0, 1),
        grow("p1", "a", 1, 0),   # only generator "a" produced a wrong solution
    ]
    chosen, _ = select_rows(rows, meta(["p1"]))
    wrong = [r for r in chosen if r["outcome"] == 0]
    assert len(wrong) == 1 and wrong[0]["generator"] == "a"


def test_cap_fills_math_before_gsm8k():
    rows, m = [], {}
    for i in range(5):
        rows += mixed_problem(f"m{i}")
        m[f"m{i}"] = {"source": "math", "level": "Level 2"}
        rows += mixed_problem(f"g{i}")
        m[f"g{i}"] = {"source": "gsm8k", "level": "gsm8k"}
    chosen, stats = select_rows(rows, m, max_problems=5)
    assert stats["n_mixed"] == 10
    assert stats["mixed_by_source"] == {"gsm8k": 5, "math": 5}
    assert stats["chosen_by_source"] == {"math": 5}
    assert {r["problem_id"] for r in chosen} == {f"m{i}" for i in range(5)}
    assert stats["n_chosen_rows"] == 10


def test_deterministic_for_a_seed():
    rows = []
    for i in range(20):
        rows += mixed_problem(f"p{i}")
        rows.append(grow(f"p{i}", "a", 2, 0))  # a second wrong solution to choose among
    m = meta([f"p{i}" for i in range(20)])
    a, _ = select_rows(rows, m, seed=7)
    b, _ = select_rows(rows, m, seed=7)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    c, _ = select_rows(rows, m, max_problems=10, seed=7)
    # the cap is a prefix of the same order
    assert {r["problem_id"] for r in c} <= {r["problem_id"] for r in a}


def test_unknown_problems_still_selected_last():
    rows = mixed_problem("m0") + mixed_problem("x0")
    m = {"m0": {"source": "math", "level": "Level 1"}}
    chosen, stats = select_rows(rows, m, max_problems=1)
    assert stats["mixed_by_source"] == {"math": 1, "unknown": 1}
    assert {r["problem_id"] for r in chosen} == {"m0"}


def rich_problem(pid, gens=("a", "b"), n_correct=2, n_wrong=2):
    """n_correct + n_wrong eligible solutions per generator."""
    rows = []
    for g in gens:
        for k in range(n_correct):
            rows.append(grow(pid, g, k, 1))
        for k in range(n_wrong):
            rows.append(grow(pid, g, 100 + k, 0))
    return rows


def test_per_class_one_is_the_default():
    rows = rich_problem("p1") + rich_problem("p2")
    assert select_rows(rows, meta(["p1", "p2"])) == select_rows(
        rows, meta(["p1", "p2"]), per_class=1)


def test_per_class_two_takes_two_of_each_without_repeats():
    pids = [f"p{i}" for i in range(4)]
    rows = [r for p in pids for r in rich_problem(p)]
    chosen, stats = select_rows(rows, meta(pids), per_class=2)
    assert stats["per_class"] == 2
    assert stats["n_chosen_rows"] == len(chosen) == 4 * 4      # 2+2 per problem
    assert stats["n_problems_short_of_per_class"] == 0
    assert len({r["id"] for r in chosen}) == len(chosen)       # never the same row twice
    for p in pids:
        got = [r for r in chosen if r["problem_id"] == p]
        assert sorted(r["outcome"] for r in got) == [0, 0, 1, 1]
        # both generators are used for each class of each problem (they have equal supply)
        assert {r["generator"] for r in got if r["outcome"] == 1} == {"a", "b"}
        assert {r["generator"] for r in got if r["outcome"] == 0} == {"a", "b"}
    assert stats["correct_by_generator"] == {"a": 4, "b": 4}
    assert stats["wrong_by_generator"] == {"a": 4, "b": 4}


def test_per_class_two_falls_back_when_a_class_is_short():
    """A problem with a single correct solution still contributes it (mixed, 1 + 2)."""
    rows = [grow("p1", "a", 0, 1)] + [grow("p1", g, 100 + k, 0)
                                      for g in ("a", "b") for k in range(2)]
    chosen, stats = select_rows(rows, meta(["p1"]), per_class=2)
    assert [r["outcome"] for r in chosen] == [1, 0, 0]
    assert stats["n_chosen_rows"] == 3
    assert stats["n_problems_short_of_per_class"] == 1
    # the first wrong pick still prefers the other generator
    assert chosen[1]["generator"] == "b"
    assert len({r["id"] for r in chosen}) == 3


def test_per_class_larger_than_supply_is_capped():
    rows = rich_problem("p1", n_correct=1, n_wrong=1)          # one of each per generator
    chosen, stats = select_rows(rows, meta(["p1"]), per_class=5)
    assert stats["n_chosen_rows"] == 4                          # 2 correct + 2 wrong exist
    assert sorted(r["outcome"] for r in chosen) == [0, 0, 1, 1]
    assert stats["n_problems_short_of_per_class"] == 1
