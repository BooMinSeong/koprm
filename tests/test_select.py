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
