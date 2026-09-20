"""§4.5 training-set assembly: nesting, 1:1 balance, A+B halves."""
import pytest

from koprm.trainsets import build_trainsets, order_pool, split_counts, summarize, take

SIZES = {"s": 10, "m": 20, "l": 40}


def lrow(arm, i, y, problem=None):
    return {
        "id": f"{arm}/{i}",
        "problem_id": problem or f"{arm}/p{i}",
        "problem_ko": "문제",
        "problem_en": "problem",
        "arm": arm,
        "generator": "prm800k" if arm == "A" else "exaone-1.2b",
        "solution_steps": ["단계 1", "단계 2"],
        "solution_steps_en": ["step 1", "step 2"],
        "step_labels": [1, y],
        "outcome": y,
    }


def pool(arm, n_correct, n_wrong):
    rows = [lrow(arm, i, 1) for i in range(n_correct)]
    rows += [lrow(arm, 1000 + i, 0) for i in range(n_wrong)]
    return rows


def ids(rows):
    return {r["id"] for r in rows}


def test_split_counts():
    assert split_counts(100, 100, 10) == (5, 5)
    assert split_counts(100, 100, 7) == (4, 3)
    assert split_counts(100, 100, 7, odd_to_correct=False) == (3, 4)
    assert split_counts(2, 100, 10) == (2, 8)      # correct is short -> wrong fills in
    assert split_counts(100, 3, 10) == (7, 3)      # and the other way round
    assert split_counts(2, 3, 10) == (2, 3)        # both short: take what there is


def test_sizes_balance_and_nesting():
    pools = {"A": pool("A", 40, 40), "B": pool("B", 40, 40)}
    sets = build_trainsets(pools, SIZES)
    assert set(sets) == {f"{a}_{t}" for a in ("A", "B", "AB") for t in SIZES}
    for arm in ("A", "B", "AB"):
        for tag, n in SIZES.items():
            rows = sets[f"{arm}_{tag}"]
            assert len(rows) == n
            n_c = sum(r["outcome"] for r in rows)
            assert n_c == n - n_c  # exactly 1:1 when the pool allows it
        assert ids(sets[f"{arm}_s"]) < ids(sets[f"{arm}_m"]) < ids(sets[f"{arm}_l"])


def test_ab_is_half_of_each_arm():
    pools = {"A": pool("A", 40, 40), "B": pool("B", 40, 40)}
    sets = build_trainsets(pools, SIZES)
    for tag, n in SIZES.items():
        rows = sets[f"AB_{tag}"]
        arms = [r["arm"] for r in rows]
        assert arms.count("A") == n // 2 and arms.count("B") == n - n // 2
        # each half keeps its own balance
        for arm in ("A", "B"):
            half = [r for r in rows if r["arm"] == arm]
            n_c = sum(r["outcome"] for r in half)
            assert abs(n_c - (len(half) - n_c)) <= 1


def test_shortfall_is_filled_by_the_other_class_and_stays_nested():
    pools = {"B": pool("B", 3, 100)}
    sets = build_trainsets(pools, SIZES)
    for tag, n in SIZES.items():
        rows = sets[f"B_{tag}"]
        assert len(rows) == n
        assert sum(r["outcome"] for r in rows) == 3
    assert ids(sets["B_s"]) < ids(sets["B_m"]) < ids(sets["B_l"])
    assert "A_s" not in sets and "AB_s" not in sets  # no A pool -> no A or A+B sets


def test_pool_smaller_than_the_target_size():
    pools = {"B": pool("B", 4, 4)}
    sets = build_trainsets(pools, {"s": 10})
    assert len(sets["B_s"]) == 8


def test_deterministic_and_seed_sensitive():
    pools = {"A": pool("A", 40, 40), "B": pool("B", 40, 40)}
    a = build_trainsets(pools, SIZES, seed=1)
    b = build_trainsets(pools, SIZES, seed=1)
    c = build_trainsets(pools, SIZES, seed=2)
    for k in a:
        assert [r["id"] for r in a[k]] == [r["id"] for r in b[k]]
    assert any(ids(a[k]) != ids(c[k]) for k in a)


def test_order_pool_is_a_stable_prefix_order():
    rows = pool("B", 10, 10)
    o1 = order_pool(rows, seed=3)
    o2 = order_pool(list(reversed(rows)), seed=3)  # input order must not matter
    assert [r["id"] for r in o1[1]] == [r["id"] for r in o2[1]]
    assert ids(take(o1, 4)) < ids(take(o1, 8))


def test_summary_table_rows():
    pools = {"A": pool("A", 40, 40), "B": pool("B", 40, 40)}
    s = summarize(build_trainsets(pools, {"s": 10}))
    by_name = {r["set"]: r for r in s}
    assert by_name["AB_s"]["arms"] == "A+B"
    assert by_name["A_s"]["n"] == 10 and by_name["A_s"]["correct"] == 5


def test_parse_sizes_and_arms():
    from koprm.trainsets import parse_arms, parse_sizes

    assert parse_sizes("3k,6k,12k") == {"3k": 3000, "6k": 6000, "12k": 12000}
    assert parse_sizes("12k, 24K ,48000") == {"12k": 12000, "24K": 24000, "48000": 48000}
    assert list(parse_sizes("48k,12k")) == ["48k", "12k"]      # order is kept
    for bad in ("", "0k", "-3k", "3kk", "abc", ","):
        with pytest.raises(ValueError):
            parse_sizes(bad)

    assert parse_arms("A,B") == ["A", "B"]
    assert parse_arms("b") == ["B"]
    for bad in ("", "C", "A,C"):
        with pytest.raises(ValueError):
            parse_arms(bad)


def test_custom_sizes_stay_nested():
    from koprm.trainsets import parse_sizes

    b_pool = pool("B", 120, 120)
    sizes = parse_sizes("40,80")
    sets = build_trainsets({"B": b_pool}, sizes=sizes)
    assert sorted(sets) == ["B_40", "B_80"]                    # B only, tags as written
    assert len(sets["B_40"]) == 40 and len(sets["B_80"]) == 80
    small, big = ids(sets["B_40"]), ids(sets["B_80"])
    assert small < big                                          # 40 nests inside 80
    for name in ("B_40", "B_80"):
        n_c = sum(1 for r in sets[name] if r["outcome"] == 1)
        assert n_c == len(sets[name]) // 2                      # still 1:1
