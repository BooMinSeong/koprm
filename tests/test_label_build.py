"""§2.3-2.4 / §4.3-4.5 label building and the audit, on fake teacher log-odds.

The fake teacher is generous (z ~ 5) on good steps and collapses (z ~ -5) from the first
error on, so the reference distribution D is tight around 0 and every collapsed step is a
candidate: the kernel's labels are then predictable and the human labels are known by
construction.
"""
import json

import numpy as np
import pytest

from koprm.label.audit import audit_metrics, report, variant_metrics
from koprm.label.build import (
    build_A_rows,
    build_labels,
    fit_refs,
    load_refs,
    prepare,
    refs_path,
    save_refs,
)

T = 6


def z_ok(rng, base=5.0, T=T):
    return list(np.round(rng.normal(base, 0.1, size=T), 6))


def z_bad(rng, first_err, base=5.0, T=T):
    z = rng.normal(base, 0.1, size=T)
    z[first_err:] -= 10.0
    return list(np.round(z, 6))


# --------------------------------------------------------------------------- arm B


def b_fixture(n_correct=80, wrong_first_errors=(1, 2, 3, 4), generator="g1", base=5.0,
              seed=0):
    """(rows, trans, teacher, problems) for one generator."""
    rng = np.random.default_rng(seed)
    rows, trans, teacher = [], {}, {}

    def add(i, outcome, z):
        pid = f"math/train/algebra/{i}"
        rid = f"{pid}#{generator}#0"
        rows.append({"id": rid, "problem_id": pid, "generator": generator,
                     "steps": [f"단계 {t}" for t in range(T)], "outcome": outcome})
        trans[rid] = {"id": rid, "steps_en": [f"step {t}" for t in range(T)],
                      "mask_restore_ok": True}
        teacher[rid] = {"id": rid, "teacher_logodds": z, "teacher_model": "fake"}
        return rid

    for i in range(n_correct):
        add(i, 1, z_ok(rng, base))
    wrong_ids = []
    for j, fe in enumerate(wrong_first_errors):
        wrong_ids.append(add(1000 + j, 0, z_bad(rng, fe, base)))
    problems = {r["problem_id"]: {"source": "math", "problem_en": "p", "problem_ko": "문제"}
                for r in rows}
    return rows, trans, teacher, problems, wrong_ids


def test_build_labels_schema_and_kernel_labels():
    rows, trans, teacher, problems, wrong_ids = b_fixture()
    out, refs, stats = build_labels(rows, trans, teacher, problems, arm="B")
    assert stats["n_labeled"] == len(rows) and stats["dropped"] == {}
    assert set(refs) == {"g1"}
    by_id = {r["id"]: r for r in out}

    expected_keys = {"problem_id", "problem_source", "problem_ko", "problem_en", "arm",
                     "generator", "solution_steps", "solution_steps_en", "teacher_logodds",
                     "outcome", "ceiling", "candidates", "tail_probs", "r_lo", "r_hi",
                     "step_labels", "mask_restore_ok"}
    assert expected_keys <= set(out[0])
    assert out[0]["arm"] == "B" and out[0]["problem_ko"] == "문제"
    assert out[0]["problem_source"] == "math"
    assert out[0]["step_labels"] == [1] * T          # y=1 -> every prefix is proven (§2.1)
    assert out[0]["solution_steps"] == [f"단계 {t}" for t in range(T)]

    for fe, rid in zip((1, 2, 3, 4), wrong_ids):
        labels = by_id[rid]["step_labels"]
        assert labels == [1] * fe + [0] * (T - fe), (fe, labels)
        assert by_id[rid]["candidates"] == list(range(fe, T))
        assert len(by_id[rid]["r_lo"]) == T and len(by_id[rid]["tail_probs"]) == T
        # json round trip: no numpy types leak into the file
        json.dumps(by_id[rid])


def test_stats_composition_and_no_candidates():
    rng = np.random.default_rng(1)
    rows, trans, teacher, problems, _ = b_fixture()
    # a wrong solution the teacher never flags: no candidates -> only the last step is 0
    rid = "math/train/algebra/2000#g1#0"
    rows.append({"id": rid, "problem_id": "math/train/algebra/2000", "generator": "g1",
                 "steps": [f"단계 {t}" for t in range(T)], "outcome": 0})
    trans[rid] = {"id": rid, "steps_en": [f"step {t}" for t in range(T)], "mask_restore_ok": True}
    teacher[rid] = {"id": rid, "teacher_logodds": z_ok(rng)}
    problems["math/train/algebra/2000"] = {"source": "math", "problem_ko": "문제"}

    out, _, stats = build_labels(rows, trans, teacher, problems)
    by_id = {r["id"]: r for r in out}
    assert by_id[rid]["step_labels"] == [None] * (T - 1) + [0]
    assert stats["n_wrong"] == 5
    assert stats["wrong_no_candidates_frac"] == pytest.approx(1 / 5)
    comp = stats["wrong_label_composition"]
    assert sum(comp.values()) == pytest.approx(1.0)
    assert comp["mask"] == pytest.approx(5 / 30)
    assert stats["per_generator"]["g1"]["n_ref_steps"] == 80 * T
    assert set(stats["per_generator"]["g1"]["ceiling_q"]) == {"q5", "q25", "q50", "q75", "q95"}


def test_rows_without_a_usable_bridge_are_dropped_and_counted():
    rows, trans, teacher, problems, _ = b_fixture(n_correct=40)
    rng = np.random.default_rng(2)

    def add(rid, **kw):
        rows.append({"id": rid, "problem_id": rid.split("#")[0], "generator": "g1",
                     "steps": [f"단계 {t}" for t in range(T)], "outcome": 0, **kw})

    add("x1#g1#0")
    trans["x1#g1#0"] = {"id": "x1#g1#0", "steps_en": None, "mask_restore_ok": False}
    teacher["x1#g1#0"] = {"id": "x1#g1#0", "teacher_logodds": z_bad(rng, 2)}
    add("x2#g1#0")                                  # no translation row at all
    add("x3#g1#0")
    trans["x3#g1#0"] = {"id": "x3#g1#0", "steps_en": ["s"] * T, "mask_restore_ok": True}
    teacher["x3#g1#0"] = {"id": "x3#g1#0", "teacher_logodds": None}
    add("x4#g1#0")
    trans["x4#g1#0"] = {"id": "x4#g1#0", "steps_en": ["s"] * T, "mask_restore_ok": True}
    teacher["x4#g1#0"] = {"id": "x4#g1#0", "teacher_logodds": z_bad(rng, 2)[: T - 1]}
    add("x5#g1#0", steps=[])

    kept, drops = prepare(rows, trans, teacher)
    assert dict(drops) == {"mask_restore_fail": 1, "no_translation": 1, "no_teacher": 1,
                           "len_mismatch": 1, "no_steps": 1}
    out, _, stats = build_labels(rows, trans, teacher, problems)
    assert len(out) == len(kept) == len(rows) - 5
    assert stats["dropped"]["len_mismatch"] == 1


def test_reference_is_per_generator_and_reusable(tmp_path):
    """A generator the teacher scores lower everywhere must not look wrong (§2.3)."""
    rows_a, trans_a, teach_a, prob_a, _ = b_fixture(generator="g1", base=5.0, seed=3)
    rows_b, trans_b, teach_b, prob_b, wrong_b = b_fixture(generator="g2", base=-1.0, seed=4)
    rows = rows_a + rows_b
    trans = {**trans_a, **trans_b}
    teacher = {**teach_a, **teach_b}
    problems = {**prob_a, **prob_b}
    out, refs, _stats = build_labels(rows, trans, teacher, problems)
    assert set(refs) == {"g1", "g2"}
    by_id = {r["id"]: r for r in out}
    for fe, rid in zip((1, 2, 3, 4), wrong_b):
        assert by_id[rid]["step_labels"] == [1] * fe + [0] * (T - fe)
    assert refs["g2"].b_min < refs["g1"].b_min

    p = refs_path(tmp_path / "B.jsonl")
    save_refs(p, refs)
    again = load_refs(p)
    assert set(again) == {"g1", "g2"}
    out2, _, _ = build_labels(rows, trans, teacher, problems, ref=again)
    assert [r["step_labels"] for r in out2] == [r["step_labels"] for r in out]


def test_a_single_shared_reference_can_be_passed_in():
    rows, trans, teacher, problems, wrong_ids = b_fixture(seed=5)
    kept, _ = prepare(rows, trans, teacher)
    ref = fit_refs(kept)["g1"]
    out, _refs, _ = build_labels(rows, trans, teacher, problems, ref=ref)
    by_id = {r["id"]: r for r in out}
    assert by_id[wrong_ids[1]]["step_labels"] == [1, 1, 0, 0, 0, 0]


# --------------------------------------------------------------------------- arm A


def a_pool(n_correct=10, n_wrong=10, big_problem_solutions=0):
    rows, trans = [], {}

    def add(i, outcome, first_err, problem=None):
        rid = f"prm800k/{i}"
        labels = [1] * T if first_err < 0 else [1] * first_err + [0] * (T - first_err)
        rows.append({
            "id": rid,
            "problem_id": problem or f"prm800k/prob/{i}",
            "problem_en": "problem",
            "answer": "1",
            "steps_en": [f"step {t}" for t in range(T)],
            "human_first_error": first_err,
            "step_labels": labels,
            "pre_generated_answer": "1",
            "finish_reason": "solution" if first_err < 0 else "found_error",
            "outcome": outcome,
        })
        trans[rid] = {"id": rid, "steps_ko": [f"단계 {t}" for t in range(T)],
                      "mask_restore_ok": True}
        return rid

    for i in range(n_correct):
        add(i, 1, -1)
    for i in range(n_wrong):
        add(100 + i, 0, 2)
    for i in range(big_problem_solutions):
        add(200 + i, 1, -1, problem="prm800k/prob/shared")
    return rows, trans


def a_problems(rows):
    """problem_id -> the Korean problem text (arm A needs one per row, §4.5)."""
    return {r["problem_id"]: {"problem_ko": "문제", "source": "prm800k"} for r in rows}


def test_build_A_rows_schema_and_balance():
    rows, trans = a_pool(10, 4)
    problems = a_problems(rows)
    out, stats = build_A_rows(rows, trans, problems, n_per_class=6)
    assert stats["n_correct"] == stats["n_wrong"] == 4      # 1:1, wrong is the short side
    assert len(out) == 8
    r = out[0]
    assert r["arm"] == "A" and r["generator"] == "prm800k"
    assert r["teacher_logodds"] is None and r["candidates"] == [] and r["ceiling"] is None
    assert r["solution_steps"] == [f"단계 {t}" for t in range(T)]
    assert r["solution_steps_en"] == [f"step {t}" for t in range(T)]
    assert r["problem_ko"] == "문제"
    wrong = next(x for x in out if x["outcome"] == 0)
    assert wrong["step_labels"] == [1, 1, 0, 0, 0, 0]       # §4.3 human mapping
    assert all(x["mask_restore_ok"] for x in out)


def _per_problem(out):
    d = {}
    for r in out:
        d[r["problem_id"]] = d.get(r["problem_id"], 0) + 1
    return d


def test_build_A_rows_prefers_two_solutions_per_problem():
    rows, trans = a_pool(4, 8, big_problem_solutions=6)
    out, stats = build_A_rows(rows, trans, a_problems(rows), n_per_class=6)
    assert _per_problem(out)["prm800k/prob/shared"] == 2
    assert stats["solutions_per_problem_max"] == 2
    assert stats["n_correct"] == stats["n_wrong"] == 6


def test_build_A_rows_relaxes_the_cap_only_to_fill_the_quota():
    rows, trans = a_pool(4, 8, big_problem_solutions=6)
    out, stats = build_A_rows(rows, trans, a_problems(rows), n_per_class=8)
    # 4 + 2 capped picks are not enough for 8 correct, so the cap gives way
    assert _per_problem(out)["prm800k/prob/shared"] == 4
    assert stats["n_correct"] == stats["n_wrong"] == 8


def test_build_A_rows_drops_bad_translations():
    rows, trans = a_pool(4, 4)
    trans["prm800k/0"]["mask_restore_ok"] = False
    trans["prm800k/1"]["steps_ko"] = ["단계"] * (T - 1)     # length mismatch
    del trans["prm800k/2"]
    out, stats = build_A_rows(rows, trans, a_problems(rows), n_per_class=8)
    assert stats["dropped"] == {"mask_restore_fail": 1, "len_mismatch": 1, "no_translation": 1}
    assert stats["n_correct"] == 1 and len(out) == 2        # only one correct survives
    assert stats["n_usable"] == {"0": 4, "1": 1}


def test_build_A_rows_drops_rows_without_a_korean_problem():
    """A problem that was never translated cannot be trained on; it goes before balancing."""
    rows, trans = a_pool(4, 4)
    problems = a_problems(rows)
    del problems[rows[0]["problem_id"]]        # this correct solution has no Korean problem
    out, stats = build_A_rows(rows, trans, problems, n_per_class=8)
    assert stats["dropped"] == {"no_problem_ko": 1}
    assert rows[0]["id"] not in {r["id"] for r in out}
    assert stats["n_usable"] == {"0": 4, "1": 3}
    assert stats["n_correct"] == stats["n_wrong"] == 3      # 1:1 from what is left
    assert stats["n_missing_problem_ko"] == 0
    assert all(r["problem_ko"] for r in out)


def test_build_A_rows_is_deterministic():
    rows, trans = a_pool(20, 20)
    a, _ = build_A_rows(rows, trans, a_problems(rows), n_per_class=5, seed=11)
    b, _ = build_A_rows(rows, trans, a_problems(rows), n_per_class=5, seed=11)
    c, _ = build_A_rows(rows, trans, a_problems(rows), n_per_class=5, seed=12)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["id"] for r in a] != [r["id"] for r in c]


# --------------------------------------------------------------------------- audit


def audit_rows(n_correct=200, wrong=((1, 1), (2, 2), (3, 3), (4, 4)), seed=7, direct=True):
    """wrong: (human_first_error, teacher's collapse point) pairs."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_correct):
        z = z_ok(rng)
        rows.append({"id": f"a{i}", "outcome": 1, "finish_reason": "solution",
                     "step_labels": [1] * T, "human_first_error": -1,
                     "steps_en": [f"step {t}" for t in range(T)],
                     "steps_ko": [f"단계 {t}" for t in range(T)],
                     "steps_en_rt": [f"step {t}" for t in range(T)],
                     "teacher_logodds_rt": z, "mask_restore_ok": True})
        if direct:
            rows[-1]["teacher_logodds_direct"] = z_ok(rng)
    for j, (fe, collapse) in enumerate(wrong):
        rows.append({"id": f"w{j}", "outcome": 0, "finish_reason": "found_error",
                     "step_labels": [1] * fe + [0] * (T - fe), "human_first_error": fe,
                     "steps_en": [f"step {t}" for t in range(T)],
                     "steps_ko": [f"단계 {t}" for t in range(T)],
                     "steps_en_rt": [f"step {t}" for t in range(T)],
                     "teacher_logodds_rt": z_bad(rng, collapse), "mask_restore_ok": True})
        if direct:
            rows[-1]["teacher_logodds_direct"] = z_bad(rng, fe)
    return rows


def test_audit_metrics_perfect_case():
    m = audit_metrics(audit_rows(direct=False))
    rt = m["rt"]
    assert rt["n_correct"] == 200 and rt["n_wrong"] == 4
    assert rt["coverage"] == 1.0
    assert rt["precision"] == 1.0
    assert rt["precision_1"] == 1.0 and rt["precision_0"] == 1.0
    assert rt["n_eval_steps"] == 4 * (T - 1)            # the last step never counts
    assert rt["first_error_exact_frac"] == 1.0
    assert rt["no_candidate_frac"] == 0.0
    assert "direct" not in m and "bridge_cost" not in m


def test_audit_metrics_counts_disagreement():
    # the teacher collapses two steps after the human's first error
    rows = audit_rows(wrong=((1, 3), (2, 2)))
    m = audit_metrics(rows)
    rt = m["rt"]
    assert rt["precision"] < 1.0
    assert rt["precision_0"] == 1.0          # every 0 is still at or after the real error
    assert rt["precision_1"] < 1.0           # but two steps were wrongly called 1
    assert rt["n_pred_1"] == 3 + 2           # w0: steps 0..2, w1: steps 0..1
    assert rt["precision_1"] == pytest.approx(3 / 5)
    # the same rows scored without the bridge get the human's position right
    assert m["direct"]["precision"] == 1.0
    assert m["bridge_cost"]["precision"] > 0
    assert set(m["bridge_cost"]) == {"precision", "precision_1", "precision_0", "coverage",
                                     "first_error_exact_frac"}


def test_audit_masks_when_the_evidence_is_weak():
    """No candidate at all -> every non-last step is masked, coverage drops to 0."""
    rows = audit_rows(wrong=((2, T),))     # collapse never happens before the end
    rt = audit_metrics(rows)["rt"]
    assert rt["no_candidate_frac"] == 1.0
    assert rt["coverage"] == 0.0
    assert rt["n_unmasked"] == 0
    assert rt["precision"] == 0.0          # nothing predicted: precision is vacuous


def test_audit_report_has_both_threshold_settings():
    rep = report(audit_rows(wrong=((1, 3), (2, 2))))
    assert set(rep["thresholds"]) == {"0.9/0.1", "0.95/0.05"}
    for m in rep["thresholds"].values():
        assert "_per_row" not in m["rt"]
        assert 0.0 <= m["rt"]["coverage"] <= 1.0
    # tighter thresholds can only mask more
    a = rep["thresholds"]["0.9/0.1"]["rt"]
    b = rep["thresholds"]["0.95/0.05"]["rt"]
    assert b["coverage"] <= a["coverage"]
    json.dumps(rep)


def test_audit_drops_unusable_rows():
    rows = audit_rows(wrong=((2, 2),))
    rows.append({"id": "bad1", "outcome": 0, "step_labels": [1] * T,
                 "teacher_logodds_rt": None, "mask_restore_ok": True})
    rows.append({"id": "bad2", "outcome": 0, "step_labels": [1] * T,
                 "teacher_logodds_rt": [0.0] * T, "mask_restore_ok": False})
    rows.append({"id": "bad3", "outcome": 0, "step_labels": [1] * T,
                 "teacher_logodds_rt": [0.0] * (T - 2), "mask_restore_ok": True})
    m = variant_metrics(rows, "teacher_logodds_rt", 0.9, 0.1)
    assert m["dropped"] == {"no_teacher": 1, "mask_restore_fail": 1, "len_mismatch": 1}
    assert m["n_wrong"] == 1
