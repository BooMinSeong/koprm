"""§7 확장: distribution-shift joins and the ProcessBench-style first-error metrics."""
import json
from pathlib import Path

import pytest

from koprm.io import write_jsonl
from koprm.paths import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_KO, SYSTEM_PROMPTS
from koprm.shift import (
    first_error_report,
    group_completions,
    math500_en_rows,
    pb_metrics,
    processbench_rows,
    student_probs,
    teacher_probs,
)


class FakeScorer:
    """Returns canned per-step probabilities and records what it was asked."""

    def __init__(self, probs):
        self.probs = probs
        self.seen_problems: list[str] = []
        self.seen_steps: list[list[str]] = []

    def score(self, problems, steps, **kw):
        self.seen_problems = list(problems)
        self.seen_steps = [list(s) for s in steps]
        return self.probs


def _pb_row(rid, split, label, n_steps=3):
    return {"id": rid, "split": split, "generator": "g",
            "problem_en": f"problem {rid}", "steps_en": [f"step {i}" for i in range(n_steps)],
            "label": label, "final_answer_correct": label == -1}


def _trans(rid, ok=True, n_steps=3):
    return ({"id": rid, "problem_ko": f"문제 {rid}", "mask_restore_ok": ok},
            {"id": rid, "steps_ko": [f"단계 {i}" for i in range(n_steps)],
             "mask_restore_ok": ok})


# --------------------------------------------------------------------------- prompts


def test_system_prompts():
    assert SYSTEM_PROMPTS == {"ko": SYSTEM_PROMPT_KO, "en": SYSTEM_PROMPT_EN}
    for marker in ("## Step 1: [Concise description]", "[Brief explanation and calculations]",
                   "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.",
                   "Where [answer] is just the final number or expression"):
        assert marker in SYSTEM_PROMPT_EN, marker
    # the two prompts mirror each other: same number of blocks, no Korean left in the English
    assert SYSTEM_PROMPT_EN.count("\n\n") == SYSTEM_PROMPT_KO.count("\n\n")
    assert not any("가" <= ch <= "힣" for ch in SYSTEM_PROMPT_EN)


def test_generate_and_bon_expose_the_prompt_flag():
    """The flags exist and default to the trained Korean template."""
    from koprm.eval import bon
    from koprm.gen import generate as gen

    sig = gen.generate.__code__.co_varnames
    assert "problem_field" in sig and "system_prompt" in sig
    for mod, flags in ((gen, ("--problem-field", "--system-prompt")),
                       (bon, ("--system-prompt",))):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for f in flags:
            assert f in src, (mod.__name__, f)


def test_teacher_score_file_takes_field_names():
    import koprm.teacher.score as ts

    assert "problem_field" in ts.score_file.__code__.co_varnames
    assert "steps_field" in ts.score_file.__code__.co_varnames


# ------------------------------------------------------------------- math500-en / bon-jsonl


def test_math500_en_rows():
    rows = math500_en_rows([{"problem_id": "math500/1", "problem_en": "A?", "answer": "1",
                             "problem_ko": "가?", "level": "3"}])
    assert rows == [{"problem_id": "math500/1", "problem_en": "A?", "answer": "1"}]


def test_group_completions_orders_and_drops():
    gen = []
    for pid, n in (("p1", 4), ("p2", 4), ("p3", 2)):        # p3 is short
        for k in reversed(range(n)):                         # deliberately out of order
            gen.append({"id": f"{pid}#g#{k}", "problem_id": pid, "sample_idx": k,
                        "text": f"{pid}-{k}"})
    problems = {"p1": {"problem_id": "p1", "problem_en": "A?", "answer": "1"},
                "p3": {"problem_id": "p3", "problem_en": "C?", "answer": "3"}}
    rows, stats = group_completions(gen, problems, "problem_en")
    assert [r["problem_id"] for r in rows] == ["p1"]
    assert rows[0]["completions"] == ["p1-0", "p1-1", "p1-2", "p1-3"]
    assert rows[0]["problem_ko"] == "A?"                     # bon.py reads the query here
    assert rows[0]["answer"] == "1"
    assert stats["n_completions"] == 4
    assert stats["dropped"] == {"short": 1, "no_problem_text": 1}   # p3 short, p2 unknown
    assert stats["completion_counts"] == {2: 1, 4: 2}

    # an explicit count cuts instead of guessing
    rows2, stats2 = group_completions(gen, problems, "problem_en", n_completions=2)
    assert len(rows2[0]["completions"]) == 2 and stats2["dropped"] == {"no_problem_text": 1}


# --------------------------------------------------------------------- processbench-rows


def test_processbench_rows_join_and_outcome():
    pb = [_pb_row("a", "gsm8k", 1), _pb_row("b", "math", -1),
          _pb_row("c", "math", 0), _pb_row("d", "olympiadbench", 2)]
    ko_p, ko_s = {}, {}
    for rid, ok, n in (("a", True, 3), ("b", True, 3), ("c", False, 3), ("d", True, 2)):
        tp, ts = _trans(rid, ok, n)
        ko_p[rid], ko_s[rid] = tp, ts
    rows, stats = processbench_rows(pb, ko_p, ko_s)

    assert [r["id"] for r in rows] == ["a", "b"]
    a, b = rows
    assert (a["outcome"], a["human_first_error"], a["label"]) == (0, 1, 1)
    assert (b["outcome"], b["human_first_error"]) == (1, -1)
    assert a["n_steps"] == 3 and a["problem_ko"] == "문제 a" and a["steps_ko"][0] == "단계 0"
    assert a["steps_en"][0] == "step 0" and a["mask_restore_ok"] is True
    assert stats["in_by_split"] == {"gsm8k": 1, "math": 2, "olympiadbench": 1}
    assert stats["kept_by_split"] == {"gsm8k": 1, "math": 1}
    assert stats["dropped"] == {"math/mask_restore_fail": 1,
                                "olympiadbench/len_mismatch": 1}

    missing = processbench_rows([_pb_row("z", "math", 0)], {}, {})[1]
    assert missing["dropped"] == {"math/no_translation": 1}


# ------------------------------------------------------------------------- first-error


def test_pb_metrics_is_the_processbench_triple():
    rows = [{"outcome": 0, "human_first_error": 1}, {"outcome": 0, "human_first_error": 2},
            {"outcome": 1, "human_first_error": -1}, {"outcome": 1, "human_first_error": -1}]
    m = pb_metrics(rows, [1, 3, -1, 0])
    assert (m["n_error_rows"], m["n_correct_rows"]) == (2, 2)
    assert m["err_acc"] == 0.5                       # only the first error row is exact
    assert m["within1"] == 1.0                       # the second is off by one
    assert m["corr_acc"] == 0.5                      # one correct row is falsely flagged
    assert m["f1"] == pytest.approx(0.5)
    assert pb_metrics([{"outcome": 1, "human_first_error": -1}], [0])["f1"] == 0.0


def test_first_error_report_per_split_and_skips():
    rows = [
        {"id": "a", "split": "gsm8k", "outcome": 0, "human_first_error": 1},
        {"id": "b", "split": "gsm8k", "outcome": 1, "human_first_error": -1},
        {"id": "c", "split": "math", "outcome": 0, "human_first_error": 0},
        {"id": "d", "split": "math", "outcome": 1, "human_first_error": -1},
    ]
    probs = [[0.9, 0.2, 0.8], [0.9, 0.9, 0.9], [0.4, 0.9, 0.9], None]
    rep = first_error_report(rows, probs, threshold=0.5, n_skipped=1)
    assert rep["n_rows"] == 4 and rep["n_scored"] == 3 and rep["n_skipped"] == 1
    assert rep["overall"]["err_acc"] == 1.0          # both error rows are found exactly
    assert rep["overall"]["corr_acc"] == 1.0         # the one scored correct row is silent
    assert rep["overall"]["f1"] == 1.0
    assert set(rep["by_split"]) == {"gsm8k", "math"}
    assert rep["by_split"]["math"]["n_correct_rows"] == 0   # row d was skipped
    assert rep["by_split"]["gsm8k"]["n_error_rows"] == 1


def test_student_and_teacher_probability_paths(tmp_path):
    rows = [
        {"id": "a", "split": "math", "problem_ko": "문제 a", "steps_ko": ["s1", "s2"],
         "outcome": 0, "human_first_error": 1},
        {"id": "b", "split": "math", "problem_ko": "문제 b", "steps_ko": ["s1", "s2"],
         "outcome": 1, "human_first_error": -1},
    ]
    scorer = FakeScorer([[0.9, 0.2], [0.8, 0.7]])
    probs, skipped = student_probs(rows, scorer)
    assert skipped == 0 and probs == [[0.9, 0.2], [0.8, 0.7]]
    assert scorer.seen_problems == ["문제 a", "문제 b"]        # problem_ko is the query
    assert scorer.seen_steps == [["s1", "s2"], ["s1", "s2"]]
    rep = first_error_report(rows, probs)
    assert rep["overall"]["err_acc"] == 1.0 and rep["overall"]["corr_acc"] == 1.0

    path = tmp_path / "teacher.jsonl"
    write_jsonl(path, [{"id": "a", "teacher_logodds": [2.0, -2.0]},
                       {"id": "b", "teacher_logodds": [2.0]}])   # wrong step count
    t_probs, t_skipped = teacher_probs(rows, {r["id"]: r for r in json_rows(path)})
    assert t_skipped == 1 and t_probs[1] is None
    assert t_probs[0][0] > 0.85 and t_probs[0][1] < 0.15
    rep = first_error_report(rows, t_probs, n_skipped=t_skipped)
    assert rep["n_scored"] == 1 and rep["n_skipped"] == 1
    assert rep["overall"]["err_acc"] == 1.0


def json_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
