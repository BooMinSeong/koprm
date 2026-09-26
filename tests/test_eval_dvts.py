import re

import pytest

from koprm.eval.bon import evaluate, load_jsonl_rows, split_steps
from koprm.eval.dvts import (
    Candidate,
    SearchConfig,
    Subtree,
    advance,
    best_index,
    build_row,
    dvts_search,
    expand_pool,
    finish_reason,
    has_boxed_answer,
    make_requests,
    prefix_text,
    request_seed,
    score_candidates,
)
from koprm.eval.scorer import token_batches
from koprm.io import write_jsonl

PROMPT = "<prompt>"


def fake_score(problems, steps_lists):
    """Per-step score = the number after 's' in the step text (0.5 if none)."""
    out = []
    for steps in steps_lists:
        out.append([float(m.group(1)) if (m := re.search(r"s([0-9.]+)", s)) else 0.5
                    for s in steps])
    return out


class FakeGenerator:
    """Candidate j at depth d (number of steps in the prefix) is `d{d} c{j} s{score}`;
    `scores[d][j]` sets its score. At depth `final_depth` the candidates end the solution."""

    def __init__(self, scores, final_depth, eos=True, m=4):
        self.scores, self.final_depth, self.eos, self.m = scores, final_depth, eos, m
        self.requests = []

    def __call__(self, requests):
        self.requests.append(requests)
        out = []
        for r in requests:
            prefix = r.prompt[len(PROMPT):]
            d = len(split_steps(prefix))
            cs = []
            for j in range(self.m):
                sc = self.scores[min(d, len(self.scores) - 1)][j]
                if d >= self.final_depth:
                    text = f"d{d} c{j} s{sc} 답은 $\\boxed{{{j}}}$"
                    cs.append(Candidate(text, 7, "stop", None if self.eos else "\n\n"))
                else:
                    cs.append(Candidate(f"d{d} c{j} s{sc}", 5, "stop", "\n\n"))
            out.append(cs)
        return out


def _problem(pid="p1"):
    return {"problem_id": pid, "problem_ko": "문제", "answer": "0", "level": "1"}


def test_prefix_text_round_trips_through_split_steps():
    steps = ["## 단계 1: a\n계산", "## 단계 2: b", "따라서 $\\boxed{3}$"]
    assert prefix_text([]) == ""
    assert prefix_text(steps) == "\n\n".join(steps) + "\n\n"
    assert split_steps("\n\n".join(steps)) == steps
    assert split_steps(prefix_text(steps)) == steps


def test_best_index_first_wins_ties():
    assert best_index([0.2, 0.9, 0.9, 0.1]) == 1
    assert best_index([0.5, 0.5]) == 0
    assert best_index([0.1]) == 0


def test_has_boxed_answer():
    assert has_boxed_answer(["x", "따라서 $\\boxed{\\frac{1}{2}}$입니다."])
    assert not has_boxed_answer(["\\boxed{unclosed"])
    assert not has_boxed_answer(["no answer"])


def test_finish_reason():
    cfg = SearchConfig(max_tokens=100)
    step = Candidate("a", 5, "stop", "\n\n")
    assert finish_reason(step, ["a"], 10, cfg, False) is None
    assert finish_reason(Candidate("a", 5, "stop", None), ["a"], 10, cfg, False) == "eos"
    assert finish_reason(Candidate("a", 5, "length", None), ["a"], 10, cfg, False) == "length"
    assert finish_reason(step, ["a"], 100, cfg, False) == "length"  # total budget used up
    assert finish_reason(step, [], 10, cfg, False) == "empty"
    assert finish_reason(step, ["$\\boxed{1}$"], 10, cfg, False) == "boxed"
    no_box = SearchConfig(max_tokens=100, stop_at_boxed=False)
    assert finish_reason(step, ["$\\boxed{1}$"], 10, no_box, False) is None
    assert finish_reason(step, ["a"], 10, cfg, True) == "max_iter"


def test_make_requests_caps_tokens_and_drops_stop_on_last_iteration():
    cfg = SearchConfig(max_iterations=3, step_tokens=512, max_tokens=600)
    t = Subtree(problem=0, index=1, steps=["a", "b"], path_tokens=200)
    (r,) = make_requests([t], [PROMPT], ["p1"], cfg, 0)
    assert r.prompt == PROMPT + "a\n\nb\n\n"
    assert r.max_tokens == 400 and r.stop
    (r,) = make_requests([t], [PROMPT], ["p1"], cfg, 2)
    assert r.max_tokens == 400 and not r.stop
    t0 = Subtree(problem=0, index=0)
    (r,) = make_requests([t0], [PROMPT], ["p1"], cfg, 0)
    assert r.prompt == PROMPT and r.max_tokens == 512


def test_request_seed_differs_per_subtree_and_iteration():
    s = {request_seed(0, "p1", j, it) for j in range(4) for it in range(3)}
    assert len(s) == 12
    assert request_seed(0, "p1", 1, 2) == request_seed(0, "p1", 1, 2)
    assert request_seed(1, "p1", 1, 2) != request_seed(0, "p1", 1, 2)


def test_score_candidates_dedupes_and_keeps_prefix_scores_for_empty():
    calls = []

    def scorer(ps, ss):
        calls.append(len(ss))
        return fake_score(ps, ss)

    t = Subtree(problem=0, index=0, steps=["x s0.7"], step_scores=[0.7])
    cands = [Candidate("y s0.2", 3, "stop", "\n\n"), Candidate("y s0.2", 3, "stop", "\n\n"),
             Candidate("\n", 1, "stop", "\n\n"), Candidate("z s0.9\n\nw s0.4", 6, "stop", None)]
    (res,) = score_candidates([t], [cands], ["문제"], scorer)
    assert calls == [2]  # duplicate scored once, empty candidate not scored
    assert res[0] == (["x s0.7", "y s0.2"], [0.7, 0.2])
    assert res[1] == res[0]
    assert res[2] == (["x s0.7"], [0.7])
    assert res[3] == (["x s0.7", "z s0.9", "w s0.4"], [0.7, 0.9, 0.4])


def test_advance_keeps_best_candidate_and_bookkeeping():
    cfg = SearchConfig()
    t = Subtree(problem=0, index=0, steps=["a s0.9"], step_scores=[0.9], path_tokens=5)
    cands = [Candidate(f"b s{v}", 4 + j, "stop", "\n\n") for j, v in enumerate([0.3, 0.8, 0.8, 0.1])]
    scored = score_candidates([t], [cands], ["문제"], fake_score)[0]
    advance(t, cands, scored, cfg, False)
    assert t.steps == ["a s0.9", "b s0.8"] and t.step_scores == [0.9, 0.8]
    assert t.path_tokens == 5 + 5 and t.gen_tokens == 4 + 5 + 6 + 7
    assert t.iterations == 1 and not t.finished
    fc = t.final_candidates
    assert fc["chosen"] == 1 and fc["base_n_steps"] == 1
    assert fc["finish"] == ["step"] * 4


def test_dvts_search_end_to_end_with_fakes():
    # depth 0: candidate 2 best; depth 1: candidate 0 best; depth 2: candidate 3 best (EOS)
    scores = [[0.1, 0.2, 0.9, 0.3], [0.8, 0.2, 0.1, 0.3], [0.1, 0.2, 0.3, 0.95]]
    gen = FakeGenerator(scores, final_depth=2)
    done = []
    cfg = SearchConfig(beam_width=4, max_iterations=10)
    rows = dvts_search([_problem("p1"), _problem("p2")], [PROMPT, PROMPT], gen, fake_score,
                       n=8, cfg=cfg, on_done=done.append, log=lambda *_: None)
    assert len(rows) == 2 and len(done) == 2
    r = rows[0]
    assert r["n"] == 8 and len(r["completions"]) == 2  # N/M subtrees
    assert r["steps"][0] == ["d0 c2 s0.9", "d1 c0 s0.8", "d2 c3 s0.95 답은 $\\boxed{3}$"]
    assert r["scores"][0] == [0.9, 0.8, 0.95]
    assert split_steps(r["completions"][0]) == r["steps"][0]
    assert r["finish"] == ["eos", "eos"] and r["iterations"] == [3, 3]
    assert r["path_tokens"] == [5 + 5 + 7] * 2
    assert r["gen_tokens"] == 2 * 4 * (5 + 5 + 7)
    assert len(gen.requests) == 3 and len(gen.requests[0]) == 4  # 2 problems x 2 subtrees
    assert len({q.seed for q in gen.requests[0]}) == 4
    # S&L pool: each subtree's pre-final prefix + all M final candidates
    comps, sc = expand_pool(r)
    assert len(comps) == 8
    assert split_steps(comps[1]) == ["d0 c2 s0.9", "d1 c0 s0.8", "d2 c1 s0.2 답은 $\\boxed{1}$"]
    assert sc[1] == [0.9, 0.8, 0.2]


def test_dvts_search_boxed_finish_and_last_iteration():
    scores = [[0.5, 0.6, 0.7, 0.8]]
    gen = FakeGenerator(scores, final_depth=1, eos=False)  # boxed step, not EOS
    cfg = SearchConfig(beam_width=4, max_iterations=10)
    (r,) = dvts_search([_problem()], [PROMPT], gen, fake_score, n=4, cfg=cfg,
                       log=lambda *_: None)
    assert r["finish"] == ["boxed"] and r["iterations"] == [2]

    gen = FakeGenerator(scores, final_depth=99)  # never ends by itself
    cfg = SearchConfig(beam_width=4, max_iterations=3)
    (r,) = dvts_search([_problem()], [PROMPT], gen, fake_score, n=4, cfg=cfg,
                       log=lambda *_: None)
    assert r["finish"] == ["max_iter"] and r["iterations"] == [3]
    assert [q.stop for reqs in gen.requests for q in reqs] == [True, True, False]


def test_dvts_search_rejects_bad_budget():
    with pytest.raises(ValueError):
        dvts_search([_problem()], [PROMPT], FakeGenerator([[0.1] * 4], 1), fake_score, n=6,
                    cfg=SearchConfig(beam_width=4), log=lambda *_: None)


def test_rows_load_and_evaluate_with_bon(tmp_path):
    cfg = SearchConfig(beam_width=2)
    trees = [Subtree(problem=0, index=j, steps=["a", f"$\\boxed{{{v}}}$"], step_scores=[0.5, s],
                     finish="eos", iterations=2, final_candidates={"base_n_steps": 1,
                     "texts": ["x"], "scores": [[0.5, 0.1]]})
             for j, (v, s) in enumerate([(1, 0.9), (2, 0.2), (2, 0.3)])]
    row = build_row({"problem_id": "p", "problem_ko": "문제", "answer": "1"}, trees, cfg, 6)
    write_jsonl(tmp_path / "d.jsonl", [row])
    rows = load_jsonl_rows(str(tmp_path / "d.jsonl"))
    res = evaluate(rows, [r["scores"] for r in rows], agg="last", verbose=False, ns=[1, 2, 3])
    assert res["ns"] == [1, 2, 3]
    assert res["metrics"]["3"]["naive"] == 1.0  # boxed 1 has the top last-step score
    assert res["metrics"]["3"]["maj"] == 0.0  # two votes for 2
    assert res["metrics"]["3"]["weighted"] == 1.0  # 0.9 > 0.2 + 0.3


def test_token_batches():
    order = [3, 1, 0, 2]
    lengths = [10, 10, 30, 40]
    assert token_batches(lengths, order, 8) == [[3, 1, 0, 2]]
    assert token_batches(lengths, order, 2) == [[3, 1], [0, 2]]
    # 3 rows x 30 = 90 > 64 -> split before the third
    assert token_batches(lengths, order, 8, max_batch_tokens=64) == [[3, 1], [0], [2]]
    assert token_batches([100], [0], 8, max_batch_tokens=10) == [[0]]
