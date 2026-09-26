"""Diverse Verifier Tree Search (DVTS) with the student PRM, against BoN rescoring (§2.4).

DVTS (Beeching et al., HF search-and-learn `sal/search/diverse_verifier_tree_search.py`)
splits a budget of N samples per problem into N/M independent subtrees of beam width M.
Each iteration, every unfinished subtree's current prefix gets M candidate next steps from
the generator (T=0.8, top_p=1.0, stopped at the step separator "\\n\\n"); every candidate
prefix (steps so far + candidate step) is scored with the PRM and the subtree keeps its best
candidate (first one wins ties). A candidate's score is the PRM probability at its *last* step:
the student's labels are cumulative ("the prefix is still correct"), so the last step is the
prefix score, the same `last` aggregation BoN uses. A subtree finishes when its chosen
candidate ends the solution (EOS), carries a closed \\boxed{} answer (--stop-at-boxed, the
default), produces no new step, or hits a length cap (--step-tokens per step,
--max-tokens per solution). The last iteration (--max-iterations) generates to completion
without the step stop string. One subtree gives one solution, so a problem ends with N/M
solutions; answers are then picked by naive (highest last-step score) / weighted (score sum
per answer-equivalence group) / majority vote, with koprm.eval.bon's machinery.

Prompt and step format match the stored KO MATH500 generations: the generator sees
`build_prompts` (komath's SYSTEM_PROMPT_KO, EXAONE-4.0 non-thinking by default) followed by
the prefix, i.e. the chosen steps joined with "\\n\\n" plus a trailing "\\n\\n". Steps are kept
canonical (`split_steps`: split on "\\n\\n", stripped, empties dropped), so the step list the
PRM scored is exactly `split_steps` of the final completion "\\n\\n".join(steps).

Engineering: one process per GPU holds the vLLM generator (small `gpu_memory_utilization`,
prefix caching) and the StudentScorer (bf16) side by side. Every iteration makes one
`llm.generate` call for all active subtrees of all problems in the shard
(`SamplingParams(n=M)` per prefix, a per-request seed derived from
(seed, problem, subtree, iteration) so the subtrees of one problem differ and a rerun is
reproducible) and one token-budgeted scoring pass; duplicate candidate prefixes are scored
once. A problem's row is appended to the shard file as soon as all its subtrees finish, so a
restart skips finished problems.

    # one GPU, smoke test
    CUDA_VISIBLE_DEVICES=0 python -m koprm.eval.dvts run --n 16 --beam-width 4 --limit 5 \\
        --scorer data/ckpt/qwen3-8b_B_24k_soft/epoch3 --out data/eval/dvts/smoke/n16.jsonl
    # 8 shards (one per GPU): writes <out stem>.shard{i}of8.jsonl
    CUDA_VISIBLE_DEVICES=$i python -m koprm.eval.dvts run ... --shard $i --num-shards 8
    # merge + metrics + paired bootstrap against BoN of the same student
    python -m koprm.eval.dvts eval --in 'data/eval/dvts/x.shard*of8.jsonl' \\
        --bon-scores data/eval/math500_7b/qwen3-8b_B_24k_soft_ep3_EXAONE-4.0-1.2B.scores.jsonl \\
        --out data/eval/dvts/x.json

Rows: {problem_id, problem_ko, answer, level, n, beam_width, completions (N/M), steps,
scores (per-step PRM probabilities of each final path), finish, iterations, path_tokens,
gen_tokens (all M candidates of every iteration: the compute to compare with BoN's
tokens), final_candidates (per subtree the M candidates of its last iteration:
base_n_steps, texts, scores, n_tokens, finish)}. `scores`/`completions` load with
`bon.load_jsonl_rows`.

Deviations from search-and-learn: M samples per prefix via n=M instead of M copies of the
prompt; per-step token cap (512) and a total cap (2048) instead of 2048 per step; the
boxed-answer finish and the canonical step text (S&L concatenates raw text, including the
stop string); S&L returns N = (N/M) x M completions by pairing each subtree's pre-final
prefix with all M final candidates. The run keeps both: `completions` are the N/M chosen
solutions and `expand_pool` rebuilds the S&L pool from `final_candidates` (`pool_all` in
`eval`). The S&L pool is the standard DVTS output and what the report uses; the chosen-path
pool votes over only N/M solutions and is kept as a diagnostic. `eval` takes smaller budgets
from the first b/M subtrees; scripts/dvts_compare.py averages over random subsets of b/M
subtrees instead and compares against komath's English-PRM DVTS.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from koprm.data.sources import last_boxed
from koprm.eval.bon import (METHODS, bootstrap_ci, evaluate, load_hf_rows, load_scores,
                            n_grid, split_steps)
from koprm.io import load_jsonl, write_jsonl
from koprm.paths import EVAL, GENERATORS, STEP_SEP, SYSTEM_PROMPT_KO

DATASET = "ENSEONG/ko-ko-math-500-test-EXAONE-4.0-1.2B-bon"
DATASET_CONFIG = "ENSEONG_ko-math-500-test--T-0.8--top_p-1.0--n-64--seed-0--agg_strategy-last"
BON_SCORES = EVAL / "math500_7b" / "qwen3-8b_B_24k_soft_ep3_EXAONE-4.0-1.2B.scores.jsonl"


# ------------------------------------------------------------------ data types
@dataclass
class Request:
    prompt: str
    max_tokens: int
    stop: bool  # stop at STEP_SEP (False on the last iteration)
    seed: int


@dataclass
class Candidate:
    text: str
    n_tokens: int
    finish_reason: str  # vLLM: "stop" or "length"
    stop_reason: str | int | None = None  # the stop string, or None when the model ended (EOS)


@dataclass
class SearchConfig:
    beam_width: int = 4
    max_iterations: int = 40
    step_tokens: int = 512
    max_tokens: int = 2048
    stop_at_boxed: bool = True
    seed: int = 0


@dataclass
class Subtree:
    problem: int  # index into the problem list
    index: int
    steps: list[str] = field(default_factory=list)
    step_scores: list[float] = field(default_factory=list)
    path_tokens: int = 0
    gen_tokens: int = 0
    iterations: int = 0
    finish: str | None = None
    final_candidates: dict | None = None

    @property
    def finished(self) -> bool:
        return self.finish is not None


# ------------------------------------------------------------- pure helpers
def prefix_text(steps: list[str]) -> str:
    """The assistant text after the template: steps joined by STEP_SEP, plus a trailing one."""
    return STEP_SEP.join(steps) + STEP_SEP if steps else ""


def request_seed(seed: int, problem_key, subtree: int, iteration: int) -> int:
    return zlib.crc32(f"{seed}|{problem_key}|{subtree}|{iteration}".encode()) & 0x7FFFFFFF


def has_boxed_answer(steps: list[str]) -> bool:
    return any("\\boxed{" in s and last_boxed(s) is not None for s in steps)


def candidate_end(c: Candidate) -> str:
    """How a generation ended: "eos" (the model ended), "length" (a token cap), "step"."""
    if c.finish_reason == "length":
        return "length"
    return "eos" if c.stop_reason is None else "step"


def finish_reason(cand: Candidate, new_steps: list[str], path_tokens: int,
                  cfg: SearchConfig, last_iteration: bool) -> str | None:
    """Why the subtree ends after choosing `cand` (None: keep searching)."""
    end = candidate_end(cand)
    if end == "eos":
        return "eos"
    if end == "length" or path_tokens >= cfg.max_tokens:
        return "length"
    if not new_steps:
        return "empty"
    if cfg.stop_at_boxed and has_boxed_answer(new_steps):
        return "boxed"
    if last_iteration:
        return "max_iter"
    return None


def best_index(scores: list[float]) -> int:
    """argmax, first one wins ties (np.argmax in search-and-learn)."""
    return max(range(len(scores)), key=lambda j: (scores[j], -j))


def make_requests(trees: list[Subtree], prompts: list[str], keys: list, cfg: SearchConfig,
                  iteration: int) -> list[Request]:
    last = iteration == cfg.max_iterations - 1
    reqs = []
    for t in trees:
        remaining = cfg.max_tokens - t.path_tokens
        cap = remaining if last else min(cfg.step_tokens, remaining)
        reqs.append(Request(prompt=prompts[t.problem] + prefix_text(t.steps),
                            max_tokens=max(1, cap), stop=not last,
                            seed=request_seed(cfg.seed, keys[t.problem], t.index, iteration)))
    return reqs


def score_candidates(trees: list[Subtree], cands: list[list[Candidate]],
                     problems: list[str], score_fn) -> list[list[tuple[list[str], list[float]]]]:
    """Per subtree, per candidate: (full step list, per-step PRM scores). Candidates that add
    no step keep the prefix's scores; identical prefixes are scored once."""
    todo: dict[tuple[int, tuple[str, ...]], int] = {}
    flat_p: list[str] = []
    flat_s: list[list[str]] = []
    full: list[list[list[str]]] = []
    for t, cs in zip(trees, cands):
        row = []
        for c in cs:
            steps = t.steps + split_steps(c.text)
            row.append(steps)
            key = (t.problem, tuple(steps))
            if len(steps) > len(t.steps) and key not in todo:
                todo[key] = len(flat_p)
                flat_p.append(problems[t.problem])
                flat_s.append(steps)
        full.append(row)
    flat_scores = score_fn(flat_p, flat_s) if flat_p else []
    out = []
    for t, row in zip(trees, full):
        res = []
        for steps in row:
            if len(steps) > len(t.steps):
                res.append((steps, list(flat_scores[todo[(t.problem, tuple(steps))]])))
            else:
                res.append((steps, list(t.step_scores)))
        out.append(res)
    return out


def advance(t: Subtree, cands: list[Candidate], scored: list[tuple[list[str], list[float]]],
            cfg: SearchConfig, last_iteration: bool) -> None:
    """Keep the best candidate of one subtree and update its bookkeeping in place."""
    last_scores = [s[-1] if s else 0.0 for _, s in scored]
    b = best_index(last_scores)
    base = len(t.steps)
    steps, scores = scored[b]
    new_steps = steps[base:]
    t.final_candidates = {
        "base_n_steps": base,
        "texts": [c.text for c in cands],
        "scores": [s for _, s in scored],
        "n_tokens": [c.n_tokens for c in cands],
        "finish": [candidate_end(c) for c in cands],
        "chosen": b,
    }
    t.steps, t.step_scores = steps, scores
    t.path_tokens += cands[b].n_tokens
    t.gen_tokens += sum(c.n_tokens for c in cands)
    t.iterations += 1
    t.finish = finish_reason(cands[b], new_steps, t.path_tokens, cfg, last_iteration)


def build_row(problem: dict, trees: list[Subtree], cfg: SearchConfig, n: int) -> dict:
    return {
        "problem_id": problem["problem_id"],
        "problem_ko": problem["problem_ko"],
        "answer": problem["answer"],
        "level": problem.get("level"),
        "n": n,
        "beam_width": cfg.beam_width,
        "completions": [STEP_SEP.join(t.steps) for t in trees],
        "steps": [t.steps for t in trees],
        "scores": [t.step_scores for t in trees],
        "finish": [t.finish for t in trees],
        "iterations": [t.iterations for t in trees],
        "path_tokens": [t.path_tokens for t in trees],
        "gen_tokens": sum(t.gen_tokens for t in trees),
        "final_candidates": [t.final_candidates for t in trees],
    }


def expand_pool(row: dict) -> tuple[list[str], list[list[float]]]:
    """search-and-learn's N = (N/M) x M pool: each subtree's pre-final prefix + each of its
    last-iteration candidates (only the chosen one is necessarily a finished solution)."""
    comps, scores = [], []
    for steps, fc in zip(row["steps"], row["final_candidates"]):
        base = steps[: fc["base_n_steps"]]
        for text, sc in zip(fc["texts"], fc["scores"]):
            comps.append(STEP_SEP.join(base + split_steps(text)))
            scores.append(sc)
    return comps, scores


# ------------------------------------------------------------------ search loop
def dvts_search(problems: list[dict], prompts: list[str], generate_fn, score_fn, n: int,
                cfg: SearchConfig, on_done=None, log=print) -> list[dict]:
    """Run DVTS over all `problems` at once. `generate_fn(list[Request]) -> list[list[Candidate]]`
    (beam_width candidates per request), `score_fn(problems_ko, steps_lists) -> per-step
    scores`. `on_done(row)` is called as soon as a problem's subtrees have all finished."""
    if n % cfg.beam_width:
        raise ValueError(f"n={n} is not a multiple of beam_width={cfg.beam_width}")
    n_sub = n // cfg.beam_width
    keys = [p["problem_id"] for p in problems]
    texts = [p["problem_ko"] for p in problems]
    trees = [Subtree(problem=i, index=j) for i in range(len(problems)) for j in range(n_sub)]
    by_problem = [trees[i * n_sub:(i + 1) * n_sub] for i in range(len(problems))]
    open_problems = set(range(len(problems)))
    rows: list[dict | None] = [None] * len(problems)
    for it in range(cfg.max_iterations):
        active = [t for t in trees if not t.finished]
        if not active:
            break
        last = it == cfg.max_iterations - 1
        t0 = time.time()
        cands = generate_fn(make_requests(active, prompts, keys, cfg, it))
        t1 = time.time()
        for t, cs in zip(active, cands):
            if len(cs) != cfg.beam_width:
                raise RuntimeError(f"expected {cfg.beam_width} candidates, got {len(cs)}")
        scored = score_candidates(active, cands, texts, score_fn)
        t2 = time.time()
        for t, cs, sc in zip(active, cands, scored):
            advance(t, cs, sc, cfg, last)
        for i in sorted(open_problems):
            if all(t.finished for t in by_problem[i]):
                open_problems.discard(i)
                rows[i] = build_row(problems[i], by_problem[i], cfg, n)
                if on_done is not None:
                    on_done(rows[i])
        n_gen = sum(c.n_tokens for cs in cands for c in cs)
        log(f"[dvts] iter {it + 1}: {len(active)} subtrees, {n_gen} tokens, gen {t1 - t0:.1f}s, "
            f"score {t2 - t1:.1f}s, {len(open_problems)} problems open")
    return [r for r in rows if r is not None]


# --------------------------------------------------------------- GPU backends
class VLLMGenerator:
    def __init__(self, model: str, beam_width: int, temperature: float = 0.8,
                 top_p: float = 1.0, gpu_memory_utilization: float = 0.25,
                 max_model_len: int = 4096, seed: int = 0):
        from vllm import LLM  # koprm.paths (imported above) set VLLM_USE_FLASHINFER_SAMPLER

        self.llm = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enable_prefix_caching=True, seed=seed)
        self.tokenizer = self.llm.get_tokenizer()
        self.m = beam_width
        self.temperature = temperature
        self.top_p = top_p

    def __call__(self, requests: list[Request]) -> list[list[Candidate]]:
        from vllm import SamplingParams

        sps = [SamplingParams(n=self.m, temperature=self.temperature, top_p=self.top_p,
                              max_tokens=r.max_tokens, seed=r.seed,
                              stop=[STEP_SEP] if r.stop else None) for r in requests]
        outs = self.llm.generate([r.prompt for r in requests], sps, use_tqdm=False)
        return [[Candidate(c.text, len(c.token_ids), c.finish_reason, c.stop_reason)
                 for c in o.outputs] for o in outs]


def shard_path(out: str | Path, shard: int, num_shards: int) -> Path:
    out = Path(out)
    if num_shards > 1:
        out = out.with_name(f"{out.stem}.shard{shard}of{num_shards}{out.suffix}")
    return out


def run(args) -> None:
    from koprm.gen.generate import build_prompts

    rows = load_hf_rows(args.dataset, args.dataset_config)
    problems = [{"problem_id": r["problem_id"], "problem_ko": r["problem_ko"],
                 "answer": r["answer"], "level": r.get("level")} for r in rows]
    problems = [p for i, p in enumerate(problems) if i % args.num_shards == args.shard]
    if args.limit:
        problems = problems[: args.limit]
    out = shard_path(args.out, args.shard, args.num_shards)
    done = {r["problem_id"] for r in load_jsonl(out)} if out.exists() else set()
    todo = [p for p in problems if p["problem_id"] not in done]
    print(f"[dvts] n={args.n} M={args.beam_width}: {len(todo)} problems to do "
          f"({len(done)} already done) -> {out}")
    if not todo:
        return

    gen = VLLMGenerator(GENERATORS.get(args.generator, args.generator), args.beam_width,
                        temperature=args.temperature, top_p=args.top_p,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        max_model_len=args.max_model_len, seed=args.seed)
    from koprm.eval.scorer import StudentScorer

    scorer = StudentScorer(args.scorer, device=args.device, max_len=args.max_model_len)

    def score_fn(ps, ss):
        return scorer.score(ps, ss, batch_size=args.score_batch_size,
                            max_batch_tokens=args.score_max_tokens)

    chat_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None
    prompts = build_prompts(gen.tokenizer, [p["problem_ko"] for p in todo], SYSTEM_PROMPT_KO,
                            chat_kwargs)
    cfg = SearchConfig(beam_width=args.beam_width, max_iterations=args.max_iterations,
                       step_tokens=args.step_tokens, max_tokens=args.max_tokens,
                       stop_at_boxed=not args.no_stop_at_boxed, seed=args.seed)
    t0 = time.time()
    for b in range(0, len(todo), args.batch_problems or len(todo)):
        chunk = todo[b: b + (args.batch_problems or len(todo))]
        chunk_prompts = prompts[b: b + len(chunk)]
        dvts_search(chunk, chunk_prompts, gen, score_fn, args.n, cfg,
                    on_done=lambda row: write_jsonl(out, [row], append=True))
        print(f"[dvts] {b + len(chunk)}/{len(todo)} problems, {time.time() - t0:.0f}s")
    print(f"[dvts] done in {time.time() - t0:.0f}s (truncated PRM inputs: {scorer.n_truncated})")


# ------------------------------------------------------------------- evaluation
def merge_shards(patterns: list[str]) -> list[dict]:
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"no files match {patterns}")
    rows: dict = {}
    for p in paths:
        for r in load_jsonl(p):
            rows.setdefault(r["problem_id"], r)
    print(f"[dvts] {len(rows)} problems from {len(paths)} files")
    return list(rows.values())


def paired_diff(a: list[int], b: list[int]) -> dict:
    return bootstrap_ci([x - y for x, y in zip(a, b)])


def bon_tokens(rows: list[dict], generator: str) -> list[list[int]]:
    from koprm.train.model import load_tokenizer

    tok = load_tokenizer(GENERATORS.get(generator, generator))
    return [[len(ids) for ids in tok(list(r["completions"]), add_special_tokens=False)["input_ids"]]
            for r in rows]


def evaluate_dvts(args) -> dict:
    cache_dir = None if args.no_cache else EVAL / "cache"
    dv = merge_shards(args.inputs)
    hf = load_hf_rows(args.dataset, args.dataset_config)
    order = {r["problem_id"]: i for i, r in enumerate(hf)}
    unknown = [r["problem_id"] for r in dv if r["problem_id"] not in order]
    if unknown:
        raise SystemExit(f"{len(unknown)} DVTS problems are not in {args.dataset}: {unknown[:3]}")
    dv.sort(key=lambda r: order[r["problem_id"]])
    idx = [order[r["problem_id"]] for r in dv]
    if len(dv) < len(hf):
        print(f"[dvts] WARNING: {len(dv)}/{len(hf)} problems; BoN is restricted to the same ones")
    n, m = dv[0]["n"], dv[0]["beam_width"]
    pool = min(len(r["completions"]) for r in dv)

    # Subtrees are independent, so the first b/M subtrees of an N-budget run are a DVTS run
    # at budget b (search-and-learn evaluates smaller n the same way, from one large-N run).
    budgets = sorted({b for b in args.bon_ns if b % m == 0 and b // m <= pool} | {n})
    res_pool = evaluate(dv, [r["scores"] for r in dv], agg="last", cache_dir=cache_dir,
                        ns=n_grid(pool) + [pool] + [b // m for b in budgets], verbose=False)
    exp = [expand_pool(r) for r in dv]
    rows_all = [dict(r, completions=c, scores=s) for r, (c, s) in zip(dv, exp)]
    n_all = min(len(c) for c, _ in exp)
    res_all = evaluate(rows_all, [s for _, s in exp], agg="last", cache_dir=cache_dir,
                       ns=[b for b in budgets if b <= n_all] + [n_all], verbose=False)

    bon = evaluate(hf, load_scores(args.bon_scores, hf), agg="last", cache_dir=cache_dir,
                   verbose=False)
    bon_pp = {k: [v[i] for i in idx] for k, v in bon["per_problem"].items()}
    bon_ns = [b for b in args.bon_ns if f"naive@{b}" in bon_pp]

    def block(res, k):
        return {meth: {**bootstrap_ci(res["per_problem"][f"{meth}@{k}"])}
                for meth in (*METHODS, "pass")}

    out = {
        "n": n, "beam_width": m, "pool": pool, "n_problems": len(dv),
        "source": args.inputs, "bon_scores": str(args.bon_scores),
        "dvts": block(res_pool, pool),
        "dvts_curve": {str(k): res_pool["metrics"][str(k)] for k in res_pool["ns"]},
        "pool_all": {"n": n_all, **block(res_all, n_all)},
        # per compute budget b: DVTS from the first b/M subtrees, the S&L pool from its first
        # b completions (the same subtrees), and BoN@b, compared at equal b
        "budgets": {str(b): {"subtrees": b // m,
                             "dvts": block(res_pool, b // m),
                             "pool_all": block(res_all, b) if b <= n_all else None}
                    for b in budgets},
        "bon": {str(b): {meth: bootstrap_ci(bon_pp[f"{meth}@{b}"]) for meth in (*METHODS, "pass")}
                for b in bon_ns},
        "diff_vs_bon": {str(b): {meth: paired_diff(res_pool["per_problem"][f"{meth}@{b // m}"],
                                                   bon_pp[f"{meth}@{b}"]) for meth in METHODS}
                        for b in bon_ns if b in budgets},
        "diff_pool_all_vs_bon": {
            str(b): {meth: paired_diff(res_all["per_problem"][f"{meth}@{b}"],
                                       bon_pp[f"{meth}@{b}"]) for meth in METHODS}
            for b in bon_ns if b in budgets and b <= n_all},
    }
    fin: dict[str, int] = {}
    for r in dv:
        for f in r["finish"]:
            fin[f] = fin.get(f, 0) + 1
    it = [x for r in dv for x in r["iterations"]]
    out["stats"] = {
        "finish": fin,
        "iterations_mean": sum(it) / len(it),
        "iterations_max": max(it),
        "gen_tokens_mean": sum(r["gen_tokens"] for r in dv) / len(dv),
        "path_tokens_mean": sum(sum(r["path_tokens"]) for r in dv) / len(dv),
    }
    if not args.no_bon_tokens:
        toks = bon_tokens([hf[i] for i in idx], args.generator)
        out["stats"]["bon_tokens_mean"] = {str(b): sum(sum(t[:b]) for t in toks) / len(toks)
                                           for b in bon_ns}
    out["per_problem"] = {"problem_ids": [r["problem_id"] for r in dv],
                          **{f"{meth}@{b}": res_pool["per_problem"][f"{meth}@{b // m}"]
                             for meth in METHODS for b in budgets}}
    return out


def print_eval(out: dict) -> None:
    def f(c):
        return f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"

    print(f"\n[dvts] N={out['n']} M={out['beam_width']}  pool={out['pool']}  "
          f"problems={out['n_problems']}")
    for b, bb in out["budgets"].items():
        print(f"  budget {b} ({bb['subtrees']} subtrees)")
        for meth in (*METHODS, "pass"):
            line = f"    {meth:>8}: DVTS {f(bb['dvts'][meth])}"
            if bb["pool_all"]:
                line += f"  S&L pool {bb['pool_all'][meth]['mean']:.3f}"
            if b in out["bon"]:
                line += f"  BoN {out['bon'][b][meth]['mean']:.3f}"
            print(line)
    for name, key in (("DVTS", "diff_vs_bon"), ("S&L pool", "diff_pool_all_vs_bon")):
        for b, d in out[key].items():
            print(f"  {name} - BoN@{b}: " + "  ".join(
                f"{meth} {d[meth]['mean']:+.3f} [{d[meth]['lo']:+.3f}, {d[meth]['hi']:+.3f}]"
                for meth in METHODS))
    print(f"  stats: {json.dumps(out['stats'], ensure_ascii=False)}")


# ------------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="DVTS search on one shard")
    r.add_argument("--dataset", default=DATASET, help="stored BoN dataset (problem source)")
    r.add_argument("--dataset-config", default=DATASET_CONFIG)
    r.add_argument("--generator", default="exaone-1.2b")
    r.add_argument("--scorer", required=True, help="student checkpoint dir")
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=16)
    r.add_argument("--beam-width", type=int, default=4)
    r.add_argument("--max-iterations", type=int, default=40)
    r.add_argument("--step-tokens", type=int, default=512)
    r.add_argument("--max-tokens", type=int, default=2048, help="per solution")
    r.add_argument("--temperature", type=float, default=0.8)
    r.add_argument("--top-p", type=float, default=1.0)
    r.add_argument("--no-stop-at-boxed", action="store_true",
                   help="do not end a subtree at a step with a closed \\boxed{} answer")
    r.add_argument("--chat-kwargs", default=None,
                   help='json for apply_chat_template, e.g. \'{"enable_thinking": false}\'')
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--batch-problems", type=int, default=0,
                   help="problems searched together (0 = the whole shard)")
    r.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    r.add_argument("--max-model-len", type=int, default=4096)
    r.add_argument("--device", default="cuda")
    r.add_argument("--score-batch-size", type=int, default=64)
    r.add_argument("--score-max-tokens", type=int, default=32768,
                   help="padded tokens per PRM batch")

    e = sub.add_parser("eval", help="merge shards, metrics, paired bootstrap vs BoN")
    e.add_argument("--in", dest="inputs", nargs="+", required=True, help="files or globs")
    e.add_argument("--out", required=True)
    e.add_argument("--dataset", default=DATASET)
    e.add_argument("--dataset-config", default=DATASET_CONFIG)
    e.add_argument("--bon-scores", default=str(BON_SCORES),
                   help="bon.py --save-scores file of the same student on --dataset")
    e.add_argument("--bon-ns", type=int, nargs="+", default=[16, 64])
    e.add_argument("--generator", default="exaone-1.2b", help="tokenizer for BoN token counts")
    e.add_argument("--no-bon-tokens", action="store_true")
    e.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.cmd == "run":
        run(args)
    else:
        out = evaluate_dvts(args)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print_eval(out)
        print(f"[dvts] wrote {args.out}")


if __name__ == "__main__":
    main()
