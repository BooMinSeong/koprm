"""§4.2 On-policy Korean solution generation with vLLM.

Input : problems jsonl with `problem_id`, `problem_ko`, `answer`.
Output: jsonl rows {id, problem_id, generator, sample_idx, text, steps, finish_reason,
        n_tokens} where id = f"{problem_id}#{generator}#{sample_idx}".

A second pass over the same problems (a different --seed, more samples) must not reuse
those ids: --sample-offset N shifts sample_idx (and therefore the id suffix) by N, so
pass 2 with --n 2 --sample-offset 2 continues where pass 1 stopped.

Sharding: --shard i --num-shards k takes every k-th problem, so several processes
(one per GPU) can run side by side; output files are per shard and resumable.
Steps are split on blank lines (§3): the same convention the komath harness uses
for Qwen-style PRMs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from koprm.io import load_jsonl, write_jsonl
from koprm.paths import GENERATORS, STEP_SEP, SYSTEM_PROMPT_KO


def split_steps(text: str) -> list[str]:
    steps = [s.strip() for s in text.split(STEP_SEP)]
    return [s for s in steps if s]


def build_prompts(tokenizer, problems: list[str], system_prompt: str = SYSTEM_PROMPT_KO) -> list[str]:
    convs = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
        for p in problems
    ]
    return tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)


def generate(
    problems: list[dict],
    generator: str,
    n: int,
    out_path: Path,
    temperature: float = 0.8,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    max_model_len: int = 4096,
    seed: int = 0,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 1,
    batch_problems: int = 512,
    sample_offset: int = 0,
) -> None:
    from vllm import LLM, SamplingParams

    model_path = GENERATORS.get(generator, generator)
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        seed=seed,
        tensor_parallel_size=tensor_parallel_size,
    )
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)

    done = set()
    if out_path.exists():
        done = {r["problem_id"] for r in load_jsonl(out_path)}
    todo = [p for p in problems if p["problem_id"] not in done]
    print(f"[gen] {generator}: {len(todo)} problems to do ({len(done)} already done) -> {out_path}")
    for b in range(0, len(todo), batch_problems):
        chunk = todo[b : b + batch_problems]
        prompts = build_prompts(tok, [p["problem_ko"] for p in chunk])
        outs = llm.generate(prompts, sp, use_tqdm=True)
        rows = []
        for p, o in zip(chunk, outs):
            for j, c in enumerate(o.outputs):
                k = j + sample_offset
                rows.append(
                    {
                        "id": f"{p['problem_id']}#{generator}#{k}",
                        "problem_id": p["problem_id"],
                        "generator": generator,
                        "sample_idx": k,
                        "text": c.text,
                        "steps": split_steps(c.text),
                        "finish_reason": c.finish_reason,
                        "n_tokens": len(c.token_ids),
                    }
                )
        write_jsonl(out_path, rows, append=True)
        print(f"[gen] wrote {len(rows)} rows ({b + len(chunk)}/{len(todo)} problems)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--generator", required=True, choices=list(GENERATORS) + ["custom"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--sample-offset", type=int, default=0,
                    help="shift sample_idx (and the id suffix) so a second pass cannot collide")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    problems = load_jsonl(args.problems)
    problems = [p for i, p in enumerate(problems) if i % args.num_shards == args.shard]
    if args.limit:
        problems = problems[: args.limit]
    out = Path(args.out)
    if args.num_shards > 1:
        out = out.with_name(f"{out.stem}.shard{args.shard}of{args.num_shards}{out.suffix}")
    generate(
        problems,
        args.model_path or args.generator,
        args.n,
        out,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        seed=args.seed + args.shard,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tp,
        sample_offset=args.sample_offset,
    )


if __name__ == "__main__":
    main()
