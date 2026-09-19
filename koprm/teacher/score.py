"""§2.2 Teacher log-odds with vLLM pooling.

Input is assembled directly: "<extra_0>".join(steps_en) + "<extra_0>" inside the Qwen chat
template with the model-card system prompt. The pooler returns raw 2-class logits at each
<extra_0> position (use_activation=False, head in float32); we store z_t = l1 - l0.

Rows in : {id, problem_en, steps_en}
Rows out: {id, teacher_logodds: [z_1..z_T], teacher_model}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from koprm.io import load_jsonl, write_jsonl
from koprm.paths import TEACHER_MODEL, TEACHER_SYSTEM_PROMPT

STEP_TAG = "<extra_0>"


def build_teacher_input(tokenizer, problem_en: str, steps_en: list[str]) -> str:
    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": problem_en},
        {"role": "assistant", "content": STEP_TAG.join(s.strip() for s in steps_en) + STEP_TAG},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


class Teacher:
    def __init__(
        self,
        model: str = TEACHER_MODEL,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        quantization: str | None = None,
    ):
        from vllm import LLM
        from vllm.config import PoolerConfig

        self.model = model
        self.llm = LLM(
            model=model,
            runner="pooling",
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            enable_prefix_caching=False,
            # vLLM 0.15: the kwarg is `pooler_config=` (`override_pooler_config` was removed).
            # use_activation=False -> the 2-class head returns raw logits, not softmax probs.
            pooler_config=PoolerConfig(use_activation=False),
            # `head_dtype` is read off the HF config by vllm/config/model.py::_get_head_dtype,
            # so it has to arrive through hf_overrides (there is no LLM(head_dtype=...) kwarg).
            # For runner="pooling" the default is ALREADY torch.float32 whenever the platform
            # supports fp32, so this is belt-and-braces; "model" would be the way to turn it off.
            hf_overrides={"head_dtype": "float32"},
        )
        self.tok = self.llm.get_tokenizer()
        self.step_tag_id = self.tok.encode(STEP_TAG, add_special_tokens=False)
        assert len(self.step_tag_id) == 1, self.step_tag_id
        self.step_tag_id = self.step_tag_id[0]
        self.max_model_len = max_model_len

    def score(self, problems_en: list[str], steps_en: list[list[str]]) -> list[np.ndarray | None]:
        from vllm import PoolingParams

        texts = [build_teacher_input(self.tok, p, s) for p, s in zip(problems_en, steps_en)]
        keep, inputs = [], []
        for i, t in enumerate(texts):
            n_tok = len(self.tok.encode(t, add_special_tokens=False))
            if n_tok <= self.max_model_len:
                keep.append(i)
                inputs.append(t)
        pp = PoolingParams(use_activation=False, step_tag_id=self.step_tag_id)
        outs = self.llm.reward(inputs, pooling_params=pp, use_tqdm=True) if inputs else []
        result: list[np.ndarray | None] = [None] * len(texts)
        for i, o in zip(keep, outs):
            data = o.outputs.data
            if hasattr(data, "float"):
                data = data.float().cpu().numpy()
            data = np.asarray(data, dtype=np.float32)
            if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] != len(steps_en[i]):
                result[i] = None
                continue
            result[i] = data[:, 1] - data[:, 0]
        return result


def score_file(
    in_path: str,
    out_path: str,
    model: str,
    tp: int,
    quantization: str | None,
    batch: int = 2048,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
) -> None:
    rows = load_jsonl(in_path)
    out = Path(out_path)
    done = set()
    if out.exists():
        done = {r["id"] for r in load_jsonl(out)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"[teacher] {len(todo)} to score ({len(done)} cached) with {model}")
    if not todo:
        return
    teacher = Teacher(model, tp, max_model_len, gpu_memory_utilization, quantization)
    for b in range(0, len(todo), batch):
        chunk = todo[b : b + batch]
        zs = teacher.score([r["problem_en"] for r in chunk], [r["steps_en"] for r in chunk])
        write_jsonl(
            out,
            (
                {"id": r["id"], "teacher_logodds": None if z is None else z.tolist(), "teacher_model": model}
                for r, z in zip(chunk, zs)
            ),
            append=True,
        )
        print(f"[teacher] {b + len(chunk)}/{len(todo)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=TEACHER_MODEL)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--quantization", default=None, help="e.g. fp8 (weight-only Marlin on Ampere)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=2048)
    args = ap.parse_args()
    score_file(args.inp, args.out, args.model, args.tp, args.quantization, args.batch,
               args.max_model_len, args.gpu_memory_utilization)


if __name__ == "__main__":
    main()
