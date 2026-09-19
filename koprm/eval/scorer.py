"""Score Korean solutions with a trained StepPRM checkpoint.

`score()` returns one P(positive) per step - the softmax of the 2-class logits at
that step's SEP token (§6.1).  Aggregation into a solution score (last / min,
§7) is the caller's job; see koprm/eval/bon.py.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch

from koprm.paths import SYSTEM_PROMPT_KO
from koprm.train.model import StepPRM, encode_example


class StudentScorer:
    def __init__(self, ckpt_dir: str, device: str = "cpu",
                 dtype: torch.dtype | None = None, max_len: int = 4096,
                 system_prompt: str = SYSTEM_PROMPT_KO):
        if dtype is None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.model = StepPRM.from_pretrained(ckpt_dir, device=device, dtype=dtype)
        self.model.eval()
        self.device = device
        self.max_len = max_len
        self.system_prompt = system_prompt
        self.n_truncated = 0
        pad = self.model.tokenizer.pad_token_id
        self.pad_id = int(pad if pad is not None else (self.model.tokenizer.eos_token_id or 0))
        self._use_bf16 = str(device).startswith("cuda") and dtype == torch.bfloat16

    # ------------------------------------------------------------------ utils
    def _encode(self, problem_ko: str, steps: list[str]) -> tuple[list[int], list[int], int]:
        if not steps:
            return [], [], 0
        ids, pos = encode_example(self.model.tokenizer, problem_ko, steps,
                                  self.model.sep_token, self.system_prompt)
        n_full = len(pos)
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            pos = [p for p in pos if p < self.max_len]
            self.n_truncated += 1
        return ids, pos, n_full

    @torch.no_grad()
    def _score_batch(self, items: list[tuple[list[int], list[int]]]) -> list[list[float]]:
        L = max(len(i[0]) for i in items)
        B = len(items)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for b, (ids, _) in enumerate(items):
            input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[b, : len(ids)] = 1
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if self._use_bf16 else nullcontext())
        with ctx:
            logits = self.model(input_ids, attn)  # [B, L, 2] float32
        probs = torch.softmax(logits.float(), dim=-1)[..., 1].cpu()
        return [probs[b, torch.tensor(pos, dtype=torch.long)].tolist() if pos else []
                for b, (_, pos) in enumerate(items)]

    # ------------------------------------------------------------------- main
    def score(self, problems_ko: list[str], solutions_steps: list[list[str]],
              batch_size: int = 8) -> list[list[float]]:
        if len(problems_ko) != len(solutions_steps):
            raise ValueError("problems_ko and solutions_steps must have the same length")
        enc: list[tuple[list[int], list[int], int]] = [
            self._encode(p, s) for p, s in zip(problems_ko, solutions_steps)
        ]
        order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))  # length-sorted batching
        out: list[list[float]] = [[] for _ in enc]
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            items = [(enc[i][0], enc[i][1]) for i in idx]
            live = [j for j, i in enumerate(idx) if items[j][1]]
            if not live:
                continue
            res = self._score_batch([items[j] for j in live])
            for j, r in zip(live, res):
                out[idx[j]] = r
        # steps whose SEP fell outside max_len keep the last surviving score; if the
        # prompt alone is longer than max_len nothing survives and we fall back to 0.5
        for i, (_, pos, n_full) in enumerate(enc):
            if n_full and not out[i]:
                out[i] = [0.5] * n_full
            elif out[i] and len(out[i]) < n_full:
                out[i] = out[i] + [out[i][-1]] * (n_full - len(out[i]))
        return out
