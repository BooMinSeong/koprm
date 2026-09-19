#!/usr/bin/env python
"""§6.2 day-1 student backbone check (CPU, float32).

For each backbone:
  1. the step separator encodes to a single token id,
  2. a 2-example batch runs forward and backward,
  3. the logits at the SEP positions come out as [n_steps, 2],
  4. the §6.1 loss is finite and produces finite gradients.

    python scripts/check_student.py                       # EXAONE (default) + Qwen fallback
    python scripts/check_student.py --backbone LGAI-EXAONE/EXAONE-4.0-1.2B
"""
from __future__ import annotations

import argparse
import sys
import traceback

import torch

from koprm.paths import STUDENT_BACKBONE, STUDENT_BACKBONE_FALLBACK
from koprm.train.data import Collator, build_dataset
from koprm.train.loss import prm_loss, step_probs
from koprm.train.model import StepPRM

ROWS = [
    {
        "problem_ko": "1부터 10까지의 자연수의 합을 구하시오.",
        "solution_steps": [
            "## 단계 1: 공식 적용\n$1+2+\\cdots+n = \\frac{n(n+1)}{2}$입니다.",
            "## 단계 2: 계산\n$n=10$이므로 $\\frac{10\\cdot 11}{2}=55$입니다.",
            "따라서 최종 답은: $\\boxed{55}$입니다.",
        ],
        "step_labels": [1, None, 1],
        "outcome": 1,
        "teacher_logodds": [2.0, 0.5, 3.0],
    },
    {
        "problem_ko": "$x^2 - 5x + 6 = 0$의 두 근의 합을 구하시오.",
        "solution_steps": [
            "## 단계 1: 인수분해\n$x^2-5x+6=(x-2)(x-3)$입니다.",
            "## 단계 2: 근\n두 근은 $2$와 $3$입니다.",
            "## 단계 3: 합\n$2+3=6$입니다.",
            "따라서 최종 답은: $\\boxed{6}$입니다.",
        ],
        "step_labels": [1, 1, 0, 0],
        "outcome": 0,
        "teacher_logodds": [2.5, 1.5, -3.0, -4.0],
    },
]


def check(backbone: str, max_len: int = 2048) -> bool:
    print(f"\n{'=' * 70}\n[check] backbone = {backbone}\n{'=' * 70}")
    ok = True
    model = StepPRM.from_backbone(backbone, dtype=torch.float32, device="cpu")
    tok = model.tokenizer
    sep = model.sep_token

    ids = tok.encode(sep, add_special_tokens=False)
    single = len(ids) == 1
    print(f"[1] SEP token {sep!r} -> ids {ids}  single_id={single}  sep_id={model.sep_id}")
    print(f"    tokenizer len={len(tok)}  embedding rows="
          f"{model.backbone.get_input_embeddings().weight.shape[0]}  "
          f"hidden={model.backbone.config.hidden_size}  head_dtype="
          f"{next(model.head.parameters()).dtype}")
    ok &= single

    ds, stats = build_dataset(ROWS, tok, sep, max_len=max_len, with_soft=True, verbose=False)
    print(f"[2] dataset: {stats}")
    ok &= len(ds) == len(ROWS)
    batch = Collator(pad_id=tok.pad_token_id or tok.eos_token_id or 0, with_soft=True)(
        list(ds.examples))
    print(f"    input_ids {tuple(batch['input_ids'].shape)}  "
          f"sep_positions {batch['sep_positions'].tolist()}  "
          f"n_steps {batch['n_steps'].tolist()}  targets {batch['targets'].tolist()}")

    # per-example step logits, as the label pipeline reads them (§2.2)
    per_ex = model.step_logits_list(batch["input_ids"], batch["attention_mask"],
                                    [e.sep_positions for e in ds.examples])
    shapes = [tuple(x.shape) for x in per_ex]
    print(f"[3] step logits per example: {shapes}")
    ok &= all(s == (len(e.sep_positions), 2) for s, e in zip(shapes, ds.examples))

    sl = model.step_logits(batch["input_ids"], batch["attention_mask"],
                           batch["sep_positions"], batch["step_mask"])
    print(f"    padded step logits {tuple(sl.shape)}  "
          f"P(pos) {[[round(v, 3) for v in r] for r in step_probs(sl).tolist()]}")

    loss = prm_loss(sl, batch["targets"], batch["n_steps"], return_parts=True)
    soft = prm_loss(sl, batch["targets"], batch["n_steps"],
                    soft_targets=batch["soft_targets"])
    oo = prm_loss(sl, batch["targets"], batch["n_steps"], outcome_only=True)
    print(f"[4] loss={loss.loss.item():.4f} (outcome={loss.outcome.item():.4f}, "
          f"steps={loss.steps.item():.4f}, |M| terms={loss.n_step_terms})  "
          f"soft={soft.item():.4f}  outcome_only={oo.item():.4f}")
    ok &= bool(torch.isfinite(loss.loss)) and bool(torch.isfinite(soft)) \
        and bool(torch.isfinite(oo))

    loss.loss.backward()
    head_g = model.head.net[0].weight.grad
    emb_g = model.backbone.get_input_embeddings().weight.grad
    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    print(f"[5] backward ok: head grad norm={head_g.norm().item():.4e}, "
          f"embedding grad norm={emb_g.norm().item():.4e}, params without grad={n_none}")
    ok &= bool(torch.isfinite(head_g).all()) and head_g.norm().item() > 0

    print(f"[{'PASS' if ok else 'FAIL'}] {backbone}")
    return bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", action="append", default=None)
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()
    backbones = args.backbone or [STUDENT_BACKBONE, STUDENT_BACKBONE_FALLBACK]
    results = {}
    for b in backbones:
        try:
            results[b] = check(b, args.max_len)
        except Exception:
            traceback.print_exc()
            results[b] = False
    print("\n==== summary ====")
    for b, r in results.items():
        print(f"  {'PASS' if r else 'FAIL'}  {b}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
