"""§6.2 student training: plain PyTorch, no TRL.

lr 1e-5, warmup 3%, cosine decay, effective batch 64 solutions (gradient
accumulation), max length 2048, bf16 autocast on GPU, 3 epochs, grad clip 1.0.
The same values are used for every run; only --train / --out and the ablation flag
change. The label-form variants are: none (hard kernel/human labels), --soft (teacher
sigma(z) as the step target), --soft-y (the same, but a y=1 solution is 1.0 everywhere,
§2.1) and --outcome-only (no step term at all). They are mutually exclusive.

Smoke test (CPU, no GPU needed):

    python -m koprm.train.train --train x.jsonl --out /tmp/ck \
        --device cpu --limit 8 --micro-batch 2 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from koprm.paths import STUDENT_BACKBONE
from koprm.train.data import Collator, build_dataset
from koprm.train.loss import prm_loss
from koprm.train.model import StepPRM

EFFECTIVE_BATCH = 64  # solutions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(step: int, total: int, warmup_ratio: float = 0.03) -> float:
    warmup = max(1, round(total * warmup_ratio))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    use_bf16 = (not args.no_bf16) and device.startswith("cuda")
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = StepPRM.from_backbone(args.backbone, sep_token=args.sep_token,
                                  dtype=dtype, device=device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    print(f"[train] backbone={args.backbone} sep={model.sep_token!r} id={model.sep_id} "
          f"dtype={dtype} device={device}")

    soft_mode = "soft_y" if args.soft_y else ("soft" if args.soft else None)
    ds, stats = build_dataset(args.train, model.tokenizer, model.sep_token,
                              max_len=args.max_len, limit=args.limit,
                              soft_mode=soft_mode)
    if len(ds) == 0:
        raise SystemExit("no usable training examples")

    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id or 0
    collate = Collator(pad_id=pad_id, soft_mode=soft_mode)
    g = torch.Generator()
    g.manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=args.micro_batch, shuffle=True, collate_fn=collate,
                    generator=g, num_workers=args.num_workers, drop_last=False)

    accum = args.grad_accum or max(1, EFFECTIVE_BATCH // args.micro_batch)
    micro_per_epoch = len(dl)
    opt_steps_per_epoch = max(1, math.ceil(micro_per_epoch / accum))
    total_opt_steps = opt_steps_per_epoch * args.epochs

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_opt_steps, args.warmup_ratio))

    log_path = out_dir / "train_log.jsonl"
    (out_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "soft_mode": soft_mode, "accum": accum,
                    "effective_batch": accum * args.micro_batch,
                    "total_opt_steps": total_opt_steps, "n_examples": len(ds),
                    "data_stats": stats.as_dict()}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"[train] {len(ds)} solutions, micro={args.micro_batch} x accum={accum} "
          f"= effective {args.micro_batch * accum}, {total_opt_steps} optimizer steps")

    model.train()
    gstep = 0
    t0 = time.time()
    running: list[float] = []
    logf = log_path.open("a", encoding="utf-8")
    for epoch in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if use_bf16 else nullcontext())
            with ctx:
                sl = model.step_logits(batch["input_ids"], batch["attention_mask"],
                                       batch["sep_positions"], batch["step_mask"])
            parts = prm_loss(sl, batch["targets"], batch["n_steps"],
                             soft_targets=batch.get("soft_targets"),
                             outcome_only=args.outcome_only, return_parts=True)
            (parts.loss / accum).backward()
            running.append(float(parts.loss.detach()))

            is_last = (i + 1) == micro_per_epoch
            if (i + 1) % accum == 0 or is_last:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % args.log_every == 0 or gstep == total_opt_steps:
                    rec = {
                        "step": gstep, "epoch": epoch,
                        "loss": sum(running) / len(running),
                        "outcome": float(parts.outcome), "steps": float(parts.steps),
                        "lr": sched.get_last_lr()[0], "grad_norm": float(gnorm),
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
                    print(f"[train] {rec}")
                    running = []
        ck = out_dir / f"epoch{epoch + 1}"
        model.save_pretrained(ck)
        print(f"[train] saved {ck}")
    logf.close()
    return {"out": str(out_dir), "steps": gstep, "n_examples": len(ds)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="training jsonl (§4.5 schema)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default=STUDENT_BACKBONE)
    ap.add_argument("--sep-token", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=0, help="0 = 64 // micro-batch")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    variant = ap.add_mutually_exclusive_group()
    variant.add_argument("--soft", action="store_true", help="ablation: soft teacher targets")
    variant.add_argument("--soft-y", action="store_true",
                         help="ablation: soft teacher targets, but 1.0 everywhere when y=1")
    variant.add_argument("--outcome-only", action="store_true",
                         help="ablation: outcome term only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rows")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
