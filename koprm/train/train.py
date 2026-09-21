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

A 7-8B student needs several GPUs: `--fsdp` shards parameters, gradients and optimizer
state with FSDP2 (`fully_shard` per decoder layer plus the root), bf16 compute over fp32
master shards, fp32 gradient reduction, and the 2-class head kept in float32. The data is
split across ranks (seeded shuffle, then rank::world_size) and `--micro-batch` is per rank,
so the effective batch stays 64 solutions globally.

    torchrun --nproc_per_node=4 -m koprm.train.train --train big.jsonl --out data/ckpt/r \
        --init-from-prm Qwen/Qwen2.5-Math-PRM-7B --fsdp --micro-batch 1 --grad-ckpt
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

EFFECTIVE_BATCH = 64  # solutions, globally (FSDP: across all ranks)


# --------------------------------------------------------------------------- distributed
def setup_distributed() -> tuple[int, int, int]:
    """(rank, world_size, local_rank) from the torchrun environment."""
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def grad_accum_for(micro_batch: int, world_size: int = 1,
                   effective: int = EFFECTIVE_BATCH) -> int:
    """Accumulation steps so that micro_batch x world_size x accum == the effective batch."""
    per_step = micro_batch * world_size
    if world_size > 1 and effective % per_step:
        raise SystemExit(
            f"--micro-batch {micro_batch} x {world_size} ranks does not divide the "
            f"effective batch {effective}; pick a micro-batch from "
            f"{[b for b in range(1, effective + 1) if effective % (b * world_size) == 0]}")
    return max(1, effective // per_step)


def decoder_layers(backbone) -> list:
    """The repeated transformer blocks, whatever the HF model calls them."""
    for attr in ("layers", "h", "blocks"):
        mod = getattr(backbone, attr, None)
        if mod is not None and len(list(mod)) > 0:
            return list(mod)
    inner = getattr(backbone, "model", None)
    return decoder_layers(inner) if inner is not None else []


def shard_model(model, mesh=None):
    """FSDP2: bf16 compute over fp32 master shards; the head stays float32 (§6.2)."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    bf16 = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fp32 = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32)
    kw = {"mesh": mesh} if mesh is not None else {}
    for layer in decoder_layers(model.backbone):  # innermost first
        fully_shard(layer, mp_policy=bf16, **kw)
    fully_shard(model.backbone, mp_policy=bf16, **kw)
    fully_shard(model.head, mp_policy=fp32, **kw)  # 2-class head: fp32 params and grads
    fully_shard(model, mp_policy=bf16, **kw)
    return model


def clip_grad_norm(model, clip: float) -> float:
    """Works for plain and DTensor (FSDP2) gradients."""
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if hasattr(gnorm, "full_tensor"):
        gnorm = gnorm.full_tensor()
    return float(gnorm)


def reduce_mean(value: float, device: str) -> float:
    import torch.distributed as dist

    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


def gather_full_state_dict(model) -> dict:
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


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
    fsdp = bool(getattr(args, "fsdp", False))
    rank, world, local_rank = setup_distributed() if fsdp else (0, 1, 0)
    is_main = rank == 0
    out_dir = Path(args.out)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{local_rank}" if fsdp else args.device
    use_bf16 = (not args.no_bf16) and device.startswith("cuda")
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    # FSDP keeps the master shards in the dtype the model is built with and casts to bf16
    # for compute (mp_policy), so under --fsdp the model is built float32 on CPU and the
    # optimizer state ends up fp32; the autocast context is then not used.
    build_dtype = torch.float32 if fsdp else dtype
    build_device = "cpu" if fsdp else device

    src = args.init_from_prm or args.backbone
    if args.init_from_prm:
        model = StepPRM.from_prm(args.init_from_prm, sep_token=args.sep_token,
                                 dtype=build_dtype, device=build_device)
    else:
        model = StepPRM.from_backbone(args.backbone, sep_token=args.sep_token,
                                      dtype=build_dtype, device=build_device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()  # before sharding
    if fsdp:
        model = shard_model(model)
    if is_main:
        print(f"[train] backbone={src} sep={model.sep_token!r} id={model.sep_id} "
              f"dtype={dtype} device={device} fsdp={fsdp} world={world}")

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
    sampler = None
    if fsdp:
        from torch.utils.data.distributed import DistributedSampler

        # seeded shuffle, then rank::world_size; the sampler pads so every rank runs the
        # same number of micro-batches (unequal counts would hang the collectives).
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True,
                                     seed=args.seed, drop_last=False)
        dl = DataLoader(ds, batch_size=args.micro_batch, sampler=sampler, collate_fn=collate,
                        num_workers=args.num_workers, drop_last=False)
    else:
        g = torch.Generator()
        g.manual_seed(args.seed)
        dl = DataLoader(ds, batch_size=args.micro_batch, shuffle=True, collate_fn=collate,
                        generator=g, num_workers=args.num_workers, drop_last=False)

    accum = args.grad_accum or grad_accum_for(args.micro_batch, world)
    micro_per_epoch = len(dl)
    opt_steps_per_epoch = max(1, math.ceil(micro_per_epoch / accum))
    total_opt_steps = opt_steps_per_epoch * args.epochs

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_opt_steps, args.warmup_ratio))

    log_path = out_dir / "train_log.jsonl"
    if is_main:
        (out_dir / "run_config.json").write_text(
            json.dumps({**vars(args), "soft_mode": soft_mode, "accum": accum,
                        "world_size": world,
                        "effective_batch": accum * args.micro_batch * world,
                        "total_opt_steps": total_opt_steps, "n_examples": len(ds),
                        "data_stats": stats.as_dict()}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"[train] {len(ds)} solutions, micro={args.micro_batch} x {world} ranks "
              f"x accum={accum} = effective {args.micro_batch * world * accum}, "
              f"{total_opt_steps} optimizer steps")

    model.train()
    gstep = 0
    t0 = time.time()
    running: list[float] = []
    logf = log_path.open("a", encoding="utf-8") if is_main else None
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if (use_bf16 and not fsdp) else nullcontext())
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
                gnorm = clip_grad_norm(model, args.clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % args.log_every == 0 or gstep == total_opt_steps:
                    loss = sum(running) / len(running)
                    if fsdp:
                        loss = reduce_mean(loss, device)
                    rec = {
                        "step": gstep, "epoch": epoch, "loss": loss,
                        "outcome": float(parts.outcome), "steps": float(parts.steps),
                        "lr": sched.get_last_lr()[0], "grad_norm": gnorm,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    running = []
                    if is_main:
                        logf.write(json.dumps(rec) + "\n")
                        logf.flush()
                        print(f"[train] {rec}")
        ck = out_dir / f"epoch{epoch + 1}"
        if fsdp:
            full = gather_full_state_dict(model)  # collective: every rank calls it
            if is_main:
                model.save_pretrained(ck, state_dict=full)
        else:
            model.save_pretrained(ck)
        if is_main:
            print(f"[train] saved {ck}")
    if logf is not None:
        logf.close()
    if fsdp:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    return {"out": str(out_dir), "steps": gstep, "n_examples": len(ds)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="training jsonl (§4.5 schema)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default=STUDENT_BACKBONE)
    ap.add_argument("--init-from-prm", default=None,
                    help="start from a Qwen2.5-Math-PRM checkpoint (backbone + 2-class head)")
    ap.add_argument("--fsdp", action="store_true",
                    help="shard with FSDP2; run under torchrun --nproc_per_node=N")
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
