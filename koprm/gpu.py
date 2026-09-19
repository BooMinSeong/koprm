"""Pick free GPUs on a shared machine (no scheduler here)."""
from __future__ import annotations

import os
import subprocess


def gpu_memory_used() -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return [int(x.strip()) for x in out.strip().splitlines()]


def free_gpus(max_used_mib: int = 1024) -> list[int]:
    return [i for i, m in enumerate(gpu_memory_used()) if m <= max_used_mib]


def claim_gpus(n: int, max_used_mib: int = 1024) -> list[int]:
    """Return n free GPU indices and set CUDA_VISIBLE_DEVICES accordingly."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x != ""]
        if len(ids) < n:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES has {len(ids)} GPUs, need {n}")
        return ids[:n]
    free = free_gpus(max_used_mib)
    if len(free) < n:
        raise RuntimeError(f"need {n} free GPUs, only {free} are free")
    chosen = free[:n]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, chosen))
    return chosen
