"""Day-1 check (§5.2): vLLM pooling log-odds vs. a reference.

Runs the 7B PRM on a few PRM800K solutions and compares the per-step log-odds against
either the HF reference implementation (default) or a previously saved run (--ref-json).

Measured 2026-09-20 on the 7B (16 solutions, 148 steps), max |p_hf - sigmoid(z_vllm)|:
    vLLM vs HF-bf16    0.050   (mean 0.004)
    vLLM vs HF-fp32    0.026   (mean 0.002)
    HF-bf16 vs HF-fp32 0.040   (mean 0.004)
i.e. vLLM is *closer* to the fp32 reference than HF-bf16 is; ~0.05 here is the bf16
noise floor of two independent implementations, not a bug in the pooling path.

The HF path needs the model-card remote code, which may not run under transformers 5. For
the vLLM 0.29 / transformers 5 evaluation, save the 0.15 run and diff against it:

    # in the 0.15 venv
    python scripts/check_teacher.py --save-json data/reports/teacher_z_vllm015.json
    # in the 0.29 venv
    python scripts/check_teacher.py --ref-json data/reports/teacher_z_vllm015.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from koprm.data.prm800k import parse_phase2
from koprm.teacher.score import Teacher, build_teacher_input

DEFAULT_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"


def load_rows(n: int, max_steps: int, scan: int) -> list[dict]:
    rows = [r for r in parse_phase2(max_rows=scan) if len(r["steps_en"]) <= max_steps]
    return rows[:n]


def run_vllm(model: str, quant: str | None, rows: list[dict], gpu_mem: float = 0.6):
    teacher = Teacher(model, tensor_parallel_size=1, gpu_memory_utilization=gpu_mem,
                      quantization=quant)
    zs = teacher.score([r["problem_en"] for r in rows], [r["steps_en"] for r in rows])
    for r, z in zip(rows[:4], zs[:4]):
        print(r["finish_reason"], r["human_first_error"], None if z is None else np.round(z, 2))
    del teacher
    return zs


def save_json(path: str, model: str, rows: list[dict], zs) -> None:
    out = {
        "model": model,
        "rows": [
            {"id": r["id"], "n_steps": len(r["steps_en"]),
             "z": None if z is None else [float(v) for v in z]}
            for r, z in zip(rows, zs)
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[check] wrote {len(out['rows'])} solutions to {path}")


def compare_ref_json(path: str, rows: list[dict], zs) -> None:
    """Compare against a saved run (same solutions, by id): steps match and max |dz|."""
    with open(path, encoding="utf-8") as f:
        ref = json.load(f)
    by_id = {r["id"]: r for r in ref["rows"]}
    n_cmp = n_steps_match = n_missing = 0
    max_dz = 0.0
    for r, z in zip(rows, zs):
        ref_row = by_id.get(r["id"])
        if ref_row is None or ref_row["z"] is None or z is None:
            n_missing += 1
            continue
        zr = np.asarray(ref_row["z"], dtype=np.float64)
        n_cmp += 1
        if len(zr) != len(z):
            print(f"[check] step count differs for {r['id']}: {len(zr)} vs {len(z)}")
            continue
        n_steps_match += 1
        max_dz = max(max_dz, float(np.max(np.abs(zr - np.asarray(z, dtype=np.float64)))))
    print(f"[check] reference {path} (model={ref.get('model')})")
    print(f"[check] compared {n_cmp} solutions, steps match: {n_steps_match}/{n_cmp}"
          f" (skipped {n_missing})")
    print(f"[check] max |z_ref - z_vllm| = {max_dz:.4f}")


def compare_hf(model: str, rows: list[dict], zs) -> None:
    """The model-card reference implementation (needs the repo's remote code)."""
    import gc

    import torch
    from transformers import AutoModel, AutoTokenizer

    gc.collect()
    torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    hf = AutoModel.from_pretrained(model, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                   device_map="cuda").eval()
    sep = tok.encode("<extra_0>")[0]
    max_abs = max_dz = 0.0
    z_hf = z = None
    for r, z in zip(rows, zs):
        if z is None:
            continue
        text = build_teacher_input(tok, r["problem_en"], r["steps_en"])
        ids = tok.encode(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            # use_cache=False: the model-card remote code builds a DynamicCache and calls
            # get_usable_length(), which transformers 4.57 removed. No cache -> no such call.
            logits = hf(input_ids=ids, use_cache=False)[0].float()
        lg = logits[ids == sep]  # [T, 2]
        z_hf = (lg[:, 1] - lg[:, 0]).cpu().numpy()
        prob_hf = torch.softmax(lg, -1)[:, 1].cpu().numpy()
        prob_v = 1 / (1 + np.exp(-z))
        assert len(z_hf) == len(z), (len(z_hf), len(z))
        max_abs = max(max_abs, float(np.max(np.abs(prob_hf - prob_v))))
        max_dz = max(max_dz, float(np.max(np.abs(z_hf - z))))
    print(f"steps match ({len(rows)} solutions); max |p_hf - sigmoid(z_vllm)| =", max_abs)
    print("max |z_hf - z_vllm| =", round(max_dz, 4))
    if z_hf is not None:
        print("sample z_hf", np.round(z_hf, 2))
        print("sample z_v ", np.round(z, 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quant", default=None, help="e.g. fp8 (weight-only Marlin on Ampere)")
    ap.add_argument("--save-json", default=None, help="dump the vLLM z lists here and stop")
    ap.add_argument("--ref-json", default=None, help="compare against this saved run, not HF")
    ap.add_argument("--n", type=int, default=16, help="solutions to score")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--scan", type=int, default=3000, help="PRM800K rows to scan for candidates")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = ap.parse_args()

    rows = load_rows(args.n, args.max_steps, args.scan)
    zs = run_vllm(args.model, args.quant, rows, args.gpu_memory_utilization)
    if args.save_json:
        save_json(args.save_json, args.model, rows, zs)
    if args.ref_json:
        compare_ref_json(args.ref_json, rows, zs)
    elif not args.save_json:
        compare_hf(args.model, rows, zs)


if __name__ == "__main__":
    main()
