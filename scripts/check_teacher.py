"""Day-1 check (§5.2): vLLM pooling log-odds vs. the HF reference implementation.

Runs the 7B PRM on a few PRM800K solutions both ways and compares sigmoid(z) with the
model-card probabilities.

Measured 2026-09-20 on the 7B (16 solutions, 148 steps), max |p_hf - sigmoid(z_vllm)|:
    vLLM vs HF-bf16    0.050   (mean 0.004)
    vLLM vs HF-fp32    0.026   (mean 0.002)
    HF-bf16 vs HF-fp32 0.040   (mean 0.004)
i.e. vLLM is *closer* to the fp32 reference than HF-bf16 is; ~0.05 here is the bf16
noise floor of two independent implementations, not a bug in the pooling path.
"""
import sys

import numpy as np
import torch

from koprm.data.prm800k import parse_phase2
from koprm.teacher.score import Teacher, build_teacher_input

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-Math-PRM-7B"
quant = sys.argv[2] if len(sys.argv) > 2 else None
rows = [r for r in parse_phase2(max_rows=3000) if len(r["steps_en"]) <= 12][:16]

teacher = Teacher(model, tensor_parallel_size=1, gpu_memory_utilization=0.6, quantization=quant)
zs = teacher.score([r["problem_en"] for r in rows], [r["steps_en"] for r in rows])
for r, z in zip(rows[:4], zs[:4]):
    print(r["finish_reason"], r["human_first_error"], np.round(z, 2))
del teacher
import gc; gc.collect(); torch.cuda.empty_cache()

# HF reference (model card code)
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
hf = AutoModel.from_pretrained(model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda").eval()
sep = tok.encode("<extra_0>")[0]
max_abs = 0.0
max_dz = 0.0
for r, z in zip(rows, zs):
    text = build_teacher_input(tok, r["problem_en"], r["steps_en"])
    ids = tok.encode(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        # use_cache=False: the model-card remote code builds a DynamicCache and calls
        # get_usable_length(), which transformers 4.57 removed. No cache -> no such call.
        logits = hf(input_ids=ids, use_cache=False)[0].float()
    m = ids == sep
    lg = logits[m]  # [T, 2]
    z_hf = (lg[:, 1] - lg[:, 0]).cpu().numpy()
    prob_hf = torch.softmax(lg, -1)[:, 1].cpu().numpy()
    prob_v = 1 / (1 + np.exp(-z))
    assert len(z_hf) == len(z), (len(z_hf), len(z))
    max_abs = max(max_abs, float(np.max(np.abs(prob_hf - prob_v))))
    max_dz = max(max_dz, float(np.max(np.abs(z_hf - z))))
print(f"steps match ({len(rows)} solutions); max |p_hf - sigmoid(z_vllm)| =", max_abs)
print("max |z_hf - z_vllm| =", round(max_dz, 4))
print("sample z_hf", np.round(z_hf, 2)); print("sample z_v ", np.round(z, 2))
