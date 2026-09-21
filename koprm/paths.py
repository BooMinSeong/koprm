"""Project paths and shared constants."""
from __future__ import annotations

import os
from pathlib import Path

# vLLM 0.29 JIT-compiles the flashinfer sampler, which needs ninja and nvcc on PATH; this box
# has neither, so the build fails at startup. Every vLLM user imports this module first.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("KOPRM_DATA", ROOT / "data"))
SPLITS = DATA / "splits"
GEN = DATA / "gen"
TRANS = DATA / "trans"
TEACHER = DATA / "teacher"
LABELS = DATA / "labels"
TRAINSETS = DATA / "trainsets"
CKPT = DATA / "ckpt"
EVAL = DATA / "eval"
REPORTS = DATA / "reports"

MATH_LOCAL = Path(os.environ.get("KOPRM_MATH_DIR", Path.home() / "projects/math_gen/MATH"))

# Models (see Plan.md §3, §4.2, §6.2)
GENERATORS = {
    "exaone-1.2b": "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",  # hold-out generator (eval only)
    # Stronger generators for AIME, where the small ones barely solve anything (§15.8c).
    # Both templates take enable_thinking: pass --chat-kwargs '{"enable_thinking": false}'
    # for the non-reasoning mode (Qwen3 defaults to thinking on, EXAONE-4.0 to off).
    "exaone-32b": "LGAI-EXAONE/EXAONE-4.0-32B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}
TEACHER_MODEL = "Qwen/Qwen2.5-Math-PRM-72B"
TEACHER_MODEL_SMALL = "Qwen/Qwen2.5-Math-PRM-7B"  # pilot / smoke tests only
# One translator, no fallback (Plan §3): gemma-4-12B-it with an instruction prompt.
TRANSLATOR = "google/gemma-4-12B-it"
STUDENT_BACKBONE = "LGAI-EXAONE/EXAONE-4.0-1.2B"
STUDENT_BACKBONE_FALLBACK = "Qwen/Qwen2.5-1.5B-Instruct"

# Korean generation prompt: identical to komath (src/sal/config.py) so that training
# solutions come from the same distribution the KO MATH500 harness scores.
SYSTEM_PROMPT_KO = (
    "다음 수학 문제를 효율적이고 명확하게 풀어주세요:\n\n"
    "- 간단한 문제 (2단계 이하):\n간단한 설명과 함께 간결한 해결책을 제공하세요.\n\n"
    "- 복잡한 문제 (3단계 이상):\n다음 단계별 형식을 사용하세요:\n\n"
    "## 단계 1: [간결한 설명]\n[간단한 설명과 계산]\n\n"
    "## 단계 2: [간결한 설명]\n[간단한 설명과 계산]\n\n...\n\n"
    "접근 방식과 관계없이 항상 다음과 같이 마무리하세요:\n\n"
    "따라서 최종 답은: $\\boxed{답}$입니다. 맞기를 바랍니다.\n\n"
    "여기서 [답]은 문제를 푸는 최종 숫자나 식입니다."
)
# The English original of the same prompt (komath/sal), for scoring English solutions.
SYSTEM_PROMPT_EN = (
    "Solve the following math problem efficiently and clearly:\n\n"
    "- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal "
    "explanation.\n\n"
    "- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n"
    "## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n"
    "## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\n"
    "Regardless of the approach, always conclude with:\n\n"
    "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
    "Where [answer] is just the final number or expression that solves the problem."
)
SYSTEM_PROMPTS = {"ko": SYSTEM_PROMPT_KO, "en": SYSTEM_PROMPT_EN}

# Teacher (Qwen2.5-Math-PRM) system prompt from the model card.
TEACHER_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
STEP_SEP = "\n\n"


def ensure_dirs() -> None:
    for d in (SPLITS, GEN, TRANS, TEACHER, LABELS, TRAINSETS, CKPT, EVAL, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
