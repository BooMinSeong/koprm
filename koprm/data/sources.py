"""Load MATH train, GSM8K train and KO/EN MATH500 into a common record format.

Record: {problem_id, source, problem_en, answer, level, subject, solution_en}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from koprm.paths import MATH_LOCAL


def last_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} (brace-balanced), or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    # \boxed 5 style (rare)
    while i < len(text) and text[i] == " ":
        i += 1
    if i >= len(text) or text[i] != "{":
        m = re.match(r"([^\s$]+)", text[i:])
        return m.group(1) if m else None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j]
    return None


def load_math_train() -> list[dict]:
    """MATH train (7,500) from the local copy of the original release."""
    rows = []
    train_dir = MATH_LOCAL / "train"
    if train_dir.exists():
        for fp in sorted(train_dir.glob("*/*.json")):
            d = json.load(open(fp, encoding="utf-8"))
            ans = last_boxed(d["solution"])
            if ans is None:
                continue
            rows.append(
                {
                    "problem_id": f"math/train/{fp.parent.name}/{fp.stem}",
                    "source": "math",
                    "problem_en": d["problem"],
                    "answer": ans,
                    "level": d["level"],
                    "subject": d["type"],
                    "solution_en": d["solution"],
                }
            )
    else:  # fallback: HF mirror
        from datasets import load_dataset

        for cfg in [
            "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus",
        ]:
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
            for i, d in enumerate(ds):
                ans = last_boxed(d["solution"])
                if ans is None:
                    continue
                rows.append(
                    {
                        "problem_id": f"math/train/{cfg}/{i}",
                        "source": "math",
                        "problem_en": d["problem"],
                        "answer": ans,
                        "level": d["level"],
                        "subject": d["type"],
                        "solution_en": d["solution"],
                    }
                )
    return rows


def load_gsm8k_train() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for i, d in enumerate(ds):
        ans = d["answer"].split("####")[-1].strip().replace(",", "")
        sol = re.sub(r"<<[^>]*>>", "", d["answer"].split("####")[0]).strip()
        rows.append(
            {
                "problem_id": f"gsm8k/train/{i}",
                "source": "gsm8k",
                "problem_en": d["question"],
                "answer": ans,
                "level": "gsm8k",
                "subject": "arithmetic",
                "solution_en": sol,
            }
        )
    return rows


def load_math500(korean: bool = True) -> list[dict]:
    """KO MATH500 (ENSEONG/ko-math-500-test) joined with the English original by id."""
    from datasets import load_dataset

    en = {d["unique_id"]: d for d in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    rows = []
    if korean:
        ko = load_dataset("ENSEONG/ko-math-500-test", split="test")
        for d in ko:
            e = en[d["id"]]
            rows.append(
                {
                    "problem_id": f"math500/{d['id']}",
                    "source": "math500",
                    "problem_en": e["problem"],
                    "problem_ko": d["problem"],
                    "answer": e["answer"],
                    "level": str(e["level"]),
                    "subject": e["subject"],
                    "solution_en": e["solution"],
                }
            )
    else:
        for uid, e in en.items():
            rows.append(
                {
                    "problem_id": f"math500/{uid}",
                    "source": "math500",
                    "problem_en": e["problem"],
                    "answer": e["answer"],
                    "level": str(e["level"]),
                    "subject": e["subject"],
                    "solution_en": e["solution"],
                }
            )
    return rows
