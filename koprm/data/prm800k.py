"""§4.3 / §4.4 PRM800K phase-2 parsing.

Each phase-2 row rates the steps of one pre-generated solution until the first error
(finish_reason=found_error) or to the end (finish_reason=solution). We keep the *full*
pre-generated solution (`pre_generated_steps`) and map the human ratings onto it:

  found_error: steps before the first -1 -> 1, from that step to the end -> 0
  solution   : all steps -> 1
  neutral (0) ratings count as 1 (they do not kill the prefix)

Output record:
  {id, problem_en, answer, steps_en, human_first_error (0-based idx or -1), step_labels,
   outcome_claimed (pre_generated_answer), finish_reason}
"""
from __future__ import annotations

import glob
import json
import os

from huggingface_hub import hf_hub_download


def phase2_path() -> str:
    hits = glob.glob(
        os.path.expandvars("$HF_HOME/hub/datasets--tasksource--PRM800K/snapshots/*/phase2_train.jsonl")
    )
    if hits:
        return hits[0]
    return hf_hub_download("tasksource/PRM800K", "phase2_train.jsonl", repo_type="dataset")


def _rating_of(step_label: dict, pre_step: str) -> int | None:
    """Rating of the pre-generated step at this position.

    While walking, `chosen_completion` points at the pre-generated step. At the final
    (error) step the labeler rated several alternatives and nothing is chosen, so we
    match the completion by text.
    """
    comps = step_label.get("completions") or []
    if step_label.get("chosen_completion") is not None:
        c = comps[step_label["chosen_completion"]]
        if c["text"].strip() != pre_step.strip():
            return None
        return c.get("rating")
    if step_label.get("human_completion") is not None:
        return None  # labeler wrote their own step: solution diverges from pre_generated
    for c in comps:
        if c["text"].strip() == pre_step.strip():
            return c.get("rating")
    return None


def parse_phase2(max_rows: int | None = None) -> list[dict]:
    rows = []
    with open(phase2_path(), encoding="utf-8") as f:
        for li, line in enumerate(f):
            if max_rows and li >= max_rows:
                break
            d = json.loads(line)
            if d.get("is_quality_control_question") or d.get("is_initial_screening_question"):
                continue
            fr = d["label"]["finish_reason"]
            if fr not in ("found_error", "solution"):
                continue
            q = d["question"]
            steps = q.get("pre_generated_steps") or []
            if not steps:
                continue
            labels = d["label"]["steps"]
            # Walk ratings; every rated step must be the pre-generated step at that position.
            first_err = -1
            ok = True
            for i, sl in enumerate(labels):
                if i >= len(steps):
                    ok = False
                    break
                r = _rating_of(sl, steps[i])
                if r is None:
                    ok = False
                    break
                if r == -1:
                    first_err = i
                    break
            if not ok:
                continue
            if fr == "found_error" and first_err < 0:
                continue
            if fr == "solution" and (first_err >= 0 or len(labels) < len(steps)):
                continue
            step_labels = [1] * len(steps)
            if first_err >= 0:
                for t in range(first_err, len(steps)):
                    step_labels[t] = 0
            rows.append(
                {
                    "id": f"prm800k/{li}",
                    "problem_en": q["problem"],
                    "answer": q["ground_truth_answer"],
                    "steps_en": [s.strip() for s in steps],
                    "human_first_error": first_err,
                    "step_labels": step_labels,
                    "pre_generated_answer": q.get("pre_generated_answer"),
                    "finish_reason": fr,
                }
            )
    return rows
