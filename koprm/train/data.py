"""§4.5 training rows -> tokenized step-PRM examples.

Input jsonl rows (extra fields are ignored):

    {"problem_ko": str,
     "solution_steps": [str, ...],
     "step_labels": [1 | 0 | null, ...],     # same length as solution_steps
     "outcome": 0 | 1,
     "teacher_logodds": [float, ...]}        # optional, only for the --soft ablation

Per example we keep the SEP positions and one target per step:

    targets[t] = step_labels[t]  (or -100 when masked)   for t < T-1
    targets[T-1] = outcome                               (§6.1: CE(y, p_T))

Truncation: the sequence is cut from the right at `max_len`; examples whose SEP
positions would be cut are dropped (the count is reported, not silently hidden).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

from koprm.io import read_jsonl
from koprm.paths import SYSTEM_PROMPT_KO
from koprm.train.loss import IGNORE_INDEX
from koprm.train.model import encode_example


@dataclass
class Example:
    input_ids: list[int]
    sep_positions: list[int]
    targets: list[int]
    soft_targets: list[float] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.sep_positions)


@dataclass
class BuildStats:
    n_rows: int = 0
    n_kept: int = 0
    n_no_steps: int = 0
    n_label_len_mismatch: int = 0
    n_sep_mismatch: int = 0
    n_truncated_drop: int = 0
    n_truncated_ok: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def __str__(self) -> str:
        return (
            f"rows={self.n_rows} kept={self.n_kept} "
            f"dropped(no_steps={self.n_no_steps}, label_len={self.n_label_len_mismatch}, "
            f"sep_mismatch={self.n_sep_mismatch}, truncation={self.n_truncated_drop}) "
            f"truncated_but_kept={self.n_truncated_ok}"
        )


def row_targets(step_labels: list, outcome: int) -> list[int]:
    t = [IGNORE_INDEX if l is None else int(l) for l in step_labels]
    t[-1] = int(outcome)
    return t


def build_example(
    tokenizer,
    sep_token: str,
    row: dict,
    max_len: int = 2048,
    system_prompt: str = SYSTEM_PROMPT_KO,
    stats: BuildStats | None = None,
    with_soft: bool = False,
) -> Example | None:
    stats = stats or BuildStats()
    steps = [s for s in (row.get("solution_steps") or []) if str(s).strip()]
    if not steps:
        stats.n_no_steps += 1
        return None
    labels = row.get("step_labels")
    if labels is None:
        labels = [None] * len(steps)
    if len(labels) != len(steps):
        stats.n_label_len_mismatch += 1
        return None

    ids, sep_pos = encode_example(tokenizer, row["problem_ko"], steps, sep_token, system_prompt)
    if len(sep_pos) != len(steps):
        stats.n_sep_mismatch += 1
        return None

    if len(ids) > max_len:
        if sep_pos[-1] >= max_len:
            stats.n_truncated_drop += 1
            return None
        ids = ids[:max_len]
        stats.n_truncated_ok += 1

    targets = row_targets(labels, int(row["outcome"]))
    soft = None
    if with_soft:
        z = row.get("teacher_logodds")
        if z is not None and len(z) == len(steps):
            soft = [1.0 / (1.0 + math.exp(-float(v))) for v in z]
        else:
            soft = [float("nan")] * len(steps)
    stats.n_kept += 1
    return Example(
        input_ids=ids,
        sep_positions=sep_pos,
        targets=targets,
        soft_targets=soft,
        meta={"problem_id": row.get("problem_id"), "arm": row.get("arm")},
    )


class StepPRMDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


def build_dataset(
    path_or_rows,
    tokenizer,
    sep_token: str,
    max_len: int = 2048,
    limit: int | None = None,
    system_prompt: str = SYSTEM_PROMPT_KO,
    with_soft: bool = False,
    verbose: bool = True,
) -> tuple[StepPRMDataset, BuildStats]:
    rows = (list(read_jsonl(path_or_rows))
            if isinstance(path_or_rows, (str, bytes, os.PathLike))
            else list(path_or_rows))
    if limit:
        rows = rows[:limit]
    stats = BuildStats(n_rows=len(rows))
    out: list[Example] = []
    for r in rows:
        ex = build_example(tokenizer, sep_token, r, max_len=max_len,
                           system_prompt=system_prompt, stats=stats, with_soft=with_soft)
        if ex is not None:
            out.append(ex)
    if verbose:
        print(f"[data] {stats}")
    return StepPRMDataset(out), stats


class Collator:
    """Pads on the right; SEP positions and targets are padded to the longest solution."""

    def __init__(self, pad_id: int, with_soft: bool = False):
        self.pad_id = int(pad_id)
        self.with_soft = with_soft

    def __call__(self, batch: list[Example]) -> dict:
        L = max(len(e.input_ids) for e in batch)
        S = max(e.n_steps for e in batch)
        B = len(batch)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, L), dtype=torch.long)
        sep_positions = torch.zeros((B, S), dtype=torch.long)
        step_mask = torch.zeros((B, S), dtype=torch.bool)
        targets = torch.full((B, S), IGNORE_INDEX, dtype=torch.long)
        soft = torch.full((B, S), float("nan"), dtype=torch.float32)
        n_steps = torch.zeros((B,), dtype=torch.long)
        for b, e in enumerate(batch):
            n = len(e.input_ids)
            k = e.n_steps
            input_ids[b, :n] = torch.tensor(e.input_ids, dtype=torch.long)
            attention_mask[b, :n] = 1
            sep_positions[b, :k] = torch.tensor(e.sep_positions, dtype=torch.long)
            step_mask[b, :k] = True
            targets[b, :k] = torch.tensor(e.targets, dtype=torch.long)
            n_steps[b] = k
            if self.with_soft and e.soft_targets is not None:
                soft[b, :k] = torch.tensor(e.soft_targets, dtype=torch.float32)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sep_positions": sep_positions,
            "step_mask": step_mask,
            "targets": targets,
            "n_steps": n_steps,
        }
        if self.with_soft:
            out["soft_targets"] = soft
        return out
