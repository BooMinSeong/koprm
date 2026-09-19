"""§4.1 Fixed splits. dev = 500 MATH-train problems, level-stratified; train pool = rest + GSM8K."""
from __future__ import annotations

import random
from collections import defaultdict

from koprm.data.sources import load_gsm8k_train, load_math500, load_math_train
from koprm.io import write_jsonl
from koprm.paths import SPLITS, ensure_dirs

DEV_SIZE = 500
SEED = 20260920


def make_splits(seed: int = SEED) -> dict[str, list[dict]]:
    ensure_dirs()
    math = load_math_train()
    gsm = load_gsm8k_train()
    m500 = load_math500(korean=True)
    m500_problems = {r["problem_en"].strip() for r in m500}
    # Safety: MATH train is disjoint from test, but never let a test problem leak.
    math = [r for r in math if r["problem_en"].strip() not in m500_problems]

    rng = random.Random(seed)
    by_level = defaultdict(list)
    for r in math:
        by_level[r["level"]].append(r)
    total = len(math)
    dev, dev_ids = [], set()
    levels = sorted(by_level)
    quota = {lv: round(DEV_SIZE * len(by_level[lv]) / total) for lv in levels}
    # fix rounding so the total is exactly DEV_SIZE
    diff = DEV_SIZE - sum(quota.values())
    for lv in sorted(levels, key=lambda l: -len(by_level[l])):
        if diff == 0:
            break
        quota[lv] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
    for lv in levels:
        pool = sorted(by_level[lv], key=lambda r: r["problem_id"])
        rng.shuffle(pool)
        for r in pool[: quota[lv]]:
            dev.append(r)
            dev_ids.add(r["problem_id"])
    train_pool = [r for r in math if r["problem_id"] not in dev_ids] + gsm
    out = {"dev": dev, "train_pool": train_pool, "math500": m500}
    for name, rows in out.items():
        write_jsonl(SPLITS / f"{name}.jsonl", rows)
    return out


if __name__ == "__main__":
    out = make_splits()
    for k, v in out.items():
        print(k, len(v))
