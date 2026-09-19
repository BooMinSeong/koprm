"""train/data.py on a stub tokenizer (no model download, no network)."""
import math

import torch

from koprm.train.data import BuildStats, Collator, build_dataset, build_example, row_targets
from koprm.train.loss import IGNORE_INDEX

SEP = "<sep>"
PREFIX = [1, 1, 1, 1]   # system + user template
SUFFIX = [9, 9]         # end-of-turn tokens after the last SEP


class FakeTok:
    """Each step contributes len(step) filler tokens (5) followed by the SEP id (7)."""
    pad_token_id = 0
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return [7] if text == SEP else [5] * len(text)

    def convert_tokens_to_ids(self, t):
        return 7 if t == SEP else 5

    def apply_chat_template(self, conv, tokenize=True, add_generation_prompt=False):
        body = conv[-1]["content"]
        ids = list(PREFIX)
        for step in body.split(SEP)[:-1]:
            ids += [5] * len(step) + [7]
        return ids + list(SUFFIX)


def _row(steps, labels, outcome=1, z=None):
    r = {"problem_ko": "문제", "solution_steps": steps, "step_labels": labels,
         "outcome": outcome}
    if z is not None:
        r["teacher_logodds"] = z
    return r


def test_row_targets_last_is_outcome():
    assert row_targets([1, None, 1], 0) == [1, IGNORE_INDEX, 0]
    assert row_targets([1, 1, 0], 0) == [1, 1, 0]
    assert row_targets([None], 1) == [1]


def test_build_example_positions_and_targets():
    tok = FakeTok()
    ex = build_example(tok, SEP, _row(["ab", "cde", "f"], [1, None, 0], outcome=0))
    assert ex.sep_positions == [6, 10, 12]
    assert ex.targets == [1, IGNORE_INDEX, 0]
    assert ex.n_steps == 3
    assert ex.input_ids[-2:] == SUFFIX


def test_label_length_mismatch_is_dropped():
    tok = FakeTok()
    st = BuildStats()
    assert build_example(tok, SEP, _row(["a", "b"], [1]), stats=st) is None
    assert st.n_label_len_mismatch == 1
    st2 = BuildStats()
    assert build_example(tok, SEP, _row([], []), stats=st2) is None
    assert st2.n_no_steps == 1


def test_truncation_drops_only_when_a_sep_is_cut():
    tok = FakeTok()
    row = _row(["ab", "cde"], [1, 0], outcome=0)   # seps at 6 and 10, total len 13
    st = BuildStats()
    ex = build_example(tok, SEP, row, max_len=12, stats=st)   # cuts the suffix only
    assert ex is not None and len(ex.input_ids) == 12 and st.n_truncated_ok == 1
    st2 = BuildStats()
    assert build_example(tok, SEP, row, max_len=10, stats=st2) is None  # cuts sep at 10
    assert st2.n_truncated_drop == 1


def test_build_dataset_stats_and_soft_targets():
    tok = FakeTok()
    rows = [
        _row(["ab", "cd"], [1, 0], outcome=0, z=[2.0, -2.0]),
        _row(["ab"], [1], outcome=1),          # no teacher_logodds -> NaN
        _row(["ab", "cd"], [1], outcome=1),    # dropped
    ]
    ds, stats = build_dataset(rows, tok, SEP, max_len=2048, with_soft=True, verbose=False)
    assert (stats.n_rows, stats.n_kept, stats.n_label_len_mismatch) == (3, 2, 1)
    assert ds.examples[0].soft_targets[0] > 0.88
    assert all(math.isnan(v) for v in ds.examples[1].soft_targets)

    batch = Collator(pad_id=0, with_soft=True)(list(ds.examples))
    assert batch["input_ids"].shape == (2, 12)
    assert batch["n_steps"].tolist() == [2, 1]
    assert batch["step_mask"].tolist() == [[True, True], [True, False]]
    assert batch["targets"].tolist() == [[1, 0], [1, IGNORE_INDEX]]
    # padding of the shorter solution is masked out everywhere
    assert batch["attention_mask"][1].sum().item() == len(ds.examples[1].input_ids)
    assert torch.isnan(batch["soft_targets"][1, 1])


def test_encode_example_accepts_a_batchencoding():
    """transformers 5 returns a BatchEncoding (a UserDict, not a dict) from the template."""
    from collections import UserDict

    from koprm.train.model import encode_example

    class MappingTok(FakeTok):
        def apply_chat_template(self, conv, tokenize=True, add_generation_prompt=False):
            ids = super().apply_chat_template(conv, tokenize, add_generation_prompt)
            return UserDict({"input_ids": [ids], "attention_mask": [[1] * len(ids)]})

    ids, pos = encode_example(MappingTok(), "문제", ["가나다", "라마"], SEP)
    assert ids == encode_example(FakeTok(), "문제", ["가나다", "라마"], SEP)[0]
    assert pos == [len(PREFIX) + 3, len(PREFIX) + 3 + 1 + 2]  # the two SEP positions
    assert all(ids[p] == 7 for p in pos)
