"""train/data.py on a stub tokenizer (no model download, no network)."""
import math
from pathlib import Path

import pytest
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
    ds, stats = build_dataset(rows, tok, SEP, max_len=2048, soft_mode="soft",
                              verbose=False)
    assert (stats.n_rows, stats.n_kept, stats.n_label_len_mismatch) == (3, 2, 1)
    assert ds.examples[0].soft_targets[0] > 0.88
    assert all(math.isnan(v) for v in ds.examples[1].soft_targets)

    batch = Collator(pad_id=0, soft_mode="soft")(list(ds.examples))
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


def test_soft_y_overrides_the_teacher_on_correct_solutions():
    """§2.1: y=1 proves every prefix, so soft_y targets are 1.0 there; y=0 is unchanged."""
    tok = FakeTok()
    correct = _row(["ab", "cd"], [1, 1], outcome=1, z=[2.0, -2.0])
    wrong = _row(["ab", "cd"], [1, 0], outcome=0, z=[2.0, -2.0])

    soft_c = build_example(tok, SEP, correct, soft_mode="soft").soft_targets
    soft_y_c = build_example(tok, SEP, correct, soft_mode="soft_y").soft_targets
    assert soft_c[0] == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert soft_c[1] == pytest.approx(1 / (1 + math.exp(2.0)))
    assert soft_y_c == [1.0, 1.0]

    # a y=0 row is identical under both modes, and its last target stays the outcome
    ex_soft = build_example(tok, SEP, wrong, soft_mode="soft")
    ex_soft_y = build_example(tok, SEP, wrong, soft_mode="soft_y")
    assert ex_soft.soft_targets == ex_soft_y.soft_targets == pytest.approx(soft_c)
    assert ex_soft_y.targets[-1] == 0

    # y=1 needs no teacher under soft_y (plain soft has to fall back to NaN)
    no_teacher = _row(["ab", "cd"], [1, 1], outcome=1)
    assert build_example(tok, SEP, no_teacher, soft_mode="soft_y").soft_targets == [1.0, 1.0]
    assert all(math.isnan(v) for v in
               build_example(tok, SEP, no_teacher, soft_mode="soft").soft_targets)

    # hard mode keeps no soft targets at all, and an unknown mode is a programming error
    assert build_example(tok, SEP, correct).soft_targets is None
    with pytest.raises(ValueError):
        build_example(tok, SEP, correct, soft_mode="softish")


def test_collator_emits_soft_targets_for_soft_y():
    tok = FakeTok()
    rows = [_row(["ab", "cd"], [1, 1], outcome=1, z=[2.0, -2.0]),
            _row(["ab"], [0], outcome=0, z=[-2.0])]
    ds, _ = build_dataset(rows, tok, SEP, soft_mode="soft_y", verbose=False)
    batch = Collator(pad_id=0, soft_mode="soft_y")(list(ds.examples))
    assert batch["soft_targets"][0].tolist()[:2] == [1.0, 1.0]
    assert batch["soft_targets"][1, 0].item() == pytest.approx(1 / (1 + math.exp(2.0)))
    assert torch.isnan(batch["soft_targets"][1, 1])            # padding stays NaN
    assert "soft_targets" not in Collator(pad_id=0)(list(ds.examples))


# ---------------------------------------------- §12.2 PRM head init / FSDP plumbing


def test_load_prm_head_state_maps_score_to_the_head(tmp_path):
    """score.0/score.2 of Qwen2ForProcessRewardModel are PRMHead.net[0]/net[2]."""
    import json as _json

    from safetensors.torch import save_file

    from koprm.train.model import PRMHead, load_prm_head_state

    h = 8
    tensors = {
        "score.0.weight": torch.arange(h * h, dtype=torch.float32).reshape(h, h),
        "score.0.bias": torch.ones(h, dtype=torch.float32),
        "score.2.weight": torch.full((2, h), 0.5, dtype=torch.float32),
        "score.2.bias": torch.tensor([0.25, -0.25], dtype=torch.float32),
        "model.embed_tokens.weight": torch.zeros(4, h),
    }
    save_file({k: v for k, v in tensors.items() if k.startswith("score")},
              str(tmp_path / "model-00002-of-00002.safetensors"))
    save_file({"model.embed_tokens.weight": tensors["model.embed_tokens.weight"]},
              str(tmp_path / "model-00001-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(_json.dumps({
        "weight_map": {"model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                       **{k: "model-00002-of-00002.safetensors" for k in tensors
                          if k.startswith("score")}}}), encoding="utf-8")

    sd = load_prm_head_state(tmp_path)
    assert set(sd) == {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"}
    assert torch.equal(sd["net.0.weight"], tensors["score.0.weight"])
    assert torch.equal(sd["net.2.bias"], tensors["score.2.bias"])
    head = PRMHead(h)
    head.load_state_dict(sd)                       # shapes line up with the real head
    assert torch.equal(head.net[2].bias.detach(), tensors["score.2.bias"])

    # a single-file checkpoint works too, and a missing key is an error, not a silent skip
    (tmp_path / "model.safetensors.index.json").unlink()
    save_file({k: v for k, v in tensors.items() if k.startswith("score")},
              str(tmp_path / "model.safetensors"))
    assert torch.equal(load_prm_head_state(tmp_path)["net.0.bias"], tensors["score.0.bias"])
    save_file({"score.0.weight": tensors["score.0.weight"]},
              str(tmp_path / "model.safetensors"))
    with pytest.raises(KeyError):
        load_prm_head_state(tmp_path)


def test_split_state_dict():
    from koprm.train.model import split_state_dict

    backbone, head = split_state_dict({"backbone.layers.0.w": 1, "head.net.0.weight": 2,
                                       "other": 3})
    assert backbone == {"layers.0.w": 1} and head == {"net.0.weight": 2}


def test_grad_accum_for_effective_batch():
    from koprm.train.train import EFFECTIVE_BATCH, grad_accum_for

    assert EFFECTIVE_BATCH == 64
    assert grad_accum_for(4) == 16 and grad_accum_for(6) == 10      # single GPU: unchanged
    assert grad_accum_for(1, 4) == 16 and grad_accum_for(2, 4) == 8
    assert grad_accum_for(16, 4) == 1
    for bad in ((3, 4), (5, 2), (6, 4)):
        with pytest.raises(SystemExit):
            grad_accum_for(*bad)


def test_decoder_layers_finds_the_blocks():
    from koprm.train.train import decoder_layers

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])

    class Nested(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Wrapped()

    assert len(decoder_layers(Wrapped())) == 3
    assert len(decoder_layers(Nested())) == 3
    assert decoder_layers(torch.nn.Linear(2, 2)) == []


def test_clean_backbone_config_drops_remote_code_hooks():
    from transformers import AutoConfig

    from koprm.train.model import clean_backbone_config, needs_config_cleanup

    cfg = AutoConfig.for_model("qwen2", hidden_size=8, num_hidden_layers=1,
                               num_attention_heads=2, num_key_value_heads=2,
                               vocab_size=32, intermediate_size=16)
    assert not needs_config_cleanup(cfg)
    assert clean_backbone_config(cfg) is cfg                 # nothing to clean: untouched

    cfg.auto_map = {"AutoConfig": "configuration_qwen2_rm.Qwen2RMConfig",
                    "AutoModel": "modeling_qwen2_rm.Qwen2ForProcessRewardModel"}
    cfg.architectures = ["Qwen2ForProcessRewardModel"]
    clean = clean_backbone_config(cfg, "Qwen2Model")
    d = clean.to_dict()
    assert "auto_map" not in d and d["architectures"] == ["Qwen2Model"]
    assert d["model_type"] == "qwen2" and d["hidden_size"] == 8   # the rest survives


def test_saved_config_never_carries_auto_map(tmp_path):
    """save_pretrained cleans the config, so from_pretrained needs no remote code."""
    import json as _json

    from transformers import AutoConfig

    from koprm.train.model import HEAD_FILE, PRM_CONFIG_FILE, StepPRM

    cfg = AutoConfig.for_model("qwen2", hidden_size=8, num_hidden_layers=1,
                               num_attention_heads=2, num_key_value_heads=2,
                               vocab_size=32, intermediate_size=16)
    cfg.auto_map = {"AutoModel": "modeling_qwen2_rm.Qwen2ForProcessRewardModel"}

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = cfg
            self.lin = torch.nn.Linear(8, 8)

        def save_pretrained(self, d, state_dict=None):
            (Path(d) / "config.json").write_text(
                _json.dumps(self.config.to_dict()), encoding="utf-8")

    class Tok:
        def convert_tokens_to_ids(self, t):
            return 7

        def save_pretrained(self, d):
            (Path(d) / "tokenizer.json").write_text("{}", encoding="utf-8")

    model = StepPRM(Backbone(), Tok(), "<extra_0>", "Qwen/Qwen2.5-Math-PRM-7B")
    model.save_pretrained(tmp_path)
    saved = _json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert "auto_map" not in saved
    assert saved["architectures"] == ["Backbone"]      # the live backbone class
    assert (tmp_path / HEAD_FILE).exists() and (tmp_path / PRM_CONFIG_FILE).exists()

    # the same holds for the FSDP path, where a gathered state dict is passed in
    model.save_pretrained(tmp_path / "fsdp",
                          state_dict={"backbone.lin.weight": torch.zeros(8, 8),
                                      "head.net.0.weight": torch.zeros(8, 8)})
    saved = _json.loads((tmp_path / "fsdp/config.json").read_text(encoding="utf-8"))
    assert "auto_map" not in saved


def test_take_decoder_and_input_embeddings():
    from koprm.train.model import input_embeddings, take_decoder

    class Decoder(torch.nn.Module):
        def __init__(self, attr):
            super().__init__()
            setattr(self, attr, torch.nn.Embedding(8, 4))
            self.h = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

    class CausalLM(torch.nn.Module):
        def __init__(self, attr="transformer", getter=False):
            super().__init__()
            setattr(self, attr, Decoder("wte"))
            self.lm_head = torch.nn.Linear(4, 8)
            if getter:
                self.get_decoder = lambda: getattr(self, attr)

    for attr in ("model", "transformer", "base_model"):
        dec = take_decoder(CausalLM(attr))
        assert isinstance(dec, Decoder) and not hasattr(dec, "lm_head")
    assert isinstance(take_decoder(CausalLM(getter=True)), Decoder)   # get_decoder wins
    lonely = torch.nn.Linear(4, 4)
    assert take_decoder(lonely) is lonely                              # nothing to unwrap

    # the getter may be missing or raise (remote classes in transformers 5); fall back
    dec = Decoder("wte")
    assert input_embeddings(dec) is dec.wte
    class NoGetter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.odd_name = torch.nn.Embedding(4, 2)

        def get_input_embeddings(self):
            raise NotImplementedError

    assert isinstance(input_embeddings(NoGetter()), torch.nn.Embedding)
    assert input_embeddings(torch.nn.Linear(2, 2)) is None


def test_copy_remote_code_follows_auto_map(tmp_path):
    from koprm.train.model import copy_remote_code

    src = tmp_path / "src"
    src.mkdir()
    (src / "modeling_exaone.py").write_text("# model", encoding="utf-8")
    (src / "configuration_exaone.py").write_text("# config", encoding="utf-8")
    (src / "unrelated.py").write_text("# no", encoding="utf-8")

    class Cfg:
        def __init__(self):
            self.auto_map = {"AutoConfig": "configuration_exaone.ExaoneConfig",
                             "AutoModelForCausalLM": "modeling_exaone.ExaoneForCausalLM"}

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()

    dest = tmp_path / "ckpt"
    dest.mkdir()
    copied = copy_remote_code(Backbone(), dest, str(src))
    assert sorted(set(copied)) == ["configuration_exaone.py", "modeling_exaone.py"]
    assert not (dest / "unrelated.py").exists()

    class Clean(torch.nn.Module):          # no auto_map -> nothing to copy
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {})()

    assert copy_remote_code(Clean(), dest, str(src)) == []


def test_hsdp_mesh_shape():
    from koprm.train.train import hsdp_mesh_shape

    assert hsdp_mesh_shape(4, 1) is None and hsdp_mesh_shape(1, 1) is None   # pure FSDP
    assert hsdp_mesh_shape(8, 2) == (2, 4)      # two 4-GPU shard groups
    assert hsdp_mesh_shape(8, 4) == (4, 2)
    assert hsdp_mesh_shape(8, 8) == (8, 1)      # full replication, no sharding
    for world, n in ((8, 3), (8, 5), (6, 4)):
        with pytest.raises(SystemExit, match="does not divide"):
            hsdp_mesh_shape(world, n)
    for world, n in ((4, 8), (4, 0), (4, -1)):
        with pytest.raises(SystemExit, match="must be in"):
            hsdp_mesh_shape(world, n)
