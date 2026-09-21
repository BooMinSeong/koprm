"""§6.2 Student: a causal-LM backbone with a Qwen2ForProcessRewardModel-style head.

Structure (copied from the public `modeling_qwen2_rm.py`):

    backbone (unmodified causal LM body, no LM head)
      -> Linear(h, h) -> ReLU -> Linear(h, 2)

The model emits 2-class logits at *every* position; the caller reads only the
positions of a single step-separator token (SEP), exactly like `<extra_0>` for
Qwen2.5-Math-PRM.  The head is kept in float32 even when the backbone is bf16,
so that the log-odds z = l1 - l0 are not quantised near the ceiling (§2.2).

Input text (§6.2):

    chat_template([system: SYSTEM_PROMPT_KO,
                   user:   problem_ko,
                   assistant: SEP.join(steps) + SEP])

SEP choice per backbone:
  * Qwen2.5-*          : "<extra_0>" (added to the tokenizer when missing; the
                         Instruct checkpoints ship 151665 tokens but a 151936-row
                         embedding matrix, so no resize is needed).
  * EXAONE-4.0-1.2B    : "[unused0]" - the tokenizer reserves [unused0..99]
                         (ids 62..161), already in the 102400-row embedding
                         matrix and never produced by the model.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from koprm.paths import SYSTEM_PROMPT_KO

HEAD_FILE = "head.pt"
PRM_CONFIG_FILE = "prm_config.json"

# Preferred separator per backbone family, tried in order; the first one that
# encodes to exactly one id wins.
SEP_CANDIDATES = ("<extra_0>", "[unused0]", "<|unused_0|>", "<|reserved_special_token_0|>")
_RESERVED_RE = re.compile(r"unused|extra|reserved|placeholder", re.IGNORECASE)


def _local_snapshot(name: str) -> str:
    """Cached snapshot dir for `name` (transformers 4.57.3 still calls the hub for a
    plain repo id even under HF_HUB_OFFLINE=1; a local path skips that)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(name, local_files_only=True)


def _offline_retry(loader, name, **kw):
    try:
        return loader(name, **kw)
    except Exception as e:  # only the offline case is retried
        if "offline" not in str(e).lower() or os.path.isdir(str(name)):
            raise
        return loader(_local_snapshot(str(name)), **kw)


def load_tokenizer(name, **kw):
    from transformers import AutoTokenizer

    kw.setdefault("trust_remote_code", True)
    return _offline_retry(AutoTokenizer.from_pretrained, name, **kw)


def find_sep_token(tokenizer, sep_token: str | None = None) -> tuple[str, bool]:
    """Return (sep_token, added) for `tokenizer`.

    Prefers an existing token that already encodes to a single id; falls back to
    scanning the added vocabulary for an unused/reserved token; finally adds
    "<extra_0>" as a new special token (caller must then resize embeddings).
    """
    def is_single(t: str) -> bool:
        ids = tokenizer.encode(t, add_special_tokens=False)
        return len(ids) == 1 and ids[0] != getattr(tokenizer, "unk_token_id", None)

    if sep_token is not None:
        if is_single(sep_token):
            return sep_token, False
        tokenizer.add_special_tokens({"additional_special_tokens": [sep_token]})
        return sep_token, True

    for cand in SEP_CANDIDATES:
        if is_single(cand):
            return cand, False
    # any reserved-looking added token that is not already in use
    added = sorted(tokenizer.get_added_vocab().items(), key=lambda kv: kv[1])
    for tokstr, _ in added:
        if _RESERVED_RE.search(tokstr) and is_single(tokstr):
            return tokstr, False
    tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_0>"]})
    return "<extra_0>", True


def build_conversation(problem_ko: str, steps: list[str], sep_token: str,
                       system_prompt: str = SYSTEM_PROMPT_KO) -> list[dict]:
    body = sep_token.join(s.strip() for s in steps) + sep_token
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_ko},
        {"role": "assistant", "content": body},
    ]


def encode_example(tokenizer, problem_ko: str, steps: list[str], sep_token: str,
                   system_prompt: str = SYSTEM_PROMPT_KO) -> tuple[list[int], list[int]]:
    """Return (input_ids, sep_positions).  No truncation here (see train/data.py)."""
    conv = build_conversation(problem_ko, steps, sep_token, system_prompt)
    ids = tokenizer.apply_chat_template(conv, tokenize=True, add_generation_prompt=False)
    # transformers 5 returns a BatchEncoding (a UserDict, so *not* a dict) from
    # apply_chat_template(tokenize=True); Mapping covers both it and a plain dict.
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if len(ids) and isinstance(ids[0], list):
        ids = ids[0]
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    pos = [i for i, t in enumerate(ids) if t == sep_id]
    return list(ids), pos


# Qwen2ForProcessRewardModel stores the head as `score.0` (Linear h->h) and `score.2`
# (Linear h->2) -- exactly PRMHead.net[0] and net[2] (§12.2).
PRM_HEAD_KEYS = {
    "score.0.weight": "net.0.weight",
    "score.0.bias": "net.0.bias",
    "score.2.weight": "net.2.weight",
    "score.2.bias": "net.2.bias",
}
PRM_SEP_TOKEN = "<extra_0>"  # id 151651 in the PRM tokenizer, a single token


def resolve_model_dir(name: str | os.PathLike) -> Path:
    """A local directory for `name` (a path, or a cached hub snapshot)."""
    return Path(name) if os.path.isdir(str(name)) else Path(_local_snapshot(str(name)))


def load_prm_head_state(model_dir: str | os.PathLike) -> dict:
    """The two head Linears of a PRM checkpoint as a `PRMHead` state dict (float32)."""
    from safetensors import safe_open

    d = resolve_model_dir(model_dir)
    index = d / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        per_file: dict[str, list[str]] = {}
        for src in PRM_HEAD_KEYS:
            if src not in weight_map:
                raise KeyError(f"{d}: no {src} in the safetensors index")
            per_file.setdefault(weight_map[src], []).append(src)
    else:
        per_file = {"model.safetensors": list(PRM_HEAD_KEYS)}
    out: dict[str, torch.Tensor] = {}
    for fname, keys in per_file.items():
        with safe_open(str(d / fname), framework="pt") as f:
            available = set(f.keys())
            for src in keys:
                if src not in available:
                    raise KeyError(f"{d / fname}: no {src}")
                out[PRM_HEAD_KEYS[src]] = f.get_tensor(src).to(torch.float32)
    return out


class PRMHead(nn.Module):
    """Linear(h, h) -> ReLU -> Linear(h, 2), always float32."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        ).to(torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states.to(torch.float32))


# A backbone taken out of a remote-code checkpoint (Qwen2ForProcessRewardModel) keeps that
# checkpoint's `auto_map`, so a saved config would send the loader looking for
# configuration_qwen2_rm.py that we never copy. The backbone itself is a plain Qwen2Model.
REMOTE_CODE_KEYS = ("auto_map", "custom_pipelines")


def needs_config_cleanup(config) -> bool:
    d = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return any(k in d for k in REMOTE_CODE_KEYS) or any(k.startswith("custom_") for k in d)


def clean_backbone_config(config, architecture: str | None = None):
    """Rebuild a config without the remote-code hooks; a no-op when there are none."""
    if not needs_config_cleanup(config):
        return config
    from transformers import AutoConfig

    d = config.to_dict()
    model_type = d.pop("model_type", None) or getattr(config, "model_type", None)
    for key in (*REMOTE_CODE_KEYS, "architectures", "transformers_version", "_name_or_path"):
        d.pop(key, None)
    for key in [k for k in d if k.startswith("custom_")]:
        d.pop(key)
    try:
        clean = AutoConfig.for_model(model_type, **d)
    except Exception:  # noqa: BLE001 - unknown model type: keep the remote code
        return config
    clean.architectures = [architecture or "Qwen2Model"]
    return clean


def split_state_dict(state_dict: dict) -> tuple[dict, dict]:
    """A gathered StepPRM state dict -> (backbone state dict, head state dict)."""
    backbone = {k[len("backbone."):]: v for k, v in state_dict.items()
                if k.startswith("backbone.")}
    head = {k[len("head."):]: v for k, v in state_dict.items() if k.startswith("head.")}
    return backbone, head


class StepPRM(nn.Module):
    def __init__(self, backbone, tokenizer, sep_token: str, backbone_name: str = ""):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.sep_token = sep_token
        self.sep_id = int(tokenizer.convert_tokens_to_ids(sep_token))
        self.backbone_name = backbone_name
        self.head = PRMHead(backbone.config.hidden_size)

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_backbone(cls, backbone_name: str, sep_token: str | None = None,
                      dtype: torch.dtype | None = None, device: str = "cpu",
                      attn_implementation: str | None = None) -> StepPRM:
        from transformers import AutoModel

        tokenizer = load_tokenizer(backbone_name)
        sep_token, added = find_sep_token(tokenizer, sep_token)
        kw = {"trust_remote_code": True}
        if dtype is not None:
            kw["dtype"] = dtype
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        backbone = _offline_retry(AutoModel.from_pretrained, backbone_name, **kw)
        n_rows = backbone.get_input_embeddings().weight.shape[0]
        sep_id = tokenizer.convert_tokens_to_ids(sep_token)
        if added and sep_id >= n_rows:
            backbone.resize_token_embeddings(len(tokenizer))
        model = cls(backbone, tokenizer, sep_token, backbone_name)
        return model.to(device)

    @classmethod
    def from_prm(cls, prm_name: str, sep_token: str | None = PRM_SEP_TOKEN,
                 dtype: torch.dtype | None = None, device: str = "cpu",
                 attn_implementation: str | None = None) -> StepPRM:
        """§12.2: start from Qwen2.5-Math-PRM (backbone *and* the trained 2-class head)."""
        from transformers import AutoConfig, AutoModel

        d = resolve_model_dir(prm_name)
        tokenizer = load_tokenizer(d)
        sep_token, _ = find_sep_token(tokenizer, sep_token)
        kw: dict = {}
        if dtype is not None:
            kw["dtype"] = dtype
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        cfg = AutoConfig.from_pretrained(d, trust_remote_code=True)
        if getattr(cfg, "model_type", None) == "qwen2":
            # the config resolves to Qwen2Model: no remote code, and the `model.` prefix of
            # the ForProcessRewardModel checkpoint is stripped by the base-model loader.
            backbone = AutoModel.from_pretrained(d, trust_remote_code=False, **kw)
        else:
            full = AutoModel.from_pretrained(d, trust_remote_code=True, **kw)
            backbone = getattr(full, "model", full)
        backbone.config = clean_backbone_config(backbone.config, type(backbone).__name__)
        model = cls(backbone, tokenizer, sep_token, str(prm_name))
        model.head.load_state_dict(load_prm_head_state(d))
        return model.to(device)

    def save_pretrained(self, save_dir: str | os.PathLike, state_dict: dict | None = None
                        ) -> None:
        """`state_dict` (full, CPU) comes from FSDP's gather; without it the live model."""
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        # never write a config that points at remote code we do not ship
        self.backbone.config = clean_backbone_config(self.backbone.config,
                                                     type(self.backbone).__name__)
        if state_dict is None:
            self.backbone.save_pretrained(d)
            head_sd = self.head.state_dict()
        else:
            backbone_sd, head_sd = split_state_dict(state_dict)
            if not backbone_sd or not head_sd:
                raise ValueError("state_dict has no backbone.*/head.* entries")
            self.backbone.save_pretrained(d, state_dict=backbone_sd)
        self.tokenizer.save_pretrained(d)
        torch.save(head_sd, d / HEAD_FILE)
        (d / PRM_CONFIG_FILE).write_text(
            json.dumps(
                {
                    "sep_token": self.sep_token,
                    "sep_id": self.sep_id,
                    "backbone": self.backbone_name,
                    "hidden_size": int(self.backbone.config.hidden_size),
                    "system_prompt": SYSTEM_PROMPT_KO,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, ckpt_dir: str | os.PathLike, device: str = "cpu",
                        dtype: torch.dtype | None = None) -> StepPRM:
        from transformers import AutoModel

        d = Path(ckpt_dir)
        cfg = json.loads((d / PRM_CONFIG_FILE).read_text(encoding="utf-8"))
        tokenizer = load_tokenizer(d)
        kw = {}
        if dtype is not None:
            kw["dtype"] = dtype
        try:  # a clean config needs no remote code; only fall back when it does
            backbone = AutoModel.from_pretrained(d, trust_remote_code=False, **kw)
        except Exception:  # noqa: BLE001 - any loader error means remote code is needed
            backbone = AutoModel.from_pretrained(d, trust_remote_code=True, **kw)
        model = cls(backbone, tokenizer, cfg["sep_token"], cfg.get("backbone", ""))
        model.head.load_state_dict(torch.load(d / HEAD_FILE, map_location="cpu"))
        return model.to(device)

    # ------------------------------------------------------------------ apply
    def to(self, *args, **kwargs):  # keep the head in float32
        super().to(*args, **kwargs)
        self.head.to(torch.float32)
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """-> logits [B, L, 2] (float32)."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return self.head(out.last_hidden_state)

    def step_logits_list(self, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor | None = None,
                         sep_positions: list[list[int]] | None = None
                         ) -> list[torch.Tensor]:
        """Per-example [n_steps, 2] logits at the SEP positions."""
        logits = self.forward(input_ids, attention_mask)
        if sep_positions is None:
            sep_positions = [
                (row == self.sep_id).nonzero(as_tuple=True)[0].tolist() for row in input_ids
            ]
        return [logits[b, torch.as_tensor(p, dtype=torch.long, device=logits.device)]
                for b, p in enumerate(sep_positions)]

    def step_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                    sep_positions: torch.Tensor | list[list[int]] | None = None,
                    step_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Padded step logits [B, S, 2].

        `sep_positions` may be a padded LongTensor [B, S] (with `step_mask` marking
        the real entries) - that is what the collator in train/data.py produces.
        """
        logits = self.forward(input_ids, attention_mask)
        if sep_positions is None:
            rows = [(row == self.sep_id).nonzero(as_tuple=True)[0] for row in input_ids]
            S = max((len(r) for r in rows), default=0)
            sep_positions = torch.zeros(len(rows), S, dtype=torch.long, device=logits.device)
            step_mask = torch.zeros(len(rows), S, dtype=torch.bool, device=logits.device)
            for b, r in enumerate(rows):
                sep_positions[b, : len(r)] = r
                step_mask[b, : len(r)] = True
        elif not torch.is_tensor(sep_positions):
            S = max((len(p) for p in sep_positions), default=0)
            pad = torch.zeros(len(sep_positions), S, dtype=torch.long, device=logits.device)
            step_mask = torch.zeros(len(sep_positions), S, dtype=torch.bool, device=logits.device)
            for b, p in enumerate(sep_positions):
                pad[b, : len(p)] = torch.as_tensor(p, dtype=torch.long, device=logits.device)
                step_mask[b, : len(p)] = True
            sep_positions = pad
        sep_positions = sep_positions.to(logits.device)
        gathered = torch.gather(
            logits, 1, sep_positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
        )
        if step_mask is not None:
            gathered = gathered * step_mask.to(gathered.dtype).unsqueeze(-1)
        return gathered

    def gradient_checkpointing_enable(self, **kw) -> None:
        self.backbone.gradient_checkpointing_enable(**kw)
