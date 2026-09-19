"""§3 Step-level translation with masking and verification.

One translator does the whole job. Backends (both via vLLM):
  * translategemma : google/translategemma-* (takes no system prompt; masking is the only
                     control we have, see §3). Its chat template renders a translation
                     instruction from `source_lang_code`/`target_lang_code`.
  * instruct       : any chat model (Qwen2.5-7B-Instruct, gemma-3-12b-it) with a strict
                     translation prompt.

`translate_steps` translates a flat list of steps (one item = one step, no context) and
returns (translations, statuses). A step whose placeholders cannot be restored, or whose
output holds script that does not belong in the target language, gets text None and status
"fail:<reason>". A row with any failed item has `mask_restore_ok` False and is dropped
downstream -- there is no second pass.

A step that is nothing but masked formulas/markdown (no letters left after masking, e.g.
a display equation on its own line, or "---") is passed through verbatim with status
"verbatim": there is nothing to translate, and asking a model to translate a lone
placeholder invites hallucination (gemma-3-12b-it answered a made-up problem for "---").

§3.2 also checks the raw output for script that does not belong in the target language
(untranslated Korean left in an English output, Chinese meta-commentary from Qwen in a
Korean one). That is a failure like a broken placeholder: status "fail:lang". Verbatim steps
skip the check.

Statuses, per item: "ok" / "order" (restored, the latter with the placeholders reordered),
"verbatim", or "fail:<lang|missing|duplicate|extra>".

For a list-valued field a failed row keeps the per-item translations it did get (nulls where
an item failed); `mask_restore_ok` stays the single gate on whether a row is usable. Output is
written a chunk at a time and keyed by id, so an interrupted run resumes where it stopped.

Sharding: --shard i --num-shards k takes every k-th row, so several processes (one per GPU)
can run side by side; output files are per shard and resumable, exactly as in koprm.gen.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from koprm.io import load_jsonl, write_jsonl
from koprm.translate.mask import mask, restore

LANG_NAME = {"ko": "Korean", "en": "English"}

INSTRUCT_PROMPT = (
    "You are a professional {src_name}-to-{tgt_name} translator of mathematics solutions.\n"
    "Translate the text between the markers into {tgt_name}.\n"
    "Rules:\n"
    "- Placeholders that look like ⟦M1⟧, ⟦M2⟧, ... stand for formulas or numbers. Copy each one exactly, "
    "unchanged and in the same order. Do not add, drop, or renumber any placeholder.\n"
    "- Keep markdown such as '## ' headings.\n"
    "- Translate everything else faithfully; do not solve, explain, correct, or add anything.\n"
    "- Output only the translation, nothing else.\n"
    "<text>\n{text}\n</text>"
)


def _is_multimodal(model: str) -> bool:
    """True for checkpoints with a vision tower (gemma-3-*-it, translategemma-*).

    vLLM loads those as Gemma3ForConditionalGeneration and profiles image memory unless
    told otherwise; we only ever send text, so cap image inputs at 0.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except Exception:
        return False
    return hasattr(cfg, "vision_config") or hasattr(getattr(cfg, "text_config", None), "vision_config")


def gemma3_rope_compat(config):
    """Rewrite Gemma 3's Transformers-v5 nested RoPE config into the v4 fields vLLM reads.

    TranslateGemma's config.json (saved by transformers 4.57) stores RoPE per layer type:
    `rope_parameters = {"full_attention": {...}, "sliding_attention": {...}}`. On a
    transformers 4.x install vLLM 0.15's patch_rope_parameters() first injects the top-level
    `rope_theta` into that dict, which breaks its own "is this nested?" check and then trips
    `ValueError: rope_parameters should have a 'rope_type' key`, so the model never loads.

    vLLM's own Gemma3Attention already handles the v4 shape: global layers take `rope_scaling`
    with `rope_theta`, sliding layers always use rope_type "default" with `rope_local_base_freq`.
    That is exactly what the nested config says, so hand vLLM the v4 shape instead. Passed as
    `hf_overrides`, which runs immediately before the patch. A no-op for a v4 config.
    """
    text = config.get_text_config()
    rope = getattr(text, "rope_parameters", None)
    if isinstance(rope, dict) and set(rope) & {"full_attention", "sliding_attention"}:
        text.rope_scaling = dict(rope.get("full_attention", {"rope_type": "default"}))
        del text.rope_parameters
    return config


def _needs_translation(masked_text: str) -> bool:
    """False when the masked step holds no letters at all (only formulas/markdown/digits)."""
    from koprm.translate.mask import PH_RE

    return any(ch.isalpha() for ch in PH_RE.sub(" ", masked_text))


# Hangul syllables + jamo, and CJK ideographs. Math lives in placeholders during the check,
# so anything matched here is prose the model failed to translate (or added).
_HANGUL_RE = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
_CJK_RE = re.compile(r"[一-鿿]")


def has_foreign_script(text: str, tgt: str) -> bool:
    """True when `text` holds script that a translation into `tgt` must not contain.

    en: any Hangul (untranslated source) or CJK ideograph. ko: any CJK ideograph -- Hanja is
    not used in the corpus, so ideographs mean the model drifted into Chinese.
    """
    if tgt == "en":
        return bool(_HANGUL_RE.search(text) or _CJK_RE.search(text))
    if tgt == "ko":
        return bool(_CJK_RE.search(text))
    return False


def _strip_wrappers(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^<text>\s*|\s*</text>$", "", s).strip()
    s = re.sub(r"^```[a-z]*\n|\n```$", "", s).strip()
    return s


class Translator:
    def __init__(self, model: str, backend: str, tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85, max_model_len: int = 4096):
        from vllm import LLM

        self.backend = backend
        self.model = model
        kw = {"hf_overrides": gemma3_rope_compat}
        if backend == "translategemma" or _is_multimodal(model):
            kw["limit_mm_per_prompt"] = {"image": 0}
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            seed=0,
            **kw,
        )
        self.tok = self.llm.get_tokenizer()
        if backend == "translategemma" and not getattr(self.tok, "chat_template", None):
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(model)

    def _messages(self, text: str, src: str, tgt: str) -> list[dict]:
        if self.backend == "translategemma":
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "source_lang_code": src, "target_lang_code": tgt, "text": text}
                    ],
                }
            ]
        return [
            {
                "role": "user",
                "content": INSTRUCT_PROMPT.format(
                    src_name=LANG_NAME[src], tgt_name=LANG_NAME[tgt], text=text
                ),
            }
        ]

    def translate_raw(self, texts: list[str], src: str, tgt: str, max_tokens: int = 1024) -> list[str]:
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        msgs = [self._messages(t, src, tgt) for t in texts]
        if self.backend == "translategemma":
            # llm.chat() flattens a content part down to {"type","text"} and drops every other
            # key (chat_utils._parse_chat_message_content_part), which would strip exactly the
            # source_lang_code/target_lang_code the TranslateGemma template reads. Render the
            # template here instead; it emits its own <bos>, so tokenize through it too.
            prompts = [
                {"prompt_token_ids": self.tok.apply_chat_template(
                    m, tokenize=True, add_generation_prompt=True)}
                for m in msgs
            ]
            outs = self.llm.generate(prompts, sp, use_tqdm=True)
        else:
            outs = self.llm.chat(msgs, sp, use_tqdm=True)
        return [_strip_wrappers(o.outputs[0].text) for o in outs]


def translate_steps(
    steps: list[str], src: str, tgt: str, translator: Translator, mask_numbers: bool = True,
) -> tuple[list[str | None], list[str]]:
    masked = [mask(s, numbers=mask_numbers) for s in steps]
    out_text: list[str | None] = [None] * len(steps)
    status = ["fail"] * len(steps)
    todo = []
    for i, m in enumerate(masked):
        if _needs_translation(m.text):
            todo.append(i)
        else:
            out_text[i], status[i] = steps[i], "verbatim"
    raw = translator.translate_raw([masked[i].text for i in todo], src, tgt) if todo else []
    for i, r in zip(todo, raw):
        t, st = (None, "lang") if has_foreign_script(r, tgt) else restore(r, masked[i].spans)
        out_text[i] = t
        status[i] = st if t is not None else f"fail:{st}"
    return out_text, status


def row_items(value) -> tuple[list[str], bool]:
    """A row's field as a flat list of items, plus whether the field itself was a list."""
    is_list = isinstance(value, list)
    return (list(value) if is_list else [value]), is_list


def assemble_row(row_id, out_field: str, texts: list[str | None], statuses: list[str],
                 is_list: bool) -> dict:
    """The one output-row shape (key order is part of the file format)."""
    return {
        "id": row_id,
        out_field: texts if is_list else texts[0],
        f"{out_field}_status": statuses,
        "mask_restore_ok": all(t is not None for t in texts),
    }


def shard_path(out_path: str | Path, shard: int, num_shards: int) -> Path:
    """Per-shard output file, so parallel shards never write to the same file."""
    out = Path(out_path)
    if num_shards <= 1:
        return out
    return out.with_name(f"{out.stem}.shard{shard}of{num_shards}{out.suffix}")


def translate_file(
    in_path: str, out_path: str, field: str, out_field: str, src: str, tgt: str,
    translator: Translator, batch: int = 4000, only_ids: set | None = None,
    shard: int = 0, num_shards: int = 1,
) -> None:
    """Translate `field` (a string or a list of step strings) for each row, resumable."""
    rows = load_jsonl(in_path)
    if num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % num_shards == shard]
        print(f"[translate] shard {shard} of {num_shards}: {len(rows)} rows")
    if only_ids is not None:
        rows = [r for r in rows if r["id"] in only_ids]
        print(f"[translate] --only-ids: {len(rows)} of the input rows selected")
    out = Path(out_path)
    done = set()
    if out.exists():
        done = {r["id"] for r in load_jsonl(out)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"[translate] {len(todo)} rows to do ({len(done)} cached)")
    for b in range(0, len(todo), batch):
        chunk = todo[b : b + batch]
        flat, shapes = [], []
        for r in chunk:
            items, is_list = row_items(r[field])
            shapes.append((r["id"], len(items), is_list))
            flat.extend(items)
        texts, statuses = translate_steps(flat, src, tgt, translator)
        k = 0
        results = []
        for row_id, n, is_list in shapes:
            results.append(assemble_row(row_id, out_field, texts[k : k + n], statuses[k : k + n], is_list))
            k += n
        write_jsonl(out, results, append=True)
        print(f"[translate] {b + len(chunk)}/{len(todo)}")


def _load_ids(path: str) -> set:
    return {ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", required=True, help="problem_en | steps | ...")
    ap.add_argument("--out-field", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--primary-model", required=True)
    ap.add_argument("--primary-backend", default="instruct", choices=["translategemma", "instruct"])
    ap.add_argument("--only-ids", default=None, help="file with one id per line; limits the run")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--batch", type=int, default=4000)
    args = ap.parse_args()
    out = shard_path(args.out, args.shard, args.num_shards)
    translator = Translator(args.primary_model, args.primary_backend, args.tp,
                            args.gpu_memory_utilization)
    only = _load_ids(args.only_ids) if args.only_ids else None
    translate_file(args.inp, str(out), args.field, args.out_field, args.src, args.tgt,
                   translator, args.batch, only, args.shard, args.num_shards)


if __name__ == "__main__":
    main()
