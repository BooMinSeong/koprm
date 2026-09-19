"""§3 Translation bridge: one instruct translator over masked text, resumable jsonl.

The translator is gemma-4-12B-it with an instruction prompt (Plan §3, HANDOFF). There is
no fallback model, no retry pass and no similarity check: a unit either restores cleanly or
it is reported as a failure.

Per unit (a problem string, or one step of a solution):
    mask()              -> "⟦M1⟧" placeholders for math spans and standalone numbers
    formula-only?       -> keep the original verbatim, status "passthrough" (the translator
                           invents content when there is nothing to translate)
    chat prompt         -> per-unit system prompt (placeholder / step-header rules only when
                           that unit has them), temperature 0, fixed seed, one sample
    unwrap              -> strip surrounding whitespace, one layer of quotes or code fence
    cut                 -> a "\n## ..." continuation the model invented for a source without a
                           header, and a trailing run of placeholders it invented
    script check        -> target-language script only outside placeholders, else "script"
    restore()           -> "ok"/"order" pass, "missing"/"duplicate"/"extra" fail

Rows in : {<id-key>, <field>} where <field> is a string (problem) or a list (steps).
Rows out: {<id-key>, [problem_id, generator, outcome], <out-field> (None if the row failed),
           mask_restore_ok, restore_status (str, or one per step), translator,
           raw_failed (failed rows only: the unwrapped model output, or one per step with
           None where the step was fine)}
Failed rows are written too, so a re-run resumes past them and the failure rate is visible.

    python -m koprm.translate.translate --in data/trans/problems_in.jsonl \
        --out data/trans/problems_ko.jsonl --field problem_en --out-field problem_ko \
        --src en --tgt ko --primary-model google/gemma-4-12B-it --primary-backend instruct
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from koprm.io import done_ids, load_jsonl, write_jsonl
from koprm.paths import TRANSLATOR
from koprm.translate.mask import PH_RE, has_step_header, mask, restore

LANG = {"en": "English", "ko": "Korean"}
CARRY_FIELDS = ("problem_id", "generator", "outcome")
OK_STATUSES = ("ok", "order", "passthrough")

# Hangul (syllables, jamo) and CJK Han: the scripts that may not survive a translation.
_HANGUL_RE = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
_HAN_RE = re.compile(r"[一-鿿]")
_FENCE_RE = re.compile(r"^```[A-Za-z0-9_+-]*\n(?P<body>.*)\n```$", re.DOTALL)
# A markdown header in the source, and one the model started on its own after the translation.
_HEADER_LINE_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_CONTINUATION_RE = re.compile(r"\n#{1,6}\s")
# A placeholder at the very end, with only whitespace or punctuation after it.
_TAIL_PH_RE = re.compile(rf"\s*{PH_RE.pattern}[\s,;:.·、。]*$")
_QUOTES = (('"', '"'), ("'", "'"), ("“", "”"), ("「", "」"))

# The prompt is assembled per unit: a rule about placeholders or step headers in a unit that
# has neither invites the model to invent them (the pilot's "extra"/"duplicate" failures were
# hallucinated "## 단계 2: ⟦M2⟧ ..." continuations). Four variants only, so prefix caching still
# pays off.
_HEAD = "You are a professional translator. Translate the user's message from {src} to {tgt}."
_RULE_PLACEHOLDERS = (
    "Every placeholder of the form ⟦M1⟧, ⟦M2⟧, ... is a frozen chunk of mathematics. Copy "
    "each one verbatim, exactly once, with the same number, and put it at the place in the "
    "{tgt} sentence where its content belongs. Never translate, split, merge, drop, add or "
    "renumber a placeholder."
)
_RULE_HEADER = (
    "Keep the markdown structure of the message, including line breaks. A step header stays "
    'a step header: "## 단계 3:" becomes "## Step 3:" and "## Step 3:" becomes "## 단계 3:", '
    "keeping the same number."
)
_RULE_WHOLE = (
    "Translate the whole message and nothing more. Do not continue it, add steps, add "
    "placeholders, or add notes."
)
_RULE_SCRIPT = "Write natural {tgt}. Use no {src} words and no other script in your answer{exc}."
_RULE_OUTPUT = (
    "Output only the translation. No preamble, no quotes around it, no code fences, no "
    "notes, no explanation."
)


def system_prompt(src: str, tgt: str, placeholders: bool = True, header: bool = True) -> str:
    """The instruction for one unit: only the rules that unit's masked text can need."""
    rules = []
    if placeholders:
        rules.append(_RULE_PLACEHOLDERS)
    if header:
        rules.append(_RULE_HEADER)
    rules.append(_RULE_WHOLE)
    rules.append(_RULE_SCRIPT.format(exc=", except inside the placeholders" if placeholders else "",
                                     src="{src}", tgt="{tgt}"))
    rules.append(_RULE_OUTPUT)
    body = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, start=1))
    return f"{_HEAD}\nRules:\n{body}".format(src=LANG[src], tgt=LANG[tgt])


def needs_translation(masked_text: str) -> bool:
    """False for a formula-only unit: nothing left but placeholders and punctuation."""
    return any(ch.isalpha() for ch in PH_RE.sub(" ", masked_text))


def unwrap(text: str) -> str:
    """Strip whitespace, and one layer of code fence or matching quotes around the whole output."""
    t = text.strip()
    m = _FENCE_RE.match(t)
    if m:
        return m.group("body").strip()
    for a, b in _QUOTES:
        inner = t[1:-1]
        if len(t) >= 2 and t.startswith(a) and t.endswith(b) and a not in inner and b not in inner:
            return inner.strip()
    return t


def script_ok(text: str, tgt: str) -> bool:
    """Target-script check on the still-masked translation (§3: math is inside placeholders)."""
    bare = PH_RE.sub(" ", text)
    if _HAN_RE.search(bare):
        return False
    return not (tgt == "en" and _HANGUL_RE.search(bare))


def cut_continuation(text: str) -> str:
    """Cut a hallucinated "\\n## ..." continuation off a unit whose source had no header."""
    m = _CONTINUATION_RE.search(text)
    return text[: m.start()].rstrip() if m else text


def strip_extra_tail(text: str, n_spans: int) -> str:
    """Drop a trailing run of invented placeholders (numbers above the unit's span count)."""
    out = text
    while True:
        m = _TAIL_PH_RE.search(out)
        if m is None or int(m.group(1)) <= n_spans:
            return out
        out = out[: m.start()].rstrip()


def finish_unit(
    raw: str, spans: list[str], tgt: str, has_header: bool = False
) -> tuple[str | None, str, str]:
    """Post-process one model output: unwrap -> cut hallucinations -> script check -> restore.

    Returns (restored_text, status, unwrapped_output). The unwrapped output is the text
    *before* the cuts, so a failure (and what was cut) can be diagnosed from the jsonl.
    """
    text = unwrap(raw)
    cut = text if has_header else cut_continuation(text)
    cut = strip_extra_tail(cut, len(spans))
    if not script_ok(cut, tgt):
        return None, "script", text
    restored, status = restore(cut, spans)
    return restored, status, text


class InstructTranslator:
    """gemma-4-12B-it through vLLM: one greedy sample per masked unit."""

    def __init__(
        self,
        model: str = TRANSLATOR,
        max_tokens: int = 1024,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        seed: int = 0,
    ):
        from vllm import LLM, SamplingParams

        self.model = model
        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            seed=seed,
            tensor_parallel_size=tensor_parallel_size,
            # gemma-4-12B-it is a Gemma4UnifiedForConditionalGeneration checkpoint (vLLM >= 0.29);
            # we only ever send text, so the vision tower must not reserve multimodal memory.
            limit_mm_per_prompt={"image": 0},
        )
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=seed)

    def build_prompts(self, texts: list[str], systems: list[str]) -> list[str]:
        # The Gemma chat template accepts a system turn and folds it into the first user turn.
        convs = [
            [{"role": "system", "content": s}, {"role": "user", "content": t}]
            for t, s in zip(texts, systems)
        ]
        return self.tok.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)

    def translate(self, texts: list[str], systems: list[str]) -> list[str]:
        """One greedy sample per unit, each with that unit's system prompt."""
        if not texts:
            return []
        outs = self.llm.generate(self.build_prompts(texts, systems), self.sp, use_tqdm=True)
        return [o.outputs[0].text for o in outs]


def translate_units(
    texts: list[str], translator, src: str, tgt: str
) -> tuple[list[str | None], list[str], list[str | None]]:
    """Mask, translate what needs it, restore.

    Returns (texts_out, statuses, raw_outputs) aligned with `texts`; raw_outputs holds the
    unwrapped model output of every translated unit (None for a passthrough unit).
    """
    masked = [mask(t) for t in texts]
    todo = [i for i, m in enumerate(masked) if needs_translation(m.text)]
    systems = [
        system_prompt(src, tgt, bool(masked[i].spans), has_step_header(masked[i].text))
        for i in todo
    ]
    raws = translator.translate([masked[i].text for i in todo], systems) if todo else []
    out: list[str | None] = list(texts)
    statuses = ["passthrough"] * len(texts)
    outputs: list[str | None] = [None] * len(texts)
    for i, raw in zip(todo, raws):
        has_header = _HEADER_LINE_RE.search(masked[i].text) is not None
        out[i], statuses[i], outputs[i] = finish_unit(raw, masked[i].spans, tgt, has_header)
    return out, statuses, outputs


def translate_rows(
    rows: list[dict],
    translator,
    field: str,
    out_field: str,
    src: str,
    tgt: str,
    id_key: str = "id",
    translator_name: str = TRANSLATOR,
) -> tuple[list[dict], Counter]:
    """One vLLM call for all units of `rows`; one output row per input row."""
    units: list[str] = []
    owners: list[tuple[int, int]] = []  # (row index, unit index within the row)
    is_list = []
    for ri, r in enumerate(rows):
        v = r.get(field)
        is_list.append(isinstance(v, list))
        if isinstance(v, list):
            for ui, s in enumerate(v):
                units.append(s)
                owners.append((ri, ui))
        elif isinstance(v, str) and v:
            units.append(v)
            owners.append((ri, 0))

    texts, statuses, outputs = translate_units(units, translator, src, tgt)
    per_row: list[list[tuple[str | None, str, str | None]]] = [[] for _ in rows]
    for (ri, _), t, st, raw in zip(owners, texts, statuses, outputs):
        per_row[ri].append((t, st, raw))

    stats: Counter = Counter()
    out_rows = []
    for ri, r in enumerate(rows):
        got = per_row[ri]
        stats["steps_translated"] += sum(1 for _, st, _ in got if st != "passthrough")
        stats["steps_passthrough"] += sum(1 for _, st, _ in got if st == "passthrough")
        raw_failed: str | list[str | None] | None = None
        if not got:  # missing or empty field: nothing to translate, nothing to hand on
            value, status, ok = None, "empty", False
        elif is_list[ri]:
            status = [st for _, st, _ in got]
            ok = all(st in OK_STATUSES for st in status)
            value = [t for t, _, _ in got] if ok else None
            if not ok:  # the raw output of the steps that failed, aligned with the steps
                raw_failed = [None if st in OK_STATUSES else raw for _, st, raw in got]
        else:
            value, status, raw = got[0]
            ok = status in OK_STATUSES
            if not ok:
                raw_failed = raw
        out = {id_key: r[id_key]}
        out.update({k: r[k] for k in CARRY_FIELDS if k in r})
        out.update(
            {
                out_field: value,
                "mask_restore_ok": ok,
                "restore_status": status,
                "translator": translator_name,
            }
        )
        if not ok:  # keep the evidence: failures are fixed in the parsing/logic, not re-run
            out["raw_failed"] = raw_failed
        out_rows.append(out)
        stats["rows"] += 1
        stats["ok" if ok else "failed"] += 1
        for st in status if isinstance(status, list) else [status]:
            if st not in OK_STATUSES:
                stats[f"status_{st}"] += 1
    return out_rows, stats


def run(
    rows: list[dict],
    translator,
    out_path: Path,
    field: str,
    out_field: str,
    src: str,
    tgt: str,
    id_key: str = "id",
    batch: int = 256,
    translator_name: str = TRANSLATOR,
) -> Counter:
    done = done_ids(out_path, key=id_key)
    todo = [r for r in rows if r[id_key] not in done]
    print(f"[trans] {len(todo)} rows to do ({len(done)} already done) -> {out_path}")
    total: Counter = Counter()
    for b in range(0, len(todo), batch):
        chunk = todo[b : b + batch]
        out_rows, stats = translate_rows(
            chunk, translator, field, out_field, src, tgt, id_key, translator_name
        )
        write_jsonl(out_path, out_rows, append=True)
        total.update(stats)
        print(f"[trans] {b + len(chunk)}/{len(todo)} rows; ok={total['ok']} failed={total['failed']}")
    return total


def print_stats(stats: Counter) -> None:
    n = max(stats["rows"], 1)
    print(f"[trans] rows={stats['rows']} ok={stats['ok']} ({stats['ok'] / n:.3f}) "
          f"failed={stats['failed']}")
    fails = {k[len("status_"):]: v for k, v in sorted(stats.items()) if k.startswith("status_")}
    print(f"[trans] failures by status: {fails or '{}'}")
    print(f"[trans] units: translated={stats['steps_translated']} "
          f"passthrough={stats['steps_passthrough']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", required=True, help="source field: a string or a list of strings")
    ap.add_argument("--out-field", required=True)
    ap.add_argument("--src", choices=["en", "ko"], required=True)
    ap.add_argument("--tgt", choices=["en", "ko"], required=True)
    ap.add_argument("--primary-model", default=TRANSLATOR)
    # Kept for CLI compatibility; there is one translator and one backend (Plan §3).
    ap.add_argument("--primary-backend", choices=["instruct"], default="instruct")
    ap.add_argument("--id-key", default="id")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=256, help="rows per vLLM call")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.src == args.tgt:
        ap.error("--src and --tgt must differ")

    rows = load_jsonl(args.inp)
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    if args.num_shards > 1:
        out = out.with_name(f"{out.stem}.shard{args.shard}of{args.num_shards}{out.suffix}")

    translator = InstructTranslator(
        model=args.primary_model,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tp,
        seed=args.seed,
    )
    stats = run(
        rows,
        translator,
        out,
        args.field,
        args.out_field,
        args.src,
        args.tgt,
        id_key=args.id_key,
        batch=args.batch,
        translator_name=args.primary_model,
    )
    print_stats(stats)


if __name__ == "__main__":
    main()
