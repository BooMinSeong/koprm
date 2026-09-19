"""§3.2 target-language purity check, statuses, and the resumable row writer (no model)."""
from koprm.io import load_jsonl, write_jsonl
from koprm.translate.translate import (
    assemble_row,
    has_foreign_script,
    row_items,
    translate_file,
    translate_steps,
)


class FakeTranslator:
    """Stands in for Translator: the pipeline only ever calls translate_raw()."""

    def __init__(self, fn):
        self.fn = fn
        self.seen: list[str] = []

    def translate_raw(self, texts, src, tgt, max_tokens=1024):
        self.seen.extend(texts)
        return [self.fn(t) for t in texts]


def test_has_foreign_script_en():
    assert has_foreign_script("## Step 2: Add ⟦M1⟧, so the answer is ⟦M2⟧.", "en") is False
    assert has_foreign_script("## Step 2: 계산한다", "en") is True  # Hangul syllables
    assert has_foreign_script("Step ㄱ", "en") is True  # compatibility jamo
    assert has_foreign_script("Step ᄀ", "en") is True  # conjoining jamo
    assert has_foreign_script("Step 2: 首先", "en") is True  # CJK ideographs


def test_has_foreign_script_ko():
    assert has_foreign_script("## 단계 2: ⟦M1⟧을 더하면 ⟦M2⟧이다.", "ko") is False
    assert has_foreign_script("## 단계 2: Add the numbers", "ko") is False  # latin is fine
    assert has_foreign_script("## 단계 2: 首先计算", "ko") is True


def test_has_foreign_script_unknown_target():
    assert has_foreign_script("anything 아무거나 什么", "fr") is False


STEP = "## 단계 1: 계산\n$x = 1$이다."
MASKED = "## 단계 1: 계산\n⟦M1⟧이다."
CLEAN_EN = "## Step 1: Compute\n⟦M1⟧."
LEAKY_EN = "## Step 1: Compute\n계산하면 ⟦M1⟧이다."


def test_clean_output_is_ok():
    t = FakeTranslator(lambda s: CLEAN_EN)
    texts, statuses = translate_steps([STEP], "ko", "en", t)
    assert statuses == ["ok"]
    assert texts == ["## Step 1: Compute\n$x = 1$."]
    assert t.seen == [MASKED]  # the masked text is what the model sees


def test_reordered_placeholders_are_restored_as_order():
    step = "$a$ plus $b$"
    t = FakeTranslator(lambda s: "⟦M2⟧ 더하기 ⟦M1⟧")
    texts, statuses = translate_steps([step], "en", "ko", t)
    assert statuses == ["order"]
    assert texts == ["$b$ 더하기 $a$"]


def test_leaked_korean_is_fail_lang():
    t = FakeTranslator(lambda s: LEAKY_EN)
    texts, statuses = translate_steps([STEP], "ko", "en", t)
    assert statuses == ["fail:lang"]
    assert texts == [None]


def test_chinese_in_korean_output_is_fail_lang():
    t = FakeTranslator(lambda s: "## 단계 1: 계산\n首先, ⟦M1⟧.")
    texts, statuses = translate_steps(["## Step 1: Compute\n$x = 1$."], "en", "ko", t)
    assert statuses == ["fail:lang"]
    assert texts == [None]


def test_dropped_placeholder_is_fail_missing():
    t = FakeTranslator(lambda s: "## Step 1: Compute the value.")
    texts, statuses = translate_steps([STEP], "ko", "en", t)
    assert statuses == ["fail:missing"]
    assert texts == [None]


def test_lang_check_runs_before_restore():
    """A leak wins over a placeholder problem in the same output: the reason is "lang"."""
    t = FakeTranslator(lambda s: "## Step 1: 계산하면 끝.")  # Hangul *and* no ⟦M1⟧
    texts, statuses = translate_steps([STEP], "ko", "en", t)
    assert statuses == ["fail:lang"]
    assert texts == [None]


def test_verbatim_step_bypasses_lang_check():
    """Nothing to translate -> passed through untouched, even when it holds foreign script."""
    steps = ["$$x = 1$$", "---", "$\\boxed{\\text{漢}}$"]
    t = FakeTranslator(lambda s: "翻訳されない")
    texts, statuses = translate_steps(steps, "en", "ko", t)
    assert statuses == ["verbatim"] * 3
    assert texts == steps
    assert t.seen == []


def test_mixed_batch_keeps_indices_aligned():
    steps = [STEP, "---", "## 단계 2: 답을 적는다"]
    replies = {MASKED: LEAKY_EN, "## 단계 2: 답을 적는다": "## Step 2: Write the answer"}
    t = FakeTranslator(lambda s: replies[s])
    texts, statuses = translate_steps(steps, "ko", "en", t)
    assert statuses == ["fail:lang", "verbatim", "ok"]
    assert texts == [None, "---", "## Step 2: Write the answer"]


# ------------------------------------------------------------------ output row shape

def test_row_items():
    assert row_items(["a", "b"]) == (["a", "b"], True)
    assert row_items("a") == (["a"], False)


def test_assemble_row_scalar():
    assert assemble_row("r1", "problem_ko", ["번역"], ["ok"], is_list=False) == {
        "id": "r1", "problem_ko": "번역", "problem_ko_status": ["ok"], "mask_restore_ok": True,
    }
    bad = assemble_row("r2", "problem_ko", [None], ["fail:missing"], is_list=False)
    assert bad["problem_ko"] is None and bad["mask_restore_ok"] is False


def test_assemble_row_keeps_partial_list():
    """A failed row still reports the items that did translate; the row is dropped on the flag."""
    row = assemble_row("r1", "steps_en", ["## Step 1", "---", None],
                       ["ok", "verbatim", "fail:missing"], is_list=True)
    assert row["steps_en"] == ["## Step 1", "---", None]
    assert row["mask_restore_ok"] is False


# ------------------------------------------------------------------ file-level pass

def test_translate_file_writes_rows_and_resumes(tmp_path):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(inp, [
        {"id": "a", "problem_en": "Compute $x$."},
        {"id": "b", "problem_en": "Find $y$."},
    ])
    t = FakeTranslator(lambda s: s.replace("Compute", "계산").replace("Find", "구하라"))
    translate_file(str(inp), str(out), "problem_en", "problem_ko", "en", "ko", t)
    rows = load_jsonl(out)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0] == {"id": "a", "problem_ko": "계산 $x$.",
                       "problem_ko_status": ["ok"], "mask_restore_ok": True}

    # A second run has nothing to do: rows already in the out file are not re-translated.
    t2 = FakeTranslator(lambda s: "should not be called")
    translate_file(str(inp), str(out), "problem_en", "problem_ko", "en", "ko", t2)
    assert t2.seen == []
    assert load_jsonl(out) == rows


def test_translate_file_only_ids(tmp_path):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(inp, [
        {"id": "a", "problem_en": "Compute $x$."},
        {"id": "b", "problem_en": "Find $y$."},
    ])
    t = FakeTranslator(lambda s: s)
    translate_file(str(inp), str(out), "problem_en", "problem_ko", "en", "ko", t,
                   only_ids={"b"})
    assert [r["id"] for r in load_jsonl(out)] == ["b"]


# ------------------------------------------------------------------ sharding

def test_shard_path():
    from koprm.translate.translate import shard_path

    assert str(shard_path("data/trans/problems_ko.jsonl", 0, 1)) == "data/trans/problems_ko.jsonl"
    assert str(shard_path("data/trans/problems_ko.jsonl", 2, 4)) == \
        "data/trans/problems_ko.shard2of4.jsonl"


def test_translate_file_shards_by_row_index(tmp_path):
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, [{"id": str(i), "problem_en": f"Value $x_{i}$."} for i in range(5)])
    seen = {}
    for s in range(2):
        out = tmp_path / f"out{s}.jsonl"
        t = FakeTranslator(lambda x: x)
        translate_file(str(inp), str(out), "problem_en", "problem_ko", "en", "ko", t,
                       shard=s, num_shards=2)
        seen[s] = [r["id"] for r in load_jsonl(out)]
    assert seen == {0: ["0", "2", "4"], 1: ["1", "3"]}  # every k-th row, no overlap
