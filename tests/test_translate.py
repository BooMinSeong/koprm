"""§3 translation pipeline with a fake translator: prompts, passthrough, cuts, restore."""
from koprm.translate.translate import (
    OK_STATUSES,
    cut_continuation,
    needs_translation,
    run,
    script_ok,
    strip_extra_tail,
    system_prompt,
    translate_rows,
    translate_units,
    unwrap,
)


class FakeTranslator:
    """Records the masked texts and system prompts it is given; replays canned outputs."""

    def __init__(self, outputs: list[str] | None = None):
        self.outputs = outputs
        self.calls: list[list[str]] = []
        self.seen: list[str] = []
        self.systems: list[str] = []

    def translate(self, texts: list[str], systems: list[str]) -> list[str]:
        assert len(texts) == len(systems)
        self.calls.append(list(texts))
        self.seen.extend(texts)
        self.systems.extend(systems)
        if self.outputs is None:
            return list(texts)
        out = self.outputs[: len(texts)]
        self.outputs = self.outputs[len(texts) :]
        return out


# --- the per-unit system prompt ---------------------------------------------------------


def test_system_prompt_names_both_languages():
    s = system_prompt("en", "ko")
    assert "English" in s and "Korean" in s and "⟦M1⟧" in s


def test_system_prompt_drops_rules_the_unit_cannot_need():
    """A unit with no placeholder and no header must not be told about either (it invents them)."""
    plain = system_prompt("en", "ko", placeholders=False, header=False)
    assert "⟦M1⟧" not in plain
    assert "step header" not in plain
    assert "except inside the placeholders" not in plain

    with_ph = system_prompt("en", "ko", placeholders=True, header=False)
    assert "⟦M1⟧" in with_ph and "step header" not in with_ph

    with_header = system_prompt("en", "ko", placeholders=False, header=True)
    assert "⟦M1⟧" not in with_header and "step header" in with_header


def test_every_variant_forbids_continuing_the_message():
    for ph in (True, False):
        for hd in (True, False):
            s = system_prompt("ko", "en", placeholders=ph, header=hd)
            assert "Translate the whole message and nothing more." in s
            assert "Do not continue it, add steps, add placeholders, or add notes." in s
            # rules are numbered from 1 with no gaps
            nums = [int(line.split(".")[0]) for line in s.splitlines() if line[:1].isdigit()]
            assert nums == list(range(1, len(nums) + 1))


def test_prompt_variant_is_chosen_per_unit():
    tr = FakeTranslator()
    translate_units(["How many ways are there?", "## Step 1: add $x$"], tr, "en", "ko")
    plain, rich = tr.systems
    assert "⟦M1⟧" not in plain and "step header" not in plain
    assert "⟦M1⟧" in rich and "step header" in rich


# --- the pipeline -----------------------------------------------------------------------


def test_needs_translation():
    assert needs_translation("Add ⟦M1⟧ to ⟦M2⟧.")
    assert not needs_translation("⟦M1⟧")
    assert not needs_translation("⟦M1⟧ = ⟦M2⟧ + ⟦M3⟧.")
    assert not needs_translation("  ")


def test_unwrap_quotes_and_fences():
    assert unwrap('  "Add ⟦M1⟧."  ') == "Add ⟦M1⟧."
    assert unwrap("```\nAdd ⟦M1⟧.\n```") == "Add ⟦M1⟧."
    assert unwrap("```markdown\n## Step 1: go\n```") == "## Step 1: go"
    assert unwrap("“Add ⟦M1⟧.”") == "Add ⟦M1⟧."
    # Only one layer, and never a quote that is not matched on both sides.
    assert unwrap('"a" and "b"') == '"a" and "b"'


def test_script_ok():
    assert script_ok("Add 3 apples", "en")
    assert not script_ok("Add 3 사과", "en")
    assert not script_ok("Add 3 漢", "en")
    assert script_ok("사과 3개", "ko")
    assert not script_ok("사과 漢", "ko")
    # Korean inside a placeholder (a masked \text{인치}) is fine: it is not translator output.
    assert script_ok("⟦M1⟧ inches", "en")


def test_cut_continuation():
    assert cut_continuation("번역문입니다.\n## 단계 2: ⟦M2⟧을 계산합니다") == "번역문입니다."
    assert cut_continuation("번역문입니다.\n### Step 2: more") == "번역문입니다."
    assert cut_continuation("번역문입니다.") == "번역문입니다."
    # "#" inside a line is not a header.
    assert cut_continuation("번호 #3 입니다.") == "번호 #3 입니다."


def test_strip_extra_tail():
    assert strip_extra_tail("몇 가지일까요? ⟦M1⟧", 0) == "몇 가지일까요?"
    assert strip_extra_tail("답: ⟦M1⟧, ⟦M2⟧, ⟦M3⟧, ⟦M4⟧.", 0) == "답:"
    # A placeholder the unit really has is never touched.
    assert strip_extra_tail("답은 ⟦M3⟧.", 3) == "답은 ⟦M3⟧."
    assert strip_extra_tail("답은 ⟦M2⟧이다.", 1) == "답은 ⟦M2⟧이다."  # not at the very end
    assert strip_extra_tail("답은 ⟦M2⟧ ⟦M3⟧", 2) == "답은 ⟦M2⟧"


def test_formula_only_unit_is_not_sent_to_the_model():
    tr = FakeTranslator()
    texts, statuses, _ = translate_units(["$x = 2$", "Add 3 apples."], tr, "en", "ko")
    assert statuses == ["passthrough", "ok"]
    assert texts[0] == "$x = 2$"  # kept verbatim
    assert tr.seen == ["Add ⟦M1⟧ apples."]  # only the second unit reached the translator


def test_identity_translation_restores_the_original():
    texts, statuses, _ = translate_units(
        ["$x = 2 + 3$이므로 답은 5이다."], FakeTranslator(), "ko", "ko"
    )
    assert statuses == ["ok"]
    assert texts == ["$x = 2 + 3$이므로 답은 5이다."]


def test_missing_placeholder_fails():
    tr = FakeTranslator(["The answer is here."])
    texts, statuses, raws = translate_units(["답은 $x$이다."], tr, "ko", "en")
    assert statuses == ["missing"] and texts == [None]
    assert raws == ["The answer is here."]  # the evidence survives the failure


def test_hangul_left_in_english_output_fails_with_script():
    tr = FakeTranslator(["답은 ⟦M1⟧이다."])
    texts, statuses, _ = translate_units(["답은 $x$이다."], tr, "ko", "en")
    assert statuses == ["script"] and texts == [None]


def test_quoted_output_is_unwrapped_before_restore():
    tr = FakeTranslator(['"So ⟦M1⟧ is the answer."'])
    texts, statuses, _ = translate_units(["따라서 $x$가 답이다."], tr, "ko", "en")
    assert statuses == ["ok"]
    assert texts == ["So $x$ is the answer."]


def test_reordered_placeholders_count_as_ok():
    tr = FakeTranslator(["⟦M2⟧ and ⟦M1⟧"])
    texts, statuses, _ = translate_units(["$a$와 $b$"], tr, "ko", "en")
    assert statuses == ["order"]
    assert texts == ["$b$ and $a$"]
    assert all(s in OK_STATUSES for s in statuses)


def test_invented_continuation_is_cut_instead_of_failing():
    """The pilot's "extra" failures: the model writes its own next step after the translation."""
    tr = FakeTranslator(["⟦M1⟧와 ⟦M2⟧를 더합니다.\n## 단계 2: ⟦M3⟧을 계산합니다"])
    texts, statuses, raws = translate_units(["Add $a$ and $b$."], tr, "en", "ko")
    assert statuses == ["ok"]
    assert texts == ["$a$와 $b$를 더합니다."]
    assert raws[0].endswith("## 단계 2: ⟦M3⟧을 계산합니다")  # raw keeps what was cut


def test_invented_trailing_placeholders_are_cut():
    tr = FakeTranslator(["몇 가지일까요? ⟦M1⟧, ⟦M2⟧."])
    texts, statuses, _ = translate_units(["How many ways are there?"], tr, "en", "ko")
    assert statuses == ["ok"]
    assert texts == ["몇 가지일까요?"]


def test_header_unit_keeps_its_own_header():
    tr = FakeTranslator(["## 단계 1: ⟦M1⟧를 더한다\n따라서 끝."])
    texts, statuses, _ = translate_units(["## Step 1: add $x$\nSo done."], tr, "en", "ko")
    assert statuses == ["ok"]
    assert texts == ["## 단계 1: $x$를 더한다\n따라서 끝."]


def test_translate_rows_string_field():
    rows = [{"id": "p1", "problem_id": "p1", "problem_en": "Add 2 and 3."}]
    out, stats = translate_rows(rows, FakeTranslator(), "problem_en", "problem_ko", "en", "ko")
    assert out[0]["id"] == "p1" and out[0]["problem_id"] == "p1"
    assert out[0]["problem_ko"] == "Add 2 and 3."
    assert out[0]["mask_restore_ok"] is True
    assert out[0]["restore_status"] == "ok"
    assert out[0]["translator"]
    assert "raw_failed" not in out[0]  # only failures carry the raw output
    assert stats["rows"] == 1 and stats["ok"] == 1


def test_raw_output_of_a_failed_row_is_kept_before_the_cuts():
    tr = FakeTranslator(["```\n답은 ⟦M1⟧ ⟦M1⟧이다.\n## 단계 2: 계속\n```"])
    rows = [{"id": "p1", "problem_en": "The answer is $x$."}]
    out, _ = translate_rows(rows, tr, "problem_en", "problem_ko", "en", "ko")
    assert out[0]["problem_ko"] is None and out[0]["mask_restore_ok"] is False
    assert out[0]["restore_status"] == "duplicate"  # the cut does not repair a duplicate
    # unwrapped (no fence), but still showing the continuation that was cut before restore
    assert out[0]["raw_failed"] == "답은 ⟦M1⟧ ⟦M1⟧이다.\n## 단계 2: 계속"


def test_translate_rows_list_field_one_bad_step_fails_the_row():
    rows = [
        {
            "id": "s1",
            "problem_id": "p1",
            "generator": "exaone-1.2b",
            "outcome": 0,
            "steps": ["$x = 1$", "답은 $x$이다.", "3을 더한다"],
        }
    ]
    tr = FakeTranslator(["The answer is here.", "Add ⟦M1⟧"])
    out, stats = translate_rows(rows, tr, "steps", "steps_en", "ko", "en")
    r = out[0]
    assert r["restore_status"] == ["passthrough", "missing", "ok"]
    assert r["mask_restore_ok"] is False
    assert r["steps_en"] is None
    assert r["generator"] == "exaone-1.2b" and r["outcome"] == 0
    assert r["raw_failed"] == [None, "The answer is here.", None]  # aligned with the steps
    assert stats["steps_passthrough"] == 1 and stats["steps_translated"] == 2
    assert stats["failed"] == 1 and stats["status_missing"] == 1


def test_translate_rows_list_field_all_good():
    rows = [{"id": "s1", "steps": ["$x = 1$", "Add 3 apples."]}]
    out, _ = translate_rows(rows, FakeTranslator(), "steps", "steps_ko", "en", "ko")
    assert out[0]["steps_ko"] == ["$x = 1$", "Add 3 apples."]
    assert out[0]["mask_restore_ok"] is True
    assert "raw_failed" not in out[0]


def test_empty_field_is_a_failed_row():
    out, stats = translate_rows([{"id": "s1", "steps": []}], FakeTranslator(), "steps",
                                "steps_en", "ko", "en")
    assert out[0]["steps_en"] is None
    assert out[0]["restore_status"] == "empty"
    assert out[0]["mask_restore_ok"] is False
    assert out[0]["raw_failed"] is None  # nothing was ever sent to the model
    assert stats["failed"] == 1


def test_run_is_resumable(tmp_path):
    rows = [{"id": f"p{i}", "problem_en": f"Add {i} apples."} for i in range(4)]
    out = tmp_path / "ko.jsonl"
    tr = FakeTranslator()
    run(rows, tr, out, "problem_en", "problem_ko", "en", "ko", batch=2)
    first = out.read_text(encoding="utf-8").splitlines()
    assert len(first) == 4
    assert len(tr.calls) == 2  # one vLLM call per batch of rows

    tr2 = FakeTranslator()
    stats = run(rows + [{"id": "p4", "problem_en": "Add 4 apples."}], tr2, out,
                "problem_en", "problem_ko", "en", "ko", batch=2)
    assert stats["rows"] == 1  # the four done ids are skipped
    assert out.read_text(encoding="utf-8").splitlines()[:4] == first
