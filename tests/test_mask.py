"""§3.1 masking: what gets a placeholder, what stays plain text, and round-tripping."""
from koprm.translate.mask import PH_RE, mask, restore


def _placeholders(text: str) -> list[int]:
    return [int(m.group(1)) for m in PH_RE.finditer(text)]


def test_step_header_number_stays_plain():
    """"## 단계 2:" keeps its digit: the translator rewrites the header as "## Step 2:"."""
    m = mask("## 단계 2: 정리한다")
    assert m.text == "## 단계 2: 정리한다"
    assert m.spans == []


def test_step_header_variants():
    for text in ["# 단계 1: 설명", "###### 단계 12: 설명", "## Step 3: explain",
                 "## step 3: explain", "## STEP 3: explain", "##단계4:설명",
                 "   ## 단계 5 : 설명"]:
        m = mask(text)
        assert m.text == text, text
        assert m.spans == [], text


def test_header_without_hash_is_masked():
    """The pattern requires markdown hashes; a bare "단계 2:" line is ordinary prose."""
    m = mask("단계 2: 숫자 7")
    assert m.text == "단계 ⟦M1⟧: 숫자 ⟦M2⟧"
    assert m.spans == ["2", "7"]


def test_body_numbers_and_math_still_masked():
    step = "## 단계 2: 3을 더한다\n$x = 2 + 3$이므로 답은 5이다."
    m = mask(step)
    assert m.text == "## 단계 2: ⟦M1⟧을 더한다\n⟦M2⟧이므로 답은 ⟦M3⟧이다."
    assert m.spans == ["3", "$x = 2 + 3$", "5"]
    # The header number is not among the spans, and appears once as plain text.
    assert m.text.startswith("## 단계 2: ")


def test_number_in_header_body_after_colon_is_masked():
    m = mask("## 단계 2: 위의 2를 쓴다")
    assert m.text == "## 단계 2: 위의 ⟦M1⟧를 쓴다"
    assert m.spans == ["2"]


def test_placeholders_numbered_in_order_of_appearance():
    step = "## 단계 3: 값\n먼저 7, 그다음 $a+1$, 마지막으로 $$b$$ 그리고 9."
    m = mask(step)
    assert _placeholders(m.text) == [1, 2, 3, 4]
    assert m.spans == ["7", "$a+1$", "$$b$$", "9"]


def test_nested_boxed_span():
    step = "## 단계 4: 답은 $\\boxed{\\frac{1}{2}}$이다."
    m = mask(step)
    assert m.text == "## 단계 4: 답은 ⟦M1⟧이다."
    assert m.spans == ["$\\boxed{\\frac{1}{2}}$"]
    back, st = restore(m.text, m.spans)
    assert (back, st) == (step, "ok")


def test_bare_nested_boxed_span():
    step = "## 단계 5: \\boxed{\\frac{1}{2}} 가 최종 답이다."
    m = mask(step)
    assert m.spans == ["\\boxed{\\frac{1}{2}}"]
    assert "\\frac" not in m.text
    assert restore(m.text, m.spans) == (step, "ok")


def test_restore_round_trip_ok():
    steps = [
        "## 단계 2: 3을 더한다\n$x = 2 + 3$이므로 답은 5이다.",
        "## Step 3: add 4\nSo $y = 4$.",
        "## 단계 1: 30%를 구한다\n$$\\frac{3}{10}$$",
    ]
    for step in steps:
        m = mask(step)
        back, st = restore(m.text, m.spans)
        assert st == "ok", step
        assert back == step, step


def test_restore_after_faithful_translation():
    m = mask("## 단계 2: 3을 더한다\n$x = 2 + 3$이므로 답은 5이다.")
    translated = "## Step 2: Add ⟦M1⟧\nSince ⟦M2⟧, the answer is ⟦M3⟧."
    back, st = restore(translated, m.spans)
    assert st == "ok"
    assert back == "## Step 2: Add 3\nSince $x = 2 + 3$, the answer is 5."


def test_step_with_no_maskable_content():
    for step in ["## 단계 1: 정리한다", "따라서 참이다.", "---"]:
        m = mask(step)
        assert m.text == step
        assert m.spans == []
        assert restore(m.text, m.spans) == (step, "ok")


def test_restore_failure_statuses():
    m = mask("## 단계 2: $x$와 $y$를 더하면 7이다.")
    assert len(m.spans) == 3
    assert restore("⟦M1⟧ and ⟦M2⟧", m.spans) == (None, "missing")
    assert restore("⟦M1⟧ ⟦M1⟧ ⟦M2⟧ ⟦M3⟧", m.spans) == (None, "duplicate")
    assert restore("⟦M1⟧ ⟦M2⟧ ⟦M3⟧ ⟦M9⟧", m.spans) == (None, "extra")
    assert restore("⟦M2⟧ ⟦M1⟧ ⟦M3⟧", m.spans)[1] == "order"


def test_mask_numbers_off_keeps_all_digits():
    step = "## 단계 2: 3을 더한다\n$x = 3$"
    m = mask(step, numbers=False)
    assert m.spans == ["$x = 3$"]
    assert m.text == "## 단계 2: 3을 더한다\n⟦M1⟧"


# --- content-based "$...$" judgment (§3) ------------------------------------------------


def test_currency_pair_is_not_a_formula():
    """"$5 ... $8": both dollars open a price; the numbers are still masked one by one."""
    m = mask("She paid $5 for apples and $8 for pears.")
    assert m.spans == ["5", "8"]
    assert m.text == "She paid $⟦M1⟧ for apples and $⟦M2⟧ for pears."


def test_single_currency_without_closing_dollar():
    m = mask("It costs $5.")
    assert m.spans == ["5"]
    assert m.text == "It costs $⟦M1⟧."


def test_unmatched_dollar_does_not_swallow_the_rest():
    """A lone "$" is a plain character: what follows it is still scanned normally."""
    m = mask("It costs $5, so \\boxed{5} is the answer.")
    assert m.spans == ["5", "\\boxed{5}"]
    assert m.text == "It costs $⟦M1⟧, so ⟦M2⟧ is the answer."


def test_short_formula_still_a_formula():
    m = mask("$x = 2 + 3$")
    assert m.spans == ["$x = 2 + 3$"]
    assert m.text == "⟦M1⟧"
    assert restore(m.text, m.spans) == ("$x = 2 + 3$", "ok")


def test_single_letters_are_not_prose():
    m = mask("$a b c$")
    assert m.spans == ["$a b c$"]


def test_text_command_is_still_a_formula():
    for step in ["$\\text{inches}$", "$\\text{인치}$", "$5 \\text{ inches per foot}$"]:
        m = mask(step)
        assert m.spans == [step], step
        assert m.text == "⟦M1⟧", step


def test_three_prose_words_are_not_a_formula():
    m = mask("$x is the number of apples$ here")
    assert all(not s.startswith("$") for s in m.spans)
    assert "$" in m.text


def test_newline_inside_candidate_is_not_a_formula():
    m = mask("The price is $5\nand the tax is $1.")
    assert all(not s.startswith("$") for s in m.spans)
    assert m.spans == ["5", "1"]


def test_hangul_inside_candidate_is_not_a_formula():
    m = mask("$x는 5$")
    assert m.spans == ["5"]
    assert m.text == "$x는 ⟦M1⟧$"


def test_asy_diagram_is_one_span():
    """An [asy]...[/asy] block is one opaque placeholder; math outside it is masked normally."""
    step = ("Find $x$ in the figure.\n"
            "[asy]\ndraw((0,0)--(1,2));\nlabel(\"$5$\", (1,1));\n[/ASY]")
    m = mask(step)
    assert len(m.spans) == 2
    assert m.spans[0] == "$x$"
    assert m.spans[1].startswith("[asy]") and m.spans[1].endswith("[/ASY]")
    assert m.text == "Find ⟦M1⟧ in the figure.\n⟦M2⟧"
    assert restore(m.text, m.spans) == (step, "ok")


def test_unclosed_asy_tag_is_ordinary_text():
    m = mask("[asy] draw(3);")
    assert m.spans == ["3"]
    assert m.text == "[asy] draw(⟦M1⟧);"
