"""§3.1 Formula masking for translation.

Math spans ($...$, $$...$$, \\[...\\], \\(...\\), \\boxed{...}) and standalone numbers are
replaced by placeholders before translation and restored after. The placeholder shape is
configurable because MT models sometimes rewrite unusual symbols; `restore()` verifies that
every placeholder appears exactly once and in order.

One exception to number masking: the step number in a step header ("## 단계 2:", "### Step 2:")
stays plain text. The translator rewrites such a header as a whole ("## 단계 2:" -> "## Step 2:")
and keeps the digit, while a placeholder there is one more thing it can drop or reorder.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PH_FMT = "⟦M{}⟧"
PH_RE = re.compile(r"⟦\s*M\s*(\d+)\s*⟧")

_NUMBER = r"(?<![A-Za-z\\{}_^])[-+]?\d+(?:[.,]\d+)*(?:%|°)?(?![A-Za-z\d{}])"

# "## 단계 2:" / "### Step 2:" (any of 1-6 hashes, "step" case-insensitive). Group `num` is the
# step number, which mask() leaves untouched.
_STEP_HEADER = re.compile(
    r"^\s*#{1,6}\s*(?:단계|Step)\s*(?P<num>\d+)\s*:", re.MULTILINE | re.IGNORECASE
)
# One pass over the text: placeholders are skipped whole so their digits are never re-masked.
_NUM_OR_PH = re.compile(f"(?P<ph>{PH_RE.pattern})|(?P<num>{_NUMBER})")


@dataclass
class Masked:
    text: str
    spans: list[str] = field(default_factory=list)


def _find_math_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("$$", i):
            j = text.find("$$", i + 2)
            if j < 0:
                break
            spans.append((i, j + 2))
            i = j + 2
        elif c == "$":
            j = text.find("$", i + 1)
            if j < 0:
                break
            spans.append((i, j + 1))
            i = j + 1
        elif text.startswith("\\[", i):
            j = text.find("\\]", i + 2)
            if j < 0:
                break
            spans.append((i, j + 2))
            i = j + 2
        elif text.startswith("\\(", i):
            j = text.find("\\)", i + 2)
            if j < 0:
                break
            spans.append((i, j + 2))
            i = j + 2
        elif text.startswith("\\boxed", i):
            k = i + len("\\boxed")
            while k < n and text[k] == " ":
                k += 1
            if k < n and text[k] == "{":
                depth = 0
                for j in range(k, n):
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                        if depth == 0:
                            spans.append((i, j + 1))
                            i = j + 1
                            break
                else:
                    break
            else:
                i = k
        else:
            i += 1
    return spans


def mask(text: str, numbers: bool = True) -> Masked:
    spans = _find_math_spans(text)
    out, pieces, last = [], [], 0
    for a, b in spans:
        out.append(text[last:a])
        pieces.append(text[a:b])
        out.append(PH_FMT.format(len(pieces)))
        last = b
    out.append(text[last:])
    masked = "".join(out)
    if numbers:
        # Mask standalone numbers in the non-placeholder text, except a step header's number.
        protected = [m.span("num") for m in _STEP_HEADER.finditer(masked)]

        def repl(m: re.Match) -> str:
            if m.group("ph") is not None:  # leave existing placeholders alone
                return m.group(0)
            if any(a <= m.start() < b for a, b in protected):
                return m.group(0)
            pieces.append(m.group(0))
            return PH_FMT.format(len(pieces))

        masked = _NUM_OR_PH.sub(repl, masked)
    # Renumber placeholders in order of appearance so a faithful translation is "ok".
    order = [int(m.group(1)) for m in PH_RE.finditer(masked)]
    remap = {old: new for new, old in enumerate(order, start=1)}
    masked = PH_RE.sub(lambda m: PH_FMT.format(remap[int(m.group(1))]), masked)
    pieces = [pieces[old - 1] for old in order]
    return Masked(masked, pieces)


def restore(translated: str, spans: list[str]) -> tuple[str | None, str]:
    """Return (restored_text, status). status in {ok, missing, duplicate, order, extra}."""
    found = [int(m.group(1)) for m in PH_RE.finditer(translated)]
    expected = list(range(1, len(spans) + 1))
    if len(found) != len(set(found)):
        return None, "duplicate"
    if set(found) - set(expected):
        return None, "extra"
    if set(expected) - set(found):
        return None, "missing"
    status = "ok" if found == expected else "order"
    text = PH_RE.sub(lambda m: spans[int(m.group(1)) - 1], translated)
    return text, status
