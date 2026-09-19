"""§5 glue joins: splits -> translation input -> generator input, and the teacher input."""
from koprm.io import write_jsonl
from koprm.prep import prm800k_problems_in, problems_in, problems_ko, teacher_in


def _write(tmp_path, name, rows):
    p = tmp_path / name
    write_jsonl(p, rows)
    return str(p)


def test_problems_in_dedupes_by_problem_id(tmp_path):
    dev = _write(tmp_path, "dev.jsonl", [
        {"problem_id": "math/train/1", "source": "math", "problem_en": "A?", "answer": "1",
         "level": "Level 3"},
    ])
    pool = _write(tmp_path, "pool.jsonl", [
        {"problem_id": "math/train/1", "source": "math", "problem_en": "A?", "answer": "1"},
        {"problem_id": "gsm8k/train/0", "source": "gsm8k", "problem_en": "B?", "answer": "2"},
    ])
    rows = problems_in([dev, pool])
    assert [r["problem_id"] for r in rows] == ["math/train/1", "gsm8k/train/0"]
    assert rows[0] == {"id": "math/train/1", "problem_id": "math/train/1", "source": "math",
                       "problem_en": "A?", "answer": "1"}


def test_problems_ko_keeps_only_clean_translations():
    rows_in = [
        {"id": "p1", "problem_id": "p1", "source": "math", "problem_en": "A?", "answer": "1"},
        {"id": "p2", "problem_id": "p2", "source": "gsm8k", "problem_en": "B?", "answer": "2"},
        {"id": "p3", "problem_id": "p3", "source": "math", "problem_en": "C?", "answer": "3"},
    ]
    trans = [
        {"id": "p1", "problem_id": "p1", "problem_ko": "가?", "mask_restore_ok": True},
        {"id": "p2", "problem_id": "p2", "problem_ko": None, "mask_restore_ok": False},
        # p3 has no translation row at all
    ]
    out, dropped = problems_ko(rows_in, trans)
    assert dropped == 2
    assert out == [{"problem_id": "p1", "problem_ko": "가?", "answer": "1", "source": "math"}]


def test_prm800k_problems_in_is_unique_per_problem():
    rows = [
        {"id": "prm800k/1", "problem_id": "h1", "problem_en": "A?"},
        {"id": "prm800k/2", "problem_id": "h1", "problem_en": "A?"},
        {"id": "prm800k/3", "problem_id": "h2", "problem_en": "B?"},
    ]
    out = prm800k_problems_in(rows)
    assert out == [
        {"id": "h1", "problem_id": "h1", "problem_en": "A?"},
        {"id": "h2", "problem_id": "h2", "problem_en": "B?"},
    ]


def test_teacher_in_from_round_trip_translations():
    rows = [
        {"id": "s1", "problem_id": "h1", "problem_en": "A?", "steps_en": ["a", "b"]},
        {"id": "s2", "problem_id": "h1", "problem_en": "A?", "steps_en": ["c"]},
    ]
    steps_by_id = {
        "s1": {"id": "s1", "steps_en_rt": ["a2", "b2"], "mask_restore_ok": True},
        "s2": {"id": "s2", "steps_en_rt": None, "mask_restore_ok": False},
    }
    out, dropped = teacher_in(rows, steps_by_id, "steps_en_rt")
    assert dropped == 1
    assert out == [{"id": "s1", "problem_en": "A?", "steps_en": ["a2", "b2"]}]


def test_teacher_in_direct_variant_and_problem_lookup():
    rows = [{"id": "s1", "problem_id": "h1", "steps_en": ["a", "b"]}]
    problems = {"h1": {"problem_id": "h1", "problem_en": "A?"}}
    out, dropped = teacher_in(rows, None, "steps_en", problems)
    assert dropped == 0
    assert out == [{"id": "s1", "problem_en": "A?", "steps_en": ["a", "b"]}]

    # No problem text anywhere -> the row cannot be scored.
    out, dropped = teacher_in(rows, None, "steps_en", {})
    assert out == [] and dropped == 1
