"""Small jsonl helpers with resumable, sharded writes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict], append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def done_ids(path: str | Path, key: str = "id") -> set:
    path = Path(path)
    if not path.exists():
        return set()
    return {r[key] for r in read_jsonl(path)}
