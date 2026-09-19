"""Outcome scoring (y) with math_verify, in parallel with per-item timeouts.

Follows komath (exp/analysis/core.py, src/sal/utils/math.py) but:
  * extracts the LAST brace-balanced \\boxed{...} (Korean completions have no reliable
    "final answer" sentence), and
  * runs in a process pool with a hard timeout per item, because sympy can hang.
"""
from __future__ import annotations

from concurrent.futures import TimeoutError as FutTimeout

from koprm.data.sources import last_boxed


def _verify_one(gold: str, completion: str) -> tuple[bool, str | None]:
    from math_verify import parse, verify

    pred = last_boxed(completion)
    if pred is None or not pred.strip():
        return False, None
    try:
        g = parse("\\boxed{" + gold + "}")
        p = parse("\\boxed{" + pred + "}")
        return bool(verify(g, p)), pred
    except Exception:  # noqa: BLE001 - math_verify raises anything
        return False, pred


def verify_one(gold: str, completion: str, timeout: float = 5.0) -> tuple[bool, str | None]:
    """Single verification with a timeout (uses SIGALRM; main thread only)."""
    import signal

    def handler(signum, frame):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return _verify_one(gold, completion)
    except TimeoutError:
        return False, last_boxed(completion)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def verify_many(
    golds: list[str], completions: list[str], workers: int = 16, timeout: float = 5.0
) -> list[tuple[bool, str | None]]:
    """Parallel verification; order preserved; timeouts/errors count as incorrect."""
    from pebble import ProcessPool

    results: list[tuple[bool, str | None] | None] = [None] * len(golds)
    with ProcessPool(max_workers=workers) as pool:
        futs = [pool.schedule(_verify_one, args=(g, c), timeout=timeout) for g, c in zip(golds, completions)]
        for i, fut in enumerate(futs):
            try:
                results[i] = fut.result()
            except (FutTimeout, Exception):  # noqa: BLE001 - worker may raise anything
                results[i] = (False, last_boxed(completions[i]))
    return results  # type: ignore[return-value]


def answers_equal(a: str, b: str, timeout: float = 5.0) -> bool:
    return verify_one(a, "\\boxed{" + b + "}", timeout=timeout)[0]
