"""§2.3–2.4 Reference distribution and y-conditional kernel.

All scores are teacher log-odds z_t = l1 - l0 (float32). Labels: 1, 0, or None (mask).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CEIL_Q = 0.75      # per-solution ceiling b = 75th percentile of z (top 25% point)
BMIN_Q = 0.05      # b_min = 5th percentile of ceilings over y=1 solutions
CAND_P = 0.05      # candidate threshold on tail prob
HI = 0.9           # r_lo >= HI -> 1
LO = 0.1           # r_hi <= LO -> 0


def ceiling(z: np.ndarray) -> float:
    return float(np.quantile(np.asarray(z, dtype=np.float64), CEIL_Q))


@dataclass
class RefDist:
    """Per-generator reference: b_min and the empirical drop distribution D_g."""
    b_min: float
    drops: np.ndarray  # sorted ascending

    @classmethod
    def fit(cls, correct_solutions_z: list[np.ndarray]) -> RefDist:
        ceils = np.array([ceiling(z) for z in correct_solutions_z])
        b_min = float(np.quantile(ceils, BMIN_Q))
        drops = []
        for z in correct_solutions_z:
            b = max(ceiling(z), b_min)
            drops.append(b - np.asarray(z, dtype=np.float64))
        return cls(b_min=b_min, drops=np.sort(np.concatenate(drops)))

    def tail_prob(self, d: np.ndarray) -> np.ndarray:
        """pi = P_D(drop >= d), with +1 smoothing so pi is never exactly 0."""
        n = len(self.drops)
        idx = np.searchsorted(self.drops, d, side="left")  # count of drops < d
        return (n - idx + 1) / (n + 1)

    def to_dict(self) -> dict:
        return {"b_min": self.b_min, "drops": self.drops.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> RefDist:
        return cls(b_min=d["b_min"], drops=np.asarray(d["drops"], dtype=np.float64))


@dataclass
class KernelOut:
    labels: list[int | None]
    ceiling: float
    candidates: list[int]
    tail_probs: list[float]
    r_lo: list[float]
    r_hi: list[float]


def kernel(z: np.ndarray, y: int, ref: RefDist, hi: float = HI, lo: float = LO) -> KernelOut:
    """Hard labels for one solution.

    y=1: all steps 1. y=0: candidate steps from the tail probability, first-error weights,
    r_lo/r_hi bounds, then thresholds.
    """
    z = np.asarray(z, dtype=np.float64)
    T = len(z)
    b = ceiling(z)
    d = max(b, ref.b_min) - z
    pi = ref.tail_prob(d)
    if y == 1:
        return KernelOut([1] * T, b, [], pi.tolist(), [1.0] * T, [1.0] * T)
    C = [t for t in range(T) if pi[t] < CAND_P]
    p_prime = {t: pi[t] / CAND_P for t in C}
    w = {}
    running = 1.0
    for k in C:
        w[k] = (1.0 - p_prime[k]) * running
        running *= p_prime[k]
    S = running  # first error somewhere outside C
    r_lo, r_hi, labels = [], [], []
    for t in range(T):
        lo_t = sum(w[k] for k in C if k > t)
        hi_t = lo_t + S
        r_lo.append(lo_t)
        r_hi.append(hi_t)
        if t == T - 1:
            labels.append(0)
        elif lo_t >= hi:
            labels.append(1)
        elif hi_t <= lo:
            labels.append(0)
        else:
            labels.append(None)
    return KernelOut(labels, b, C, pi.tolist(), r_lo, r_hi)
