#!/usr/bin/env python3
"""Build the three report figures into data/reports/figs/.

Dependency-light (json + pathlib + matplotlib) and deterministic: every number
is read out of the eval JSONs, nothing is hardcoded.

Run:  .venv/bin/python scripts/report_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
BIG = ROOT / "data/eval/math500_big"
SEVEN = ROOT / "data/eval/math500_7b"
AGG = ROOT / "data/eval/agg"
PB = ROOT / "data/shift/pb"
OUT = ROOT / "data/reports/figs"

NS = [1, 2, 4, 8, 16, 32, 64]
SPLITS = ["gsm8k", "math", "olympiadbench", "omnimath"]
SPLIT_LABELS = ["GSM8K", "MATH", "OlympiadBench", "OmniMATH"]

FIGSIZE = (7.0, 4.2)
DPI = 160

# One colour per scorer family, shared across all three figures.
C_SOFT = "#1f77b4"
C_OUTCOME = "#2ca02c"
C_SOFTY = "#9467bd"
C_HARD = "#d62728"
C_EXISTING = "#7f7f7f"
C_QWEN8B = "#ff7f0e"      # Qwen3-8B student, 24k soft
C_QWEN8B_48 = "#a0522d"   # Qwen3-8B student, 48k soft
C_PRM7B_INIT = "#17becf"
C_PRM72B = "#4d4d4d"

# Values we print at the end so the caller can cross-check them by hand.
REPORTED: dict[str, dict[str, float]] = {}


# --------------------------------------------------------------------------- io


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def naive_at(path: Path, n: int) -> float:
    """naive@n under the `last` aggregation."""
    return float(load(path)["last"]["metrics"][str(n)]["naive"])


def curve(path: Path, key: str = "naive") -> list[float]:
    metrics = load(path)["last"]["metrics"]
    return [float(metrics[str(n)][key]) for n in NS]


def setup_fonts() -> bool:
    """Use a Korean-capable font when one exists; otherwise stay in English."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("NanumGothic", "Noto Sans CJK KR", "NanumBarunGothic", "Malgun Gothic"):
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def finish(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------------------ fig 1


def fig1_scaling() -> Path:
    sizes = [12, 24, 48]
    suffix = "_ep3_EXAONE-4.0-1.2B.json"
    series = [
        ("soft (1.2B)", "_soft", C_SOFT, "o", "-"),
        ("outcome-only (1.2B)", "_outcome", C_OUTCOME, "s", "-"),
        ("soft+y (1.2B)", "_soft_y", C_SOFTY, "^", "-"),
        ("hard kernel (1.2B)", "", C_HARD, "v", "-"),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    values: dict[str, float] = {}

    for label, tag, color, marker, ls in series:
        ys = []
        for size in sizes:
            path = BIG / f"B_{size}k{tag}{suffix}"
            y = naive_at(path, 64)
            ys.append(y)
            values[f"B_{size}k{tag}"] = y
        ax.plot(sizes, ys, marker=marker, linestyle=ls, color=color, label=label,
                linewidth=1.8, markersize=6)

    existing = naive_at(AGG / "existing.json", 64)
    values["existing"] = existing
    ax.axhline(existing, color=C_EXISTING, linestyle="--", linewidth=1.5,
               label=f"existing EN PRM-7B ({existing:.3f})")

    q24 = naive_at(SEVEN / "qwen3-8b_B_24k_soft_ep3_EXAONE-4.0-1.2B.json", 64)
    q48 = naive_at(SEVEN / "qwen3-8b_B_48k_soft_ep3_EXAONE-4.0-1.2B.json", 64)
    values["qwen3-8b_B_24k_soft"] = q24
    values["qwen3-8b_B_48k_soft"] = q48
    ax.plot([24, 48], [q24, q48], linestyle=":", color=C_QWEN8B, linewidth=1.5,
            marker="*", markersize=15, label="Qwen3-8B soft")

    p24 = naive_at(SEVEN / "prm7b_B_24k_soft_ep3_EXAONE-4.0-1.2B.json", 64)
    values["prm7b_B_24k_soft"] = p24
    ax.plot([24], [p24], linestyle="none", color=C_PRM7B_INIT, marker="D",
            markersize=8, label="PRM-7B-init soft")

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}k" for s in sizes])
    ax.set_xlim(10.5, 56)
    ax.set_xlabel("training solutions")
    ax.set_ylabel("naive@64 accuracy")
    ax.set_title("Label format x data size (1.2B) and bigger backbones\n"
                 "KO MATH500, EXAONE-4.0-1.2B generator, last aggregation",
                 fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              frameon=False, borderaxespad=0)

    REPORTED["fig1"] = values
    return finish(fig, "fig1_scaling.png")


# ------------------------------------------------------------------------ fig 2


def fig2_bon_curve() -> Path:
    series = [
        ("existing EN PRM-7B", AGG / "existing.json", C_EXISTING, "o", "-"),
        ("hard kernel 1.2B (12k)", BIG / "B_12k_ep3_EXAONE-4.0-1.2B.json", C_HARD, "v", "-"),
        ("soft 1.2B (48k)", BIG / "B_48k_soft_ep3_EXAONE-4.0-1.2B.json", C_SOFT, "s", "-"),
        ("outcome-only 1.2B (48k)", BIG / "B_48k_outcome_ep3_EXAONE-4.0-1.2B.json", C_OUTCOME, "^", "-"),
        ("Qwen3-8B soft (24k)", SEVEN / "qwen3-8b_B_24k_soft_ep3_EXAONE-4.0-1.2B.json", C_QWEN8B, "*", "-"),
        ("Qwen3-8B soft (48k)", SEVEN / "qwen3-8b_B_48k_soft_ep3_EXAONE-4.0-1.2B.json", C_QWEN8B_48, "D", "-"),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, path, color, marker, ls in series:
        ax.plot(NS, curve(path, "naive"), marker=marker, linestyle=ls, color=color,
                label=label, linewidth=1.7, markersize=6)

    ax.plot(NS, curve(AGG / "existing.json", "maj"), linestyle="--", color="#8c8c8c",
            linewidth=1.5, label="majority vote")
    ax.plot(NS, curve(AGG / "existing.json", "pass"), linestyle=":", color="#bdbdbd",
            linewidth=1.8, label="pass@n (oracle)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel("n (best-of-n)")
    ax.set_ylabel("naive@n accuracy")
    ax.set_title("Best-of-n selection on KO MATH500\n"
                 "EXAONE-4.0-1.2B generator, last aggregation", fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              frameon=False, borderaxespad=0)
    return finish(fig, "fig2_bon_curve.png")


# ------------------------------------------------------------------------ fig 3


def fig3_processbench() -> Path:
    scorers = [
        ("PRM-72B direct", "prm72b", C_PRM72B),
        ("PRM-7B direct (existing)", "prm7b", C_EXISTING),
        ("Qwen3-8B student (48k soft)", "qwen3-8b_B_48k_soft", C_QWEN8B_48),
        ("Qwen3-8B student (24k soft)", "qwen3-8b_B_24k_soft", C_QWEN8B),
        ("1.2B student (48k soft)", "B_48k_soft", C_SOFT),
    ]

    data = {}
    for _, key, _c in scorers:
        for lang in ("ko", "en"):
            data[key, lang] = load(PB / f"{key}_{lang}.json")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    width = 0.15
    xs = list(range(len(SPLITS)))

    for ax, lang, panel in zip(axes, ("ko", "en"), ("Korean", "English")):
        for i, (label, key, color) in enumerate(scorers):
            d = data[key, lang]
            ys = [float(d["by_split"][s]["f1"]) for s in SPLITS]
            offs = [x + (i - (len(scorers) - 1) / 2) * width for x in xs]
            ax.bar(offs, ys, width=width, color=color, label=label if lang == "ko" else None)
        ax.set_xticks(xs)
        ax.set_xticklabels(SPLIT_LABELS, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{panel} ProcessBench", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("first-error F1")

    # Overall F1 (Korean panel) goes into the legend labels.
    handles, _ = axes[0].get_legend_handles_labels()
    labels = [
        f"{label} (KO F1 {float(data[key, 'ko']['overall']['f1']):.3f}"
        f" / EN F1 {float(data[key, 'en']['overall']['f1']):.3f})"
        for label, key, _c in scorers
    ]
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("ProcessBench first-error F1 by split", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return finish(fig, "fig3_processbench.png")


def main() -> None:
    korean = setup_fonts()
    print(f"Korean-capable font: {'yes' if korean else 'no (figures stay in English)'}")
    for path in (fig1_scaling(), fig2_bon_curve(), fig3_processbench()):
        print(f"wrote {path}")
    print("\nfig1 plotted values (naive@64, last agg, EXAONE generator):")
    for name, value in REPORTED["fig1"].items():
        print(f"  {name:28s} {value:.3f}")


if __name__ == "__main__":
    main()
