"""Generate the README performance charts from the measured kernel timings.

Usage: python scripts/plot_results.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

# Measured on Apple M1 GPU, 2048x2048 float image, 3x3 Gaussian blur.
WORK_GROUPS = ["8x8", "16x8", "16x16"]
NAIVE_MS = [0.035, 0.035, 0.035]
OPTIMIZED_MS = [0.077, 0.048, 0.060]

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

OUT = Path(__file__).resolve().parent.parent / "images"


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=TEXT, pad=14, fontweight="bold")
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.set_xlabel("Work-group size", color=TEXT_2)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_2)


def bar_labels(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=TEXT)


def execution_time():
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    x = range(len(WORK_GROUPS))
    w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], NAIVE_MS, w, color=BLUE, label="Naive (global memory)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], OPTIMIZED_MS, w, color=ORANGE, label="Optimized (local memory)")
    bar_labels(ax, b1, "{:.3f}")
    bar_labels(ax, b2, "{:.3f}")
    ax.set_xticks(list(x), WORK_GROUPS)
    ax.set_ylim(0, max(OPTIMIZED_MS) * 1.2)
    style(ax, "Kernel execution time (lower is better)", "Time (ms)")
    ax.legend(frameon=False, labelcolor=TEXT, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "execution_time.png", dpi=160, facecolor=SURFACE)


def speedup():
    s = [n / o for n, o in zip(NAIVE_MS, OPTIMIZED_MS)]
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    bars = ax.bar(WORK_GROUPS, s, 0.5, color=BLUE)
    bar_labels(ax, bars, "{:.2f}x")
    ax.axhline(1.0, color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(0.99, 1.02, "1.0x = break-even with naive", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9, color=TEXT_2)
    ax.set_ylim(0, 1.2)
    style(ax, "Speedup of local-memory kernel over naive", "Speedup (naive time / optimized time)")
    fig.tight_layout()
    fig.savefig(OUT / "speedup.png", dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    execution_time()
    speedup()
    print(f"Charts written to {OUT}")
