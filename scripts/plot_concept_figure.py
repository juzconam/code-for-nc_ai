"""Conceptual figure for the introduction (second attempt: tighter geometry, clear panel c)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "paper_figures" / "uncertainty_study" / "main_figures"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "font.size": 7.5,
})

INK = "#1a1a1a"
BLUE = "#4c78a8"
GREEN = "#54a24b"
RED = "#e45756"
GREY = "0.55"


def panel_label(axis, text, x=-0.10, y=1.02):
    axis.text(x, y, text, transform=axis.transAxes, fontsize=10.5,
              fontweight="bold", va="bottom", ha="left")


def main():
    figure = plt.figure(figsize=(7.4, 5.0))
    grid = figure.add_gridspec(
        2, 3, height_ratios=[0.52, 1.0], hspace=0.34, wspace=0.34,
        left=0.098, right=0.978, top=0.945, bottom=0.075,
    )

    # ------------------------------------------------------------ a: the template
    axis = figure.add_subplot(grid[0, :])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    panel_label(axis, "a", x=-0.02, y=0.92)

    width, height, bottom = 0.208, 0.42, 0.46
    gap = 0.043
    steps = (
        ("① 从先验里采样\nK 个可能的场", GREY),
        ("② 用物理法则\n修正这些场", GREY),
        ("③ 逐个场读出\n关心的量", GREY),
        ("④ 用候选的差别\n做误差棒", RED),
    )
    for index, (text, colour) in enumerate(steps):
        x = 0.004 + index * (width + gap)
        axis.add_patch(FancyBboxPatch(
            (x, bottom), width, height,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor=colour, edgecolor=INK, linewidth=0.7, alpha=0.15,
        ))
        axis.text(x + width / 2, bottom + height / 2, text, ha="center", va="center",
                  fontsize=7.2, color=INK, linespacing=1.6)
        if index < 3:
            axis.add_patch(FancyArrowPatch(
                (x + width + 0.004, bottom + height / 2),
                (x + width + gap - 0.004, bottom + height / 2),
                arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color=INK,
            ))

    left_end = 0.004 + 2 * (width + gap) + width
    axis.plot([0.004, left_end], [0.36, 0.36], color=INK, lw=0.8)
    axis.text((0.004 + left_end) / 2, 0.22, "大家检查的是这一段：重建得准不准",
              ha="center", va="center", fontsize=7.2, color=INK)
    right_start = 0.004 + 3 * (width + gap)
    axis.plot([right_start, 0.996], [0.36, 0.36], color=RED, lw=1.2)
    axis.text((right_start + 0.996) / 2, 0.22, "本文要检验的一步", ha="center",
              va="center", fontsize=7.2, color=RED)

    # ------------------------------------------------------------ b: the assumption
    axis = figure.add_subplot(grid[1, 0])
    panel_label(axis, "b")
    rng = np.random.default_rng(7)
    line = np.linspace(0.05, 1.0, 20)
    axis.plot([0, 1.05], [0, 1.05], ls="--", lw=0.8, color="0.6", zorder=1)
    axis.scatter(line, line * 0.96 + rng.normal(0, 0.025, line.size),
                 s=13, color=GREEN, edgecolor="k", linewidth=0.25, zorder=3)
    axis.scatter(0.86 + rng.normal(0, 0.028, 16), rng.uniform(0.05, 0.95, 16),
                 s=13, color=RED, edgecolor="k", linewidth=0.25, zorder=3)
    axis.set_xlim(0, 1.08)
    axis.set_ylim(0, 1.05)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("候选之间的差别", fontsize=7)
    axis.set_ylabel("真实误差", fontsize=7)
    axis.text(0.05, 0.95, "差别跟着误差走 → 有用", fontsize=6.6, color=GREEN, va="top")
    axis.text(0.60, 0.26, "差别不动 → 没用", fontsize=6.6, color=RED)

    # ------------------------------------------------------------ c: why it is silent
    axis = figure.add_subplot(grid[1, 1])
    panel_label(axis, "c")
    axis.set_xlim(0.05, 1.30)
    axis.set_ylim(-0.35, 1.85)
    axis.axis("off")

    prediction, truth = 0.42, 0.52

    axis.axvline(truth, color=INK, lw=0.8, ls=":", ymin=0.10, ymax=0.88)
    axis.axvline(prediction, color=INK, lw=0.8, ymin=0.10, ymax=0.88)

    rows = (
        (1.25, 0.14, BLUE, "常数区间", "半宽 0.14"),
        (0.55, 0.33, RED, "用分歧做的区间", "半宽 0.33"),
    )
    for y, half, colour, label, note in rows:
        axis.add_patch(Rectangle((prediction - half, y - 0.13), 2 * half, 0.26,
                                 facecolor=colour, edgecolor=INK, linewidth=0.7,
                                 alpha=0.30))
        axis.plot([prediction], [y], marker="o", ms=3.2, color=INK)
        axis.text(prediction - half - 0.03, y, label, ha="right", va="center",
                  fontsize=6.8, color=colour)
        axis.text(prediction + half + 0.03, y, note, ha="left", va="center",
                  fontsize=6.4, color=colour)

    axis.text(truth, -0.16, "真值", ha="center", fontsize=6.6)
    axis.text(0.60, 0.90, "两个都盖住真值\n→ 覆盖率相同，宽度差 2.5 倍",
              fontsize=6.8, color=INK, linespacing=1.6)

    # ------------------------------------------------------------ d: the criterion
    axis = figure.add_subplot(grid[1, 2])
    panel_label(axis, "d")
    axis.set_xlim(0.0, 1.88)
    axis.set_ylim(-0.72, 0.86)
    axis.axis("off")

    axis.add_patch(Rectangle((0.02, 0.46), 0.98, 0.12, facecolor=RED, alpha=0.12))
    axis.add_patch(Rectangle((1.00, 0.46), 0.86, 0.12, facecolor=GREEN, alpha=0.12))
    axis.plot([0.02, 1.86], [0.52, 0.52], color=INK, lw=0.9)

    marks = ((0.067, "0.067", RED, "最坏"),
             (0.42, "0.42", BLUE, "退化"),
             (1.00, "1.00", GREY, "打平"),
             (1.74, "1.74", GREEN, "理想"))
    for value, label, colour, note in marks:
        axis.plot([value, value], [0.45, 0.59], color=colour, lw=2.4, solid_capstyle="butt")
        axis.text(value, 0.69, label, ha="center", fontsize=7.6, color=colour,
                  fontweight="bold")
        axis.text(value, 0.32, note, ha="center", fontsize=6.4, color=colour)

    axis.text(0.02, 0.02, "R < 1\n不如常数宽度", fontsize=6.8, color=RED, va="top",
              linespacing=1.5)
    axis.text(1.86, 0.02, "R > 1\n分歧有用", fontsize=6.8, color=GREEN, ha="right",
              va="top", linespacing=1.5)

    axis.text(0.94, -0.42, "R = 用分歧的区间宽度 ÷ 常数区间宽度", ha="center",
              fontsize=7.2, color=INK)
    axis.text(0.94, -0.60, "（同一覆盖率下比较）", ha="center", fontsize=6.4, color="0.4")

    path = OUT / "fig0_concept.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
