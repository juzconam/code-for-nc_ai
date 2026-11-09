"""Figure for the three held-out domains: example fields plus the gate comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "paper_figures" / "uncertainty_study" / "tables"
OUT = ROOT / "paper_figures" / "uncertainty_study" / "main_figures"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "font.size": 6.6,
})

ROWS = [("JHTDB_mhd1024_hd", "MHD 湍流"), ("JHTDB_channel_hd", "槽道（高清）"), ("uniform", "均匀流")]
TITLES = ["真值", "重建", "候选分歧", "绝对误差", "门控前 / 门控后"]


def main() -> None:
    data = np.load(TABLES / "holdout_examples.npz")
    stats = json.loads((TABLES / "holdout_all.json").read_text(encoding="utf-8"))

    figure, axes = plt.subplots(3, 5, figsize=(7.4, 4.5),
                                gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5]})

    for row, (key, label) in enumerate(ROWS):
        for column, (name, cmap, signed) in enumerate([
                ("truth", "RdBu_r", True), ("recon", "RdBu_r", True),
                ("spread", "viridis", False), ("error", "magma", False)]):
            field = data[f"{key}::{name}"]
            if field.ndim == 3:
                field = np.sqrt((field ** 2).sum(axis=0))
                signed = False
            axis = axes[row, column]
            if signed:
                centre = float(np.mean(field))
                spread = float(np.std(field)) + 1e-12
                im = axis.imshow(field, cmap=cmap, vmin=centre - 2 * spread,
                                 vmax=centre + 2 * spread)
            else:
                im = axis.imshow(field, cmap=cmap)
            plt.colorbar(im, ax=axis, fraction=0.046, pad=0.02).ax.tick_params(labelsize=4)
            if float(np.std(field)) < 1e-6 * (abs(float(np.mean(field))) + 1.0):
                axis.text(0.5, 0.5, "空间均匀\n速度恒为 2.83", transform=axis.transAxes,
                          fontsize=6.2, ha="center", va="center", color="0.25")
                axis.set_facecolor("0.93")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(TITLES[column], fontsize=7)
            letter = chr(ord("a") + row * 5 + column)
            axis.text(-0.08, 1.05, letter, transform=axis.transAxes, fontsize=7.5,
                      fontweight="bold", va="bottom", ha="right")

        axis = axes[row, 4]
        entry = stats.get(key)
        if entry:
            quantities = entry["quantities"]
            names = [quantities[k]["name"] for k in quantities]
            before = [quantities[k]["R_none"] for k in quantities]
            after = [quantities[k]["R_gate"] for k in quantities]
            y = np.arange(len(names))
            for yi, b, a in zip(y, before, after):
                axis.plot([b, a], [yi, yi], color="0.65", lw=0.8, zorder=1)
            axis.scatter(before, y, s=15, color="#4c78a8", zorder=2, label="不门控")
            axis.scatter(after, y, s=15, color="#e45756", zorder=2, label="门控")
            axis.axvline(1.0, color="0.4", lw=0.6, ls="--")
            axis.set_yticks(y)
            axis.set_yticklabels(names, fontsize=5.6)
            axis.yaxis.tick_right()
            axis.invert_yaxis()
            axis.set_xlabel("宽度比 R", labelpad=1, fontsize=6.2)
            axis.tick_params(axis="x", labelsize=5.5)
            axis.tick_params(axis="y", length=0, pad=1)
            # colours are explained in the caption
        letter = chr(ord("a") + row * 5 + 4)
        axis.text(-0.12, 1.05, letter, transform=axis.transAxes, fontsize=7.5,
                  fontweight="bold", va="bottom", ha="right")

        axes[row, 0].set_ylabel(label, fontsize=8, labelpad=16)

    figure.subplots_adjust(left=0.07, right=0.86, top=0.93, bottom=0.10, hspace=0.24, wspace=0.18)
    path = OUT / "fig_holdout_domains.png"
    figure.savefig(path)
    plt.close(figure)
    print("wrote", path)


if __name__ == "__main__":
    main()
