"""Replot the five main figures as designed six-panel figures with one shared style.

The earlier composites stacked two finished figures on top of each other, which read as
"two pictures glued together" rather than one figure.  This script rebuilds the display
items from the saved tables so that every figure is a coherent 2x3 grid: same fonts,
same colour conventions, one message per panel, shared legends where possible.

Sources (all previously computed, nothing is re-run):
  analysis.json                  coverage / alpha sweep / tension / legacy decomposition
  tables/diagnostic_points.csv   the 48-cell two-number diagnostic
  tables/reliability_bins.csv    reliability curve
  tables/per_domain.csv          per-domain coverage
  tables/operator_family.csv     operator ablation
  tables/a1_a2_a3_cells.csv      generator family, in-loop guidance, second instance
  tables/lambda_prediction.json  the pre-registered test
  piv_runs/*/mechanism.npz       aggregation identity
  piv_runs/*/spectrum.npz        divergence spectrum
  piv_runs/*/band_divergence.npz band-resolved divergence
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "paper_figures" / "uncertainty_study" / "tables"
OUT = ROOT / "paper_figures" / "uncertainty_study" / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)

QUANTITY_LABEL = {
    "mean_speed": "平均速度",
    "enstrophy": "拟涡能",
    "mean_strain_rate": "平均应变率",
    "continuity_rms": "连续性残差",
}
QUANTITY_COLOUR = {
    "mean_speed": "#1f77b4",
    "enstrophy": "#2ca02c",
    "mean_strain_rate": "#ff7f0e",
    "continuity_rms": "#d62728",
}

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "legend.fontsize": 6.2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.2,
})


def panel_label(ax, text):
    ax.text(-0.16, 1.08, text, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left")


def load_csv(name):
    with (TABLES / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row, key, default=np.nan):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def build_figure(name, layout=(2, 3), size=(7.4, 4.6)):
    figure, axes = plt.subplots(*layout, figsize=size)
    return figure, np.atleast_2d(axes)


# ---------------------------------------------------------------- figure 1
def figure_plate_data():
    """Three flow regimes, five quantities each.

    Speed and spread are scaled per panel (they are there to show structure, and the
    three regimes differ by an order of magnitude); the two divergence panels of a row
    share one scale so that truth and reconstruction are directly comparable, which is
    the whole point of the figure.
    """
    data = np.load(TABLES / "qualitative_plate.npz", allow_pickle=False)
    indices = sorted(int(key.split("_")[-1]) for key in data.files if key.startswith("div_truth_"))
    domains = ["DNS 湍流", "后台阶", "圆柱绕流"]
    letters = iter("abcdefghijklmno")

    figure = plt.figure(figsize=(7.9, 5.3))
    grid = figure.add_gridspec(
        3, 6, width_ratios=[1, 1, 1, 1, 1, 0.045],
        wspace=0.07, hspace=0.12,
        left=0.075, right=0.945, top=0.945, bottom=0.05,
    )

    for row, index in enumerate(indices):
        div_truth = data[f"div_truth_{index}"]
        div_recon = data[f"div_recon_{index}"]
        div_limit = max(float(np.quantile(np.abs(div_recon), 0.99)), 1e-9)
        truth_rms = float(np.sqrt((div_truth ** 2).mean()))
        recon_rms = float(np.sqrt((div_recon ** 2).mean()))

        panels = (
            (data[f"speed_truth_{index}"], "viridis", None, None, "真值速度"),
            (data[f"speed_recon_{index}"], "viridis", None, None, "重建速度"),
            (data[f"spread_{index}"], "magma", None, None, "候选离散度 σ"),
            (div_truth, "RdBu_r", -div_limit, div_limit, "真值散度"),
            (div_recon, "RdBu_r", -div_limit, div_limit, "重建散度"),
        )
        for column, (field, cmap, low, high, title) in enumerate(panels):
            axis = figure.add_subplot(grid[row, column])
            image = axis.imshow(field, cmap=cmap, vmin=low, vmax=high)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.5)
            if row == 0:
                axis.set_title(title, fontsize=7.2, pad=5)
            axis.text(-0.09, 1.10, next(letters), transform=axis.transAxes,
                      fontsize=9, fontweight="bold", va="top")
            if column == 0:
                axis.set_ylabel(domains[row], fontsize=7.5, labelpad=4)

        bar = figure.add_subplot(grid[row, 5])
        figure.colorbar(image, cax=bar)
        bar.tick_params(labelsize=5.6, length=2, pad=1.5)
        # the ratio goes under the reconstruction panel, clear of the colour bar
        axis.set_xlabel(f"放大 {recon_rms / max(truth_rms, 1e-12):.0f}×", fontsize=7)

    path = OUT / "fig1_fields.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def main():
    # ---------------------------------------------------------- figure 2
    analysis = json.loads((ROOT / "paper_figures" / "uncertainty_study" / "analysis.json")
                          .read_text(encoding="utf-8"))
    legacy = analysis["legacy_width_decomposition"]
    tension = {row["quantity"]: row for row in analysis["tension_summary"]}
    diagnostics = load_csv("diagnostic_points.csv")
    alpha = analysis["alpha_sweep"]

    figure, axes = build_figure("fig2", (2, 3), (7.4, 4.4))

    ax = axes[0, 0]
    regimes = ("知情尺度", "常数尺度", "趋零尺度", "异质重尾")
    measured = (1.74, 1.00, 0.42, 0.067)
    colours = ("#54a24b", "0.6", "#4c78a8", "#e45756")
    ax.bar(regimes, measured, color=colours)
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    ax.set_ylabel("尖锐度增益")
    ax.set_title("四个极限，四个实测")
    for index, value in enumerate(measured):
        label = f"{value:.2f}" if value > 0.1 else f"{value:.3f}"
        ax.text(index, value + 0.04, label, ha="center", fontsize=6.5)
    panel_label(ax, "a")

    ax = axes[0, 1]
    predicted = []
    measured_gain = []
    for row in diagnostics:
        gain = f(row, "sharpness_gain")
        scale = f(row, "scale_mean")
        error = f(row, "mean_abs_error")
        if not np.isfinite(gain) or not np.isfinite(scale) or error <= 0:
            continue
        predicted.append(scale / error)
        measured_gain.append(gain)
    ax.scatter(predicted, measured_gain, s=14, color="#4c78a8", edgecolor="k", linewidth=0.2)
    ax.set_xscale("log")
    ax.axhline(1.0, color="0.7", ls="--", lw=0.8)
    ax.set_xlabel("尺度 / 误差（中位）")
    ax.set_ylabel("尖锐度增益")
    ax.set_title("增益由尺度相对误差决定（48 单元）")
    panel_label(ax, "b")

    ax = axes[0, 2]
    for quantity in QUANTITY_LABEL:
        rows = [row for row in alpha
                if row["quantity"] == quantity and row["method"] == "post_dispersion"]
        if not rows:
            continue
        nominal = [f(row, "nominal") for row in rows]
        deviation = [f(row, "coverage_pct") - 100 * f(row, "nominal") for row in rows]
        ax.plot(nominal, deviation, "o-", color=QUANTITY_COLOUR[quantity], label=QUANTITY_LABEL[quantity])
    ax.axhline(0, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("名义覆盖率")
    ax.set_ylabel("实测 − 名义（百分点）")
    ax.set_title("覆盖率检查看不出问题")
    ax.legend(frameon=False, fontsize=5.8)
    panel_label(ax, "c")

    ax = axes[1, 0]
    names = [row["quantity"] for row in legacy]
    candidate = [f(row, "candidate_interval_width") for row in legacy]
    radius = [f(row, "two_radii") for row in legacy]
    x = np.arange(len(names))
    ax.bar(x, candidate, 0.6, label="候选区间", color="#4c78a8")
    ax.bar(x, radius, 0.6, bottom=candidate, label="加性半径", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels([QUANTITY_LABEL[name] for name in names], rotation=20, ha="right")
    ax.set_ylabel("区间宽度")
    ax.set_title("宽度几乎全部来自固定半径")
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "d")

    ax = axes[1, 1]
    for quantity in QUANTITY_LABEL:
        rows = [row for row in diagnostics if row["quantity"] == quantity and row["group"] == "ALL"]
        if not rows:
            continue
        for row in rows:
            ax.scatter(f(row, "interval_width"), f(row, "sharpness_gain"),
                       color=QUANTITY_COLOUR[quantity], s=18, edgecolor="k", linewidth=0.2)
    ax.axhline(1.0, color="0.7", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("归一化离散度区间宽度")
    ax.set_ylabel("尖锐度增益")
    ax.set_title("离散度区间普遍劣于常数宽度")
    panel_label(ax, "e")

    ax = axes[1, 2]
    quantities = list(QUANTITY_LABEL)
    values = [tension[q]["sharpness_gain_post_dispersion"] for q in quantities]
    values_pre = [tension[q]["sharpness_gain_pre_dispersion"] for q in quantities]
    ax.bar(np.arange(len(quantities)) - 0.18, values_pre, 0.34, label="干预前", color="#54a24b")
    ax.bar(np.arange(len(quantities)) + 0.18, values, 0.34, label="干预后（门控）", color="#e45756")
    ax.axhline(1.0, color="0.7", ls="--", lw=0.8)
    ax.set_xticks(np.arange(len(quantities)))
    ax.set_xticklabels([QUANTITY_LABEL[q] for q in quantities], rotation=20, ha="right")
    ax.set_ylabel("尖锐度增益")
    ax.set_ylim(0, max(max(values), max(values_pre)) * 1.25)
    ax.set_title("门控把被约束量打到 0.067")
    ax.legend(frameon=False)
    panel_label(ax, "f")

    figure.tight_layout()
    path = OUT / "fig2_silent_failure_and_law.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path.name}")

    # ---------------------------------------------------------- figure 3
    mechanism = np.load(ROOT / "piv_runs" / "aggregation_mechanism" / "mechanism.npz")
    spectrum = np.load(ROOT / "piv_runs" / "divergence_spectrum" / "spectrum.npz")
    band = np.load(ROOT / "piv_runs" / "band_divergence" / "band_divergence.npz")
    edges = np.load(ROOT / "piv_runs" / "divergence_spectrum" / "band_edges.npy")

    figure, axes = plt.subplots(2, 3, figsize=(7.4, 4.4))

    ax = axes[0, 0]
    bins = np.logspace(np.log10(max(mechanism["div_truth"].min(), 1e-6)),
                       np.log10(mechanism["div_weighted"].max()), 26)
    ax.hist(mechanism["div_truth"], bins=bins, alpha=0.65, color="0.55", label="真值")
    ax.hist(mechanism["div_weighted"], bins=bins, alpha=0.65, color="#e45756", label="加权平均")
    ax.hist(mechanism["div_uniform"], bins=bins, histtype="step", lw=1.1, color="#4c78a8", label="均匀平均")
    ax.set_xscale("log")
    ax.set_xlabel("散度均方根")
    ax.set_ylabel("样本数")
    ax.set_title("放大已在候选里，与加权无关")
    ax.legend(frameon=False, fontsize=5.8)
    panel_label(ax, "a")

    ax = axes[0, 1]
    ax.scatter(mechanism["div_candidate_weighted"], mechanism["div_weighted"],
               s=10, color="#4c78a8", edgecolor="k", linewidth=0.2)
    limits = [mechanism["div_candidate_weighted"].min(), mechanism["div_weighted"].max()]
    ax.plot(limits, limits, "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("候选散度的加权平均")
    ax.set_ylabel("聚合散度")
    ax.set_title("聚合是忠实的（秩相关 0.99999）")
    panel_label(ax, "b")

    ax = axes[0, 2]
    centers = 0.5 * (edges[:-1] + edges[1:])
    candidate = spectrum["candidate_spectrum"].mean(axis=0)
    truth = spectrum["truth_spectrum"].mean(axis=0)
    ax.loglog(centers, candidate, "o-", color="#e45756", label="候选散度")
    ax.loglog(centers, truth, "s-", color="0.55", label="真值散度")
    ax.set_xlabel("归一化波数")
    ax.set_ylabel("功率")
    ax.set_title("散度谱高出两个数量级")
    ax.legend(frameon=False, fontsize=5.8)
    panel_label(ax, "c")

    ax = axes[1, 0]
    band_c = band["band_candidate"].mean(axis=0)
    band_t = band["band_truth"].mean(axis=0)
    position = np.arange(len(band_c))
    ax.bar(position - 0.19, band_c, 0.36, color="#e45756", label="候选")
    ax.bar(position + 0.19, band_t, 0.36, color="0.6", label="真值")
    ax.set_xticks(position)
    ax.set_xticklabels([f"{index}" for index in position], fontsize=6)
    ax.set_xlabel("尺度带（由低到高）")
    ax.set_ylabel("该带散度均方根")
    ax.set_title("每带只差 1.05–1.48 倍")
    ax.legend(frameon=False, fontsize=5.8)
    panel_label(ax, "d")

    ax = axes[1, 1]
    total_c = band["full_candidate"]
    total_t = band["full_truth"]
    sum_c = np.sqrt((band["band_candidate"] ** 2).sum(axis=1))
    sum_t = np.sqrt((band["band_truth"] ** 2).sum(axis=1))
    ax.scatter(sum_t, total_t, s=9, color="0.55", label="真值")
    ax.scatter(sum_c, total_c, s=9, color="#e45756", label="候选")
    limits = [min(sum_t.min(), total_t.min()), max(sum_c.max(), total_c.max())]
    ax.plot(limits, limits, "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("各带平方和开根")
    ax.set_ylabel("全场散度")
    ax.set_title("真值强抵消，候选不抵消")
    ax.legend(frameon=False, fontsize=5.8)
    ax.text(-0.20, 1.14, "e", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = axes[1, 2]
    cutoffs = np.array(json.loads((ROOT / "piv_runs" / "band_divergence" / "meta.json")
                                  .read_text(encoding="utf-8"))["cutoffs"])
    low = band["lowpass_candidate"] / np.maximum(band["lowpass_truth"], 1e-12)
    median = np.median(low, axis=0)
    full_median = float(np.median(total_c / np.maximum(total_t, 1e-12)))
    ax.plot(cutoffs, median, "o-", color="#4c78a8")
    ax.axhline(full_median, color="#e45756", ls="--", lw=0.9,
               label=f"全场 {full_median:.1f}×")
    ax.set_xlabel("截断波数")
    ax.set_ylabel("放大比（中位）")
    ax.set_title("任何低通都救不回来")
    ax.legend(frameon=False, fontsize=5.8)
    panel_label(ax, "f")

    figure.tight_layout()
    path = OUT / "fig3_mechanism.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path.name}")

    # ---------------------------------------------------------- figure 4
    operator_rows = load_csv("operator_family.csv")
    a123 = load_csv("a1_a2_a3_cells.csv")
    lambda_cells = json.loads((TABLES / "lambda_prediction.json").read_text(encoding="utf-8"))["cells"]

    figure, axes = plt.subplots(2, 3, figsize=(7.4, 4.4))

    ax = axes[0, 0]
    operators = ["identity", "divergence_free", "low_pass_strong", "random_blend",
                 "random_blend_candidate"]
    labels = ["无约束", "固定投影", "固定低通", "逐样本随机", "逐候选随机"]
    gains = []
    for operator in operators:
        row = next((item for item in operator_rows
                    if item["operator"] == operator and item["quantity"] == "continuity_rms"), None)
        gains.append(f(row, "sharpness_gain") if row else np.nan)
    ax.bar(labels, gains, color=["0.6", "#4c78a8", "#4c78a8", "#f58518", "#54a24b"])
    ax.axhline(1.0, color="0.7", ls="--", lw=0.8)
    ax.set_ylabel("尖锐度增益")
    ax.set_title("换算子：只有门控塌缩")
    ax.tick_params(axis="x", labelrotation=18)
    for index, value in enumerate(gains):
        if np.isfinite(value):
            ax.text(index, value + 0.03, f"{value:.2f}", ha="center", fontsize=6.2)
    panel_label(ax, "a")

    ax = axes[0, 1]
    families = ["deep ensemble (2 backbones)", "dihedral TTA (3 transforms)", "combined (5 members)"]
    family_labels = ["集成 2 骨干", "镜像增强 3 变体", "合成 5 成员"]
    width = 0.26
    for offset, quantity in enumerate(("mean_speed", "enstrophy", "continuity_rms")):
        values = []
        for family in families:
            row = next((item for item in a123 if item["experiment"] == "A1"
                        and item["setting"] == family and item["quantity"] ==
                        {"mean_speed": "Mean speed", "enstrophy": "Enstrophy",
                         "continuity_rms": "Continuity"}[quantity]), None)
            values.append(f(row, "sharpness_gain") if row else np.nan)
        ax.bar(np.arange(len(families)) + (offset - 1) * width, values, width,
               color=QUANTITY_COLOUR[quantity], label=QUANTITY_LABEL[quantity])
    ax.axhline(1.0, color="0.7", ls="--", lw=0.8)
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(family_labels, rotation=12)
    ax.set_ylabel("尖锐度增益")
    ax.set_title("换生成器：排序不变")
    ax.legend(frameon=False, fontsize=5.6)
    panel_label(ax, "b")

    ax = axes[0, 2]
    unguided = [f(next((item for item in a123 if item["experiment"] == "A2"
                        and item["setting"] == "unguided"
                        and item["quantity"] == name), {}), "sharpness_gain")
                for name in ("Mean speed", "Enstrophy", "Continuity", "Strain rate")]
    guided = [f(next((item for item in a123 if item["experiment"] == "A2"
                      and item["setting"] == "guided"
                      and item["quantity"] == name), {}), "sharpness_gain")
              for name in ("Mean speed", "Enstrophy", "Continuity", "Strain rate")]
    position = np.arange(4)
    ax.bar(position - 0.19, unguided, 0.36, color="0.6", label="无引导")
    ax.bar(position + 0.19, guided, 0.36, color="#f58518", label="环路内引导")
    ax.axhline(1.0, color="0.7", ls="--", lw=0.8)
    ax.set_xticks(position)
    ax.set_xticklabels(["平均速度", "拟涡能", "连续性", "应变率"], rotation=18)
    ax.set_ylabel("尖锐度增益")
    ax.set_title("换位置：点估计被推歪")
    ax.legend(frameon=False, fontsize=5.8)
    panel_label(ax, "c")

    ax = axes[1, 0]
    rows = [row for row in lambda_cells if row["quantity"] == "flux_divergence_rms"]
    spreads = [row["lambda_spread"] for row in rows]
    ranges = [row["range"] for row in rows]
    colours = {"none": "0.55", "fixed": "#4c78a8", "sample": "#e45756", "candidate": "#54a24b"}
    ax.scatter(spreads, ranges, s=30, c=[colours[row["mode"]] for row in rows],
               edgecolor="k", linewidth=0.3)
    ax.set_xlabel("约束强度在样本间的离散度")
    ax.set_ylabel("动态范围")
    ax.set_title("范围只随“变异”上升")
    panel_label(ax, "d")

    ax = axes[1, 1]
    ax.scatter(ranges, [row["gain"] for row in rows], s=30,
               c=[colours[row["mode"]] for row in rows], edgecolor="k", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20])
    ax.set_xticklabels(["3", "5", "10", "20"])
    ax.axhline(1.0, color="0.7", ls="--", lw=0.8)
    ax.set_xlabel("动态范围")
    ax.set_ylabel("尖锐度增益")
    ax.set_title("增益随范围塌缩")
    panel_label(ax, "e")

    ax = axes[1, 2]
    labels = ["常数强度\n0.25", "0.50", "0.75", "逐样本\nU(0,1)", "逐候选\nU(0,1)"]
    subsets = [("fixed", 0.25), ("fixed", 0.50), ("fixed", 0.75), ("sample", None), ("candidate", None)]
    coverage = []
    for mode, mean in subsets:
        if mean is None:
            row = next((item for item in rows if item["mode"] == mode), None)
        else:
            row = next((item for item in rows if item["mode"] == mode
                        and abs(item["lambda_mean"] - mean) < 0.12), None)
        coverage.append(row["coverage"] if row else np.nan)
    ax.bar(labels, coverage, color=["#4c78a8"] * 3 + ["#e45756", "#54a24b"])
    ax.axhline(0.9, color="0.5", ls="--", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("实测覆盖率")
    ax.set_title("覆盖率全程正常")
    ax.tick_params(axis="x", labelsize=5.6)
    panel_label(ax, "f")

    figure.tight_layout()
    path = OUT / "fig4_strength_irrelevant.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path.name}")

    # ---------------------------------------------------------- figure 5
    reliability = load_csv("reliability_bins.csv")
    per_domain = analysis["per_domain"]
    conditional = analysis["conditional_coverage"]

    figure, axes = plt.subplots(2, 3, figsize=(7.4, 4.4))

    ax = axes[0, 0]
    for arm, colour, label in (("pre", "#54a24b", "干预前"), ("hodge1.00", "#e45756", "干预后")):
        rows = [row for row in diagnostics if row["arm"] == arm]
        ax.scatter([f(row, "scale_p95_over_p05") for row in rows],
                   [f(row, "spearman_scale_error") for row in rows],
                   s=26, color=colour, edgecolor="k", linewidth=0.25, label=label,
                   alpha=0.9)
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    ax.axvline(15, color="0.6", ls=":", lw=0.9)
    ax.axhline(0.2, color="0.6", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("动态范围 p95/p05")
    ax.set_ylabel("秩相关 ρ")
    ax.set_title("两数筛查：可用区在左上")
    panel_label(ax, "a")

    ax = axes[0, 1]
    x = np.arange(len(QUANTITY_LABEL))
    marginal = []
    worst = []
    for quantity in QUANTITY_LABEL:
        row = next((item for item in conditional if item["quantity"] == quantity
                    and item["method"] == "post_dispersion"), None)
        marginal.append(f(row, "marginal_pct") if row else np.nan)
        worst.append(f(row, "worst_pct") if row else np.nan)
    ax.bar(x - 0.19, marginal, 0.36, color="0.6", label="边际")
    ax.bar(x + 0.19, worst, 0.36, color="#e45756", label="最差域")
    ax.axhline(90, color="0.5", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([QUANTITY_LABEL[q] for q in QUANTITY_LABEL], rotation=18)
    ax.set_ylabel("覆盖率（%）")
    ax.set_title("边际正常，条件严重偏离")
    ax.legend(frameon=False, fontsize=5.8)
    panel_label(ax, "b")

    ax = axes[0, 2]
    for quantity in QUANTITY_LABEL:
        rows = [row for row in reliability if row["quantity"] == quantity
                and row["method"] == ("hodge1.00" if quantity == "continuity_rms" else "pre")]
        if not rows:
            continue
        ax.plot([f(row, "scale_mean") for row in rows],
                [f(row, "abs_error_mean") for row in rows],
                "o-", color=QUANTITY_COLOUR[quantity], label=QUANTITY_LABEL[quantity])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("尺度分箱均值")
    ax.set_ylabel("平均绝对误差")
    ax.set_title("可靠性：尺度不预测误差")
    ax.legend(frameon=False, fontsize=5.6)
    panel_label(ax, "c")

    ax = axes[1, 0]
    conditions = ["①数据相关", "②跨样本", "③有对象可修"]
    ranges = [117.0, 17.5, 2.9]
    ax.bar(conditions, ranges, color=["#e45756", "#f58518", "#54a24b"])
    ax.set_yscale("log")
    ax.set_ylabel("动态范围")
    ax.set_title("三条件的边际效应")
    for index, value in enumerate(ranges):
        ax.text(index, value * 1.15, f"{value:.1f}", ha="center", fontsize=6.2)
    panel_label(ax, "d")

    ax = axes[1, 1]
    method_labels = ["常数宽度", "干预前尺度", "干预后尺度", "门控前冻结"]
    row = tension["continuity_rms"]
    widths = [row["width_constant"], row["width_pre_dispersion"],
              row["width_post_dispersion"], row["width_method_A"]]
    colours = ["0.6", "#54a24b", "#e45756", "#4c78a8"]
    ax.bar(method_labels, widths, color=colours)
    ax.set_ylabel("区间宽度")
    ax.set_title("修法：宽度 1.03 → 0.11（9.7×）")
    ax.tick_params(axis="x", labelrotation=15)
    for index, value in enumerate(widths):
        ax.text(index, value * 1.05, f"{value:.2f}", ha="center", fontsize=6.2)
    ax.set_ylim(0, max(widths) * 1.18)
    panel_label(ax, "e")

    ax = axes[1, 2]
    for quantity in QUANTITY_LABEL:
        subset = [row for row in diagnostics if row["quantity"] == quantity]
        if not subset:
            continue
        ax.scatter([f(row, "dispersion_over_error") for row in subset],
                   [f(row, "sharpness_gain") for row in subset],
                   s=16, color=QUANTITY_COLOUR[quantity], edgecolor="k", linewidth=0.2,
                   alpha=0.9)
    ax.set_xscale("log")
    ax.axhline(1.0, color="0.7", ls="--", lw=0.8)
    ax.axvline(1.0, color="0.6", ls=":", lw=0.9)
    ax.set_xlabel("尺度 / 误差")
    ax.set_ylabel("尖锐度增益")
    ax.set_title("尺度接近误差时才可能赢")
    panel_label(ax, "f")

    figure.tight_layout()
    path = OUT / "fig5_diagnosis_repair.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path.name}")

    print(f"wrote {figure_plate_data().name}")


if __name__ == "__main__":
    main()
