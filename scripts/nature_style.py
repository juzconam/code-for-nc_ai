"""Shared Nature Portfolio figure style.

Nature guidance for figures: sans-serif (Helvetica/Arial), 5-7 pt final text, panel letters as
bold lowercase, no chart titles, minimal framing, single column 89 mm and double column 183 mm,
300 dpi or better.  This module sets those defaults once so every supplementary figure matches.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
SINGLE = 89 * MM
ONE_HALF = 120 * MM
DOUBLE = 183 * MM

PANEL_LETTER_SIZE = 8
LABEL_SIZE = 7
TICK_SIZE = 6
LEGEND_SIZE = 6
TITLE_SIZE = 7

COLOURS = {
    "blue": "#3b6fb6",
    "orange": "#e8873a",
    "green": "#4f9d5d",
    "red": "#c94a4a",
    "grey": "#8a8a8a",
    "lightgrey": "#c8c8c8",
    "purple": "#8a6fb0",
}


def apply() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": LABEL_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
        "axes.unicode_minus": True,
    })


def panel(ax, letter: str, x: float = -0.16, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL_LETTER_SIZE,
            fontweight="bold", va="bottom", ha="left")


TRANSLATIONS = {
    # quantities
    "平均速度": "Mean speed",
    "动能代理": "Kinetic-energy proxy",
    "拟涡能": "Enstrophy",
    "平均应变率": "Mean strain rate",
    "连续性残差": "Continuity residual",
    "散度均方根": "RMS divergence",
    "环量": "Circulation",
    # regimes and operators
    "无约束": "Unconstrained",
    "固定投影": "Fixed projection",
    "固定低通": "Fixed low-pass",
    "逐样本随机": "Per-sample random",
    "逐候选随机": "Per-candidate random",
    "干预前": "Before intervention",
    "干预后": "After intervention",
    "基线（无物理）": "Baseline (no physics)",
    "训练时·损失": "Training-time, loss",
    "训练时·结构": "Training-time, architecture",
    "训练时·结构（另一种子）": "Training-time, architecture (other seed)",
    "环路内引导": "In-loop guidance",
    "无引导": "Unguided",
    # axes
    "宽度比 R（对数轴）": "Width ratio R (log scale)",
    "宽度比 R": "Width ratio R",
    "分歧相对无物理基线的倍数（对数轴）": "Spread relative to baseline (log scale)",
    "分歧的动态范围（p95/p05）": "Spread dynamic range (p95/p05)",
    "分歧的动态范围": "Spread dynamic range",
    "分歧与误差的秩相关 ρ": "Rank correlation with error, ρ",
    "秩相关 ρ": "Rank correlation ρ",
    "动态范围": "Dynamic range",
    "覆盖率（%）": "Coverage (%)",
    "平均绝对误差": "Mean absolute error",
    "尺度分箱均值": "Binned scale mean",
    "区间宽度": "Interval width",
    "尖锐度增益": "Sharpness gain",
    "尺度 / 误差": "Scale-to-error ratio",
    "分歧倍数": "Spread multiple",
    # annotations
    "阈值 15": "Threshold 15",
    "分歧的跨样本变化较小": "Little sample-to-sample variation",
    "两数阈值以内": "Within both thresholds",
    "主实例（PIV 管线，4 个量）": "Main instance (PIV pipeline, 4 quantities)",
    "FunDiff · PIV 流场（3 种强度）": "FunDiff, PIV fields (3 settings)",
    "FunDiff · Darcy（3 种强度）": "FunDiff, Darcy (3 settings)",
    "体外圆柱绕流": "In vitro cylinder",
    "体内主动脉": "In vivo aorta",
    "体内左心室": "In vivo ventricle",
    "常数宽度": "Constant width",
    "干预前尺度": "Pre-intervention scale",
    "干预后尺度": "Post-intervention scale",
    "门控前冻结": "Frozen pre-gate",
    "数据相关": "Data-correlated",
    "跨样本": "Across samples",
    "有对象可修": "Has a target to correct",
    "边际": "Marginal",
    "最差域": "Worst domain",
}
