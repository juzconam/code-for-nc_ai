"""Analyse the three-route grid: R, coverage, spread magnitude and the spread-error correlation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures" / "uncertainty_study" / "tables"
CONFIGS = ["baseline", "loss0.5", "loss5", "loss50", "arch", "arch_seed1"]
QUANT = ["mean_speed", "mean_strain_rate", "enstrophy", "divergence_rms"]
ALPHA = 0.1


def conformal_quantile(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(np.ceil((scores.size + 1) * (1 - alpha))) - 1
    return float(scores[min(max(k, 0), scores.size - 1)])


def evaluate(point, spread, truth, repeats=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(truth.size)
    ratios, cover = [], []
    for _ in range(repeats):
        perm = rng.permutation(idx)
        half = perm.size // 2
        cal, ev = perm[:half], perm[half:]
        err = np.abs(truth[cal] - point[cal])
        qc = conformal_quantile(err, ALPHA)
        qs = conformal_quantile(err / (spread[cal] + 1e-12), ALPHA)
        ratios.append(np.mean(2 * qc) / max(np.mean(2 * qs * spread[ev]), 1e-12))
        lower = point[ev] - qs * spread[ev]
        upper = point[ev] + qs * spread[ev]
        cover.append(np.mean((truth[ev] >= lower) & (truth[ev] <= upper)))
    return float(np.mean(ratios)), float(np.mean(cover))


def main():
    data = np.load(OUT / "three_routes.npz", allow_pickle=False)
    report = {}
    base_spread = {}
    for name in CONFIGS:
        row = {}
        for key in QUANT:
            point = data[f"{name}__point__{key}"]
            spread = data[f"{name}__spread__{key}"]
            truth = data[f"{name}__truth__{key}"]
            ratio, coverage = evaluate(point, spread, truth)
            error = np.abs(truth - point)
            rho = spearmanr(spread, error).statistic if spread.std() > 0 else float("nan")
            row[key] = {"R": ratio, "coverage": coverage,
                        "spread_median": float(np.median(spread)),
                        "rho_spread_error": float(rho)}
        report[name] = row
        base_spread[name] = {k: row[k]["spread_median"] for k in QUANT}

    # spread magnitude relative to the baseline
    for name in CONFIGS:
        for key in QUANT:
            report[name][key]["spread_vs_baseline"] = (
                base_spread[name][key] / max(base_spread["baseline"][key], 1e-30))

    (OUT / "three_routes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'config':12s} " + " ".join(f"{k[:12]:>26s}" for k in QUANT))
    for name in CONFIGS:
        cells = " ".join(
            f"R{report[name][k]['R']:6.2f} ρ{report[name][k]['rho_spread_error']:+.2f}"
            f" s{report[name][k]['spread_vs_baseline']:5.2f}".rjust(26) for k in QUANT)
        print(f"{name:12s} {cells}")
    print()
    for name in CONFIGS:
        cov = [report[name][k]["coverage"] for k in QUANT]
        print(f"{name:12s} coverage {min(cov):.2f}-{max(cov):.2f}")


if __name__ == "__main__":
    main()
