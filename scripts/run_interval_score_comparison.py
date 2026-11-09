"""Unified interval comparison on the main instance, with the interval (Winkler) score.

The referee's objection is that the paper reports mean width only, and that the recommended
comparators (Mondrian / per-domain calibration, CQR, frozen scale, per-candidate scale) have
never been compared under one protocol.  This script does that on the saved per-sample arrays.

Methods compared at alpha = 0.1, on a split-half protocol (calibrate on one half, evaluate on
the other, repeated):

  constant            constant-width split conformal
  pre                 scale = spread of the un-gated candidates (the frozen-scale repair)
  post                scale = spread of the gated candidates (what the pipeline actually does)
  per_candidate       scale = spread of each candidate's own repeated draws (mc_candidate)
  mondrian            constant width per domain (per-domain calibration)
  mondrian_pre        per-domain calibration of the pre-gate normalized interval
  cqr                 conformalized quantile regression on the candidate quantiles

Reported per method and quantity: coverage, mean width, mean Winkler interval score, and the
score relative to the constant baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "piv_runs" / "uncertainty_study"
OUT = ROOT / "paper_figures" / "uncertainty_study" / "tables"
ALPHA = 0.1
REPEATS = 20

QUANTITY_LABEL = {
    "mean_speed": "平均速度",
    "kinetic_energy_proxy": "动能代理",
    "circulation": "环量",
    "enstrophy": "拟涡能",
    "continuity_rms": "连续性残差",
    "mean_strain_rate": "平均应变率",
}


def conformal_quantile(scores, alpha):
    """Order statistic; never interpolated."""
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    n = scores.size
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(scores[min(max(k, 0), n - 1)])


def winkler(lower, upper, truth, alpha):
    width = upper - lower
    penalty_low = (2.0 / alpha) * np.maximum(lower - truth, 0.0)
    penalty_high = (2.0 / alpha) * np.maximum(truth - upper, 0.0)
    return width + penalty_low + penalty_high


def evaluate(point, scale, truth, cal, ev, alpha):
    """Split-conformal interval around `point`, scaled by `scale` (or constant if scale is None)."""
    error = np.abs(truth[cal] - point[cal])
    if scale is None:
        q = conformal_quantile(error, alpha)
        lower = point[ev] - q
        upper = point[ev] + q
    else:
        scores = error / (scale[cal] + 1e-12)
        q = conformal_quantile(scores, alpha)
        lower = point[ev] - q * scale[ev]
        upper = point[ev] + q * scale[ev]
    coverage = float(np.mean((truth[ev] >= lower) & (truth[ev] <= upper)))
    width = float(np.mean(upper - lower))
    score = float(np.mean(winkler(lower, upper, truth[ev], alpha)))
    return coverage, width, score


def mondrian(point, scale, truth, domain, cal, ev, alpha):
    """One quantile per domain, applied to that domain's evaluation samples."""
    coverage_hits, widths, scores = [], [], []
    for name in np.unique(domain):
        cal_d = cal[domain[cal] == name]
        ev_d = ev[domain[ev] == name]
        if cal_d.size < 20 or ev_d.size < 5:
            continue
        error = np.abs(truth[cal_d] - point[cal_d])
        if scale is None:
            q = conformal_quantile(error, alpha)
            lower = point[ev_d] - q
            upper = point[ev_d] + q
        else:
            q = conformal_quantile(error / (scale[cal_d] + 1e-12), alpha)
            lower = point[ev_d] - q * scale[ev_d]
            upper = point[ev_d] + q * scale[ev_d]
        coverage_hits.append(np.mean((truth[ev_d] >= lower) & (truth[ev_d] <= upper)) * ev_d.size)
        widths.append(np.mean(upper - lower) * ev_d.size)
        scores.append(np.mean(winkler(lower, upper, truth[ev_d], alpha)) * ev_d.size)
    total = sum(1 for _ in coverage_hits)
    if total == 0:
        return np.nan, np.nan, np.nan
    weight = sum(ev[domain[ev] == name].size for name in np.unique(domain)
                 if ev[domain[ev] == name].size >= 5 and cal[domain[cal] == name].size >= 20)
    return (float(np.sum(coverage_hits) / weight), float(np.sum(widths) / weight),
            float(np.sum(scores) / weight))


def main():
    data = np.load(RUNS / "study_test.npz", allow_pickle=True)
    meta = json.loads((RUNS / "study_test_meta.json").read_text(encoding="utf-8"))
    quantities = meta["quantities"]
    domain = data["domain"]
    truth_all = data["q_truth"]

    rng = np.random.default_rng(0)
    indices = np.arange(truth_all.shape[0])

    results = {}
    for column, name in enumerate(quantities):
        truth = truth_all[:, column].astype(np.float64)
        cand_pre = data["q_cand__pre"][:, :, column].astype(np.float64)
        cand_post = data["q_cand__hodge1.00"][:, :, column].astype(np.float64)
        point_pre = cand_pre.mean(axis=1)
        point_post = cand_post.mean(axis=1)
        sigma_pre = cand_pre.std(axis=1)
        sigma_post = cand_post.std(axis=1)
        sigma_mc = data["mc_std__pre__candidate"][:, column].astype(np.float64)
        lower_q = np.quantile(cand_pre, ALPHA / 2, axis=1)
        upper_q = np.quantile(cand_pre, 1 - ALPHA / 2, axis=1)

        specs = {
            "constant": (point_pre, None, False),
            "pre": (point_pre, sigma_pre, False),
            "post": (point_post, sigma_post, False),
            "per_candidate": (point_pre, sigma_mc, False),
            "mondrian": (point_pre, None, True),
            "mondrian_pre": (point_pre, sigma_pre, True),
        }

        rows = {method: {"coverage": [], "width": [], "score": []} for method in specs}
        rows["cqr"] = {"coverage": [], "width": [], "score": []}

        for _ in range(REPEATS):
            perm = rng.permutation(indices)
            half = perm.size // 2
            cal, ev = perm[:half], perm[half:]
            for method, (point, scale, is_mondrian) in specs.items():
                if is_mondrian:
                    cov, wid, sco = mondrian(point, scale, truth, domain, cal, ev, ALPHA)
                else:
                    cov, wid, sco = evaluate(point, scale, truth, cal, ev, ALPHA)
                rows[method]["coverage"].append(cov)
                rows[method]["width"].append(wid)
                rows[method]["score"].append(sco)

            # CQR: calibrate the residual outside the candidate quantile band
            band_low, band_high = lower_q[cal], upper_q[cal]
            residual = np.maximum(band_low - truth[cal], truth[cal] - band_high)
            q = conformal_quantile(residual, ALPHA)
            low_ev = lower_q[ev] - q
            high_ev = upper_q[ev] + q
            rows["cqr"]["coverage"].append(float(np.mean((truth[ev] >= low_ev) & (truth[ev] <= high_ev))))
            rows["cqr"]["width"].append(float(np.mean(high_ev - low_ev)))
            rows["cqr"]["score"].append(float(np.mean(winkler(low_ev, high_ev, truth[ev], ALPHA))))

        base = float(np.mean(rows["constant"]["score"]))
        results[name] = {}
        for method, values in rows.items():
            results[name][method] = {
                "coverage": float(np.mean(values["coverage"])),
                "width": float(np.mean(values["width"])),
                "interval_score": float(np.mean(values["score"])),
                "score_ratio_vs_constant": float(np.mean(values["score"]) / base) if base > 0 else np.nan,
            }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "interval_score_comparison.json").write_text(
        json.dumps({"alpha": ALPHA, "repeats": REPEATS, "results": results}, indent=2),
        encoding="utf-8",
    )

    header = f"{'quantity':22s} {'method':16s} {'cover':>6s} {'width':>10s} {'Winkler':>10s} {'vs const':>9s}"
    print(header)
    print("-" * len(header))
    for name in quantities:
        for method, row in results[name].items():
            print(f"{name:22s} {method:16s} {row['coverage']:6.3f} {row['width']:10.4f} "
                  f"{row['interval_score']:10.4f} {row['score_ratio_vs_constant']:9.3f}")
        print()
    print(f"wrote {OUT / 'interval_score_comparison.json'}")


if __name__ == "__main__":
    main()
