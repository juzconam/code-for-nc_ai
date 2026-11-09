"""Check how the width ratio R depends on the nominal level alpha.

The previous draft claimed R is independent of alpha.  That is false in general; this script
measures R at several alphas on the main instance so the paper can report the dependence
instead of asserting invariance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "piv_runs" / "uncertainty_study"
OUT = ROOT / "paper_figures" / "uncertainty_study" / "tables"


def conformal_quantile(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    n = scores.size
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(scores[min(max(k, 0), n - 1)])


def main():
    data = np.load(RUNS / "study_test.npz", allow_pickle=True)
    meta = json.loads((RUNS / "study_test_meta.json").read_text(encoding="utf-8"))
    quantities = meta["quantities"]
    truth_all = data["q_truth"]
    rng = np.random.default_rng(0)
    indices = np.arange(truth_all.shape[0])
    alphas = [0.05, 0.10, 0.20, 0.30]

    results = {}
    for column, name in enumerate(quantities):
        truth = truth_all[:, column].astype(np.float64)
        cand = data["q_cand__hodge1.00"][:, :, column].astype(np.float64)
        point = cand.mean(axis=1)
        scale = cand.std(axis=1)
        results[name] = {}
        for alpha in alphas:
            ratios = []
            for _ in range(20):
                perm = rng.permutation(indices)
                half = perm.size // 2
                cal, ev = perm[:half], perm[half:]
                error = np.abs(truth[cal] - point[cal])
                q_const = conformal_quantile(error, alpha)
                q_scaled = conformal_quantile(error / (scale[cal] + 1e-12), alpha)
                width_const = float(np.mean(2 * q_const))
                width_scaled = float(np.mean(2 * q_scaled * scale[ev]))
                ratios.append(width_const / width_scaled)
            results[name][f"{alpha:.2f}"] = {
                "R_mean": float(np.mean(ratios)),
                "R_sd": float(np.std(ratios)),
            }

    (OUT / "alpha_dependence_of_R.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"{'quantity':22s} " + " ".join(f"a={a:.2f}" for a in alphas))
    for name in quantities:
        row = " ".join(f"{results[name][f'{a:.2f}']['R_mean']:6.3f}" for a in alphas)
        spread = max(results[name][f"{a:.2f}"]["R_mean"] for a in alphas) / \
                 min(results[name][f"{a:.2f}"]["R_mean"] for a in alphas)
        print(f"{name:22s} {row}   max/min = {spread:.2f}")
    print(f"wrote {OUT / 'alpha_dependence_of_R.json'}")


if __name__ == "__main__":
    main()
