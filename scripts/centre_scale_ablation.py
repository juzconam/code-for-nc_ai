"""2x2 ablation: does the decoupling come from the centre moving, or from the scale changing?

The interval in Eq. 1 has two ingredients: the point estimate (the candidate mean) and the scale
(the candidate spread).  The gated arm changes both at once, so the earlier comparisons cannot
say which one carries the effect.  This script crosses the two factors on the saved arrays:

    centre:  pre-gate candidate mean      /  post-gate candidate mean
    scale:   pre-gate candidate spread    /  post-gate candidate spread

and reports R for all four combinations, plus coverage.  If R collapses only when the post-gate
*spread* is used, the scale is the culprit; if it collapses whenever the post-gate centre is used,
the effect is in the point estimate and the "candidate set" narrative needs re-qualifying.
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


def conformal_quantile(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(np.ceil((scores.size + 1) * (1 - alpha))) - 1
    return float(scores[min(max(k, 0), scores.size - 1)])


def main():
    data = np.load(RUNS / "study_test.npz", allow_pickle=True)
    meta = json.loads((RUNS / "study_test_meta.json").read_text(encoding="utf-8"))
    quantities = meta["quantities"]
    truth_all = data["q_truth"]
    rng = np.random.default_rng(0)
    idx = np.arange(truth_all.shape[0])

    combos = {
        "pre centre / pre scale": ("pre", "pre"),
        "pre centre / post scale": ("pre", "post"),
        "post centre / pre scale": ("post", "pre"),
        "post centre / post scale": ("post", "post"),
    }

    results = {}
    for column, name in enumerate(quantities):
        truth = truth_all[:, column].astype(np.float64)
        cand = {arm: data[f"q_cand__{ 'hodge1.00' if arm == 'post' else 'pre' }"][:, :, column]
                .astype(np.float64) for arm in ("pre", "post")}
        point = {arm: cand[arm].mean(axis=1) for arm in cand}
        scale = {arm: cand[arm].std(axis=1) for arm in cand}

        results[name] = {}
        for label, (centre_arm, scale_arm) in combos.items():
            ratios, coverages = [], []
            for _ in range(REPEATS):
                perm = rng.permutation(idx)
                half = perm.size // 2
                cal, ev = perm[:half], perm[half:]
                p, s = point[centre_arm], scale[scale_arm]
                err = np.abs(truth[cal] - p[cal])
                q_const = conformal_quantile(err, ALPHA)
                q_scaled = conformal_quantile(err / (s[cal] + 1e-12), ALPHA)
                width_const = np.mean(2 * q_const)
                width_scaled = np.mean(2 * q_scaled * s[ev])
                ratios.append(width_const / max(width_scaled, 1e-12))
                lower = p[ev] - q_scaled * s[ev]
                upper = p[ev] + q_scaled * s[ev]
                coverages.append(np.mean((truth[ev] >= lower) & (truth[ev] <= upper)))
            results[name][label] = {"R": float(np.mean(ratios)),
                                    "R_sd": float(np.std(ratios)),
                                    "coverage": float(np.mean(coverages))}

    (OUT / "centre_scale_ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    labels = list(combos)
    print(f"{'quantity':22s} " + " ".join(f"{l[:22]:>24s}" for l in labels))
    for name in quantities:
        cells = " ".join(f"{results[name][l]['R']:8.3f}"
                         f"(cov {results[name][l]['coverage']:.2f})".rjust(24) for l in labels)
        print(f"{name:22s} {cells}")

    # how much of the drop is attributable to each factor (log scale, averaged over quantities)
    print()
    for name in quantities:
        r = {l: results[name][l]["R"] for l in labels}
        centre_effect = 0.5 * (np.log(r["post centre / pre scale"] / r["pre centre / pre scale"]) +
                               np.log(r["post centre / post scale"] / r["pre centre / post scale"]))
        scale_effect = 0.5 * (np.log(r["pre centre / post scale"] / r["pre centre / pre scale"]) +
                              np.log(r["post centre / post scale"] / r["post centre / pre scale"]))
        print(f"{name:22s} centre {np.exp(centre_effect):6.2f}x   scale {np.exp(scale_effect):6.2f}x")


if __name__ == "__main__":
    main()
