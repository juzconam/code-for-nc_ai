"""Coverage and interval score by domain and by quantity.

The earlier version pooled the six derived quantities into one coverage per domain.  They are
different scientific quantities with different error scales, so pooling can hide a quantity that
misses the target in a particular domain.  This reports, for every domain x quantity cell, the
coverage, the interval width and the Winkler score for both constructions, with a bootstrap over
samples for the coverage interval.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "piv_runs" / "uncertainty_study"
OUT = ROOT / "paper_figures" / "uncertainty_study" / "tables"
NAMES = {
    "mean_speed": "平均速度", "kinetic_energy_proxy": "动能代理", "circulation": "环量",
    "enstrophy": "拟涡能", "continuity_rms": "连续性残差", "mean_strain_rate": "平均应变率",
}
ALPHA = 0.1


def conformal_quantile(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(np.ceil((scores.size + 1) * (1 - alpha))) - 1
    return float(scores[min(max(k, 0), scores.size - 1)])


def winkler(truth, lower, upper, alpha):
    width = upper - lower
    penalty = 2 / alpha * (np.maximum(lower - truth, 0) + np.maximum(truth - upper, 0))
    return float(np.mean(width + penalty))


def boot_ci(hits, repeats=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = hits.size
    draws = [hits[rng.integers(0, n, n)].mean() for _ in range(repeats)]
    return float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


def main():
    data = np.load(RUNS / "study_test.npz", allow_pickle=True)
    meta = json.loads((RUNS / "study_test_meta.json").read_text(encoding="utf-8"))
    domain = data["domain"]
    truth_all = data["q_truth"]
    point_all = data["q_cand__hodge1.00"].mean(axis=1)
    spread_all = data["q_cand__hodge1.00"].std(axis=1)

    rng = np.random.default_rng(0)
    perm = rng.permutation(truth_all.shape[0])
    half = perm.size // 2
    cal, ev = perm[:half], perm[half:]

    report = {}
    for column, key in enumerate(meta["quantities"]):
        truth = truth_all[:, column].astype(np.float64)
        point = point_all[:, column].astype(np.float64)
        spread = spread_all[:, column].astype(np.float64)

        err = np.abs(truth[cal] - point[cal])
        q_const = conformal_quantile(err, ALPHA)
        q_scaled = conformal_quantile(err / (spread[cal] + 1e-12), ALPHA)

        entry = {}
        for name in sorted(set(domain)):
            mask = (domain[ev] == name)
            n = int(mask.sum())
            t = truth[ev][mask]
            p = point[ev][mask]
            s = spread[ev][mask]

            lower_c, upper_c = p - q_const, p + q_const
            lower_s, upper_s = p - q_scaled * s, p + q_scaled * s
            hit_c = ((t >= lower_c) & (t <= upper_c)).astype(np.float64)
            hit_s = ((t >= lower_s) & (t <= upper_s)).astype(np.float64)

            entry[name] = {
                "n": n,
                "coverage_constant": float(hit_c.mean()),
                "coverage_scaled": float(hit_s.mean()),
                "ci_constant": boot_ci(hit_c),
                "ci_scaled": boot_ci(hit_s),
                "hits_constant": int(hit_c.sum()),
                "hits_scaled": int(hit_s.sum()),
                "width_constant": float(np.mean(upper_c - lower_c)),
                "width_scaled": float(np.mean(upper_s - lower_s)),
                "winkler_constant": winkler(t, lower_c, upper_c, ALPHA),
                "winkler_scaled": winkler(t, lower_s, upper_s, ALPHA),
            }
        report[key] = {"name": NAMES.get(key, key), "domains": entry}

    (OUT / "coverage_by_domain_quantity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    domains = sorted(report["mean_speed"]["domains"])
    print("coverage (constant / scaled), 90% nominal")
    print(f"{'quantity':12s} " + " ".join(f"{d[:9]:>19s}" for d in domains))
    for key, entry in report.items():
        cells = []
        for d in domains:
            e = entry["domains"][d]
            cells.append(f"{e['coverage_constant']:.2f}/{e['coverage_scaled']:.2f}".rjust(19))
        print(f"{entry['name']:12s} " + " ".join(cells))

    print()
    print("Winkler ratio scaled/constant")
    print(f"{'quantity':12s} " + " ".join(f"{d[:9]:>10s}" for d in domains))
    for key, entry in report.items():
        cells = [f"{entry['domains'][d]['winkler_scaled']/entry['domains'][d]['winkler_constant']:.2f}".rjust(10)
                 for d in domains]
        print(f"{entry['name']:12s} " + " ".join(cells))


if __name__ == "__main__":
    main()
