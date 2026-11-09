"""Paired analysis, stratified by whether the gate actually acted.

About half of the evaluation samples come back unchanged between the two arms, so pooled paired
differences are dominated by zeros.  This version reports the fraction of samples the gate acts
on, and then the paired differences on that subset, with a bootstrap over samples.
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


def boot_median(values, repeats=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = values.size
    draws = np.array([np.median(values[rng.integers(0, n, n)]) for _ in range(repeats)])
    return float(np.median(values)), float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


def paired(a, b):
    delta = b - a
    median, low, high = boot_median(delta)
    return {"median_before": float(np.median(a)), "median_after": float(np.median(b)),
            "median_delta": median, "ci90": [low, high],
            "fraction_decreased": float(np.mean(delta < 0)), "n": int(delta.size)}


def main():
    data = np.load(RUNS / "study_test.npz", allow_pickle=True)
    meta = json.loads((RUNS / "study_test_meta.json").read_text(encoding="utf-8"))
    truth = data["q_truth"]

    epe_pre = data["epe__pre"].astype(np.float64)
    epe_post = data["epe__hodge1.00"].astype(np.float64)
    active = ~np.isclose(epe_pre, epe_post, rtol=0, atol=0)

    point_pre = data["q_cand__pre"].mean(axis=1)
    point_post = data["q_cand__hodge1.00"].mean(axis=1)
    s_pre = data["q_cand__pre"].std(axis=1)
    s_post = data["q_cand__hodge1.00"].std(axis=1)

    report = {
        "n_total": int(active.size),
        "n_active": int(active.sum()),
        "fraction_active": float(active.mean()),
        "pooled": {"epe": paired(epe_pre, epe_post)},
        "active_only": {"epe": paired(epe_pre[active], epe_post[active])},
        "quantities": {},
    }

    for column, key in enumerate(meta["quantities"]):
        err_pre = np.abs(truth[:, column] - point_pre[:, column])
        err_post = np.abs(truth[:, column] - point_post[:, column])
        ratio_pre = err_pre / (s_pre[:, column] + 1e-12)
        ratio_post = err_post / (s_post[:, column] + 1e-12)
        entry = {
            "name": NAMES.get(key, key),
            "all": {
                "error": paired(err_pre, err_post),
                "ratio": paired(ratio_pre, ratio_post),
                "spread": paired(s_pre[:, column], s_post[:, column]),
            },
            "active": {
                "error": paired(err_pre[active], err_post[active]),
                "ratio": paired(ratio_pre[active], ratio_post[active]),
                "spread": paired(s_pre[active, column], s_post[active, column]),
            },
        }
        report["quantities"][key] = entry

    (OUT / "paired_intervention.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"samples {report['n_total']}, gate active on {report['n_active']} "
          f"({report['fraction_active']*100:.0f}%)")
    for scope in ("pooled", "active_only"):
        e = report[scope]["epe"]
        print(f"  EPE [{scope:11s}] {e['median_before']:.4f} -> {e['median_after']:.4f}  "
              f"Δ {e['median_delta']:+.4f} [{e['ci90'][0]:+.4f}, {e['ci90'][1]:+.4f}]  "
              f"下降比例 {e['fraction_decreased']*100:.0f}%")
    print()
    print(f"{'quantity':12s} | {'Δ误差':>26s} | {'Δ|E|/s':>26s} | {'Δ分歧':>24s}")
    for key, entry in report["quantities"].items():
        a = entry["active"]
        def fmt(d):
            return f"{d['median_delta']:+.3g} [{d['ci90'][0]:+.2g},{d['ci90'][1]:+.2g}]"
        print(f"{entry['name']:12s} | {fmt(a['error']):>26s} | {fmt(a['ratio']):>26s} | "
              f"{fmt(a['spread']):>24s}")


if __name__ == "__main__":
    main()
