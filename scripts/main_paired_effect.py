"""Paired interval for the main instance: gated versus ungated, same form as the replication."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "piv_runs" / "uncertainty_study"
OUT = ROOT / "paper_figures" / "uncertainty_study" / "tables"
ALPHA = 0.1
BOOT = 300


def cq(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(np.ceil((scores.size + 1) * (1 - alpha))) - 1
    return float(scores[min(max(k, 0), scores.size - 1)])


def ratio(point, scale, truth, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(truth.size)
    perm = rng.permutation(idx)
    h = perm.size // 2
    cal, ev = perm[:h], perm[h:]
    err = np.abs(truth[cal] - point[cal])
    qc = cq(err, ALPHA)
    qs = cq(err / (scale[cal] + 1e-12), ALPHA)
    return float(np.mean(2 * qc) / max(np.mean(2 * qs * scale[ev]), 1e-12))


def main():
    data = np.load(RUNS / "study_test.npz", allow_pickle=True)
    meta = json.loads((RUNS / "study_test_meta.json").read_text(encoding="utf-8"))
    truth_all = data["q_truth"]
    rng = np.random.default_rng(0)
    report = {}

    for column, key in enumerate(meta["quantities"]):
        truth = truth_all[:, column].astype(np.float64)
        cand_g = data["q_cand__hodge1.00"][:, :, column].astype(np.float64)
        cand_u = data["q_cand__pre"][:, :, column].astype(np.float64)
        pg, sg = cand_g.mean(axis=1), cand_g.std(axis=1)
        pu, su = cand_u.mean(axis=1), cand_u.std(axis=1)

        r_gate = ratio(pg, sg, truth)
        r_none = ratio(pu, su, truth)

        diffs = []
        for b in range(BOOT):
            pick = rng.integers(0, truth.size, truth.size)
            diffs.append(ratio(pg[pick], sg[pick], truth[pick], seed=b)
                         - ratio(pu[pick], su[pick], truth[pick], seed=b))
        report[key] = {"R_gated": r_gate, "R_ungated": r_none,
                       "difference": r_gate - r_none,
                       "ci90": [float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))]}

    (OUT / "main_paired_effect.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, e in report.items():
        print(f"{key:22s} gated {e['R_gated']:.3f}  ungated {e['R_ungated']:.3f}  "
              f"diff {e['difference']:+.3f} [{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}]")


if __name__ == "__main__":
    main()
