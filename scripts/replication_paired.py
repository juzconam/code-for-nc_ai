"""Paired uncertainty for the two-gate comparison.

The two gates were evaluated on the same 355 samples with the same per-uid candidate noise, so the
comparison is paired at the sample level.  This resamples samples (not calibration splits) to put an
interval on each R and on the difference between the two gates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "piv_runs" / "uncertainty_study" / "study_test.npz"
REPL = ROOT / "piv_runs" / "replicate_agent_v2" / "study_test.npz"
OUT = ROOT / "paper_figures" / "uncertainty_study" / "tables"
ALPHA = 0.1
BOOT = 300


def cq(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(np.ceil((scores.size + 1) * (1 - alpha))) - 1
    return float(scores[min(max(k, 0), scores.size - 1)])


def width_ratio(point, scale, truth, seed=0):
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
    main_data = np.load(MAIN, allow_pickle=True)
    repl_data = np.load(REPL, allow_pickle=True)
    main_index = {str(u): i for i, u in enumerate(main_data["uid"])}
    repl_index = {str(u): i for i, u in enumerate(repl_data["uid"])}
    shared = sorted(set(main_index) & set(repl_index))
    meta = json.loads((ROOT / "piv_runs" / "uncertainty_study" / "study_test_meta.json")
                      .read_text(encoding="utf-8"))

    mi = np.array([main_index[u] for u in shared])
    ri = np.array([repl_index[u] for u in shared])
    rng = np.random.default_rng(0)
    report = {}

    for column, key in enumerate(meta["quantities"]):
        def arrays(data, idx):
            truth = data["q_truth"][idx, column].astype(np.float64)
            cand = data["q_cand__hodge1.00"][idx, :, column].astype(np.float64)
            return truth, cand.mean(axis=1), cand.std(axis=1)

        t_main, p_main, s_main = arrays(main_data, mi)
        t_repl, p_repl, s_repl = arrays(repl_data, ri)

        r_main = width_ratio(p_main, s_main, t_main)
        r_repl = width_ratio(p_repl, s_repl, t_repl)

        diffs = []
        for b in range(BOOT):
            pick = rng.integers(0, len(shared), len(shared))
            a = width_ratio(p_main[pick], s_main[pick], t_main[pick], seed=b)
            c = width_ratio(p_repl[pick], s_repl[pick], t_repl[pick], seed=b)
            diffs.append(a - c)
        report[key] = {
            "R_main": r_main, "R_replicate": r_repl,
            "difference": r_main - r_repl,
            "ci90": [float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))],
            "n": len(shared),
        }

    (OUT / "replication_paired.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, entry in report.items():
        print(f"{key:20s} main {entry['R_main']:.3f}  repl {entry['R_replicate']:.3f}  "
              f"diff {entry['difference']:+.3f} [{entry['ci90'][0]:+.3f}, {entry['ci90'][1]:+.3f}]")


if __name__ == "__main__":
    main()
