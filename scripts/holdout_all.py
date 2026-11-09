"""Gate versus no gate on each held-out domain, with a paired bootstrap over frames."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures" / "uncertainty_study" / "tables"
ALPHA = 0.1
RUNS = [("holdout_mhd", "JHTDB_mhd1024_hd"), ("holdout_channel", "JHTDB_channel_hd"),
        ("holdout_uniform", "uniform")]
NAMES = {"mean_speed": "平均速度", "kinetic_energy_proxy": "动能代理", "circulation": "环量",
         "enstrophy": "拟涡能", "continuity_rms": "连续性残差", "mean_strain_rate": "平均应变率"}


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


def main() -> None:
    report = {}
    for folder, domain in RUNS:
        path = ROOT / "piv_runs" / folder / "study_test.npz"
        if not path.exists():
            print("missing", folder)
            continue
        data = np.load(path, allow_pickle=True)
        meta = json.loads((path.parent / "study_test_meta.json").read_text(encoding="utf-8"))
        truth_all = data["q_truth"]
        rng = np.random.default_rng(0)
        entry = {"n": int(truth_all.shape[0]), "quantities": {}}
        for column, key in enumerate(meta["quantities"]):
            truth = truth_all[:, column].astype(np.float64)
            gate = data["q_cand__hodge1.00"][:, :, column].astype(np.float64)
            none = data["q_cand__pre"][:, :, column].astype(np.float64)
            pg, sg = gate.mean(axis=1), gate.std(axis=1)
            pn, sn = none.mean(axis=1), none.std(axis=1)
            diffs = []
            for b in range(200):
                pick = rng.integers(0, truth.size, truth.size)
                diffs.append(ratio(pg[pick], sg[pick], truth[pick], seed=b)
                             - ratio(pn[pick], sn[pick], truth[pick], seed=b))
            entry["quantities"][key] = {
                "name": NAMES.get(key, key),
                "R_none": ratio(pn, sn, truth), "R_gate": ratio(pg, sg, truth),
                "ci90": [float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))],
            }
        report[domain] = entry

    (OUT / "holdout_all.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for domain, entry in report.items():
        print(f"--- {domain} (n={entry['n']})")
        for key, q in entry["quantities"].items():
            print(f"  {q['name']:8s} {q['R_none']:5.2f} -> {q['R_gate']:5.2f}  "
                  f"[{q['ci90'][0]:+.2f}, {q['ci90'][1]:+.2f}]")


if __name__ == "__main__":
    main()
