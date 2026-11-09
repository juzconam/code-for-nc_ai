"""Sensitivity of the calibration split to temporal blocking.

Conformal calibration needs the calibration and evaluation samples to be exchangeable.  If frames
come from a continuous sequence, a random split puts near-duplicate neighbours on both sides.  The
manifest carries the frame number in the file name, so contiguous time blocks can be formed.  This
recomputes R and coverage under three splits: the hash split used in the paper, a contiguous
first-half/second-half split, and an alternating-block split.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "piv_runs" / "uncertainty_study"
OUT = ROOT / "paper_figures" / "uncertainty_study" / "tables"
MANIFEST = ROOT / "splits" / "class1_primary_split_v2.json"
ALPHA = 0.1
BLOCKS = 10


def conformal_quantile(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(np.ceil((scores.size + 1) * (1 - alpha))) - 1
    return float(scores[min(max(k, 0), scores.size - 1)])


def evaluate(point, scale, truth, cal, ev):
    err = np.abs(truth[cal] - point[cal])
    qc = conformal_quantile(err, ALPHA)
    qs = conformal_quantile(err / (scale[cal] + 1e-12), ALPHA)
    ratio = np.mean(2 * qc) / max(np.mean(2 * qs * scale[ev]), 1e-12)
    lower = point[ev] - qs * scale[ev]
    upper = point[ev] + qs * scale[ev]
    coverage = float(np.mean((truth[ev] >= lower) & (truth[ev] <= upper)))
    return float(ratio), coverage


def main():
    data = np.load(RUNS / "study_test.npz", allow_pickle=True)
    meta = json.loads((RUNS / "study_test_meta.json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    frame = {}
    for sample in manifest["samples"]:
        match = re.search(r"_(\d+)_img1", sample["image1_name"])
        if match:
            frame[sample["sample_uid"]] = int(match.group(1))

    uid = [str(u) for u in data["uid"]]
    domain = data["domain"]
    truth = data["q_truth"]
    point = data["q_cand__hodge1.00"].mean(axis=1)
    scale = data["q_cand__hodge1.00"].std(axis=1)

    n = len(uid)
    order_hash = np.random.default_rng(0).permutation(n)

    # contiguous blocks by frame number, within domain
    block_id = np.zeros(n, dtype=int)
    for name in sorted(set(domain)):
        idx = np.array([i for i in range(n) if domain[i] == name])
        idx_sorted = idx[np.argsort([frame.get(uid[i], 0) for i in idx])]
        for position, i in enumerate(idx_sorted):
            block_id[i] = int(position * BLOCKS / max(len(idx_sorted), 1))

    splits = {
        "hash (paper)": (order_hash[: n // 2], order_hash[n // 2:]),
        "contiguous blocks": (np.where(block_id < BLOCKS // 2)[0], np.where(block_id >= BLOCKS // 2)[0]),
        "alternating blocks": (np.where(block_id % 2 == 0)[0], np.where(block_id % 2 == 1)[0]),
    }

    report = {}
    for column, key in enumerate(meta["quantities"]):
        entry = {}
        for name, (cal, ev) in splits.items():
            ratio, coverage = evaluate(point[:, column].astype(np.float64),
                                       scale[:, column].astype(np.float64),
                                       truth[:, column].astype(np.float64), cal, ev)
            entry[name] = {"R": ratio, "coverage": coverage}
        report[key] = entry

    (OUT / "split_sensitivity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'quantity':22s} " + " ".join(f"{k:>26s}" for k in splits))
    for key, entry in report.items():
        cells = [f"{entry[k]['R']:.2f} / {entry[k]['coverage']:.2f}" for k in splits]
        print(f"{key:22s} " + " ".join(f"{c:>26s}" for c in cells))


if __name__ == "__main__":
    main()
