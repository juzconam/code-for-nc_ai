"""Permutation null and frozen-reference check for the gate strength.

Two questions from the referee:

  (1) the gate strength is built with a median taken over the evaluation set, i.e. it depends on the
      whole batch.  If the reference is frozen on a calibration half, does the effect survive?
  (2) the "matched" control used a single permutation.  Here the real assignment is compared with
      20 random permutations, giving a null distribution for R under "same marginal, no pairing".

Both are answered offline from the saved fields, so the candidates are identical across variants.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "paper_figures" / "uncertainty_study" / "tables"
QUANTITIES = ["divergence_rms", "mean_strain_rate", "mean_speed", "enstrophy"]
ALPHA = 0.1
N_PERM = 20


def scalars(u, v):
    speed = np.sqrt(u ** 2 + v ** 2)
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    vort = dv_dx - du_dy
    strain = np.sqrt(du_dx ** 2 + dv_dy ** 2 + 0.5 * (du_dy + dv_dx) ** 2)
    div = du_dx + dv_dy
    return {"mean_speed": float(speed.mean()),
            "enstrophy": float(0.5 * (vort ** 2).mean()),
            "mean_strain_rate": float(strain.mean()),
            "divergence_rms": float(np.sqrt((div ** 2).mean()))}


def conformal_quantile(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(np.ceil((scores.size + 1) * (1 - alpha))) - 1
    return float(scores[min(max(k, 0), scores.size - 1)])


def ratio(point, scale, truth, repeats=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(truth.size)
    out = []
    for _ in range(repeats):
        perm = rng.permutation(idx)
        h = perm.size // 2
        cal, ev = perm[:h], perm[h:]
        err = np.abs(truth[cal] - point[cal])
        qc = conformal_quantile(err, ALPHA)
        qs = conformal_quantile(err / (scale[cal] + 1e-12), ALPHA)
        out.append(np.mean(2 * qc) / max(np.mean(2 * qs * scale[ev]), 1e-12))
    return float(np.mean(out))


def points_spreads(cand, proj, lam):
    """Blend candidates at strength lam per field, return per-field point and spread per quantity."""
    n, k = cand.shape[:2]
    out_p = {q: np.zeros(n) for q in QUANTITIES}
    out_s = {q: np.zeros(n) for q in QUANTITIES}
    for i in range(n):
        vals = {q: [] for q in QUANTITIES}
        for c in range(k):
            u = (1 - lam[i]) * cand[i, c, 0] + lam[i] * proj[i, c, 0]
            v = (1 - lam[i]) * cand[i, c, 1] + lam[i] * proj[i, c, 1]
            s = scalars(u, v)
            for q in QUANTITIES:
                vals[q].append(s[q])
        for q in QUANTITIES:
            out_p[q][i] = float(np.mean(vals[q]))
            out_s[q][i] = float(np.std(vals[q]))
    return out_p, out_s


def main():
    cache_path = TABLES / "permutation_fields.npz"
    if not cache_path.exists():
        print("SKIP: permutation_fields.npz is not included in this archive; "
              "the precomputed null distribution is in permutation_1000.json.")
        return
    data = np.load(cache_path)
    models = sorted({key.split("::")[0] for key in data.files})
    report = {}

    for model in models:
        cand = data[f"{model}::cand"]
        proj = data[f"{model}::proj"]
        div = data[f"{model}::div"].astype(np.float64)
        truth_fields = data[f"{model}::truth"]
        truth = {q: np.array([scalars(t[0], t[1])[q] for t in truth_fields]) for q in QUANTITIES}

        n = div.size
        rng = np.random.default_rng(0)
        cal_idx = rng.permutation(n)[: n // 2]

        lam_full = np.clip(div / (np.median(div) + 1e-12) - 1, 0, 1)
        lam_frozen = np.clip(div / (np.median(div[cal_idx]) + 1e-12) - 1, 0, 1)
        lam_zero = np.zeros(n)

        variants = {
            "no projection": lam_zero,
            "gate (evaluation-set median)": lam_full,
            "gate (frozen calibration median)": lam_frozen,
        }
        for r in range(N_PERM):
            variants[f"permuted #{r}"] = rng.permutation(lam_full)

        entry = {"lam_mean": {"full": float(lam_full.mean()), "frozen": float(lam_frozen.mean())},
                 "variants": {}}
        for name, lam in variants.items():
            p, s = points_spreads(cand, proj, lam)
            entry["variants"][name] = {q: ratio(p[q], s[q], truth[q]) for q in QUANTITIES}
        report[model] = entry

    (TABLES / "permutation_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for model, entry in report.items():
        print(f"=== {model}  (λ mean full {entry['lam_mean']['full']:.3f}, "
              f"frozen {entry['lam_mean']['frozen']:.3f})")
        base = entry["variants"]["no projection"]["divergence_rms"]
        gate = entry["variants"]["gate (evaluation-set median)"]["divergence_rms"]
        frozen = entry["variants"]["gate (frozen calibration median)"]["divergence_rms"]
        perms = [entry["variants"][f"permuted #{r}"]["divergence_rms"] for r in range(N_PERM)]
        perms = np.array(perms)
        print(f"  divergence_rms: none {base:.2f}  gate {gate:.2f}  frozen-gate {frozen:.2f}")
        print(f"  permuted null: mean {perms.mean():.2f}  [{np.percentile(perms,5):.2f}, "
              f"{np.percentile(perms,95):.2f}]   P(null <= gate) = "
              f"{np.mean(perms <= gate):.2f}")
        print(f"  separation gate vs null: {perms.mean() - gate:+.2f}")


if __name__ == "__main__":
    main()
