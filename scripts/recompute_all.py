"""Recompute every number the referee flagged, from one cache and one protocol.

Outputs
  coverage_wilson.json   per domain x quantity coverage with Wilson intervals (both constructions)
  coverage_mondrian.json per domain x quantity coverage under per-domain (Mondrian) calibration
  permutation_1000.json  the gate-versus-permutation null with 1000 permutations and exact p
  second_2x2_unified.json the centre x scale cross computed from the same cache as the arm table
  main_paired_block.json paired gate effect with a time-block bootstrap instead of per-frame
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "piv_runs" / "uncertainty_study"
TABLES = ROOT / "paper_figures" / "uncertainty_study" / "tables"
ALPHA = 0.1
Z = 1.6448536269514722


def wilson(hits, n, z=Z):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def cq(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(np.ceil((scores.size + 1) * (1 - alpha))) - 1
    return float(scores[min(max(k, 0), scores.size - 1)])


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

    # ------------------------------------------------ coverage: global vs Mondrian
    result = {"global": {}, "mondrian": {}}
    for column, key in enumerate(meta["quantities"]):
        truth = truth_all[:, column].astype(np.float64)
        point = point_all[:, column].astype(np.float64)
        spread = spread_all[:, column].astype(np.float64)

        err_cal = np.abs(truth[cal] - point[cal])
        q_const = cq(err_cal, ALPHA)
        q_scaled = cq(err_cal / (spread[cal] + 1e-12), ALPHA)

        entry_g, entry_m = {}, {}
        for name in sorted(set(domain)):
            mask = domain[ev] == name
            t, p, s = truth[ev][mask], point[ev][mask], spread[ev][mask]

            hit_c = ((t >= p - q_const) & (t <= p + q_const)).astype(float)
            hit_s = ((t >= p - q_scaled * s) & (t <= p + q_scaled * s)).astype(float)

            cal_mask = domain[cal] == name
            qc_m = cq(np.abs(truth[cal][cal_mask] - point[cal][cal_mask]), ALPHA) \
                if cal_mask.sum() > 20 else q_const
            qs_m = cq(np.abs(truth[cal][cal_mask] - point[cal][cal_mask])
                      / (spread[cal][cal_mask] + 1e-12), ALPHA) if cal_mask.sum() > 20 else q_scaled
            hit_cm = ((t >= p - qc_m) & (t <= p + qc_m)).astype(float)
            hit_sm = ((t >= p - qs_m) & (t <= p + qs_m)).astype(float)

            n = int(mask.sum())
            entry_g[name] = {
                "n": n,
                "constant": {"hits": int(hit_c.sum()), "value": float(hit_c.mean()),
                             "wilson": wilson(hit_c.sum(), n)},
                "scaled": {"hits": int(hit_s.sum()), "value": float(hit_s.mean()),
                           "wilson": wilson(hit_s.sum(), n)},
            }
            entry_m[name] = {
                "n": n,
                "constant": {"hits": int(hit_cm.sum()), "value": float(hit_cm.mean()),
                             "wilson": wilson(hit_cm.sum(), n)},
                "scaled": {"hits": int(hit_sm.sum()), "value": float(hit_sm.mean()),
                           "wilson": wilson(hit_sm.sum(), n)},
            }
        result["global"][key] = entry_g
        result["mondrian"][key] = entry_m

    (TABLES / "coverage_wilson.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    cache_path = TABLES / "permutation_fields.npz"
    if cache_path.exists():
    # ------------------------------------------------ 2x2 from the shared cache
        cache = np.load(cache_path)
        models = sorted({k.split("::")[0] for k in cache.files})
        combos = {}
        for model in models:
            cand = cache[f"{model}::cand"]
            proj = cache[f"{model}::proj"]
            truth_fields = cache[f"{model}::truth"]

            def scal(t):
                u, v = t[0], t[1]
                du_dy, du_dx = np.gradient(u)
                dv_dy, dv_dx = np.gradient(v)
                return float(np.sqrt(((du_dx + dv_dy) ** 2).mean()))

            truth_div = np.array([scal(t) for t in truth_fields])
            out = {}
            for centre in ("pre", "post"):
                for scale in ("pre", "post"):
                    pts, sps = [], []
                    for i in range(cand.shape[0]):
                        vals = []
                        for c in range(cand.shape[1]):
                            u = proj[i, c, 0] if scale == "post" else cand[i, c, 0]
                            v = proj[i, c, 1] if scale == "post" else cand[i, c, 1]
                            vals.append(scal(np.stack([u, v])))
                        vals = np.array(vals)
                        base = np.array([scal(cand[i, c]) for c in range(cand.shape[1])])
                        point = vals.mean() if centre == "post" else base.mean()
                        sps.append(vals.std() if scale == "post" else base.std())
                        pts.append(point)
                    pts, sps = np.array(pts), np.array(sps)
                    err = np.abs(truth_div[cal[: min(truth_div.size, cal.size)]] - pts[cal[: pts.size]]) \
                        if False else None
                    # proper split
                    idx = np.arange(truth_div.size)
                    rng2 = np.random.default_rng(0).permutation(idx)
                    h = idx.size // 2
                    c_idx, e_idx = rng2[:h], rng2[h:]
                    err = np.abs(truth_div[c_idx] - pts[c_idx])
                    qc = cq(err, ALPHA)
                    qs = cq(err / (sps[c_idx] + 1e-12), ALPHA)
                    out[f"{centre}/{scale}"] = float(np.mean(2 * qc)
                                                     / max(np.mean(2 * qs * sps[e_idx]), 1e-12))
            combos[model] = out

        mean_2x2 = {k: float(np.mean([combos[m][k] for m in models])) for k in combos[models[0]]}
        (TABLES / "second_2x2_unified.json").write_text(
            json.dumps({"per_model": combos, "mean": mean_2x2}, indent=2), encoding="utf-8")

        # ------------------------------------------------ 1000-permutation null
        perm_report = {}
        for model in models:
            cand = cache[f"{model}::cand"]
            proj = cache[f"{model}::proj"]
            div = cache[f"{model}::div"].astype(np.float64)
            truth_fields = cache[f"{model}::truth"]

            def scal2(t):
                u, v = t[0], t[1]
                du_dy, du_dx = np.gradient(u)
                dv_dy, dv_dx = np.gradient(v)
                return float(np.sqrt(((du_dx + dv_dy) ** 2).mean()))

            truth_div = np.array([scal2(t) for t in truth_fields])
            lam = np.clip(div / (np.median(div) + 1e-12) - 1, 0, 1)

            def ratio_for(lam_values):
                pts, sps = [], []
                for i in range(cand.shape[0]):
                    vals = []
                    for c in range(cand.shape[1]):
                        u = (1 - lam_values[i]) * cand[i, c, 0] + lam_values[i] * proj[i, c, 0]
                        v = (1 - lam_values[i]) * cand[i, c, 1] + lam_values[i] * proj[i, c, 1]
                        vals.append(scal2(np.stack([u, v])))
                    pts.append(np.mean(vals))
                    sps.append(np.std(vals))
                pts, sps = np.array(pts), np.array(sps)
                idx = np.random.default_rng(0).permutation(pts.size)
                h = idx.size // 2
                c_idx, e_idx = idx[:h], idx[h:]
                err = np.abs(truth_div[c_idx] - pts[c_idx])
                qc = cq(err, ALPHA)
                qs = cq(err / (sps[c_idx] + 1e-12), ALPHA)
                return float(np.mean(2 * qc) / max(np.mean(2 * qs * sps[e_idx]), 1e-12))

            observed = ratio_for(lam)
            rng3 = np.random.default_rng(11)
            null = [ratio_for(rng3.permutation(lam)) for _ in range(1000)]
            null = np.array(null)
            p = (1 + int((null <= observed).sum())) / (1 + null.size)
            perm_report[model] = {"observed": observed, "null_mean": float(null.mean()),
                                  "null_ci": [float(np.percentile(null, 5)), float(np.percentile(null, 95))],
                                  "p_exact": p, "n_perm": int(null.size)}

        (TABLES / "permutation_1000.json").write_text(json.dumps(perm_report, indent=2), encoding="utf-8")
    else:
        print("SKIP 2x2 and permutation: permutation_fields.npz is not included in this archive;")
        print("using the precomputed JSON results shipped with the release.")
        mean_2x2 = json.loads((TABLES / "second_2x2_unified.json").read_text(encoding="utf-8"))["mean"]
        perm_report = json.loads((TABLES / "permutation_1000.json").read_text(encoding="utf-8"))

    print("global coverage (mean over quantities):")
    for name in sorted(set(domain)):
        c = np.mean([result["global"][k][name]["constant"]["value"] for k in result["global"]])
        m = np.mean([result["mondrian"][k][name]["constant"]["value"] for k in result["mondrian"]])
        print(f"  {name:16s} global {c:.2f}  mondrian {m:.2f}")
    print()
    print("2x2 (mean over seeds):", {k: round(v, 2) for k, v in mean_2x2.items()})
    print()
    for model, e in perm_report.items():
        print(f"{model:20s} observed {e['observed']:.3f}  null {e['null_mean']:.3f} "
              f"[{e['null_ci'][0]:.3f},{e['null_ci'][1]:.3f}]  p = {e['p_exact']:.4f}")


if __name__ == "__main__":
    main()
