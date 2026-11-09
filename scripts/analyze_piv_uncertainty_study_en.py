"""Offline analysis of the constraint/tension study.

Everything here reads the per-sample records written by
``scripts/run_piv_uncertainty_study.py``; no model is run.  That separation is
deliberate: the calibration and the coverage evaluation must be auditable and
must be re-runnable at any ``alpha`` or ``K`` without touching the network.

Outputs (under ``--output-dir``):

* ``tables/*.csv`` and ``tables/*.md`` - every number quoted in the report
* ``figures/*.png``                    - the figures proposed in the revision plan
* ``summary.md``                       - the human-readable digest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import nature_style

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.conformal import (
    conformal_quantile,
    evaluate_intervals,
    fit_split_conformal,
    per_group_rows,
    predict_interval,
    scale_reliability_rows,
    spearman,
    top_bottom_error_ratio,
    wilson_interval,
)

QUANTITIES = (
    "mean_speed",
    "kinetic_energy_proxy",
    "circulation",
    "enstrophy",
    "continuity_rms",
    "mean_strain_rate",
)
PAPER_QUANTITIES = ("mean_speed", "enstrophy", "mean_strain_rate", "continuity_rms")
LABELS = {
    "mean_speed": "Mean speed",
    "enstrophy": "Enstrophy",
    "mean_strain_rate": "Mean strain rate",
    "continuity_rms": "Continuity residual",
    "kinetic_energy_proxy": "Kinetic-energy proxy",
    "circulation": "Circulation",
}
EPSILON = 1e-8
SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_split(study_dir: Path, split: str) -> dict:
    path = study_dir / f"study_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = dict(np.load(path, allow_pickle=False))
    meta_path = study_dir / f"study_{split}_meta.json"
    data["__meta__"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return data


def quantity_index(quantity: str) -> int:
    return QUANTITIES.index(quantity)


def holdout_halves(dataset: dict, seed: int = 20261018):
    """Deterministically split one study file into (calibration, evaluation) halves.

    The two halves come from *the same* mixture, so they are exchangeable by
    construction.  When the frozen split offers no calibration set that is both
    independent of model selection and representative of the test distribution,
    this is the honest way to obtain one.
    """
    import hashlib

    uids = [str(uid) for uid in dataset["uid"]]
    order = sorted(
        range(len(uids)),
        key=lambda index: hashlib.sha256(f"{seed}|{uids[index]}".encode()).hexdigest(),
    )
    half = len(order) // 2
    first = np.asarray(sorted(order[:half]))
    second = np.asarray(sorted(order[half:]))
    calibrate, evaluate = {}, {}
    for key, value in dataset.items():
        if key == "__meta__":
            calibrate[key] = value
            evaluate[key] = value
        elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(uids):
            calibrate[key] = value[first]
            evaluate[key] = value[second]
        else:
            calibrate[key] = value
            evaluate[key] = value
    return calibrate, evaluate


def k_index_of(meta: dict, k: int) -> int:
    return int(meta["k_grid"].index(int(k)))


# ---------------------------------------------------------------------------
# Method definitions
# ---------------------------------------------------------------------------
class Method:
    """One way of turning a frozen pipeline into an interval."""

    def __init__(self, name, point_arm, spread_arm=None, kind="scaled", label="",
                 floor_quantile=None, cap_quantile=None, mondrian=False):
        self.name = name
        self.point_arm = point_arm
        self.spread_arm = spread_arm
        self.kind = kind
        self.label = label or name
        self.floor_quantile = floor_quantile
        self.cap_quantile = cap_quantile
        self.mondrian = bool(mondrian)

    def raw_scale(self, dataset, quantity, k=None):
        meta = dataset["__meta__"]
        index = quantity_index(quantity)
        if self.spread_arm is None:
            return None
        if self.spread_arm.startswith("mc:"):
            tag_arm, tag = self.spread_arm[3:].split("|")
            return dataset[f"mc_std__{tag_arm}__{tag}"][:, index].astype(np.float64)
        k_value = int(meta["candidates"]) if k is None else int(k)
        return dataset[f"q_cand__{self.spread_arm}"][:, : int(k_value), index].std(axis=1).astype(np.float64)

    def scale(self, dataset, quantity, k=None):
        """Scale after the (calibration-fitted) floor / cap robustification."""
        scale = self.raw_scale(dataset, quantity, k)
        return apply_scale_limits(scale, self, quantity)


# Calibration-fitted scale limits.  A floor/cap is a deterministic monotone
# transform of s(x) fitted on the calibration split only, so the split-conformal
# guarantee still holds; it is the obvious repair for a dispersion that collapses
# to near zero or explodes on a few samples, and it is reported separately so the
# main table is not silently tuned.
_SCALE_LIMITS = {}


def apply_scale_limits(scale, method: Method, quantity: str):
    if scale is None:
        return None
    low = _SCALE_LIMITS.get((method.name, quantity, "floor"))
    high = _SCALE_LIMITS.get((method.name, quantity, "cap"))
    if low is not None:
        scale = np.maximum(scale, low)
    if high is not None:
        scale = np.minimum(scale, high)
    return scale


def register_scale_limits(cal, methods):
    for method in methods:
        if method.floor_quantile is None and method.cap_quantile is None:
            continue
        for quantity in QUANTITIES:
            raw = method.raw_scale(cal, quantity)
            if raw is None:
                continue
            if method.floor_quantile is not None:
                _SCALE_LIMITS[(method.name, quantity, "floor")] = float(
                    np.quantile(raw, method.floor_quantile)
                )
            if method.cap_quantile is not None:
                _SCALE_LIMITS[(method.name, quantity, "cap")] = float(
                    np.quantile(raw, method.cap_quantile)
                )


METHODS = (
    Method("raw_candidate_interval", "pre", None, kind="quantile",
           label="Candidate 5-95 % interval (published base)"),
    Method("legacy_additive_radius", "pre", None, kind="additive",
           label="Candidate quantiles + additive validation radius (published)"),
    Method("constant_split_conformal", "hodge1.00", None, kind="scaled",
           label="Split conformal, constant width"),
    Method("pivd_pre_dispersion", "pre", "pre", kind="scaled",
           label="Split conformal, pre-intervention dispersion (PIVID-D)"),
    Method("method_A_pre_dispersion", "hodge1.00", "pre", kind="scaled",
           label="Method A: post point estimate + pre dispersion"),
    Method("post_dispersion", "hodge1.00", "hodge1.00", kind="scaled",
           label="Split conformal, post-intervention dispersion (PIVID)"),
    Method("pixel_mc_photometric", "hodge1.00", "mc:hodge1.00|photometric", kind="scaled",
           label="Photometric per-pixel MC propagation + conformal"),
    Method("pixel_mc_candidate", "hodge1.00", "mc:hodge1.00|candidate", kind="scaled",
           label="Candidate per-pixel MC propagation + conformal"),
    Method("method_B_certificate", "hodge1.00", "pre", kind="certified",
           label="Method B: pre dispersion + Lipschitz bias budget"),
    Method("post_dispersion_floored", "hodge1.00", "hodge1.00", kind="scaled",
           label="Post dispersion with calibration 10th-percentile floor",
           floor_quantile=0.10),
    Method("pre_dispersion_floored", "hodge1.00", "pre", kind="scaled",
           label="Pre dispersion with calibration 10th-percentile floor",
           floor_quantile=0.10),
    Method("pixel_mc_photometric_capped", "hodge1.00", "mc:hodge1.00|photometric", kind="scaled",
           label="Photometric MC with calibration 90th-percentile cap",
           cap_quantile=0.90),
    Method("constant_mondrian", "hodge1.00", None, kind="scaled",
           label="Constant width, one quantile per domain (Mondrian)", mondrian=True),
    Method("method_A_mondrian", "hodge1.00", "pre", kind="scaled",
           label="Method A with one quantile per domain (Mondrian)", mondrian=True),
)
METHOD_BY_NAME = {method.name: method for method in METHODS}
ROBUST_METHODS = ("post_dispersion_floored", "pre_dispersion_floored", "pixel_mc_photometric_capped")
MONDRIAN_METHODS = ("constant_mondrian", "method_A_mondrian")


def point_values(dataset, method: Method, quantity: str, k=None):
    meta = dataset["__meta__"]
    k_value = int(meta["candidates"]) if k is None else int(k)
    return dataset[f"q_point__{method.point_arm}"][
        :, k_index_of(meta, k_value), quantity_index(quantity)
    ].astype(np.float64)


def truth_values(dataset, quantity):
    return dataset["q_truth"][:, quantity_index(quantity)].astype(np.float64)


def lipschitz_per_sample(dataset, quantity):
    if quantity in {"mean_speed"}:
        return np.ones(dataset["q_truth"].shape[0], dtype=np.float64)
    if quantity in {"continuity_rms", "mean_strain_rate"}:
        return np.full(dataset["q_truth"].shape[0], SQRT2, dtype=np.float64)
    if quantity == "enstrophy":
        return SQRT2 * (dataset["max_vort_pre"].astype(np.float64) + SQRT2 * dataset["delta__hodge1.00"])
    return np.full(dataset["q_truth"].shape[0], np.nan)


def interval_predictions(cal, test, method: Method, quantity: str, alpha: float, k=None):
    """Fit on ``cal`` only and return ``(lower, upper)`` on ``test``."""
    index = quantity_index(quantity)
    truth_cal = truth_values(cal, quantity)
    if method.kind == "quantile":
        lower = test[f"q05__{method.point_arm}"][:, index].astype(np.float64)
        upper = test[f"q95__{method.point_arm}"][:, index].astype(np.float64)
        return lower, upper, {"q": float("nan"), "n_calibration": 0}
    if method.kind == "additive":
        lower_cal = cal[f"q05__{method.point_arm}"][:, index].astype(np.float64)
        upper_cal = cal[f"q95__{method.point_arm}"][:, index].astype(np.float64)
        radius = conformal_quantile(np.maximum(lower_cal - truth_cal, truth_cal - upper_cal), alpha)
        lower = test[f"q05__{method.point_arm}"][:, index].astype(np.float64) - radius
        upper = test[f"q95__{method.point_arm}"][:, index].astype(np.float64) + radius
        return lower, upper, {"q": radius, "n_calibration": int(truth_cal.size)}
    point_cal = point_values(cal, method, quantity, k)
    point_test = point_values(test, method, quantity, k)
    scale_cal = method.scale(cal, quantity, k)
    scale_test = method.scale(test, quantity, k)
    if method.mondrian:
        return _mondrian_predictions(cal, test, method, quantity, alpha,
                                     point_cal, point_test, scale_cal, scale_test)
    fit = fit_split_conformal(point_cal, truth_cal, scale=scale_cal, alpha=alpha,
                              epsilon=EPSILON, method=method.name)
    if method.kind == "certified":
        radius = (
            fit.q * (scale_test + EPSILON)
            + lipschitz_per_sample(test, quantity) * test["delta__hodge1.00"].astype(np.float64)
        )
        lower, upper = point_test - radius, point_test + radius
    else:
        lower, upper = predict_interval(fit, point_test, scale_test)
    return lower, upper, {"q": fit.q, "n_calibration": fit.n_calibration}


def fit_and_evaluate(cal, test, method: Method, quantity: str, alpha: float, k=None):
    """Fit on ``cal`` and evaluate on ``test``; returns a ``CoverageReport``."""
    lower, upper, info = interval_predictions(cal, test, method, quantity, alpha, k)
    report = evaluate_intervals(
        quantity, method.name, truth_values(test, quantity), lower, upper, alpha,
        q=info.get("q", float("nan")), n_calibration=info.get("n_calibration", 0),
    )
    if info.get("notes"):
        report.notes = info["notes"]
    return report


def _mondrian_predictions(cal, test, method, quantity, alpha,
                          point_cal, point_test, scale_cal, scale_test):
    """Group-conditional (Mondrian) split conformal: one quantile per domain.

    Marginal coverage averages over the test distribution and can hide a domain
    whose errors are much larger than the rest.  Fitting ``Q`` per domain restores
    group-conditional coverage at the price of a wider interval in the hard domain.
    """
    truth_cal = truth_values(cal, quantity)
    truth_test = truth_values(test, quantity)
    domain_cal = np.asarray(cal["domain"]).ravel()
    domain_test = np.asarray(test["domain"]).ravel()
    lower = np.full(truth_test.shape, np.nan)
    upper = np.full(truth_test.shape, np.nan)
    quantiles, counts = {}, {}
    for domain in sorted(set(domain_test.tolist())):
        mask_cal = domain_cal == domain
        mask_test = domain_test == domain
        if not np.any(mask_cal) or not np.any(mask_test):
            continue
        fit = fit_split_conformal(
            point_cal[mask_cal], truth_cal[mask_cal],
            scale=None if scale_cal is None else scale_cal[mask_cal],
            alpha=alpha, epsilon=EPSILON, method=method.name,
        )
        lo, hi = predict_interval(
            fit, point_test[mask_test], None if scale_test is None else scale_test[mask_test]
        )
        lower[mask_test], upper[mask_test] = lo, hi
        quantiles[domain] = fit.q
        counts[domain] = fit.n_calibration
    if np.any(~np.isfinite(lower)):
        raise ValueError(f"Mondrian fit left uncovered domains for {method.name}/{quantity}")
    return lower, upper, {
        "q": float(np.mean(list(quantiles.values()))),
        "n_calibration": int(np.sum(list(counts.values()))),
        "notes": "; ".join(f"{domain}:Q={value:.4g},n={counts[domain]}"
                           for domain, value in quantiles.items()),
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def coverage_table(cal, test, alpha, quantities=PAPER_QUANTITIES, methods=METHODS):
    rows = []
    for method in methods:
        for quantity in quantities:
            try:
                report = fit_and_evaluate(cal, test, method, quantity, alpha)
            except KeyError:
                # The study directory was produced without the arrays this method
                # needs (e.g. a K-sweep run with --mc-draws 0); skip the row.
                continue
            row = report.as_row()
            row["method_label"] = method.label
            rows.append(row)
    return rows


def alpha_sweep_table(cal, test, quantities=PAPER_QUANTITIES, alphas=(0.01, 0.05, 0.10, 0.20),
                      methods=("constant_split_conformal", "pivd_pre_dispersion",
                               "method_A_pre_dispersion", "post_dispersion",
                               "pixel_mc_photometric", "raw_candidate_interval")):
    rows = []
    for method_name in methods:
        method = METHOD_BY_NAME[method_name]
        for quantity in quantities:
            for alpha in alphas:
                report = fit_and_evaluate(cal, test, method, quantity, alpha)
                row = report.as_row()
                row["alpha"] = float(alpha)
                rows.append(row)
    return rows


def k_sweep_table(cal, test, alpha, quantities=PAPER_QUANTITIES,
                  methods=("constant_split_conformal", "pivd_pre_dispersion", "post_dispersion",
                           "method_A_pre_dispersion")):
    k_grid = [int(k) for k in cal["__meta__"]["k_grid"]]
    rows = []
    for method_name in methods:
        method = METHOD_BY_NAME[method_name]
        for quantity in quantities:
            for k in k_grid:
                report = fit_and_evaluate(cal, test, method, quantity, alpha, k=k)
                row = report.as_row()
                row["k"] = int(k)
                rows.append(row)
    return rows


def reliability_table(test, quantities=PAPER_QUANTITIES,
                      arms=("pre", "hodge0.50", "hodge1.00", "hodge2.00")):
    rows, summary = [], []
    for arm in arms:
        for quantity in quantities:
            scale = test[f"q_cand__{arm}"][:, :, quantity_index(quantity)].std(axis=1).astype(np.float64)
            error = np.abs(
                test[f"q_point__{arm}"][:, -1, quantity_index(quantity)].astype(np.float64)
                - truth_values(test, quantity)
            )
            rows.extend(scale_reliability_rows(scale, error, quantity, arm, bins=5))
            with np.errstate(divide="ignore", invalid="ignore"):
                gaussian = float(np.mean(error <= 1.96 * scale))
            summary.append({
                "arm": arm,
                "quantity": quantity,
                "mean_scale": float(scale.mean()),
                "median_scale": float(np.median(scale)),
                "scale_cv": float(scale.std() / max(scale.mean(), EPSILON)),
                "scale_p95_over_p05": float(
                    np.quantile(scale, 0.95) / max(np.quantile(scale, 0.05), EPSILON)
                ),
                "mean_abs_error": float(error.mean()),
                "dispersion_over_error": float(scale.mean() / max(error.mean(), EPSILON)),
                "spearman_scale_error": spearman(scale, error),
                "top_over_bottom_quintile_error": top_bottom_error_ratio(scale, error, 0.2),
                "fraction_within_1.96_scale": gaussian,
                "n": int(scale.size),
            })
    return rows, summary


def dose_response_table(test):
    arms = ["pre", "hodge0.00", "hodge0.25", "hodge0.50", "hodge1.00", "hodge2.00", "hodge4.00",
            "hodge_only", "observation_only"]
    meta = test["__meta__"]
    rows = []
    for arm in arms:
        if f"epe__{arm}" not in test:
            continue
        row = {
            "arm": arm,
            "n": int(test["uid"].shape[0]),
            "epe": float(test[f"epe__{arm}"].mean()),
            "epe_rel_pct": float(100.0 * (test[f"epe__{arm}"].mean() / test["epe__pre"].mean() - 1.0)),
            "divergence_rms": float(test[f"div__{arm}"].mean()),
            "divergence_rel_pct": float(100.0 * (test[f"div__{arm}"].mean() / test["div__pre"].mean() - 1.0)),
            "realized_delta_rms": float(test[f"delta__{arm}"].mean()),
            "acceptance_rate": (
                float("nan") if arm == "pre" else float(test[f"accept__{arm}"].mean())
            ),
            "pixel_sigma_mean": float(test[f"sigpix__{arm}"].mean()),
            "unb": float(test[f"unb__{arm}"].mean()),
            "une": float(test[f"une__{arm}"].mean()),
        }
        for quantity in PAPER_QUANTITIES:
            index = quantity_index(quantity)
            spread = test[f"q_cand__{arm}"][:, :, index].std(axis=1).astype(np.float64)
            point = test[f"q_point__{arm}"][:, -1, index].astype(np.float64)
            error = np.abs(point - truth_values(test, quantity))
            row[f"{quantity}_spread"] = float(spread.mean())
            row[f"{quantity}_abs_error"] = float(error.mean())
            row[f"{quantity}_spread_over_error"] = float(spread.mean() / max(error.mean(), EPSILON))
            row[f"{quantity}_spearman"] = spearman(spread, error)
            row[f"{quantity}_within_1.96"] = float(np.mean(error <= 1.96 * spread))
            lower = test[f"q05__{arm}"][:, index].astype(np.float64)
            upper = test[f"q95__{arm}"][:, index].astype(np.float64)
            truth = truth_values(test, quantity)
            row[f"{quantity}_naive_width"] = float((upper - lower).mean())
            row[f"{quantity}_naive_coverage"] = float(np.mean((truth >= lower) & (truth <= upper)))
        rows.append(row)
    return rows


def per_domain_table(cal, test, alpha, quantities=PAPER_QUANTITIES,
                     methods=("constant_split_conformal", "method_A_pre_dispersion",
                              "pivd_pre_dispersion", "post_dispersion",
                              "constant_mondrian", "method_A_mondrian")):
    domains = np.asarray(test["domain"]).ravel()
    rows = []
    for method_name in methods:
        method = METHOD_BY_NAME[method_name]
        for quantity in quantities:
            truth = truth_values(test, quantity)
            lower, upper, _ = interval_predictions(cal, test, method, quantity, alpha)
            rows.extend(per_group_rows(quantity, method_name, truth, lower, upper, domains, alpha))
    # marginal reference rows
    for method_name in methods:
        method = METHOD_BY_NAME[method_name]
        for quantity in quantities:
            report = fit_and_evaluate(cal, test, method, quantity, alpha)
            row = report.as_row()
            row["group"] = "ALL"
            rows.append(row)
    return rows


def conditional_coverage_summary(domain_rows, alpha):
    """Worst and best domain coverage per quantity for the marginal-only methods."""
    summary = []
    methods = ("constant_split_conformal", "method_A_pre_dispersion", "post_dispersion",
               "constant_mondrian", "method_A_mondrian")
    for quantity in PAPER_QUANTITIES:
        for method in methods:
            selected = [row for row in domain_rows
                        if row["quantity"] == quantity and row["method"] == method
                        and row["group"] != "ALL" and row["n"] > 0]
            marginal = [row for row in domain_rows
                        if row["quantity"] == quantity and row["method"] == method
                        and row["group"] == "ALL"]
            if not selected or not marginal:
                continue
            worst = min(selected, key=lambda row: row["coverage_pct"])
            best = max(selected, key=lambda row: row["coverage_pct"])
            summary.append({
                "quantity": quantity,
                "method": method,
                "marginal_pct": marginal[0]["coverage_pct"],
                "worst_domain": worst["group"],
                "worst_pct": worst["coverage_pct"],
                "worst_ci_low_pct": worst["coverage_ci_low_pct"],
                "best_domain": best["group"],
                "best_pct": best["coverage_pct"],
                "domain_spread_pt": best["coverage_pct"] - worst["coverage_pct"],
                "nominal_pct": 100 * (1 - alpha),
            })
    return summary


def _additive_radius(cal, quantity, arm, alpha):
    truth = truth_values(cal, quantity)
    lower = cal[f"q05__{arm}"][:, quantity_index(quantity)].astype(np.float64)
    upper = cal[f"q95__{arm}"][:, quantity_index(quantity)].astype(np.float64)
    return conformal_quantile(np.maximum(lower - truth, truth - upper), alpha)


def unb_une_table(test):
    arms = ["pre", "hodge0.00", "hodge0.25", "hodge0.50", "hodge1.00", "hodge2.00", "hodge4.00",
            "hodge_only", "observation_only"]
    rows = []
    for arm in arms:
        if f"unb__{arm}" not in test:
            continue
        rows.append({
            "arm": arm,
            "unb": float(np.mean(test[f"unb__{arm}"])),
            "une": float(np.mean(test[f"une__{arm}"])),
            "une_rms": float(np.mean(test[f"unerms__{arm}"])),
            "pixel_sigma_mean": float(np.mean(test[f"sigpix__{arm}"])),
            "n_samples": int(test[f"unb__{arm}"].size),
        })
    return rows


def pixel_scale_rank_table(test, arms=("pre", "hodge0.25", "hodge0.50", "hodge1.00", "hodge_only")):
    """Does the per-pixel dispersion rank *sample difficulty* even if it is mis-scaled?

    A globally mis-scaled dispersion can still be useful if it orders samples
    correctly; a per-sample Spearman correlation between mean pixel dispersion and
    EPE tests exactly that, independently of any calibration constant.
    """
    rows = []
    for arm in arms:
        key = f"sigpix__{arm}"
        if key not in test:
            continue
        sigma = test[key].astype(np.float64)
        epe = test[f"epe__{arm}"].astype(np.float64)
        rows.append({
            "arm": arm,
            "mean_pixel_sigma": float(sigma.mean()),
            "mean_epe": float(epe.mean()),
            "sigma_over_epe": float(sigma.mean() / max(epe.mean(), EPSILON)),
            "spearman_sigma_epe": spearman(sigma, epe),
            "top_over_bottom_quintile_epe": top_bottom_error_ratio(sigma, epe, 0.2),
            "n": int(sigma.size),
        })
    return rows


def repeated_holdout_table(full, alpha, repeats=20,
                           methods=("constant_split_conformal", "method_A_pre_dispersion",
                                    "post_dispersion"),
                           quantities=PAPER_QUANTITIES):
    """Repeat the random half-split and report the sampling distribution of coverage.

    A single split can land a couple of points away from nominal because the
    benchmark is block-structured (adjacent frames are correlated), so the
    effective calibration size is much smaller than the nominal one.  The mean over
    repeats is the decisive check that the construction itself is correct: split
    conformal must average to ``ceil((n+1)(1-alpha))/(n+1)``.
    """
    rows = []
    for method_name in methods:
        method = METHOD_BY_NAME[method_name]
        for quantity in quantities:
            coverages, widths = [], []
            for repeat in range(int(repeats)):
                cal, test = holdout_halves(full, seed=7919 + repeat)
                report = fit_and_evaluate(cal, test, method, quantity, alpha)
                coverages.append(report.coverage)
                widths.append(report.mean_width)
            coverages = np.asarray(coverages)
            rows.append({
                "method": method_name,
                "quantity": quantity,
                "nominal_pct": 100 * (1 - alpha),
                "repeats": int(repeats),
                "mean_coverage_pct": 100 * float(coverages.mean()),
                "sd_coverage_pt": 100 * float(coverages.std()),
                "min_coverage_pct": 100 * float(coverages.min()),
                "max_coverage_pct": 100 * float(coverages.max()),
                "fraction_below_nominal": float(np.mean(coverages < (1 - alpha))),
                "mean_width": float(np.mean(widths)),
            })
    return rows


def legacy_decomposition_table(cal, test, alpha, quantities=PAPER_QUANTITIES, arm="pre"):
    """Split the published width into its two parts: candidate interval + radius.

    The published ``interval_width`` is ``(q95 - q05) + 2 * radius``.  Reporting
    the two terms separately shows how much of the "90 % interval" is actually the
    candidate quantiles (almost nothing) and how much is a constant fitted on the
    validation split.
    """
    rows = []
    for quantity in quantities:
        index = quantity_index(quantity)
        truth_cal = truth_values(cal, quantity)
        lower_cal = cal[f"q05__{arm}"][:, index].astype(np.float64)
        upper_cal = cal[f"q95__{arm}"][:, index].astype(np.float64)
        radius = conformal_quantile(np.maximum(lower_cal - truth_cal, truth_cal - upper_cal), alpha)
        truth = truth_values(test, quantity)
        lower = test[f"q05__{arm}"][:, index].astype(np.float64)
        upper = test[f"q95__{arm}"][:, index].astype(np.float64)
        base_width = float((upper - lower).mean())
        total_width = float((upper + radius - (lower - radius)).mean())
        rows.append({
            "arm": arm,
            "quantity": quantity,
            "candidate_interval_width": base_width,
            "additive_radius": radius,
            "two_radii": 2.0 * radius,
            "total_width": total_width,
            "radius_share_of_width": float(2.0 * radius / max(total_width, EPSILON)),
            "coverage_candidate_interval_only": float(np.mean((truth >= lower) & (truth <= upper))),
            "coverage_with_radius": float(np.mean((truth >= lower - radius) & (truth <= upper + radius))),
            "n_calibration": int(truth_cal.size),
        })
    return rows


def tension_summary_table(coverage_rows, quantities=PAPER_QUANTITIES):
    """One row per quantity: how much does dispersion-scaling cost at matched coverage?

    `sharpness_gain_*` is ``width(constant) / width(method)``: greater than 1 means
    the dispersion sharpened the interval, below 1 means it made it worse than not
    using a scale at all.
    """
    lookup = {(row["quantity"], row["method"]): row for row in coverage_rows}
    rows = []
    for quantity in quantities:
        constant = lookup.get((quantity, "constant_split_conformal"))
        pre = lookup.get((quantity, "pivd_pre_dispersion"))
        post = lookup.get((quantity, "post_dispersion"))
        method_a = lookup.get((quantity, "method_A_pre_dispersion"))
        if constant is None:
            continue
        row = {
            "quantity": quantity,
            "width_constant": constant["mean_width"],
            "width_pre_dispersion": pre["mean_width"] if pre else float("nan"),
            "width_post_dispersion": post["mean_width"] if post else float("nan"),
            "width_method_A": method_a["mean_width"] if method_a else float("nan"),
            "coverage_constant_pct": constant["coverage_pct"],
            "coverage_post_pct": post["coverage_pct"] if post else float("nan"),
            "sharpness_gain_pre_dispersion": (
                constant["mean_width"] / pre["mean_width"] if pre and pre["mean_width"] > 0 else float("nan")
            ),
            "sharpness_gain_post_dispersion": (
                constant["mean_width"] / post["mean_width"] if post and post["mean_width"] > 0 else float("nan")
            ),
            "sharpness_gain_method_A": (
                constant["mean_width"] / method_a["mean_width"] if method_a and method_a["mean_width"] > 0 else float("nan")
            ),
            "post_over_pre_width": (
                post["mean_width"] / pre["mean_width"] if pre and post and pre["mean_width"] > 0 else float("nan")
            ),
        }
        rows.append(row)
    return rows


def published_wilson_check():
    published = [
        ("mean_speed", 0.9277, 0.05313),
        ("enstrophy", 0.9859, 0.003681),
        ("mean_strain_rate", 0.8873, 0.01797),
        ("continuity_rms", 0.8620, 0.03933),
    ]
    rows = []
    for quantity, coverage, width in published:
        k = int(round(coverage * 1065))
        low, high = wilson_interval(k, 1065)
        rows.append({
            "quantity": quantity,
            "published_coverage_pct": 100.0 * coverage,
            "covered_count": k,
            "n": 1065,
            "wilson_low_pct": 100.0 * low,
            "wilson_high_pct": 100.0 * high,
            "published_width": width,
            "above_90": bool(low >= 0.90),
        })
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
QUANTITY_TITLE = {
    "mean_speed": "Mean speed",
    "enstrophy": "Enstrophy",
    "mean_strain_rate": "Mean strain rate",
    "continuity_rms": "Continuity residual",
}


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nature_style
    nature_style.apply()
    return plt


def figure_dose_response(test, output_dir: Path):
    plt = _setup_matplotlib()
    # Only arms that realise a non-zero displacement are informative: hodge2/hodge4
    # are fully rejected by the no-harm gate and coincide with the origin.
    arms = ["pre", "hodge0.25", "hodge0.50", "hodge1.00"]
    delta = [float(np.mean(test[f"delta__{arm}"])) for arm in arms]
    rejected = [arm for arm in ("hodge2.00", "hodge4.00") if f"delta__{arm}" in test]
    rejected_acceptance = [float(np.mean(test[f"accept__{arm}"])) for arm in rejected]
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    ax = axes[0, 0]
    ax.plot(delta, [100 * (np.mean(test[f"div__{arm}"]) / np.mean(test["div__pre"]) - 1) for arm in arms],
            "o-", label="divergence RMS")
    ax.plot(delta, [100 * (np.mean(test[f"epe__{arm}"]) / np.mean(test["epe__pre"]) - 1) for arm in arms],
            "s-", label="EPE")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_xlabel("realised intervention RMS (px)")
    ax.set_ylabel("change vs. no intervention (%)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    index = quantity_index("continuity_rms")
    spread = [np.mean(test[f"q_cand__{arm}"][:, :, index].std(axis=1)) for arm in arms]
    error = [np.mean(np.abs(test[f"q_point__{arm}"][:, -1, index] - truth_values(test, "continuity_rms"))) for arm in arms]
    ax.plot(delta, spread, "o-", label="candidate spread $\\sigma_q$")
    ax.plot(delta, error, "s-", label="|point error|")
    ax.set_yscale("log")
    ax.set_xlabel("realised intervention RMS (px)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for quantity, colour in zip(PAPER_QUANTITIES, ("tab:blue", "tab:orange", "tab:green", "tab:red")):
        index = quantity_index(quantity)
        fraction = [
            float(np.mean(np.abs(test[f"q_point__{arm}"][:, -1, index] - truth_values(test, quantity))
                          <= 1.96 * test[f"q_cand__{arm}"][:, :, index].std(axis=1)))
            for arm in arms
        ]
        ax.plot(delta, 100 * np.asarray(fraction), "o-", label=LABELS[quantity], color=colour)
    ax.axhline(95, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("realised intervention RMS (px)")
    ax.set_ylabel("|error| <= 1.96 $\\sigma_q$ (%)")
    ax.set_ylim(0, 100)

    ax = axes[1, 1]
    for quantity, colour in zip(PAPER_QUANTITIES, ("tab:blue", "tab:orange", "tab:green", "tab:red")):
        index = quantity_index(quantity)
        ratio = []
        for arm in arms:
            sp = np.mean(test[f"q_cand__{arm}"][:, :, index].std(axis=1))
            er = np.mean(np.abs(test[f"q_point__{arm}"][:, -1, index] - truth_values(test, quantity)))
            ratio.append(sp / max(er, EPSILON))
        ax.plot(delta, ratio, "o-", label=LABELS[quantity], color=colour)
    ax.set_yscale("log")
    ax.set_xlabel("realised intervention RMS (px)")
    ax.set_ylabel("spread / |error|")
    ax.legend(frameon=False, ncol=2)
    if rejected:
        axes[0, 0].text(
            0.02, 0.05,
            "Hodge x2 and x4 rejected by the no-harm gate\n(acceptance "
            + ", ".join(f"{100 * value:.0f}%" for value in rejected_acceptance) + ")",
            transform=axes[0, 0].transAxes, fontsize=6, color="0.4",
        )
    figure.tight_layout()
    path = output_dir / "fig1_dose_response.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_calibration_curve(alpha_rows, output_dir: Path):
    plt = _setup_matplotlib()
    methods = ["constant_split_conformal", "pivd_pre_dispersion", "method_A_pre_dispersion",
               "post_dispersion", "pixel_mc_photometric"]
    colours = {"constant_split_conformal": "0.35", "pivd_pre_dispersion": "tab:blue",
               "method_A_pre_dispersion": "tab:purple", "post_dispersion": "tab:red",
               "pixel_mc_photometric": "tab:green"}
    markers = {"constant_split_conformal": "o", "pivd_pre_dispersion": "^",
               "method_A_pre_dispersion": "v", "post_dispersion": "D",
               "pixel_mc_photometric": "s"}
    figure, axes = plt.subplots(2, 4, figsize=(9.0, 4.6), sharex=True)
    for column, quantity in enumerate(PAPER_QUANTITIES):
        top, bottom = axes[0, column], axes[1, column]
        for method in methods:
            selected = [row for row in alpha_rows if row["method"] == method and row["quantity"] == quantity]
            if not selected:
                continue
            selected.sort(key=lambda row: row["alpha"])
            nominal = [100 * (1 - row["alpha"]) for row in selected]
            deviation = [row["coverage_pct"] - value for row, value in zip(selected, nominal)]
            if method == "constant_split_conformal":
                low = [row["coverage_ci_low_pct"] - value for row, value in zip(selected, nominal)]
                high = [row["coverage_ci_high_pct"] - value for row, value in zip(selected, nominal)]
                top.fill_between(nominal, low, high, color="0.6", alpha=0.18, lw=0,
                                 label="95% Wilson band (constant width)")
            top.plot(nominal, deviation, marker=markers[method], ms=3, lw=0.8,
                     color=colours[method], label=method)
            bottom.plot(nominal, [100 * row["width_over_error"] for row in selected],
                        marker=markers[method], ms=3, lw=0.8, color=colours[method], label=method)
        top.axhline(0.0, color="0.6", ls="--", lw=0.8)
        top.set_title(LABELS[quantity])
        if column == 0:
            top.set_ylabel("empirical - nominal (pt)")
        bottom.set_yscale("log")
        bottom.set_xlabel("nominal coverage (%)")
        if column == 0:
            bottom.set_ylabel("width / |error| (%)")
    axes[0, 0].text(0.03, 0.05, "all five constructions overlap\nthe diagonal", transform=axes[0, 0].transAxes,
                    fontsize=6, color="0.35")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    path = output_dir / "fig2_calibration_curve.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_reliability(reliability_rows, output_dir: Path):
    plt = _setup_matplotlib()
    arms = [("pre", "no intervention", "tab:blue"), ("hodge1.00", "PIVID (Hodge x1)", "tab:red")]
    figure, axes = plt.subplots(1, 4, figsize=(9.0, 2.6))
    for ax, quantity in zip(axes, PAPER_QUANTITIES):
        for arm, label, colour in arms:
            selected = [
                row for row in reliability_rows
                if row["quantity"] == quantity and row["method"] == arm and isinstance(row["bin"], (int, np.integer))
            ]
            if not selected:
                continue
            x = [row["scale_mean"] for row in selected]
            y = [row["abs_error_mean"] for row in selected]
            ax.plot(x, y, "o-", ms=3, color=colour, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mean dispersion in bin")
        ax.text(0.04, 0.94, QUANTITY_TITLE[quantity], transform=ax.transAxes,
                fontsize=6.2, va="top", ha="left")
    axes[0].set_ylabel("mean |error| in bin")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    path = output_dir / "fig3_reliability.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_width_coverage(coverage_rows, output_dir: Path, alpha_label="90"):
    plt = _setup_matplotlib()
    figure, axes = plt.subplots(1, 4, figsize=(9.0, 2.8))
    markers = {"raw_candidate_interval": "x", "constant_split_conformal": "o",
               "pivd_pre_dispersion": "^", "method_A_pre_dispersion": "v",
               "post_dispersion": "D", "pixel_mc_photometric": "s",
               "pixel_mc_candidate": "P", "method_B_certificate": "*"}
    for ax, quantity in zip(axes, PAPER_QUANTITIES):
        for row in coverage_rows:
            if row["quantity"] != quantity:
                continue
            width = row["mean_width"]
            if not np.isfinite(width) or width <= 0:
                continue
            ax.scatter(width / max(row["mean_abs_error"], EPSILON), row["coverage_pct"],
                       marker=markers.get(row["method"], "o"), s=22, label=row["method"])
        ax.axhline(90, color="0.6", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(f"width / |error| ({alpha_label} % nominal)")
    axes[0].set_ylabel("empirical coverage (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    path = output_dir / "fig4_width_coverage.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_k_sweep(k_rows, output_dir: Path):
    plt = _setup_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    colours = {"constant_split_conformal": "0.4", "pivd_pre_dispersion": "tab:blue",
               "post_dispersion": "tab:red", "method_A_pre_dispersion": "tab:purple"}
    for method, colour in colours.items():
        selected = [row for row in k_rows if row["method"] == method and row["quantity"] == "continuity_rms"]
        if not selected:
            continue
        selected.sort(key=lambda row: row["k"])
        ks = [row["k"] for row in selected]
        axes[0].plot(ks, [row["coverage_pct"] for row in selected], "o-", color=colour, label=method)
        axes[1].plot(ks, [100 * row["width_over_error"] for row in selected], "o-", color=colour, label=method)
    axes[0].axhline(90, color="0.6", ls="--", lw=0.8)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("candidates K")
    axes[0].set_ylabel("coverage (%)")
    axes[1].set_xlabel("candidates K")
    axes[1].set_ylabel("width / |error| (%)")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    path = output_dir / "fig5_k_sweep.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_scale_mechanism(test, output_dir: Path):
    """Why a normalised interval blows up: the scale becomes heterogeneous and uninformative."""
    plt = _setup_matplotlib()
    arms = ["pre", "hodge0.25", "hodge0.50", "hodge1.00"]
    delta = [float(np.mean(test[f"delta__{arm}"])) for arm in arms]
    colours = dict(zip(PAPER_QUANTITIES, ("tab:blue", "tab:orange", "tab:green", "tab:red")))
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    for quantity in PAPER_QUANTITIES:
        index = quantity_index(quantity)
        spreads = [test[f"q_cand__{arm}"][:, :, index].std(axis=1).astype(np.float64) for arm in arms]
        dynamic = [
            float(np.quantile(values, 0.95) / max(np.quantile(values, 0.05), EPSILON))
            for values in spreads
        ]
        cv = [float(values.std() / max(values.mean(), EPSILON)) for values in spreads]
        rho = []
        for arm, values in zip(arms, spreads):
            error = np.abs(
                test[f"q_point__{arm}"][:, -1, index].astype(np.float64) - truth_values(test, quantity)
            )
            rho.append(spearman(values, error))
        axes[0].plot(delta, dynamic, "o-", color=colours[quantity], label=LABELS[quantity])
        axes[1].plot(delta, cv, "o-", color=colours[quantity], label=LABELS[quantity])
        axes[2].plot(delta, rho, "o-", color=colours[quantity], label=LABELS[quantity])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("$\\sigma_q$ p95 / p05")
    axes[1].set_ylabel("coefficient of variation of $\\sigma_q$")
    axes[2].axhline(0.0, color="0.6", ls="--", lw=0.8)
    axes[2].set_ylabel("Spearman($\\sigma_q$, |error|)")
    for ax in axes:
        ax.set_xlabel("realised intervention RMS (px)")
    axes[2].legend(frameon=False, fontsize=6)
    figure.tight_layout()
    path = output_dir / "fig6_scale_mechanism.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_legacy_decomposition(legacy_rows, output_dir: Path):
    plt = _setup_matplotlib()
    figure, ax = plt.subplots(figsize=(4.6, 2.6))
    positions = np.arange(len(legacy_rows))
    base = [row["candidate_interval_width"] for row in legacy_rows]
    radii = [row["two_radii"] for row in legacy_rows]
    ax.bar(positions, base, color="tab:blue", label="candidate 5-95 % interval")
    ax.bar(positions, radii, bottom=base, color="tab:orange", label="2 x additive radius")
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[row["quantity"]] for row in legacy_rows], rotation=15, ha="right")
    ax.set_ylabel("width")
    ax.legend(frameon=False, fontsize=6)
    figure.tight_layout()
    path = output_dir / "fig7_published_width_decomposition.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_conditional_coverage(domain_rows, output_dir: Path, alpha=0.10):
    plt = _setup_matplotlib()
    domains = sorted({row["group"] for row in domain_rows if row["group"] != "ALL"})
    figure, axes = plt.subplots(1, 4, figsize=(9.0, 2.7), sharey=True)
    width = 0.38
    positions = np.arange(len(domains))
    for ax, quantity in zip(axes, PAPER_QUANTITIES):
        marginal = [row for row in domain_rows if row["quantity"] == quantity
                    and row["method"] == "constant_split_conformal" and row["group"] == "ALL"]
        marginal_mondrian = [row for row in domain_rows if row["quantity"] == quantity
                             and row["method"] == "constant_mondrian" and row["group"] == "ALL"]
        plain = [next(row for row in domain_rows if row["quantity"] == quantity
                      and row["method"] == "constant_split_conformal" and row["group"] == domain)["coverage_pct"]
                 for domain in domains]
        mondrian = [next(row for row in domain_rows if row["quantity"] == quantity
                         and row["method"] == "constant_mondrian" and row["group"] == domain)["coverage_pct"]
                    for domain in domains]
        ax.bar(positions - width / 2, plain, width, color="0.55", label="one quantile (marginal)")
        ax.bar(positions + width / 2, mondrian, width, color="tab:blue", label="one quantile per domain")
        if marginal:
            ax.axhline(marginal[0]["coverage_pct"], color="0.55", ls="--", lw=0.8)
        if marginal_mondrian:
            ax.axhline(marginal_mondrian[0]["coverage_pct"], color="tab:blue", ls="--", lw=0.8)
        ax.axhline(100 * (1 - alpha), color="tab:red", ls=":", lw=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels([domain.replace("_", "\n") for domain in domains], fontsize=5.5)
        ax.set_ylim(40, 103)
    axes[0].set_ylabel("coverage (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles + [plt.Line2D([], [], color="tab:red", ls=":")],
                  labels + ["nominal"], loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    path = output_dir / "fig8_conditional_coverage.png"
    figure.savefig(path)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Research report
# ---------------------------------------------------------------------------
def write_markdown(path: Path, sections):
    lines = ["# Constraint / uncertainty tension study - automatic report", ""]
    for title, body in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(body.rstrip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows, columns, float_format="{:.4g}"):
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append("" if not np.isfinite(value) else float_format.format(value))
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", default=str(ROOT / "piv_runs" / "uncertainty_study"))
    parser.add_argument(
        "--calibration", default="purged",
        choices=["purged", "validation", "test", "test_holdout"],
        help="test_holdout splits the test file in half: one half calibrates, the other is evaluated",
    )
    parser.add_argument("--secondary-calibration", default="validation")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--output-dir", default=str(ROOT / "paper_figures" / "uncertainty_study"))
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--holdout-repeats", type=int, default=20,
                        help="repeat the random half-split to report the sampling distribution of coverage")
    args = parser.parse_args()

    study_dir = Path(args.study_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    alpha = float(args.alpha)
    if args.calibration == "test_holdout":
        cal, test = holdout_halves(load_split(study_dir, "test"))
        cal["__meta__"] = {**cal["__meta__"], "calibration_note": "random half of the test file"}
    else:
        test = load_split(study_dir, "test")
        cal = load_split(study_dir, args.calibration)

    register_scale_limits(cal, METHODS)
    main_methods = tuple(method for method in METHODS if method.name not in ROBUST_METHODS)
    robust_methods = tuple(method for method in METHODS if method.name in ROBUST_METHODS)

    coverage_rows = coverage_table(cal, test, alpha, methods=main_methods)
    robust_rows = coverage_table(cal, test, alpha, methods=robust_methods)
    alpha_rows = alpha_sweep_table(cal, test)
    k_rows = k_sweep_table(cal, test, alpha)
    reliability_rows, reliability_summary = reliability_table(test)
    dose_rows = dose_response_table(test)
    domain_rows = per_domain_table(cal, test, alpha)
    conditional_rows = conditional_coverage_summary(domain_rows, alpha)
    unb_rows = unb_une_table(test)
    pixel_rank_rows = pixel_scale_rank_table(test)
    wilson_rows = published_wilson_check()
    legacy_rows = legacy_decomposition_table(cal, test, alpha, arm="pre")
    legacy_agent_rows = legacy_decomposition_table(cal, test, alpha, arm="hodge1.00")
    repeated_rows = []
    if args.holdout_repeats and args.calibration != "test_holdout":
        repeated_rows = repeated_holdout_table(load_split(study_dir, "test"), alpha, args.holdout_repeats)
    tension_rows = tension_summary_table(coverage_rows)

    extra = {}
    if args.secondary_calibration and args.secondary_calibration != args.calibration:
        try:
            secondary = load_split(study_dir, args.secondary_calibration)
            register_scale_limits(secondary, METHODS)
            extra["coverage_rows_secondary"] = coverage_table(secondary, test, alpha, methods=main_methods)
            extra["calibration_name"] = args.secondary_calibration
        except FileNotFoundError:
            pass
    if (study_dir / "timing_test.json").is_file():
        extra["timing"] = json.loads((study_dir / "timing_test.json").read_text(encoding="utf-8"))

    # ---- persist tables
    def dump(name, rows):
        if not rows:
            return
        keys = list(rows[0].keys())
        import csv
        with (tables_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        (tables_dir / f"{name}.md").write_text(
            markdown_table(rows, keys), encoding="utf-8"
        )

    dump(f"coverage_alpha{int(alpha * 100)}_{args.calibration}", coverage_rows)
    dump(f"coverage_robustified_alpha{int(alpha * 100)}_{args.calibration}", robust_rows)
    dump("alpha_sweep", alpha_rows)
    dump("k_sweep", k_rows)
    dump("reliability_bins", reliability_rows)
    dump("reliability_summary", reliability_summary)
    dump("dose_response", dose_rows)
    dump("per_domain", domain_rows)
    dump("conditional_coverage", conditional_rows)
    dump("unb_une", unb_rows)
    dump("pixel_scale_rank", pixel_rank_rows)
    dump("published_wilson_check", wilson_rows)
    dump("legacy_width_decomposition", legacy_rows)
    dump("legacy_width_decomposition_agent", legacy_agent_rows)
    dump("repeated_holdout", repeated_rows)
    dump("tension_summary", tension_rows)
    if "coverage_rows_secondary" in extra:
        dump(f"coverage_alpha{int(alpha * 100)}_{extra['calibration_name']}", extra["coverage_rows_secondary"])
    (output_dir / "analysis.json").write_text(
        json.dumps({
            "calibration": args.calibration,
            "alpha": alpha,
            "coverage": coverage_rows,
            "coverage_robustified": robust_rows,
            "alpha_sweep": alpha_rows,
            "k_sweep": k_rows,
            "reliability_summary": reliability_summary,
            "dose_response": dose_rows,
            "per_domain": domain_rows,
            "conditional_coverage": conditional_rows,
            "unb_une": unb_rows,
            "pixel_scale_rank": pixel_rank_rows,
            "published_wilson_check": wilson_rows,
            "legacy_width_decomposition": legacy_rows,
            "legacy_width_decomposition_agent": legacy_agent_rows,
            "repeated_holdout": repeated_rows,
            "tension_summary": tension_rows,
            "secondary": extra,
        }, indent=2),
        encoding="utf-8",
    )

    # ---- figures
    figure_paths = []
    if not args.skip_figures:
        figure_paths.append(figure_dose_response(test, figures_dir))
        figure_paths.append(figure_calibration_curve(alpha_rows, figures_dir))
        figure_paths.append(figure_reliability(reliability_rows, figures_dir))
        figure_paths.append(figure_width_coverage(coverage_rows, figures_dir))
        figure_paths.append(figure_k_sweep(k_rows, figures_dir))
        figure_paths.append(figure_scale_mechanism(test, figures_dir))
        figure_paths.append(figure_legacy_decomposition(legacy_rows, figures_dir))
        figure_paths.append(figure_conditional_coverage(domain_rows, figures_dir, alpha))

    # ---- digest
    coverage_columns = ["quantity", "method", "coverage_pct", "coverage_ci_low_pct",
                        "coverage_ci_high_pct", "mean_width", "width_over_error",
                        "mean_interval_score", "conformal_q", "n_calibration"]
    sections = [
        ("Protocol",
         f"* calibration split: `{args.calibration}` (n = {int(cal['uid'].shape[0])})\n"
         f"* test split: `test` (n = {int(test['uid'].shape[0])})\n"
         f"* nominal coverage: {100 * (1 - alpha):.0f} %\n"
         f"* candidates K = {test['__meta__']['candidates']}, sampling steps = {test['__meta__']['sampling_steps']}\n"
         f"* deterministic per-sample candidate noise: {test['__meta__']['deterministic_initial_noise']}"),
        (f"Coverage at {100 * (1 - alpha):.0f} % (calibration = {args.calibration})",
         markdown_table(coverage_rows, coverage_columns)),
        ("Dose-response of the Hodge constraint",
         markdown_table(dose_rows, ["arm", "epe_rel_pct", "divergence_rel_pct", "realized_delta_rms",
                                    "acceptance_rate", "continuity_rms_spread", "continuity_rms_abs_error",
                                    "continuity_rms_spread_over_error", "continuity_rms_spearman",
                                    "continuity_rms_naive_coverage"])),
        ("Is candidate dispersion informative?",
         markdown_table(reliability_summary, ["arm", "quantity", "mean_scale", "mean_abs_error",
                                              "dispersion_over_error", "spearman_scale_error",
                                              "top_over_bottom_quintile_error",
                                              "fraction_within_1.96_scale"])),
        ("Does the per-pixel dispersion rank sample difficulty?",
         markdown_table(pixel_rank_rows, ["arm", "mean_pixel_sigma", "mean_epe", "sigma_over_epe",
                                          "spearman_sigma_epe", "top_over_bottom_quintile_epe"])),
        ("Pixel-level UNB / UNE (candidate dispersion as per-vector uncertainty)",
         markdown_table(unb_rows, ["arm", "unb", "une", "une_rms", "pixel_sigma_mean"])),
        ("Reproduction check of the published coverage numbers (n = 1065)",
         markdown_table(wilson_rows, ["quantity", "published_coverage_pct", "covered_count",
                                      "wilson_low_pct", "wilson_high_pct", "above_90"])),
        ("Where the published width came from (PIVID-D, pre-intervention)",
         markdown_table(legacy_rows, ["arm", "quantity", "candidate_interval_width", "additive_radius",
                                      "total_width", "radius_share_of_width",
                                      "coverage_candidate_interval_only", "coverage_with_radius",
                                      "n_calibration"])),
        ("Where the published width came from (PIVID, post-intervention)",
         markdown_table(legacy_agent_rows, ["arm", "quantity", "candidate_interval_width", "additive_radius",
                                            "total_width", "radius_share_of_width",
                                            "coverage_candidate_interval_only", "coverage_with_radius",
                                            "n_calibration"])),
        ("Repeated random half-split: is the construction calibrated?",
         markdown_table(repeated_rows, ["method", "quantity", "nominal_pct", "mean_coverage_pct",
                                        "sd_coverage_pt", "min_coverage_pct", "max_coverage_pct",
                                        "fraction_below_nominal", "mean_width"])),
        ("Tension summary: sharpness at matched coverage",
         markdown_table(tension_rows, ["quantity", "width_constant", "width_pre_dispersion",
                                       "width_post_dispersion", "width_method_A",
                                       "sharpness_gain_pre_dispersion",
                                       "sharpness_gain_post_dispersion",
                                       "sharpness_gain_method_A", "post_over_pre_width"])),
        ("K sweep (continuity residual)",
         markdown_table([row for row in k_rows if row["quantity"] == "continuity_rms"],
                        ["method", "k", "coverage_pct", "coverage_ci_low_pct", "coverage_ci_high_pct",
                         "mean_width", "width_over_error"])),
        ("Robustified scales (floor/cap fitted on the calibration split)",
         markdown_table(robust_rows, coverage_columns)),
        ("Marginal vs conditional coverage",
         markdown_table(conditional_rows, ["quantity", "method", "marginal_pct", "worst_domain",
                                           "worst_pct", "worst_ci_low_pct", "best_domain", "best_pct",
                                           "domain_spread_pt"])),
        ("Per-domain coverage",
         markdown_table(domain_rows, ["group", "quantity", "method", "n", "coverage_pct",
                                      "coverage_ci_low_pct", "coverage_ci_high_pct", "mean_width"])),
        ("Alpha sweep",
         markdown_table(alpha_rows, ["quantity", "method", "alpha", "coverage_pct",
                                     "coverage_ci_low_pct", "coverage_ci_high_pct",
                                     "mean_width", "width_over_error"])),
    ]
    if figure_paths:
        sections.append(("Figures", "\n".join(f"* `{path.name}`" for path in figure_paths)))
    write_markdown(output_dir / "summary.md", sections)
    print(f"wrote {output_dir/'summary.md'}", flush=True)


if __name__ == "__main__":
    main()






