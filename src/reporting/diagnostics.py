"""Scientific diagnostics for uncertainty ranking, flow regimes, and failure cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.reporting.statistics import as_float, filter_sample_rows, infer_primary_agent_method


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _safe_corr(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _quantile_groups(values, group_count):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.empty(0, dtype=np.int64), np.asarray([], dtype=np.float64)
    group_count = max(1, min(int(group_count), values.size))
    edges = np.quantile(values, np.linspace(0.0, 1.0, group_count + 1))
    group_ids = np.searchsorted(edges[1:-1], values, side="right")
    return group_ids.astype(np.int64), edges


def paired_rows(sample_rows, dataset, agent_method):
    base_rows = filter_sample_rows(sample_rows, dataset, "base_diffusion")
    agent_rows = filter_sample_rows(sample_rows, dataset, agent_method)
    if not base_rows or not agent_rows:
        return []

    def key(row):
        return str(row.get("sample_uid", row.get("sample_index", "")))

    base_by_uid = {key(row): row for row in base_rows}
    agent_by_uid = {key(row): row for row in agent_rows}
    rows = []
    for sample_uid in sorted(set(base_by_uid) & set(agent_by_uid)):
        base = base_by_uid[sample_uid]
        agent = agent_by_uid[sample_uid]
        rows.append(
            {
                "sample_uid": sample_uid,
                "dataset": dataset,
                "base_epe": as_float(base.get("final_epe")),
                "agent_epe": as_float(agent.get("final_epe")),
                "gain": as_float(base.get("final_epe")) - as_float(agent.get("final_epe")),
                "candidate_uncertainty_mean": as_float(base.get("candidate_uncertainty_mean")),
                "gt_flow_magnitude_mean": as_float(base.get("gt_flow_magnitude_mean")),
                "gt_vorticity_abs_mean": as_float(base.get("gt_vorticity_abs_mean")),
                "base_local_ncc_mean": as_float(base.get("base_local_ncc_mean")),
            }
        )
    return rows


def uncertainty_risk_rows(paired, quantile_count=5, tail_quantile=0.8):
    if not paired:
        return [], {}
    uncertainty = np.asarray([row["candidate_uncertainty_mean"] for row in paired], dtype=np.float64)
    base_error = np.asarray([row["base_epe"] for row in paired], dtype=np.float64)
    group_ids, edges = _quantile_groups(uncertainty, quantile_count)
    rows = []
    for group_id in range(len(edges) - 1):
        mask = group_ids == group_id
        if not np.any(mask):
            continue
        rows.append(
            {
                "dataset": paired[0]["dataset"],
                "diagnostic": "uncertainty_risk_bin",
                "bin": int(group_id + 1),
                "bin_lower": float(edges[group_id]),
                "bin_upper": float(edges[group_id + 1]),
                "sample_count": int(mask.sum()),
                "mean_uncertainty": float(uncertainty[mask].mean()),
                "mean_base_epe": float(base_error[mask].mean()),
                "mean_agent_epe": float(np.asarray([row["agent_epe"] for row in paired])[mask].mean()),
                "mean_gain": float(np.asarray([row["gain"] for row in paired])[mask].mean()),
            }
        )
    threshold_uncertainty = float(np.quantile(uncertainty, tail_quantile))
    threshold_error = float(np.quantile(base_error, tail_quantile))
    uncertain_tail = uncertainty >= threshold_uncertainty
    error_tail = base_error >= threshold_error
    overlap = uncertain_tail & error_tail
    summary = {
        "dataset": paired[0]["dataset"],
        "diagnostic": "uncertainty_risk_summary",
        "sample_count": len(paired),
        "tail_quantile": float(tail_quantile),
        "pearson_uncertainty_base_epe": _safe_corr(uncertainty, base_error),
        "spearman_uncertainty_base_epe": _safe_corr(_rankdata(uncertainty), _rankdata(base_error)),
        "tail_error_precision": float(overlap.sum() / max(uncertain_tail.sum(), 1)),
        "tail_error_recall": float(overlap.sum() / max(error_tail.sum(), 1)),
        "uncertainty_tail_count": int(uncertain_tail.sum()),
        "error_tail_count": int(error_tail.sum()),
    }
    return rows, summary


def risk_coverage_rows(paired, quantile_count=5):
    if not paired:
        return []
    uncertainty = np.asarray([row["candidate_uncertainty_mean"] for row in paired], dtype=np.float64)
    base_error = np.asarray([row["base_epe"] for row in paired], dtype=np.float64)
    agent_error = np.asarray([row["agent_epe"] for row in paired], dtype=np.float64)
    order = np.argsort(uncertainty, kind="mergesort")
    rows = []
    for step in range(1, max(1, int(quantile_count)) + 1):
        count = max(1, int(np.ceil(len(order) * step / max(1, int(quantile_count)))))
        retained = order[:count]
        rows.append(
            {
                "dataset": paired[0]["dataset"],
                "diagnostic": "uncertainty_risk_coverage",
                "coverage": float(count / len(order)),
                "sample_count": int(count),
                "mean_uncertainty": float(uncertainty[retained].mean()),
                "base_mean_epe": float(base_error[retained].mean()),
                "agent_mean_epe": float(agent_error[retained].mean()),
                "mean_gain": float((base_error[retained] - agent_error[retained]).mean()),
            }
        )
    return rows


def failure_case_rows(paired, max_cases=25):
    if not paired:
        return []
    max_cases = max(1, int(max_cases))
    output = []
    for category, ordered in (
        ("negative_gain", sorted(paired, key=lambda row: (row["gain"], -row["base_epe"]))),
        ("tail_base_error", sorted(paired, key=lambda row: (-row["base_epe"], row["gain"]))),
    ):
        for rank, row in enumerate(ordered[:max_cases], start=1):
            output.append(
                {
                    "dataset": row["dataset"],
                    "failure_category": category,
                    "rank": rank,
                    "sample_uid": row["sample_uid"],
                    "base_epe": row["base_epe"],
                    "agent_epe": row["agent_epe"],
                    "gain": row["gain"],
                    "candidate_uncertainty_mean": row["candidate_uncertainty_mean"],
                    "gt_flow_magnitude_mean": row["gt_flow_magnitude_mean"],
                    "gt_vorticity_abs_mean": row["gt_vorticity_abs_mean"],
                    "base_local_ncc_mean": row["base_local_ncc_mean"],
                }
            )
    return output


def regime_rows(paired, value_key, regime_name, quantile_count=3):
    if not paired:
        return []
    values = np.asarray([row[value_key] for row in paired], dtype=np.float64)
    gains = np.asarray([row["gain"] for row in paired], dtype=np.float64)
    base_error = np.asarray([row["base_epe"] for row in paired], dtype=np.float64)
    agent_error = np.asarray([row["agent_epe"] for row in paired], dtype=np.float64)
    group_ids, edges = _quantile_groups(values, quantile_count)
    output = []
    for group_id in range(len(edges) - 1):
        mask = group_ids == group_id
        if not np.any(mask):
            continue
        output.append(
            {
                "dataset": paired[0]["dataset"],
                "diagnostic": "flow_regime_gain",
                "regime": regime_name,
                "bin": int(group_id + 1),
                "bin_lower": float(edges[group_id]),
                "bin_upper": float(edges[group_id + 1]),
                "sample_count": int(mask.sum()),
                "mean_regime_value": float(values[mask].mean()),
                "base_mean_epe": float(base_error[mask].mean()),
                "agent_mean_epe": float(agent_error[mask].mean()),
                "mean_gain": float(gains[mask].mean()),
                "gain_positive_rate": float(np.mean(gains[mask] > 0.0)),
            }
        )
    return output


def build_scientific_diagnostics(sample_rows, quantile_count=5, tail_quantile=0.8, max_failure_cases=25):
    agent_method = infer_primary_agent_method(sample_rows)
    if not agent_method:
        return [], [], [], {"available": False, "reason": "no_agent_sample_metrics"}
    diagnostic_rows = []
    summary_rows = []
    failure_rows = []
    datasets = sorted({str(row.get("dataset", "")) for row in sample_rows if row.get("dataset")})
    for dataset in datasets:
        paired = paired_rows(sample_rows, dataset, agent_method)
        if not paired:
            continue
        risk_rows, risk_summary = uncertainty_risk_rows(paired, quantile_count, tail_quantile)
        diagnostic_rows.extend(risk_rows)
        diagnostic_rows.extend(risk_coverage_rows(paired, quantile_count))
        summary_rows.append(risk_summary)
        failure_rows.extend(failure_case_rows(paired, max_cases=max_failure_cases))
        diagnostic_rows.extend(regime_rows(paired, "gt_flow_magnitude_mean", "gt_flow_magnitude", min(3, quantile_count)))
        diagnostic_rows.extend(regime_rows(paired, "gt_vorticity_abs_mean", "gt_vorticity_abs", min(3, quantile_count)))
        diagnostic_rows.extend(regime_rows(paired, "base_local_ncc_mean", "base_local_ncc", min(3, quantile_count)))
    metadata = {
        "available": bool(diagnostic_rows),
        "agent_method": agent_method,
        "quantile_count": int(quantile_count),
        "tail_quantile": float(tail_quantile),
        "max_failure_cases": int(max_failure_cases),
        "failure_case_count": len(failure_rows),
        "claim_boundary": (
            "Candidate dispersion is evaluated as a risk-ranking signal, not as a calibrated probabilistic "
            "uncertainty estimate. Regime strata are descriptive and must be replicated on independent data."
        ),
    }
    return diagnostic_rows, summary_rows, failure_rows, metadata


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
