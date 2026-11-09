from __future__ import annotations

import csv
from math import exp, lgamma, log
from pathlib import Path

import numpy as np


def load_json(path: Path):
    import json

    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value, default=0.0):
    if value is None or value == "":
        return default
    return float(value)


def as_int(value, default=0):
    if value is None or value == "":
        return default
    return int(float(value))


def load_sample_metrics(run_dir: Path):
    csv_path = run_dir / "sample_metrics.csv"
    if csv_path.exists():
        return load_csv_rows(csv_path)
    json_path = run_dir / "sample_metrics.json"
    if json_path.exists():
        return load_json(json_path)
    return []


def infer_primary_agent_method(rows):
    methods = sorted({row.get("method", "") for row in rows if row.get("method", "") != "base_diffusion"})
    if "residual_agent" in methods:
        return "residual_agent"
    if "residual_agent_linear_calibrated" in methods:
        return "residual_agent_linear_calibrated"
    if methods:
        return methods[0]
    return ""


def filter_sample_rows(rows, dataset, method):
    return [row for row in rows if row.get("dataset") == dataset and row.get("method") == method]


def get_paired_sample_arrays(sample_rows, dataset, agent_method):
    base_rows = filter_sample_rows(sample_rows, dataset, "base_diffusion")
    agent_rows = filter_sample_rows(sample_rows, dataset, agent_method)
    if not base_rows or not agent_rows:
        return None
    def sample_key(row):
        sample_uid = str(row.get("sample_uid", "")).strip()
        if sample_uid:
            return sample_uid
        return str(as_int(row.get("sample_index"), -1))

    base_by_idx = {sample_key(row): row for row in base_rows}
    agent_by_idx = {sample_key(row): row for row in agent_rows}
    shared_indices = sorted(set(base_by_idx) & set(agent_by_idx))
    if not shared_indices:
        return None
    base_epe = np.asarray([as_float(base_by_idx[idx].get("final_epe")) for idx in shared_indices], dtype=np.float64)
    agent_epe = np.asarray([as_float(agent_by_idx[idx].get("final_epe")) for idx in shared_indices], dtype=np.float64)
    return {
        "shared_indices": shared_indices,
        "base_epe": base_epe,
        "agent_epe": agent_epe,
    }


def collect_paired_values(sample_rows, agent_method=None):
    agent_method = agent_method or infer_primary_agent_method(sample_rows)
    if not agent_method:
        return None
    datasets = sorted({row.get("dataset") for row in sample_rows if row.get("dataset")})
    paired_chunks = []
    used_datasets = []
    for dataset in datasets:
        pair = get_paired_sample_arrays(sample_rows, dataset, agent_method)
        if pair is None:
            continue
        paired_chunks.append(pair)
        used_datasets.append(dataset)
    if not paired_chunks:
        return None
    return {
        "agent_method": agent_method,
        "datasets": used_datasets,
        "base_epe": np.concatenate([chunk["base_epe"] for chunk in paired_chunks]),
        "agent_epe": np.concatenate([chunk["agent_epe"] for chunk in paired_chunks]),
    }


def sign_test_two_sided_pvalue(wins, total):
    if total <= 0:
        return 1.0

    def log_binom_prob(k):
        return (
            lgamma(total + 1)
            - lgamma(k + 1)
            - lgamma(total - k + 1)
            - total * log(2.0)
        )

    observed_log_prob = log_binom_prob(int(wins))
    log_tolerance = 1e-12
    p_value = 0.0
    for k in range(total + 1):
        log_prob = log_binom_prob(k)
        if log_prob <= observed_log_prob + log_tolerance:
            p_value += exp(log_prob)
    return float(min(1.0, p_value))


def bootstrap_mean_ci(values, num_bootstrap=2000, seed=3407, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        val = float(values[0])
        return val, val
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, values.size, size=(num_bootstrap, values.size))
    boot_means = values[sample_indices].mean(axis=1)
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def compute_cohen_dz(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.nan
    std = float(np.std(values, ddof=1))
    if std <= 1e-12:
        return np.nan
    return float(np.mean(values) / std)


def build_sample_statistics(sample_rows, class_label):
    if not sample_rows:
        return []
    datasets = sorted({row.get("dataset") for row in sample_rows if row.get("dataset")})
    agent_method = infer_primary_agent_method(sample_rows)
    if not agent_method:
        return []

    stats_rows = []
    for dataset in datasets:
        pair = get_paired_sample_arrays(sample_rows, dataset, agent_method)
        if pair is None:
            continue
        base_epe = pair["base_epe"]
        agent_epe = pair["agent_epe"]
        gain = base_epe - agent_epe
        non_ties = np.abs(gain) > 1e-12
        wins = int(np.sum(gain[non_ties] > 0.0))
        sign_total = int(np.sum(non_ties))
        ci_low, ci_high = bootstrap_mean_ci(gain, seed=3407 + len(dataset) + len(class_label))

        stats_rows.append(
            {
                "class_label": class_label,
                "dataset": dataset,
                "agent_method": agent_method,
                "sample_count": int(gain.size),
                "base_mean_epe": float(base_epe.mean()),
                "base_median_epe": float(np.median(base_epe)),
                "base_p95_epe": float(np.percentile(base_epe, 95)),
                "agent_mean_epe": float(agent_epe.mean()),
                "agent_median_epe": float(np.median(agent_epe)),
                "agent_p95_epe": float(np.percentile(agent_epe, 95)),
                "mean_gain": float(gain.mean()),
                "median_gain": float(np.median(gain)),
                "relative_gain_pct": float(gain.mean() / max(base_epe.mean(), 1e-8) * 100.0),
                "gain_ci_low": ci_low,
                "gain_ci_high": ci_high,
                "gain_positive_rate": float(np.mean(gain > 0.0)),
                "sign_test_pvalue": sign_test_two_sided_pvalue(wins, sign_total),
                "cohen_dz": compute_cohen_dz(gain),
            }
        )
    return stats_rows


def build_pooled_statistics_from_arrays(base_epe, agent_epe, class_label, dataset_count, agent_method, dataset_stats_rows=None):
    base_epe = np.asarray(base_epe, dtype=np.float64)
    agent_epe = np.asarray(agent_epe, dtype=np.float64)
    if base_epe.size == 0 or agent_epe.size == 0:
        return None
    gain = base_epe - agent_epe
    non_ties = np.abs(gain) > 1e-12
    wins = int(np.sum(gain[non_ties] > 0.0))
    sign_total = int(np.sum(non_ties))
    ci_low, ci_high = bootstrap_mean_ci(gain, seed=3907 + len(class_label))
    base_mean = float(base_epe.mean())
    mean_gain = float(gain.mean())
    summary = {
        "summary_level": "pooled",
        "class_label": class_label,
        "dataset_count": int(dataset_count),
        "sample_count": int(gain.size),
        "agent_method": agent_method,
        "base_mean_epe": base_mean,
        "base_median_epe": float(np.median(base_epe)),
        "agent_mean_epe": float(agent_epe.mean()),
        "agent_median_epe": float(np.median(agent_epe)),
        "mean_gain": mean_gain,
        "median_gain": float(np.median(gain)),
        "weighted_relative_gain_pct": float(mean_gain / max(base_mean, 1e-8) * 100.0),
        "weighted_relative_gain_ci_low_pct": float(ci_low / max(base_mean, 1e-8) * 100.0),
        "weighted_relative_gain_ci_high_pct": float(ci_high / max(base_mean, 1e-8) * 100.0),
        "gain_ci_low": ci_low,
        "gain_ci_high": ci_high,
        "gain_positive_rate": float(np.mean(gain > 0.0)),
        "sign_test_pvalue": sign_test_two_sided_pvalue(wins, sign_total),
        "cohen_dz": compute_cohen_dz(gain),
    }
    if dataset_stats_rows:
        summary["macro_mean_gain"] = float(np.mean([as_float(row.get("mean_gain")) for row in dataset_stats_rows]))
        summary["macro_relative_gain_pct"] = float(np.mean([as_float(row.get("relative_gain_pct")) for row in dataset_stats_rows]))
    else:
        summary["macro_mean_gain"] = np.nan
        summary["macro_relative_gain_pct"] = np.nan
    return summary


def build_pooled_statistics(sample_rows, class_label, dataset_stats_rows=None):
    paired = collect_paired_values(sample_rows)
    if paired is None:
        return None
    return build_pooled_statistics_from_arrays(
        paired["base_epe"],
        paired["agent_epe"],
        class_label,
        dataset_count=len(paired["datasets"]),
        agent_method=paired["agent_method"],
        dataset_stats_rows=dataset_stats_rows,
    )
