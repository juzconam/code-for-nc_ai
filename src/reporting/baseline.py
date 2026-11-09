"""Strict schema checks and paired analysis for externally evaluated baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.reporting.statistics import bootstrap_mean_ci, compute_cohen_dz, sign_test_two_sided_pvalue
from src.reporting.model_provenance import file_sha256


REQUIRED_BASELINE_FIELDS = ("sample_uid", "dataset", "final_epe")


def read_baseline_rows(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Baseline results file not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            rows = json.load(handle)
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Baseline results must be CSV or JSON.")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Baseline results must be a non-empty list of sample-level rows.")
    return rows, path


def _normalise_rows(rows, label):
    normalised = {}
    for row in rows:
        missing = [field for field in REQUIRED_BASELINE_FIELDS if not str(row.get(field, "")).strip()]
        if missing:
            raise ValueError(f"{label} result row is missing required fields: {missing}")
        sample_uid = str(row["sample_uid"]).strip()
        if sample_uid in normalised:
            raise ValueError(f"{label} contains duplicate sample_uid: {sample_uid}")
        normalised[sample_uid] = {
            "sample_uid": sample_uid,
            "dataset": str(row["dataset"]).strip(),
            "final_epe": float(row["final_epe"]),
        }
    return normalised


def load_upr_sample_rows(run_dir, method):
    run_dir = Path(run_dir).expanduser().resolve()
    metrics_path = run_dir / "sample_metrics.csv"
    if not metrics_path.exists():
        metrics_path = run_dir / "sample_metrics.json"
    rows, _ = read_baseline_rows(metrics_path)
    selected = [row for row in rows if row.get("method") == method]
    if not selected:
        raise ValueError(f"UPR-Flow run {run_dir} does not contain sample method '{method}'.")
    return _normalise_rows(selected, "UPR-Flow"), run_dir


def _read_task_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Baseline task CSV must contain at least one row.")
    task_rows = {}
    for row in rows:
        sample_uid = str(row.get("sample_uid", "")).strip()
        dataset = str(row.get("dataset", "")).strip()
        if not sample_uid or not dataset:
            raise ValueError("Baseline task CSV rows require sample_uid and dataset.")
        if sample_uid in task_rows:
            raise ValueError(f"Baseline task CSV contains duplicate sample_uid: {sample_uid}")
        task_rows[sample_uid] = dataset
    return task_rows


def verify_baseline_manifest(manifest_path, upr_run_dir, expected_protocol="in_domain"):
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        manifest = json.load(handle)
    required = (
        "baseline_name",
        "baseline_repository",
        "baseline_commit",
        "results_path",
        "results_sha256",
        "split_sha256",
        "evaluation_protocol",
        "preprocessing",
        "epe_definition",
        "training_data",
        "hardware",
        "task_manifest_path",
        "task_manifest_sha256",
        "runner_command",
        "model_weights",
    )
    missing = [field for field in required if not manifest.get(field)]
    if missing:
        raise ValueError(f"Baseline manifest is missing required fields: {missing}")
    run_dir = Path(upr_run_dir).expanduser().resolve()
    split_path = run_dir / "split_manifest.json"
    summary_path = run_dir / "run_summary.json"
    if not split_path.exists() or not summary_path.exists():
        raise FileNotFoundError("UPR-Flow run must contain split_manifest.json and run_summary.json.")
    with split_path.open("r", encoding="utf-8") as handle:
        split_metadata = json.load(handle)
    with summary_path.open("r", encoding="utf-8") as handle:
        run_summary = json.load(handle)
    if manifest["split_sha256"] != split_metadata["split_sha256"]:
        raise ValueError("Baseline split_sha256 does not match the UPR-Flow run split manifest.")
    if not run_summary.get("paper_eligible", False):
        raise ValueError("External baseline comparison requires a paper-eligible UPR-Flow run.")
    if run_summary.get("selection_mode") != "agent" or run_summary.get("target_domain_calibration") is not False:
        raise ValueError("External baseline comparison requires the uncalibrated residual-agent primary result.")
    if run_summary.get("paired_candidate_replay") is not True:
        raise ValueError("UPR-Flow run is missing paired DDIM-candidate replay provenance.")
    task_manifest_path = Path(manifest["task_manifest_path"]).expanduser().resolve()
    if not task_manifest_path.exists():
        raise FileNotFoundError(f"Baseline task manifest not found: {task_manifest_path}")
    if file_sha256(task_manifest_path) != str(manifest["task_manifest_sha256"]).lower():
        raise ValueError("Baseline task manifest SHA-256 does not match the declared value.")
    with task_manifest_path.open("r", encoding="utf-8-sig") as handle:
        task_manifest = json.load(handle)
    if task_manifest.get("split_sha256") != split_metadata["split_sha256"]:
        raise ValueError("Baseline task manifest split_sha256 does not match the UPR-Flow run.")
    if Path(task_manifest.get("upr_run", "")).expanduser().resolve() != run_dir:
        raise ValueError("Baseline task manifest was exported from a different UPR-Flow run.")
    if task_manifest.get("upr_method") != "residual_agent":
        raise ValueError("Baseline task manifest must target the residual_agent primary result.")
    task_csv_path = Path(task_manifest.get("task_csv", "")).expanduser().resolve()
    if not task_csv_path.exists():
        raise FileNotFoundError(f"Baseline task CSV not found: {task_csv_path}")
    if file_sha256(task_csv_path) != str(task_manifest.get("task_csv_sha256", "")).lower():
        raise ValueError("Baseline task CSV SHA-256 does not match its task manifest.")
    results_path = Path(manifest["results_path"]).expanduser().resolve()
    if not results_path.exists():
        raise FileNotFoundError(f"Baseline results file not found: {results_path}")
    if file_sha256(results_path) != str(manifest["results_sha256"]).lower():
        raise ValueError("Baseline results SHA-256 does not match the declared manifest value.")
    task_rows = _read_task_rows(task_csv_path)
    baseline_rows, _ = read_baseline_rows(results_path)
    normalised_baseline = _normalise_rows(baseline_rows, str(manifest["baseline_name"]))
    if set(normalised_baseline) != set(task_rows):
        raise ValueError("Baseline results must cover exactly the sample_uids in the exported task CSV.")
    if any(normalised_baseline[uid]["dataset"] != task_rows[uid] for uid in task_rows):
        raise ValueError("Baseline result dataset labels do not match the exported task CSV.")
    if int(task_manifest.get("sample_count", len(task_rows))) != len(task_rows):
        raise ValueError("Baseline task manifest sample_count does not match its CSV.")
    actual_protocol = run_summary.get("evaluation_protocol", "in_domain")
    if manifest["evaluation_protocol"] != actual_protocol or actual_protocol != expected_protocol:
        raise ValueError(
            f"Baseline evaluation protocol mismatch: baseline={manifest['evaluation_protocol']}, "
            f"UPR-Flow={actual_protocol}, expected={expected_protocol}."
        )
    manifest = {
        **manifest,
        "task_csv_sha256": str(task_manifest.get("task_csv_sha256", "")),
    }
    return manifest, split_metadata, run_summary


def paired_baseline_comparison(upr_rows, baseline_rows, baseline_name):
    shared_uids = sorted(set(upr_rows) & set(baseline_rows))
    if not shared_uids:
        raise ValueError("No shared sample_uid values between UPR-Flow and baseline results.")
    missing_from_baseline = sorted(set(upr_rows) - set(baseline_rows))
    missing_from_upr = sorted(set(baseline_rows) - set(upr_rows))
    if missing_from_baseline or missing_from_upr:
        raise ValueError(
            "Baseline and UPR-Flow results must cover exactly the same samples. "
            f"missing_from_baseline={len(missing_from_baseline)}, missing_from_upr={len(missing_from_upr)}"
        )
    datasets = {upr_rows[uid]["dataset"] for uid in shared_uids}
    if any(upr_rows[uid]["dataset"] != baseline_rows[uid]["dataset"] for uid in shared_uids):
        raise ValueError("Dataset labels differ for shared sample_uid values.")

    output_rows = []
    for dataset in sorted(datasets):
        ids = [uid for uid in shared_uids if upr_rows[uid]["dataset"] == dataset]
        upr_epe = np.asarray([upr_rows[uid]["final_epe"] for uid in ids], dtype=np.float64)
        baseline_epe = np.asarray([baseline_rows[uid]["final_epe"] for uid in ids], dtype=np.float64)
        gain = baseline_epe - upr_epe
        non_ties = np.abs(gain) > 1e-12
        wins = int(np.sum(gain[non_ties] > 0.0))
        ci_low, ci_high = bootstrap_mean_ci(gain, seed=4409 + len(dataset) + len(baseline_name))
        output_rows.append(
            {
                "dataset": dataset,
                "baseline": baseline_name,
                "sample_count": len(ids),
                "baseline_mean_epe": float(baseline_epe.mean()),
                "upr_mean_epe": float(upr_epe.mean()),
                "mean_gain": float(gain.mean()),
                "relative_gain_pct": float(gain.mean() / max(baseline_epe.mean(), 1e-8) * 100.0),
                "gain_ci_low": ci_low,
                "gain_ci_high": ci_high,
                "gain_positive_rate": float(np.mean(gain > 0.0)),
                "sign_test_pvalue": sign_test_two_sided_pvalue(wins, int(non_ties.sum())),
                "cohen_dz": compute_cohen_dz(gain),
            }
        )
    return output_rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
