from __future__ import annotations

import csv
import datetime
import json
import os
import platform
import sys

import torch


def make_run_dir(output_dir, run_name=None, allow_overwrite=False, prefix="upr_flow"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"{prefix}_{timestamp}"
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, run_name)
    if os.path.exists(run_dir) and not os.path.isdir(run_dir):
        raise FileExistsError(f"Run path exists and is not a directory: {run_dir}")
    if os.path.isdir(run_dir):
        existing_entries = os.listdir(run_dir)
        if existing_entries and not allow_overwrite:
            raise FileExistsError(
                f"Run directory already exists and is non-empty: {run_dir}. "
                "Use a unique --run-name or pass --allow-overwrite."
            )
    else:
        os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_metrics_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_run_config(args, run_dir, device):
    payload = dict(vars(args))
    payload.update(
        {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "hostname": platform.node() or os.environ.get("COMPUTERNAME", ""),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_resolved": str(device),
            "run_dir": os.path.abspath(run_dir),
            "data_path_resolved": os.path.abspath(args.data_path),
            "output_dir_resolved": os.path.abspath(args.output_dir),
        }
    )
    if getattr(args, "checkpoint", None):
        payload["checkpoint_resolved"] = os.path.abspath(args.checkpoint)
    if getattr(args, "agent_path", None):
        payload["agent_path_resolved"] = os.path.abspath(args.agent_path)
    if getattr(args, "agent_init_path", None):
        payload["agent_init_path_resolved"] = os.path.abspath(args.agent_init_path)
    return payload


def build_inference_run_summary(args, run_dir, results_rows, calibration_rows=None, sample_metrics_exported=False):
    calibration_rows = calibration_rows or []
    evaluation_protocol = getattr(args, "evaluation_protocol", "in_domain")
    eval_domains_text = getattr(args, "eval_domains", "") or ""
    eval_domains = sorted({item.strip() for item in str(eval_domains_text).split(",") if item.strip()})
    exclude_domains_text = getattr(args, "exclude_domains", "") or ""
    exclude_domains = sorted({item.strip() for item in str(exclude_domains_text).split(",") if item.strip()})
    domain_adapters_active = bool(
        not getattr(args, "disable_domain_adapters", False)
        and evaluation_protocol != "zero_shot_holdout"
    )
    target_domain_calibration = bool(
        getattr(args, "selection_mode", "") in ("linear_calibrated", "quality_selective")
    )
    split_metadata = {}
    split_path = os.path.join(run_dir, "split_manifest.json")
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as handle:
            split_metadata = json.load(handle)
    effective_split_seed = split_metadata.get("split_seed", getattr(args, "split_seed", None))
    effective_purge_gap = split_metadata.get("purge_gap", getattr(args, "purge_gap", 0))
    dataset_counts = {}
    dataset_metrics = {}
    for row in results_rows:
        dataset = row.get("dataset", "unknown")
        method = row.get("method", "unknown")
        count = int(row.get("count", 0) or 0)
        dataset_counts[dataset] = max(dataset_counts.get(dataset, 0), count)
        dataset_metrics.setdefault(dataset, {})[method] = {
            "count": count,
            "final_epe": float(row.get("final_epe", 0.0) or 0.0),
            "epe_improvement": float(row.get("epe_improvement", 0.0) or 0.0),
            "epe_improvement_pct": float(row.get("epe_improvement_pct", 0.0) or 0.0),
            "milliseconds_per_sample": float(row.get("milliseconds_per_sample", 0.0) or 0.0),
            "peak_cuda_memory_mb": float(row.get("peak_cuda_memory_mb", 0.0) or 0.0),
            "parameter_count": int(row.get("parameter_count", 0) or 0),
        }

    calibration_counts = {}
    for row in calibration_rows:
        dataset = row.get("dataset", "unknown")
        calibration_counts[dataset] = int(row.get("count", 0) or 0)

    smoke_reasons = []
    if args.run_kind != "full":
        smoke_reasons.append(f"run_kind={args.run_kind}")
    if int(args.max_eval_batches) > 0:
        smoke_reasons.append(f"max_eval_batches={int(args.max_eval_batches)}")
    if int(args.max_calibration_batches) > 0:
        smoke_reasons.append(f"max_calibration_batches={int(args.max_calibration_batches)}")
    if not results_rows:
        smoke_reasons.append("no_inference_metrics")

    low_eval_counts = {name: count for name, count in dataset_counts.items() if count <= 1}
    if low_eval_counts:
        smoke_reasons.append(
            "eval_counts<=1:" + ",".join(f"{name}={count}" for name, count in sorted(low_eval_counts.items()))
        )

    if args.selection_mode == "linear_calibrated":
        if not calibration_rows:
            smoke_reasons.append("missing_calibration_metrics")
        low_calibration_counts = {name: count for name, count in calibration_counts.items() if count <= 1}
        if low_calibration_counts:
            smoke_reasons.append(
                "calibration_counts<=1:"
                + ",".join(f"{name}={count}" for name, count in sorted(low_calibration_counts.items()))
            )

    paper_eligible = len(smoke_reasons) == 0
    return {
        "mode": "inference",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_dir": os.path.abspath(run_dir),
        "run_kind": args.run_kind,
        "selection_mode": args.selection_mode,
        "evaluation_protocol": evaluation_protocol,
        "paired_candidate_replay": bool(getattr(args, "evaluation_sampling_seed", None) is not None),
        "evaluation_sampling_seed": getattr(args, "evaluation_sampling_seed", None),
        "eval_domains": eval_domains,
        "exclude_domains": exclude_domains,
        "target_domain_calibration": target_domain_calibration,
        "domain_adapters_active": domain_adapters_active,
        "dataset_kind": getattr(args, "dataset_kind", "unknown"),
        "split_protocol": getattr(args, "split_protocol", "unknown"),
        "split_seed": effective_split_seed,
        "purge_gap": effective_purge_gap,
        "split_schema_version": split_metadata.get("schema_version"),
        "total_source_sample_count": split_metadata.get("total_source_sample_count"),
        "purged_sample_count": split_metadata.get("purged_sample_count", 0),
        "split_manifest": "split_manifest.json",
        "split_sha256": split_metadata.get("split_sha256"),
        "dataset_fingerprint_mode": getattr(args, "dataset_fingerprint_mode", "unknown"),
        "backbone_holdout_provenance": bool(getattr(args, "require_backbone_holdout_provenance", False)),
        "agent_split_provenance": bool(getattr(args, "require_agent_split_provenance", False)),
        "simulate_class2": bool(args.simulate_class2),
        "max_eval_batches": int(args.max_eval_batches),
        "max_calibration_batches": int(args.max_calibration_batches),
        "export_paper_artifacts": bool(args.export_paper_artifacts),
        "dataset_counts": dataset_counts,
        "calibration_counts": calibration_counts,
        "datasets": dataset_metrics,
        "is_smoke_run": not paper_eligible,
        "smoke_reasons": smoke_reasons,
        "paper_eligible": paper_eligible,
        "artifacts": {
            "inference_metrics_json": "inference_metrics.json",
            "inference_metrics_csv": "inference_metrics.csv",
            "calibration_metrics_json": "calibration_metrics.json" if calibration_rows else None,
            "sample_metrics_json": "sample_metrics.json" if sample_metrics_exported else None,
            "sample_metrics_csv": "sample_metrics.csv" if sample_metrics_exported else None,
            "zero_shot_provenance_json": "zero_shot_provenance.json"
            if evaluation_protocol == "zero_shot_holdout"
            else None,
            "backbone_holdout_provenance_json": "backbone_holdout_provenance.json"
            if getattr(args, "require_backbone_holdout_provenance", False)
            else None,
            "in_domain_agent_provenance_json": "in_domain_agent_provenance.json"
            if getattr(args, "require_agent_split_provenance", False)
            else None,
            "experimental_protocol_validation_json": "experimental_protocol_validation.json"
            if getattr(args, "dataset_kind", "") == "experimental"
            else None,
        },
    }
