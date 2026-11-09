"""Utilities for aggregating independently trained UPR-Flow runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


COMPARABLE_CONFIG_FIELDS = (
    "selection_mode",
    "evaluation_sampling_seed",
    "evaluation_protocol",
    "simulate_class2",
    "agent_refine_steps",
    "agent_matching_radius",
    "agent_scale",
    "use_candidate_reweighting",
    "use_forward_backward_consistency",
    "candidate_topk",
    "use_domain_conditioning",
    "use_domain_heads",
    "domain_head_residual_scale",
    "disable_uncertainty_input",
    "disable_kinematic_input",
    "kinematic_regularization_weight",
    "diffusion_timesteps",
    "diffusion_sampling_steps",
    "diffusion_candidates",
    "diffusion_flow_scale",
    "diffusion_denoised_clip",
    "ddim_eta",
    "pixel_size_x",
    "pixel_size_y",
    "frame_interval",
    "flow_representation",
    "disable_domain_adapters",
    "split_sha256",
    "dataset_fingerprint_mode",
    "dataset_kind",
    "require_backbone_holdout_provenance",
    "require_agent_split_provenance",
    "data_path_resolved",
)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_domains(value):
    if value is None:
        return []
    if isinstance(value, str):
        return sorted({item.strip() for item in value.split(",") if item.strip()})
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _run_protocol_signature(config, summary, metrics):
    evaluation_protocol = summary.get("evaluation_protocol", config.get("evaluation_protocol", "in_domain"))
    eval_domains = _normalise_domains(summary.get("eval_domains", config.get("eval_domains", "")))
    if not eval_domains:
        eval_domains = sorted({str(row.get("dataset", "")) for row in metrics if row.get("dataset")})
    selection_mode = summary.get("selection_mode", config.get("selection_mode"))
    signature = {
        "selection_mode": selection_mode,
        "evaluation_protocol": evaluation_protocol,
        "eval_domains": eval_domains,
        "target_domain_calibration": bool(
            summary.get(
                "target_domain_calibration",
                selection_mode in ("linear_calibrated", "quality_selective"),
            )
        ),
        "domain_adapters_active": bool(
            summary.get(
                "domain_adapters_active",
                not config.get("disable_domain_adapters", False)
                and evaluation_protocol != "zero_shot_holdout",
            )
        ),
    }
    for field in COMPARABLE_CONFIG_FIELDS:
        if field in ("selection_mode", "evaluation_protocol"):
            continue
        signature[field] = config.get(field)
    return signature


def _backbone_training_signature(training_config, manifest):
    """Return the seed-invariant backbone training settings needed for aggregation."""
    protocol = manifest.get("training_protocol", {})
    if not isinstance(protocol, dict):
        return None
    return {
        "run_kind": training_config.get("run_kind"),
        "protocol": protocol.get("protocol"),
        "source_split_sha256": protocol.get("source_split_sha256"),
        "effective_split_sha256": protocol.get("effective_split_sha256"),
        "training_domains": _normalise_domains(protocol.get("training_domains")),
        "excluded_domains": _normalise_domains(protocol.get("excluded_domains")),
        "batch_size": training_config.get("batch_size"),
        "eval_batch_size": training_config.get("eval_batch_size"),
        "epochs": training_config.get("epochs"),
        "lr": training_config.get("lr"),
        "diffusion_timesteps": training_config.get("diffusion_timesteps"),
        "diffusion_sampling_steps": training_config.get("diffusion_sampling_steps"),
        "diffusion_candidates": training_config.get("diffusion_candidates"),
        "diffusion_flow_scale": training_config.get("diffusion_flow_scale"),
        "diffusion_denoised_clip": training_config.get("diffusion_denoised_clip"),
        "ddim_eta": training_config.get("ddim_eta"),
        "external_pretraining_declaration": protocol.get("external_pretraining_declaration"),
    }


def _backbone_training_provenance(config):
    manifest_value = str(config.get("backbone_provenance_manifest", "") or "").strip()
    if not manifest_value:
        return None, None, None, None
    manifest_path = Path(manifest_value).expanduser().resolve()
    if not manifest_path.is_file():
        return str(manifest_path), None, None, None
    manifest = _read_json(manifest_path)
    checkpoint_path = Path(manifest.get("checkpoint_path", "")).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (manifest_path.parent / checkpoint_path).resolve()
    inference_checkpoint = str(config.get("checkpoint_resolved", "") or "").strip()
    if not inference_checkpoint or checkpoint_path.resolve() != Path(inference_checkpoint).expanduser().resolve():
        return str(manifest_path), None, None, None
    expected_hash = str(manifest.get("checkpoint_sha256", "") or "").lower()
    if not checkpoint_path.is_file() or not expected_hash or _file_sha256(checkpoint_path) != expected_hash:
        return str(manifest_path), None, None, None
    training_run = checkpoint_path.parent
    training_config_path = training_run / "run_config.json"
    if not training_config_path.is_file():
        return str(manifest_path), None, str(training_run), None
    training_config = _read_json(training_config_path)
    if training_config.get("mode") != "train_backbone" or "seed" not in training_config:
        return str(manifest_path), None, str(training_run), None
    return (
        str(manifest_path),
        int(training_config["seed"]),
        str(training_run),
        _backbone_training_signature(training_config, manifest),
    )


def _metric_key(row):
    dataset = str(row.get("dataset", ""))
    method = str(row.get("method", ""))
    if not dataset or not method:
        raise ValueError("Every inference metric row must include non-empty dataset and method values.")
    return dataset, method


def load_inference_run(run_dir, allow_non_paper=False):
    run_dir = Path(run_dir)
    config_path = run_dir / "run_config.json"
    summary_path = run_dir / "run_summary.json"
    metrics_path = run_dir / "inference_metrics.json"
    missing = [str(path.name) for path in (config_path, summary_path, metrics_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run {run_dir} is missing required artifacts: {', '.join(missing)}")

    config = _read_json(config_path)
    summary = _read_json(summary_path)
    metrics = _read_json(metrics_path)
    if config.get("mode") != "inference" or summary.get("mode") != "inference":
        raise ValueError(f"Run {run_dir} is not an inference run.")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"Run {run_dir} has no inference metric rows.")
    if not allow_non_paper and not summary.get("paper_eligible", False):
        reasons = ", ".join(summary.get("smoke_reasons", [])) or "paper_eligible=false"
        raise ValueError(f"Run {run_dir} is not paper-eligible: {reasons}")
    if "seed" not in config:
        raise ValueError(f"Run {run_dir} does not record a seed in run_config.json.")

    metric_map = {}
    for row in metrics:
        key = _metric_key(row)
        if key in metric_map:
            raise ValueError(f"Run {run_dir} has duplicate metric rows for {key}.")
        metric_map[key] = row
    agent_path = config.get("agent_path_resolved", config.get("agent_path"))
    agent_training_seed = None
    agent_training_run = None
    if agent_path:
        training_config_path = Path(agent_path).expanduser().resolve().parent / "run_config.json"
        if training_config_path.exists():
            training_config = _read_json(training_config_path)
            if training_config.get("mode") == "train_agent" and "seed" in training_config:
                agent_training_seed = int(training_config["seed"])
                agent_training_run = str(training_config_path.parent)
    backbone_manifest, backbone_training_seed, backbone_training_run, backbone_training_signature = (
        _backbone_training_provenance(config)
    )
    return {
        "run_dir": str(run_dir.resolve()),
        "run_name": run_dir.name,
        "seed": int(config["seed"]),
        "config": config,
        "summary": summary,
        "metrics": metric_map,
        "signature": _run_protocol_signature(config, summary, metrics),
        "agent_training_seed": agent_training_seed,
        "agent_training_run": agent_training_run,
        "backbone_provenance_manifest": backbone_manifest,
        "backbone_training_seed": backbone_training_seed,
        "backbone_training_run": backbone_training_run,
        "backbone_training_signature": backbone_training_signature,
    }


def _assert_comparable(runs):
    if len(runs) < 2:
        raise ValueError("At least two independent runs are required for multi-seed aggregation.")
    reference = runs[0]
    seen_inference_seeds = set()
    seen_training_seeds = set()
    reference_keys = set(reference["metrics"])
    backbone_signatures = [run["backbone_training_signature"] for run in runs]
    if any(signature is not None for signature in backbone_signatures) and not all(
        signature == backbone_signatures[0] for signature in backbone_signatures
    ):
        raise ValueError("Runs use different or unverifiable backbone training configurations.")
    for run in runs:
        if run["seed"] in seen_inference_seeds:
            raise ValueError(f"Duplicate seed {run['seed']} found across runs.")
        seen_inference_seeds.add(run["seed"])
        if run["agent_training_seed"] is not None:
            if run["agent_training_seed"] in seen_training_seeds:
                raise ValueError(
                    f"Duplicate agent training seed {run['agent_training_seed']} found across runs."
                )
            seen_training_seeds.add(run["agent_training_seed"])
        if run["signature"] != reference["signature"]:
            mismatches = {
                field: (reference["signature"].get(field), run["signature"].get(field))
                for field in sorted(set(reference["signature"]) | set(run["signature"]))
                if reference["signature"].get(field) != run["signature"].get(field)
            }
            raise ValueError(
                f"Run {run['run_name']} is not comparable to {reference['run_name']}: {mismatches}"
            )
        if set(run["metrics"]) != reference_keys:
            raise ValueError(
                f"Run {run['run_name']} does not contain the same dataset/method metric keys as "
                f"{reference['run_name']}."
            )
    return reference["signature"], sorted(reference_keys)


def _summary_stats(values):
    values = [float(value) for value in values]
    count = len(values)
    mean_value = float(statistics.mean(values))
    std_value = float(statistics.stdev(values)) if count > 1 else 0.0
    sem_value = std_value / math.sqrt(count) if count > 1 else 0.0
    ci_delta = 1.96 * sem_value
    return {
        "mean": mean_value,
        "std": std_value,
        "sem": sem_value,
        "ci95_normal_low": mean_value - ci_delta,
        "ci95_normal_high": mean_value + ci_delta,
    }


def aggregate_runs(runs_root, run_names, allow_non_paper=False):
    runs_root = Path(runs_root)
    run_names = [str(name).strip() for name in run_names if str(name).strip()]
    if not run_names:
        raise ValueError("At least one run name is required.")
    if len(set(run_names)) != len(run_names):
        raise ValueError("Run names must be unique.")

    runs = [load_inference_run(runs_root / name, allow_non_paper=allow_non_paper) for name in run_names]
    signature, metric_keys = _assert_comparable(runs)
    summary_rows = []
    run_rows = []
    numeric_fields = (
        "count",
        "raw_epe",
        "final_epe",
        "epe_improvement",
        "epe_improvement_pct",
        "milliseconds_per_sample",
        "peak_cuda_memory_mb",
        "parameter_count",
    )

    for dataset, method in metric_keys:
        per_run = [run["metrics"][(dataset, method)] for run in runs]
        row = {
            "dataset": dataset,
            "method": method,
            "n_seeds": len(runs),
        }
        for field in numeric_fields:
            values = [float(item[field]) for item in per_run if item.get(field) is not None]
            if len(values) != len(per_run):
                raise ValueError(f"Metric '{field}' is missing for {(dataset, method)}.")
            stats = _summary_stats(values)
            row[f"{field}_mean"] = stats["mean"]
            row[f"{field}_std"] = stats["std"]
            row[f"{field}_sem"] = stats["sem"]
            row[f"{field}_ci95_normal_low"] = stats["ci95_normal_low"]
            row[f"{field}_ci95_normal_high"] = stats["ci95_normal_high"]
        summary_rows.append(row)

        for run, metric in zip(runs, per_run):
            run_rows.append(
                {
                    "run_name": run["run_name"],
                    "inference_seed": run["seed"],
                    "agent_training_seed": run["agent_training_seed"],
                    "agent_training_run": run["agent_training_run"],
                    "dataset": dataset,
                    "method": method,
                    **{field: metric.get(field) for field in numeric_fields},
                }
            )

    has_agent_training_provenance = all(run["agent_training_seed"] is not None for run in runs)
    has_backbone_training_provenance = all(run["backbone_training_seed"] is not None for run in runs)
    backbone_signatures = [run["backbone_training_signature"] for run in runs]
    cost_reporting_available = all(
        all(
            field in metric and metric[field] is not None
            for field in ("milliseconds_per_sample", "peak_cuda_memory_mb", "parameter_count")
        )
        for run in runs
        for metric in run["metrics"].values()
    )
    backbone_seed_provenance = "unverified"
    if has_backbone_training_provenance:
        backbone_runs = {run["backbone_training_run"] for run in runs}
        paired_seeds = all(
            run["seed"] == run["agent_training_seed"] == run["backbone_training_seed"]
            for run in runs
        )
        if paired_seeds and len(backbone_runs) == len(runs):
            backbone_seed_provenance = "end_to_end_training"
        elif len(backbone_runs) == 1:
            backbone_seed_provenance = "shared_backbone"
        else:
            backbone_seed_provenance = "independent_backbone_unpaired_seeds"
    manifest = {
        "aggregation": "independent_training_seed_summary",
        "confidence_interval": "normal approximation: mean +/- 1.96 * SEM",
        "n_seeds": len(runs),
        "run_names": [run["run_name"] for run in runs],
        "run_dirs": [run["run_dir"] for run in runs],
        "inference_seeds": [run["seed"] for run in runs],
        "agent_training_seeds": [run["agent_training_seed"] for run in runs],
        "agent_training_runs": [run["agent_training_run"] for run in runs],
        "seed_provenance": "agent_training" if has_agent_training_provenance else "inference_only",
        "backbone_provenance_manifests": [run["backbone_provenance_manifest"] for run in runs],
        "backbone_training_seeds": [run["backbone_training_seed"] for run in runs],
        "backbone_training_runs": [run["backbone_training_run"] for run in runs],
        "backbone_training_signature": backbone_signatures[0] if backbone_signatures else None,
        "backbone_seed_provenance": backbone_seed_provenance,
        "cost_reporting_available": cost_reporting_available,
        "protocol_signature": signature,
        "metric_keys": [{"dataset": dataset, "method": method} for dataset, method in metric_keys],
        "allow_non_paper": bool(allow_non_paper),
    }
    return summary_rows, run_rows, manifest


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_aggregation(output_dir, summary_rows, run_rows, manifest):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "multiseed_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "multiseed_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(output_dir / "multiseed_summary.csv", summary_rows)
    write_csv(output_dir / "multiseed_run_metrics.csv", run_rows)
