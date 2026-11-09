"""Validation for a frozen primary-study manifest used by the evidence audit."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


REQUIRED_ABLATIONS = {
    "full",
    "no_candidate_reweighting",
    "no_local_matching",
    "no_uncertainty_input",
    "no_kinematic_input",
    "no_forward_backward_consistency",
}
POSTERIOR_REQUIRED_ABLATIONS = {
    "no_posterior_observation_guidance", "no_posterior_continuity_guidance", "fixed_sampling_budget"
}
REQUIRED_SENSITIVITY_FACTORS = {
    "diffusion_sampling_steps",
    "diffusion_candidates",
    "kinematic_regularization_weight",
    "agent_refine_steps",
}
POSTERIOR_REQUIRED_SENSITIVITY_FACTORS = {
    "posterior_observation_weight", "posterior_continuity_weight",
    "adaptive_risk_threshold", "particle_psf_sigma",
}


def load_study_manifest(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle), path


def validate_primary_study_manifest(manifest):
    """Validate immutable analysis commitments before accepting primary-journal evidence."""
    issues = []
    if not isinstance(manifest, dict):
        return {"available": False, "issues": ["study manifest must be a JSON object"]}
    if manifest.get("schema_version") != 1:
        issues.append("schema_version must be 1")
    if not str(manifest.get("study_id", "")).strip():
        issues.append("study_id is required")
    if manifest.get("frozen_before_training") is not True:
        issues.append("frozen_before_training must be true")
    if not str(manifest.get("code_revision", "")).strip():
        issues.append("code_revision is required")
    created_at = str(manifest.get("created_at_utc", "")).strip()
    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        issues.append("created_at_utc must be an ISO-8601 timestamp")

    protocol = manifest.get("primary_protocol", {})
    if not isinstance(protocol, dict):
        issues.append("primary_protocol must be an object")
        protocol = {}
    if protocol.get("selection_mode") != "agent":
        issues.append("primary_protocol.selection_mode must be agent")
    if not isinstance(protocol.get("evaluation_sampling_seed"), int) or protocol.get("evaluation_sampling_seed") < 0:
        issues.append("primary_protocol.evaluation_sampling_seed must be a non-negative integer")
    split_path = str(protocol.get("split_manifest_path", "")).strip()
    split_sha = str(protocol.get("split_sha256", "")).strip()
    if not split_path or not split_sha:
        issues.append("primary_protocol must declare split_manifest_path and split_sha256")
    else:
        path = Path(split_path).expanduser()
        if not path.exists():
            issues.append("primary_protocol split_manifest_path does not exist")
        else:
            with path.open("r", encoding="utf-8-sig") as handle:
                split_payload = json.load(handle)
            if split_payload.get("split_sha256") != split_sha:
                issues.append("primary_protocol split_sha256 does not match split_manifest_path")

    seeds = protocol.get("in_domain_training_seeds", [])
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds) or not all(isinstance(seed, int) for seed in seeds):
        issues.append("primary_protocol.in_domain_training_seeds must contain at least three unique integers")
    if protocol.get("in_domain_backbone_seed_policy") != "end_to_end_training":
        issues.append(
            "primary_protocol.in_domain_backbone_seed_policy must be end_to_end_training"
        )
    holdout_domains = protocol.get("holdout_domains", [])
    if not isinstance(holdout_domains, list) or len(holdout_domains) < 1 or len(set(holdout_domains)) != len(holdout_domains):
        issues.append("primary_protocol.holdout_domains must contain unique declared domains")

    posterior_method = manifest.get("method_family") == "posterior_guided_diffusion"
    required_ablations = REQUIRED_ABLATIONS | (POSTERIOR_REQUIRED_ABLATIONS if posterior_method else set())
    ablations = set(manifest.get("required_ablations", []))
    missing_ablations = sorted(required_ablations - ablations)
    if missing_ablations:
        issues.append(f"required_ablations missing: {missing_ablations}")
    baselines = manifest.get("required_baselines", [])
    if not isinstance(baselines, list) or len(baselines) < 3 or len(set(baselines)) != len(baselines):
        issues.append("required_baselines must contain at least three unique baseline names")
    sensitivity = manifest.get("sensitivity_grid", {})
    if not isinstance(sensitivity, dict):
        issues.append("sensitivity_grid must be an object")
        sensitivity = {}
    required_sensitivity = REQUIRED_SENSITIVITY_FACTORS | (
        POSTERIOR_REQUIRED_SENSITIVITY_FACTORS if posterior_method else set()
    )
    for factor in sorted(required_sensitivity):
        values = sensitivity.get(factor)
        if not isinstance(values, list) or len(values) < 2:
            issues.append(f"sensitivity_grid.{factor} must contain at least two planned values")
    cost = manifest.get("computational_reporting", {})
    if not isinstance(cost, dict) or any(
        cost.get(field) is not True
        for field in ("report_wall_clock_time", "report_peak_cuda_memory", "report_parameter_count")
    ):
        issues.append("computational_reporting must enable wall-clock, peak-memory, and parameter reporting")

    return {
        "available": not issues,
        "study_id": manifest.get("study_id", ""),
        "split_sha256": split_sha,
        "evaluation_sampling_seed": protocol.get("evaluation_sampling_seed"),
        "required_baselines": baselines if isinstance(baselines, list) else [],
        "holdout_domains": holdout_domains if isinstance(holdout_domains, list) else [],
        "minimum_seeds": len(seeds) if isinstance(seeds, list) else 0,
        "require_end_to_end_in_domain_seeds": protocol.get("in_domain_backbone_seed_policy")
        == "end_to_end_training",
        "issues": issues,
        "claim_boundary": (
            "This manifest records planned analyses and consistency checks. It does not establish that the study was "
            "externally preregistered, scientifically important, or free of all researcher degrees of freedom."
        ),
    }
