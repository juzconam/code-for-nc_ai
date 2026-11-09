"""Provenance artifacts for domain-held-out diffusion backbone training."""

from __future__ import annotations

import json
from pathlib import Path

from src.reporting.model_provenance import file_sha256


def write_backbone_provenance(
    output_path,
    checkpoint_path,
    training_domains,
    excluded_domains,
    training_data,
    training_protocol,
):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    training_domains = sorted({str(domain) for domain in training_domains})
    excluded_domains = sorted({str(domain) for domain in excluded_domains})
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")
    if not training_domains:
        raise ValueError("Backbone provenance requires at least one training domain.")
    overlap = sorted(set(training_domains) & set(excluded_domains))
    if overlap:
        raise ValueError(f"Held-out domains cannot appear in training_domains: {overlap}")
    if not str(training_data).strip() or not isinstance(training_protocol, dict):
        raise ValueError("Backbone provenance requires training_data and a structured training_protocol.")

    payload = {
        "schema_version": 1,
        "model_name": "UPR-Flow diffusion backbone",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "training_domains": training_domains,
        "excluded_domains": excluded_domains,
        "target_domain_supervision_excluded": bool(excluded_domains),
        "training_data": str(training_data).strip(),
        "training_protocol": training_protocol,
        "claim_boundary": (
            "This manifest supports only the exact checkpoint and declared held-out domains. "
            "It does not establish zero-shot performance without an independently trained agent and held-out test evaluation."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def write_backbone_holdout_provenance(
    output_path,
    checkpoint_path,
    training_domains,
    excluded_domains,
    training_data,
    training_protocol,
):
    """Compatibility entry point for strict leave-one-domain-out training."""
    if not excluded_domains:
        raise ValueError("Held-out backbone provenance requires at least one excluded domain.")
    return write_backbone_provenance(
        output_path,
        checkpoint_path,
        training_domains=training_domains,
        excluded_domains=excluded_domains,
        training_data=training_data,
        training_protocol=training_protocol,
    )
