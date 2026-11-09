"""Checkpoint provenance validation for strict held-out-domain evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_backbone_holdout_provenance(manifest_path, checkpoint_path, held_out_domains):
    manifest_path = Path(manifest_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Backbone provenance manifest not found: {manifest_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        manifest = json.load(handle)
    required = (
        "schema_version",
        "model_name",
        "checkpoint_sha256",
        "training_domains",
        "excluded_domains",
        "target_domain_supervision_excluded",
        "training_data",
        "training_protocol",
    )
    missing = [field for field in required if field not in manifest or manifest[field] in (None, "")]
    if missing:
        raise ValueError(f"Backbone provenance manifest is missing required fields: {missing}")
    actual_hash = file_sha256(checkpoint_path)
    if actual_hash != str(manifest["checkpoint_sha256"]).lower():
        raise ValueError("Backbone checkpoint SHA-256 does not match its provenance manifest.")
    held_out_domains = sorted(set(str(domain) for domain in held_out_domains))
    training_protocol = manifest["training_protocol"]
    if not isinstance(training_protocol, dict):
        raise ValueError("Backbone provenance training_protocol must be a JSON object.")
    required_protocol = (
        "protocol",
        "run_kind",
        "source_split_manifest",
        "source_split_sha256",
        "training_domains",
        "excluded_domains",
        "external_pretraining_declaration",
    )
    missing_protocol = [field for field in required_protocol if not training_protocol.get(field)]
    if missing_protocol:
        raise ValueError(f"Backbone provenance training_protocol is missing required fields: {missing_protocol}")
    if training_protocol["protocol"] != "leave_one_domain_out_backbone_training":
        raise ValueError("Backbone provenance does not record leave-one-domain-out backbone training.")
    if training_protocol["run_kind"] != "full":
        raise ValueError("Strict held-out-domain evaluation requires a full backbone training run.")
    source_split_path = Path(training_protocol["source_split_manifest"]).expanduser().resolve()
    if not source_split_path.exists():
        raise FileNotFoundError(f"Backbone provenance source split manifest not found: {source_split_path}")
    with source_split_path.open("r", encoding="utf-8-sig") as handle:
        source_split = json.load(handle)
    if str(source_split.get("split_sha256", "")) != str(training_protocol["source_split_sha256"]):
        raise ValueError("Backbone provenance source split SHA-256 does not match the referenced split manifest.")
    excluded_domains = {str(domain) for domain in manifest["excluded_domains"]}
    training_domains = {str(domain) for domain in manifest["training_domains"]}
    missing_exclusions = sorted(set(held_out_domains) - excluded_domains)
    present_in_training = sorted(set(held_out_domains) & training_domains)
    protocol_training_domains = {str(domain) for domain in training_protocol["training_domains"]}
    protocol_excluded_domains = {str(domain) for domain in training_protocol["excluded_domains"]}
    if protocol_training_domains != training_domains or protocol_excluded_domains != excluded_domains:
        raise ValueError("Backbone provenance top-level and training_protocol domain declarations differ.")
    if missing_exclusions or present_in_training or manifest["target_domain_supervision_excluded"] is not True:
        raise ValueError(
            "Backbone provenance does not support strict held-out-domain evaluation: "
            f"missing_exclusions={missing_exclusions}, present_in_training={present_in_training}, "
            f"target_domain_supervision_excluded={manifest['target_domain_supervision_excluded']}"
        )
    return {
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual_hash,
        "held_out_domains": held_out_domains,
        "source_split_sha256": training_protocol["source_split_sha256"],
        "source_split_manifest": str(source_split_path),
        "manifest": manifest,
    }
