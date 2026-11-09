"""Validation for independently acquired experimental PIV evidence."""

from __future__ import annotations


ACTIVE_SPLITS = {"train", "validation", "test"}
REQUIRED_DATASET_FIELDS = (
    "dataset_kind",
    "independent_of_model_development",
    "acquisition",
    "ground_truth_or_uncertainty_reference",
    "data_split_policy",
    "availability",
    "split_commitment",
)


def validate_experimental_protocol(dataset_manifest, split_manifest):
    """Validate declared acquisition provenance and a sealed group-aware test split."""
    issues = []
    missing_dataset_fields = [field for field in REQUIRED_DATASET_FIELDS if not dataset_manifest.get(field)]
    if missing_dataset_fields:
        issues.append(f"missing dataset manifest fields: {missing_dataset_fields}")
    if dataset_manifest.get("dataset_kind") != "experimental":
        issues.append("dataset_kind must be experimental")
    if dataset_manifest.get("independent_of_model_development") is not True:
        issues.append("independent_of_model_development must be true")
    if split_manifest.get("dataset_fingerprint_mode") != "triplet_sha256":
        issues.append("split must use dataset_fingerprint_mode=triplet_sha256")
    if split_manifest.get("grouping_strategy") != "explicit_group_manifest":
        issues.append("split grouping_strategy must be explicit_group_manifest")

    group_to_split = {}
    active_rows = [row for row in split_manifest.get("samples", []) if row.get("split") in ACTIVE_SPLITS]
    if not active_rows:
        issues.append("split has no active train/validation/test samples")
    for row in active_rows:
        group_id = str(row.get("group_id", "")).strip()
        split_name = str(row.get("split", ""))
        if not group_id:
            issues.append(f"sample {row.get('sample_uid', '<unknown>')} is missing group_id")
            continue
        prior = group_to_split.get(group_id)
        if prior is not None and prior != split_name:
            issues.append(f"group {group_id} spans {prior} and {split_name}")
        group_to_split[group_id] = split_name

    development_groups = {group for group, split in group_to_split.items() if split in {"train", "validation"}}
    test_groups = {group for group, split in group_to_split.items() if split == "test"}
    commitment = dataset_manifest.get("split_commitment", {})
    if not isinstance(commitment, dict):
        issues.append("split_commitment must be an object")
        commitment = {}
    declared_development = {str(value).strip() for value in commitment.get("development_group_ids", []) if str(value).strip()}
    declared_test = {str(value).strip() for value in commitment.get("test_group_ids", []) if str(value).strip()}
    if commitment.get("test_set_sealed_before_model_selection") is not True:
        issues.append("split_commitment.test_set_sealed_before_model_selection must be true")
    if not declared_development:
        issues.append("split_commitment must declare development_group_ids")
    if not declared_test:
        issues.append("split_commitment must declare test_group_ids")
    if declared_development & declared_test:
        issues.append("declared development and test groups overlap")
    if declared_development and declared_development != development_groups:
        issues.append("declared development_group_ids do not match train/validation groups")
    if declared_test and declared_test != test_groups:
        issues.append("declared test_group_ids do not match test groups")
    if not development_groups or not test_groups:
        issues.append("group-aware split requires both development and test groups")
    if development_groups & test_groups:
        issues.append("development and test groups overlap")

    return {
        "available": not issues,
        "dataset_name": dataset_manifest.get("dataset_name", ""),
        "split_sha256": split_manifest.get("split_sha256"),
        "active_sample_count": len(active_rows),
        "group_count": len(group_to_split),
        "development_group_count": len(development_groups),
        "test_group_count": len(test_groups),
        "issues": issues,
        "claim_boundary": (
            "This audit validates declared provenance, group isolation, and test-set sealing only. It does not "
            "establish measurement accuracy, acquisition independence, or scientific impact."
        ),
    }
