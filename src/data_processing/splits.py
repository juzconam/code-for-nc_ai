"""Deterministic, auditable dataset split utilities for supervised PIV benchmarks."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import ShuffleSplit


VALID_SPLIT_PROTOCOLS = ("legacy_global", "stratified_domain", "blocked_domain", "manifest")
VALID_SPLIT_NAMES = ("train", "validation", "test")
PURGED_SPLIT_NAME = "purged"
VALID_ASSIGNMENT_NAMES = VALID_SPLIT_NAMES + (PURGED_SPLIT_NAME,)
VALID_FINGERPRINT_MODES = ("triplet_sha256", "flow_sha256", "path_only")


def sample_uid_from_paths(image1_path, image2_path, flow_path, domain_name=None):
    domain_name = domain_name or Path(flow_path).parent.name
    canonical = "\x1f".join(
        [
            str(domain_name),
            Path(image1_path).name,
            Path(image2_path).name,
            Path(flow_path).name,
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sample_records(image1_paths, image2_paths, flow_paths, fingerprint_mode="path_only"):
    if not (len(image1_paths) == len(image2_paths) == len(flow_paths)):
        raise ValueError("Image and flow path lists must have the same length.")
    if fingerprint_mode not in VALID_FINGERPRINT_MODES:
        raise ValueError(f"Unknown dataset fingerprint mode: {fingerprint_mode}")
    records = []
    seen_uids = set()
    for index, (image1_path, image2_path, flow_path) in enumerate(zip(image1_paths, image2_paths, flow_paths)):
        domain_name = Path(flow_path).parent.name
        sample_uid = sample_uid_from_paths(image1_path, image2_path, flow_path, domain_name)
        if sample_uid in seen_uids:
            raise ValueError(
                "Duplicate sample UID detected. Filenames must be unique within a domain or an explicit "
                f"split manifest must use distinct samples: {sample_uid}"
            )
        seen_uids.add(sample_uid)
        record = {
            "index": int(index),
            "sample_uid": sample_uid,
            "domain": domain_name,
            "image1_name": Path(image1_path).name,
            "image2_name": Path(image2_path).name,
            "flow_name": Path(flow_path).name,
        }
        if fingerprint_mode in ("triplet_sha256", "flow_sha256"):
            record["flow_sha256"] = _file_sha256(flow_path)
        if fingerprint_mode == "triplet_sha256":
            record["image1_sha256"] = _file_sha256(image1_path)
            record["image2_sha256"] = _file_sha256(image2_path)
        records.append(record)
    return records


def _validate_split_fraction(test_size):
    test_size = float(test_size)
    if not 0.0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5.")
    return test_size


def _legacy_global_assignments(records, split_seed, test_size):
    total_indices = np.arange(len(records))
    source_train_indices, test_indices = next(
        ShuffleSplit(n_splits=1, test_size=test_size, random_state=int(split_seed)).split(total_indices, total_indices)
    )
    train_positions, validation_positions = next(
        ShuffleSplit(n_splits=1, test_size=test_size, random_state=int(split_seed)).split(
            source_train_indices, source_train_indices
        )
    )
    assignments = {}
    for index in source_train_indices[train_positions]:
        assignments[int(index)] = "train"
    for index in source_train_indices[validation_positions]:
        assignments[int(index)] = "validation"
    for index in total_indices[test_indices]:
        assignments[int(index)] = "test"
    return assignments


def _stratified_domain_assignments(records, split_seed, test_size):
    by_domain = defaultdict(list)
    for record in records:
        by_domain[record["domain"]].append(record["index"])
    assignments = {}
    for domain_offset, domain_name in enumerate(sorted(by_domain)):
        domain_indices = np.asarray(sorted(by_domain[domain_name]), dtype=np.int64)
        sample_count = len(domain_indices)
        if sample_count < 3:
            raise ValueError(
                f"Domain '{domain_name}' has only {sample_count} samples; at least 3 are required "
                "for stratified train/validation/test splitting."
            )
        rng = np.random.default_rng(int(split_seed) + domain_offset * 1009)
        shuffled = rng.permutation(domain_indices)
        test_count = max(1, int(round(sample_count * test_size)))
        remaining_count = sample_count - test_count
        validation_count = max(1, int(round(remaining_count * test_size)))
        train_count = remaining_count - validation_count
        if train_count < 1:
            raise ValueError(
                f"Domain '{domain_name}' cannot retain a training sample with test_size={test_size}."
            )
        for index in shuffled[:test_count]:
            assignments[int(index)] = "test"
        for index in shuffled[test_count:test_count + validation_count]:
            assignments[int(index)] = "validation"
        for index in shuffled[test_count + validation_count:]:
            assignments[int(index)] = "train"
    return assignments


def _blocked_domain_assignments(records, test_size, purge_gap):
    purge_gap = max(0, int(purge_gap))
    by_domain = defaultdict(list)
    for record in records:
        by_domain[record["domain"]].append(record["index"])
    assignments = {}
    for domain_name in sorted(by_domain):
        domain_indices = np.asarray(sorted(by_domain[domain_name]), dtype=np.int64)
        sample_count = len(domain_indices)
        test_count = max(1, int(round(sample_count * test_size)))
        validation_count = max(1, int(round((sample_count - test_count) * test_size)))
        train_count = sample_count - test_count - validation_count - 2 * purge_gap
        if train_count < 1:
            raise ValueError(
                f"Domain '{domain_name}' cannot support blocked splitting with purge_gap={purge_gap}. "
                "Reduce the gap or provide an explicit group-aware manifest."
            )
        train_end = train_count
        validation_start = train_end + purge_gap
        validation_end = validation_start + validation_count
        test_start = validation_end + purge_gap
        for index in domain_indices[:train_end]:
            assignments[int(index)] = "train"
        for index in domain_indices[validation_start:validation_end]:
            assignments[int(index)] = "validation"
        for index in domain_indices[train_end:validation_start]:
            assignments[int(index)] = PURGED_SPLIT_NAME
        for index in domain_indices[validation_end:test_start]:
            assignments[int(index)] = PURGED_SPLIT_NAME
        for index in domain_indices[test_start:test_start + test_count]:
            assignments[int(index)] = "test"
    return assignments


def _load_manifest_assignments(manifest_path, records, verify_fingerprints=True):
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    rows = payload.get("samples", payload.get("sample_splits", []))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Split manifest must contain a non-empty 'samples' or 'sample_splits' list.")
    split_by_uid = {}
    expected_fingerprint_by_uid = {}
    group_by_uid = {}
    group_split_by_id = {}
    for row in rows:
        sample_uid = str(row.get("sample_uid", ""))
        split_name = str(row.get("split", ""))
        if not sample_uid or split_name not in VALID_ASSIGNMENT_NAMES:
            raise ValueError(
                "Each split manifest row requires sample_uid and "
                "split=train|validation|test|purged."
            )
        if sample_uid in split_by_uid:
            raise ValueError(f"Duplicate sample_uid in split manifest: {sample_uid}")
        split_by_uid[sample_uid] = split_name
        group_id = str(row.get("group_id", "")).strip()
        if group_id:
            prior_split = group_split_by_id.get(group_id)
            if prior_split is not None and prior_split != split_name:
                raise ValueError(
                    f"Group '{group_id}' spans multiple split labels in the manifest: "
                    f"{prior_split} and {split_name}."
                )
            group_split_by_id[group_id] = split_name
            group_by_uid[sample_uid] = group_id
        expected_fingerprint_by_uid[sample_uid] = {
            field: str(row[field])
            for field in ("image1_sha256", "image2_sha256", "flow_sha256")
            if row.get(field)
        }
    assignments = {}
    missing_uids = []
    for record in records:
        split_name = split_by_uid.get(record["sample_uid"])
        if split_name is None:
            missing_uids.append(record["sample_uid"])
        else:
            expected_fingerprints = expected_fingerprint_by_uid.get(record["sample_uid"], {})
            if verify_fingerprints:
                for field, expected_value in expected_fingerprints.items():
                    actual_value = record.get(field)
                    if actual_value and expected_value != actual_value:
                        raise ValueError(
                            "Split manifest content fingerprint does not match the current dataset for sample_uid "
                            f"{record['sample_uid']}. Use a derived manifest for transformed data."
                        )
            assignments[record["index"]] = split_name
    if missing_uids:
        raise ValueError(
            f"Split manifest does not cover {len(missing_uids)} current samples; first UID: {missing_uids[0]}"
        )
    return assignments, str(manifest_path), group_by_uid, payload


def _apply_ratio(assignments, ratio):
    ratio = float(ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1].")
    if ratio == 1.0:
        return assignments
    retained = {
        index: split_name
        for index, split_name in assignments.items()
        if split_name not in VALID_SPLIT_NAMES
    }
    for split_name in VALID_SPLIT_NAMES:
        indices = sorted(index for index, value in assignments.items() if value == split_name)
        keep_count = max(1, int(len(indices) * ratio)) if indices else 0
        for index in indices[:keep_count]:
            retained[index] = split_name
    return retained


def build_split_metadata(
    image1_paths,
    image2_paths,
    flow_paths,
    split_protocol="stratified_domain",
    split_seed=2,
    test_size=0.1,
    ratio=1.0,
    split_manifest_path="",
    fingerprint_mode="triplet_sha256",
    verify_manifest_fingerprints=True,
    purge_gap=0,
):
    split_protocol = str(split_protocol)
    if split_protocol not in VALID_SPLIT_PROTOCOLS:
        raise ValueError(f"Unknown split protocol: {split_protocol}")
    test_size = _validate_split_fraction(test_size)
    records = build_sample_records(image1_paths, image2_paths, flow_paths, fingerprint_mode=fingerprint_mode)
    if not records:
        raise ValueError("Cannot construct splits from an empty dataset.")
    source_manifest = ""
    source_manifest_metadata = {}
    group_by_uid = {}
    effective_split_seed = int(split_seed)
    effective_test_size = test_size
    effective_purge_gap = int(purge_gap)
    if split_protocol == "legacy_global":
        assignments = _legacy_global_assignments(records, split_seed, test_size)
    elif split_protocol == "stratified_domain":
        assignments = _stratified_domain_assignments(records, split_seed, test_size)
    elif split_protocol == "blocked_domain":
        assignments = _blocked_domain_assignments(records, test_size, purge_gap)
    else:
        if not split_manifest_path:
            raise ValueError("split_protocol=manifest requires --split-manifest.")
        assignments, source_manifest, group_by_uid, source_manifest_metadata = _load_manifest_assignments(
            split_manifest_path,
            records,
            verify_fingerprints=verify_manifest_fingerprints,
        )
        effective_split_seed = int(source_manifest_metadata.get("split_seed", split_seed))
        effective_test_size = float(source_manifest_metadata.get("test_size", test_size))
        effective_purge_gap = int(source_manifest_metadata.get("purge_gap", purge_gap))
    assignments = _apply_ratio(assignments, ratio)

    rows = []
    for record in records:
        split_name = assignments.get(record["index"])
        if split_name is None:
            continue
        row = {**record, "split": split_name}
        if record["sample_uid"] in group_by_uid:
            row["group_id"] = group_by_uid[record["sample_uid"]]
        rows.append(row)
    split_counts = {
        split_name: sum(row["split"] == split_name for row in rows)
        for split_name in VALID_SPLIT_NAMES
    }
    if set(split_counts) != set(VALID_SPLIT_NAMES):
        raise ValueError(f"Split assignment must contain train, validation, and test sets; got {split_counts}")
    canonical_rows = sorted(rows, key=lambda row: row["sample_uid"])
    digest = hashlib.sha256(
        json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 2,
        "split_protocol": split_protocol,
        "split_seed": effective_split_seed,
        "test_size": effective_test_size,
        "ratio": float(ratio),
        "purge_gap": effective_purge_gap,
        "dataset_fingerprint_mode": fingerprint_mode,
        "verified_source_fingerprints": bool(verify_manifest_fingerprints),
        "source_manifest": source_manifest or None,
        "source_split_protocol": source_manifest_metadata.get("split_protocol") if source_manifest else None,
        "source_split_sha256": source_manifest_metadata.get("split_sha256") if source_manifest else None,
        "grouping_strategy": source_manifest_metadata.get(
            "grouping_strategy",
            "explicit_group_manifest" if group_by_uid else "none",
        )
        if split_protocol == "manifest"
        else ("ordered_blocks_with_purge_gap" if split_protocol == "blocked_domain" else "randomised_sample_split"),
        "sample_count": sum(split_counts.values()),
        "total_source_sample_count": len(rows),
        "purged_sample_count": sum(row["split"] == PURGED_SPLIT_NAME for row in rows),
        "split_counts": split_counts,
        "split_sha256": digest,
        "samples": sorted(rows, key=lambda row: (row["domain"], row["sample_uid"])),
    }


def split_indices_from_metadata(split_metadata):
    indices = {split_name: [] for split_name in VALID_SPLIT_NAMES}
    for row in split_metadata["samples"]:
        split_name = row["split"]
        if split_name in indices:
            indices[split_name].append(int(row["index"]))
    return {split_name: np.asarray(sorted(values), dtype=np.int64) for split_name, values in indices.items()}


def write_split_manifest(path, split_metadata):
    path = Path(path)
    path.write_text(json.dumps(split_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
