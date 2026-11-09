from __future__ import annotations

import glob
import json
import os
from typing import List, Tuple

import numpy as np
from src.data_processing.custom_dataset import FlowDataset
from src.data_processing.splits import build_split_metadata, split_indices_from_metadata


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_all(data_path: str, write_index_cache: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """Read the full dataset from root-level index JSON or by scanning directories."""
    data_path = os.path.abspath(data_path)
    img1_json = os.path.join(data_path, "img1_name_list.json")
    img2_json = os.path.join(data_path, "img2_name_list.json")
    gt_json = os.path.join(data_path, "gt_name_list.json")

    if os.path.exists(img1_json) and os.path.exists(img2_json) and os.path.exists(gt_json):
        return (
            [os.path.abspath(p) for p in _load_json(img1_json)],
            [os.path.abspath(p) for p in _load_json(img2_json)],
            [os.path.abspath(p) for p in _load_json(gt_json)],
        )

    data_dir = sorted(d for d in glob.glob(os.path.join(data_path, "*")) if os.path.isdir(d))
    gt_name_list: List[str] = []
    img1_name_list: List[str] = []
    img2_name_list: List[str] = []

    for dir_path in data_dir:
        gt_name_list.extend(sorted(os.path.abspath(p) for p in glob.glob(os.path.join(dir_path, "*flow.flo"))))
        img1_name_list.extend(sorted(os.path.abspath(p) for p in glob.glob(os.path.join(dir_path, "*img1.tif"))))
        img2_name_list.extend(sorted(os.path.abspath(p) for p in glob.glob(os.path.join(dir_path, "*img2.tif"))))

    gt_name_list.sort()
    img1_name_list.sort()
    img2_name_list.sort()

    if not (len(gt_name_list) == len(img1_name_list) == len(img2_name_list)):
        raise ValueError("Inconsistent dataset files found while scanning the directory.")

    if write_index_cache:
        _dump_json(img1_json, img1_name_list)
        _dump_json(img2_json, img2_name_list)
        _dump_json(gt_json, gt_name_list)
    return img1_name_list, img2_name_list, gt_name_list


def read_by_type(data_path: str, write_index_cache: bool = False):
    """Read per-domain lists from directory-structured datasets."""
    data_path = os.path.abspath(data_path)
    data_dir = sorted(d for d in glob.glob(os.path.join(data_path, "*")) if os.path.isdir(d))

    flow_img1_name_list = []
    flow_img2_name_list = []
    flow_gt_name_list = []
    flow_dir = []

    try:
        for dir_path in data_dir:
            dir_name = os.path.basename(dir_path)
            flow_dir.append(dir_name)
            flow_img1_name_list.append(sorted(os.path.abspath(p) for p in _load_json(os.path.join(data_path, f"{dir_name}_img1_name_list.json"))))
            flow_img2_name_list.append(sorted(os.path.abspath(p) for p in _load_json(os.path.join(data_path, f"{dir_name}_img2_name_list.json"))))
            flow_gt_name_list.append(sorted(os.path.abspath(p) for p in _load_json(os.path.join(data_path, f"{dir_name}_gt_name_list.json"))))
    except Exception:
        flow_dir = []
        flow_img1_name_list = []
        flow_img2_name_list = []
        flow_gt_name_list = []

        for dir_path in data_dir:
            dir_name = os.path.basename(dir_path)
            sub_flow_gt_name_list = sorted(os.path.abspath(p) for p in glob.glob(os.path.join(dir_path, "*flow.flo")))
            sub_flow_img1_name_list = sorted(os.path.abspath(p) for p in glob.glob(os.path.join(dir_path, "*img1.tif")))
            sub_flow_img2_name_list = sorted(os.path.abspath(p) for p in glob.glob(os.path.join(dir_path, "*img2.tif")))

            if not (len(sub_flow_gt_name_list) == len(sub_flow_img1_name_list) == len(sub_flow_img2_name_list)):
                raise ValueError(f"Path mismatch in domain {dir_name}")

            if write_index_cache:
                _dump_json(os.path.join(data_path, f"{dir_name}_img1_name_list.json"), sub_flow_img1_name_list)
                _dump_json(os.path.join(data_path, f"{dir_name}_img2_name_list.json"), sub_flow_img2_name_list)
                _dump_json(os.path.join(data_path, f"{dir_name}_gt_name_list.json"), sub_flow_gt_name_list)

            flow_dir.append(dir_name)
            flow_img1_name_list.append(sub_flow_img1_name_list)
            flow_img2_name_list.append(sub_flow_img2_name_list)
            flow_gt_name_list.append(sub_flow_gt_name_list)

    return flow_img1_name_list, flow_img2_name_list, flow_gt_name_list, flow_dir


def construct_dataset(
    img1_name_list,
    img2_name_list,
    gt_name_list,
    domain_id_list=None,
    ratio=1.0,
    test_size=0.1,
    split_seed=2,
    split_protocol="stratified_domain",
    split_manifest_path="",
    fingerprint_mode="triplet_sha256",
    verify_manifest_fingerprints=True,
    purge_gap=0,
    return_metadata=False,
    prior_root=None,
    prior_channels=0,
):
    split_metadata = build_split_metadata(
        img1_name_list,
        img2_name_list,
        gt_name_list,
        split_protocol=split_protocol,
        split_seed=split_seed,
        test_size=test_size,
        ratio=ratio,
        split_manifest_path=split_manifest_path,
        fingerprint_mode=fingerprint_mode,
        verify_manifest_fingerprints=verify_manifest_fingerprints,
        purge_gap=purge_gap,
    )
    split_indices = split_indices_from_metadata(split_metadata)
    train_data = split_indices["train"]
    validate_data = split_indices["validation"]
    test_data = split_indices["test"]

    train_dataset = FlowDataset(
        train_data,
        [img1_name_list, img2_name_list],
        targets_index_list=train_data,
        targets=gt_name_list,
        domain_ids=domain_id_list,
        sample_uid_by_index={row["index"]: row["sample_uid"] for row in split_metadata["samples"]},
        prior_root=prior_root,
        prior_channels=prior_channels,
    )
    validate_dataset = FlowDataset(
        validate_data,
        [img1_name_list, img2_name_list],
        targets_index_list=validate_data,
        targets=gt_name_list,
        domain_ids=domain_id_list,
        sample_uid_by_index={row["index"]: row["sample_uid"] for row in split_metadata["samples"]},
        prior_root=prior_root,
        prior_channels=prior_channels,
    )
    test_dataset = FlowDataset(
        test_data,
        [img1_name_list, img2_name_list],
        targets_index_list=test_data,
        targets=gt_name_list,
        domain_ids=domain_id_list,
        sample_uid_by_index={row["index"]: row["sample_uid"] for row in split_metadata["samples"]},
        prior_root=prior_root,
        prior_channels=prior_channels,
    )
    if return_metadata:
        return train_dataset, validate_dataset, test_dataset, split_metadata
    return train_dataset, validate_dataset, test_dataset
