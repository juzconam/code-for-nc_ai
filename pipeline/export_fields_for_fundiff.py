"""Export PIV velocity fields for the FunDiff cross-check.

FunDiff expects fields shaped (N, H, W, 2) with values in a roughly unit range.  We take
the test/train split used by the uncertainty study, area-average each 256x256 field down
to 64x64 (the compute budget is one CPU), and store the fields together with the derived
scalars and the pixel spacing, so that the FunDiff side can compute errors against truth
without needing torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_piv_uncertainty_study import IndexedSplit, build_datasets

OUT = ROOT / "external" / "fundiff_data"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = 64


def downsample(field: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return field
    n, h, w = field.shape
    field = field[:, : h - h % factor, : w - w % factor]
    return field.reshape(n, h // factor, factor, w // factor, factor).mean(axis=(2, 4))


def main():
    datasets, _manifest, domain_names, _uid_by_index, domain_by_index = build_datasets(
        r"D:\newpiv\sample_data", str(ROOT / "splits" / "class1_primary_split_v2.json")
    )
    dataset = datasets["test"]
    positions = list(range(len(dataset)))
    loader = DataLoader(IndexedSplit(dataset, positions), batch_size=8, shuffle=False, num_workers=0)

    fields, targets, domains = [], [], []
    for samples, target, domain_id, _indices in loader:
        # samples: [B, 2, H, W] image pair; target: [B, 2, H, W] ground-truth flow
        fields.append(target[:, :2].numpy())
        domains.append(domain_id.numpy())
    target = np.concatenate(fields, axis=0).astype(np.float32)   # (N, 2, H, W)
    domain = np.concatenate(domains, axis=0)

    factor = target.shape[-1] // SIZE
    u = downsample(target[:, 0], factor)
    v = downsample(target[:, 1], factor)
    uvw = np.stack([u, v], axis=-1).astype(np.float32)           # (N, S, S, 2)

    print(f"exported {uvw.shape}, downsample factor {factor}")
    print(f"u range [{u.min():.3f}, {u.max():.3f}]  v range [{v.min():.3f}, {v.max():.3f}]")

    np.savez_compressed(
        OUT / "piv_fields.npz",
        uvw=uvw,
        domain=domain,
        domain_names=np.array(domain_names),
        downsample_factor=factor,
    )
    print(f"wrote {OUT / 'piv_fields.npz'}")


if __name__ == "__main__":
    main()
