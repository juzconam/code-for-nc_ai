"""Diagnostics that test whether candidate samples represent useful multimodal posteriors."""

from __future__ import annotations

import torch


def posterior_sample_diagnostics(candidates, target=None, separation_threshold=1.0):
    if candidates.ndim != 5 or candidates.shape[1] < 2:
        raise ValueError("candidates must have shape [B,K,2,H,W] with K >= 2")
    flat = candidates.flatten(2)
    distances = torch.cdist(flat, flat) / (flat.shape[-1] ** 0.5)
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    pairwise = distances[upper].view(candidates.shape[0], -1)
    result = {
        "mean_pairwise_distance": pairwise.mean(dim=1),
        "multimodal_fraction": (pairwise.max(dim=1).values > float(separation_threshold)).float(),
        "candidate_dispersion": candidates.std(dim=1, unbiased=False).norm(dim=1).mean(dim=(1, 2)),
    }
    if target is not None:
        errors = (candidates - target.unsqueeze(1)).norm(dim=2).mean(dim=(2, 3))
        result.update({
            "oracle_epe": errors.min(dim=1).values,
            "mean_candidate_epe": errors.mean(dim=1),
            "oracle_gain": errors.mean(dim=1) - errors.min(dim=1).values,
        })
    return result
