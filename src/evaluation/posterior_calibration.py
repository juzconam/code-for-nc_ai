"""Validation-only calibration for candidate posterior weights and quantities."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def fit_candidate_reweighting(features, candidate_errors, steps=250, lr=0.05):
    """Fit a small, oracle-free-at-test-time candidate weighting rule.

    ``features`` has shape [N,K,D] and is computed only from images and
    physics energies.  ``candidate_errors`` is used only on the validation
    split to learn the rule; it is never needed by ``apply_candidate_weights``.
    """
    if features.ndim != 3 or candidate_errors.ndim != 2 or features.shape[:2] != candidate_errors.shape:
        raise ValueError("features must be [N,K,D] and errors must be [N,K]")
    mean = features.mean(dim=(0, 1))
    scale = features.std(dim=(0, 1), unbiased=False).clamp(min=1e-6)
    normalized = (features - mean) / scale
    coefficient = torch.zeros(features.shape[-1], dtype=features.dtype)
    log_temperature = torch.zeros((), dtype=features.dtype)
    coefficient.requires_grad_(True)
    log_temperature.requires_grad_(True)
    optimizer = torch.optim.Adam([coefficient, log_temperature], lr=float(lr))
    for _ in range(max(int(steps), 1)):
        logits = torch.einsum("nkd,d->nk", normalized, coefficient)
        weights = torch.softmax(logits / log_temperature.exp().clamp(min=0.05), dim=1)
        weighted_error = (weights * candidate_errors).sum(dim=1).mean()
        entropy_penalty = (weights * torch.log(weights.clamp(min=1e-8))).sum(dim=1).mean()
        loss = weighted_error + 0.002 * entropy_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficient": coefficient.detach().tolist(),
        "temperature": float(log_temperature.detach().exp().item()),
        "feature_names": ["particle_observation", "continuity", "ncc"],
        "validation_count": int(features.shape[0]),
        "candidate_count": int(features.shape[1]),
    }


def apply_candidate_weights(features, calibration):
    """Apply validation-fitted candidate weights to image/physics features."""
    mean = features.new_tensor(calibration["feature_mean"])
    scale = features.new_tensor(calibration["feature_scale"]).clamp(min=1e-6)
    coefficient = features.new_tensor(calibration["coefficient"])
    temperature = max(float(calibration["temperature"]), 0.05)
    normalized = (features - mean.view(1, 1, -1)) / scale.view(1, 1, -1)
    logits = torch.einsum("nkd,d->nk", normalized, coefficient) / temperature
    return torch.softmax(logits, dim=1)


def fit_conformal_expansion(values, truth, lower_quantile=0.05, upper_quantile=0.95, coverage=0.90):
    """Fit an additive split-conformal expansion for one scientific quantity."""
    if values.ndim != 2 or truth.ndim != 1 or values.shape[0] != truth.shape[0]:
        raise ValueError("values must be [N,K] and truth must be [N]")
    lower = torch.quantile(values, float(lower_quantile), dim=1)
    upper = torch.quantile(values, float(upper_quantile), dim=1)
    nonconformity = torch.maximum(lower - truth, truth - upper).clamp(min=0.0)
    rank = min(int(torch.ceil(torch.tensor((nonconformity.numel() + 1) * float(coverage))).item()), nonconformity.numel())
    radius = torch.kthvalue(nonconformity, k=max(rank, 1)).values
    return {
        "radius": float(radius.item()),
        "coverage": float(coverage),
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "count": int(nonconformity.numel()),
    }


def apply_conformal_expansion(values, calibration):
    alpha = (1.0 - float(calibration["coverage"])) / 2.0
    lower = torch.quantile(values, float(calibration["lower_quantile"]), dim=1)
    upper = torch.quantile(values, float(calibration["upper_quantile"]), dim=1)
    radius = values.new_tensor(float(calibration["radius"]))
    return lower - radius, upper + radius


def save_calibration(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
