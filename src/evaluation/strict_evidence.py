"""Auditable statistical evidence for stochastic predictions and interventions."""

from __future__ import annotations

import math

import torch


def _flatten_per_sample(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1)


def energy_score(samples: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Fair ensemble estimator of the multivariate energy score.

    The energy score is strictly proper for distributions with finite beta
    moments when 0 < beta < 2. The fair U-statistic excludes diagonal sample
    pairs, avoiding the finite-ensemble bias of the naive estimator.
    """
    if samples.ndim < 3 or target.shape != (samples.shape[0], *samples.shape[2:]):
        raise ValueError("samples must be [B,K,...] and target must be [B,...]")
    if samples.shape[1] < 2:
        raise ValueError("A fair energy score requires at least two candidates")
    if not 0.0 < float(beta) < 2.0:
        raise ValueError("beta must be in (0, 2)")
    batch, count = samples.shape[:2]
    flat_samples = samples.reshape(batch, count, -1)
    flat_target = _flatten_per_sample(target)
    dimension_scale = math.sqrt(float(flat_samples.shape[-1]))
    observation = torch.linalg.vector_norm(
        flat_samples - flat_target.unsqueeze(1), dim=2
    ).pow(float(beta)).mean(dim=1) / (dimension_scale ** float(beta))
    pairwise = torch.cdist(flat_samples, flat_samples).pow(float(beta))
    off_diagonal = ~torch.eye(count, device=samples.device, dtype=torch.bool)
    diversity = pairwise[:, off_diagonal].mean(dim=1) / (dimension_scale ** float(beta))
    return observation - 0.5 * diversity


def weighted_energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Weighted energy score for validation-fitted discrete candidate measures."""
    if samples.ndim < 3 or target.shape != (samples.shape[0], *samples.shape[2:]):
        raise ValueError("samples must be [B,K,...] and target must be [B,...]")
    if weights.shape != samples.shape[:2]:
        raise ValueError("weights must have shape [B,K]")
    if not 0.0 < float(beta) < 2.0:
        raise ValueError("beta must be in (0, 2)")
    normalized_weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    batch, count = samples.shape[:2]
    flat_samples = samples.reshape(batch, count, -1)
    flat_target = _flatten_per_sample(target)
    dimension_scale = math.sqrt(float(flat_samples.shape[-1]))
    observation_distance = torch.linalg.vector_norm(
        flat_samples - flat_target.unsqueeze(1), dim=2
    ).pow(float(beta)) / (dimension_scale ** float(beta))
    observation = (normalized_weights * observation_distance).sum(dim=1)
    pairwise = torch.cdist(flat_samples, flat_samples).pow(float(beta)) / (dimension_scale ** float(beta))
    pair_weights = normalized_weights.unsqueeze(2) * normalized_weights.unsqueeze(1)
    diversity = (pair_weights * pairwise).sum(dim=(1, 2))
    return observation - 0.5 * diversity


def deterministic_energy_score(prediction: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    dimension_scale = math.sqrt(float(prediction[0].numel()))
    return torch.linalg.vector_norm(
        _flatten_per_sample(prediction - target), dim=1
    ).pow(float(beta)) / (dimension_scale ** float(beta))


def variogram_score(samples: torch.Tensor, target: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """A spatial-dependence score using adjacent horizontal/vertical increments."""
    if samples.ndim != 5 or target.ndim != 4 or samples.shape[0] != target.shape[0]:
        raise ValueError("flow samples must be [B,K,C,H,W] and target [B,C,H,W]")
    if not 0.0 < float(p) <= 2.0:
        raise ValueError("p must be in (0, 2]")
    target_dx = (target[..., 1:] - target[..., :-1]).abs().pow(float(p))
    target_dy = (target[..., 1:, :] - target[..., :-1, :]).abs().pow(float(p))
    sample_dx = (samples[..., 1:] - samples[..., :-1]).abs().pow(float(p)).mean(dim=1)
    sample_dy = (samples[..., 1:, :] - samples[..., :-1, :]).abs().pow(float(p)).mean(dim=1)
    score_x = (sample_dx - target_dx).square().flatten(1).mean(dim=1)
    score_y = (sample_dy - target_dy).square().flatten(1).mean(dim=1)
    return 0.5 * (score_x + score_y)


def paired_bootstrap_mean_ci(values: torch.Tensor, confidence: float = 0.95, draws: int = 5000, seed: int = 3407):
    values = values.detach().flatten().cpu().float()
    if values.numel() < 2:
        raise ValueError("At least two paired effects are required")
    if not 0.0 < float(confidence) < 1.0 or int(draws) < 100:
        raise ValueError("Invalid confidence level or bootstrap draw count")
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randint(values.numel(), (int(draws), values.numel()), generator=generator)
    means = values[indices].mean(dim=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(values.mean()),
        "lower": float(torch.quantile(means, alpha)),
        "upper": float(torch.quantile(means, 1.0 - alpha)),
        "confidence": float(confidence),
        "count": int(values.numel()),
        "draws": int(draws),
    }


def paired_randomization_pvalue(effects: torch.Tensor, draws: int = 10000, seed: int = 3407) -> float:
    """Two-sided sign-flip test of the sharp null of zero unit effects."""
    effects = effects.detach().flatten().cpu().float()
    if effects.numel() < 2:
        raise ValueError("At least two paired effects are required")
    generator = torch.Generator().manual_seed(int(seed))
    observed = effects.mean().abs()
    signs = torch.randint(0, 2, (int(draws), effects.numel()), generator=generator).float() * 2.0 - 1.0
    null_statistics = (signs * effects).mean(dim=1).abs()
    return float(((null_statistics >= observed).sum() + 1) / (int(draws) + 1))


def algorithmic_intervention_effect(control_loss: torch.Tensor, treated_loss: torch.Tensor):
    """Identify the paired effect when both algorithmic potential outcomes are observed."""
    if control_loss.shape != treated_loss.shape:
        raise ValueError("control and treated losses must have identical sample coverage")
    benefit = control_loss.detach().flatten() - treated_loss.detach().flatten()
    interval = paired_bootstrap_mean_ci(benefit)
    return {
        "estimand": "E[Y(control)-Y(treated)] over the fixed benchmark population",
        "identification": "both algorithmic potential outcomes evaluated for every sample",
        "mean_benefit": interval["mean"],
        "confidence_interval": interval,
        "randomization_pvalue": paired_randomization_pvalue(benefit),
        "beneficial_sample_rate": float((benefit > 0).float().mean()),
        "harmful_sample_rate": float((benefit < 0).float().mean()),
        "zero_effect_rate": float((benefit == 0).float().mean()),
        "claim_boundary": (
            "This identifies the effect of toggling a deterministic algorithmic component on labelled benchmark "
            "samples; it does not identify a real-world causal relation between neural networks."
        ),
    }
