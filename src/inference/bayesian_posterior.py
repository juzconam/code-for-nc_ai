"""Explicit Bayesian PIV posterior with an auditable Metropolis sampler."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from src.inference.piv_posterior import particle_image_forward


@dataclass(frozen=True)
class BayesianPIVConfig:
    """All density terms are explicit and dimensionality is recorded."""

    prior_mean: float = 0.0
    prior_std: float = 2.0
    observation_std: float = 0.05
    particle_sigma: float = 1.0
    flow_units: str = "displacement_px"
    intensity_units: str = "normalized_intensity"
    observation_direction: str = "I1_given_I2_and_F"

    def validate(self):
        if self.prior_std <= 0.0 or self.observation_std <= 0.0 or self.particle_sigma <= 0.0:
            raise ValueError("prior_std, observation_std and particle_sigma must be positive")
        if self.observation_direction != "I1_given_I2_and_F":
            raise ValueError("only the declared conditional observation model is supported")

    def as_dict(self):
        return asdict(self)


def _normal_log_prob(value, mean, std):
    variance = float(std) ** 2
    return -0.5 * (((value - mean) ** 2) / variance + math.log(2.0 * math.pi * variance))


def log_prior(flow, config: BayesianPIVConfig, prior_center=None):
    config.validate()
    mean = flow.new_full(flow.shape, float(config.prior_mean)) if prior_center is None else prior_center
    if mean.shape != flow.shape:
        raise ValueError("prior_center must match flow shape")
    return _normal_log_prob(flow, mean, config.prior_std).flatten(1).sum(dim=1)


def log_likelihood(image1, image2, flow, config: BayesianPIVConfig):
    config.validate()
    if image1.ndim != 4 or image2.shape != image1.shape or flow.ndim != 4 or flow.shape[0] != image1.shape[0]:
        raise ValueError("image1/image2 must be [B,C,H,W] and flow must be [B,2,H,W]")
    prediction = particle_image_forward(image2, flow, particle_sigma=config.particle_sigma)
    return _normal_log_prob(image1, prediction, config.observation_std).flatten(1).sum(dim=1)


def log_posterior(image1, image2, flow, config: BayesianPIVConfig, prior_center=None):
    return log_prior(flow, config, prior_center=prior_center) + log_likelihood(image1, image2, flow, config)


def posterior_energy(image1, image2, flow, config: BayesianPIVConfig, prior_center=None):
    return -log_posterior(image1, image2, flow, config, prior_center=prior_center)


@torch.no_grad()
def metropolis_hastings(
    initial_flow,
    image1,
    image2,
    config: BayesianPIVConfig,
    steps: int = 200,
    burn_in: int = 100,
    thin: int = 1,
    proposal_std: float = 0.05,
    prior_center=None,
    seed: int = 3407,
):
    """Random-walk MH targeting the explicit posterior density.

    The returned chain is approximate only because of finite burn-in and
    autocorrelation; acceptance and effective sample size are reported.
    """
    config.validate()
    if initial_flow.ndim != 4 or initial_flow.shape[1] != 2:
        raise ValueError("initial_flow must have shape [B,2,H,W]")
    if image1.shape[0] != initial_flow.shape[0] or image2.shape != image1.shape:
        raise ValueError("initial_flow and images must have matching batch dimensions")
    if int(steps) <= 0 or int(burn_in) < 0 or int(thin) <= 0 or float(proposal_std) <= 0.0:
        raise ValueError("steps, burn_in, thin and proposal_std are invalid")
    generator = torch.Generator(device=initial_flow.device).manual_seed(int(seed))
    current = initial_flow.detach().clone()
    current_logp = log_posterior(image1, image2, current, config, prior_center=prior_center)
    collected = []
    accepted = 0
    total = int(burn_in) + int(steps) * int(thin)
    for iteration in range(total):
        proposal = current + torch.randn(current.shape, generator=generator, device=current.device, dtype=current.dtype) * float(proposal_std)
        proposal_logp = log_posterior(image1, image2, proposal, config, prior_center=prior_center)
        log_accept = proposal_logp - current_logp
        uniform = torch.rand(log_accept.shape, generator=generator, device=current.device, dtype=current.dtype).clamp(min=1e-8).log()
        accept = uniform < log_accept
        current = torch.where(accept.view(-1, 1, 1, 1), proposal, current)
        current_logp = torch.where(accept, proposal_logp, current_logp)
        accepted += int(accept.sum().item())
        if iteration >= int(burn_in) and ((iteration - int(burn_in)) % int(thin) == 0):
            collected.append(current.clone())
    chain = torch.stack(collected, dim=1)
    mean = chain.mean(dim=1)
    centered = chain - mean.unsqueeze(1)
    variance = centered.square().mean(dim=1)
    autocorrelation_lag1 = (
        (centered[:, :-1] * centered[:, 1:]).mean(dim=(1, 2, 3, 4))
        / centered.square().mean(dim=(1, 2, 3, 4)).clamp(min=1e-12)
    ) if chain.shape[1] > 1 else torch.zeros(chain.shape[0], device=chain.device)
    ess = chain.shape[1] / (1.0 + 2.0 * autocorrelation_lag1.clamp(min=-0.99, max=0.99))
    return {
        "samples": chain,
        "posterior_mean": mean,
        "posterior_std": torch.sqrt(variance + 1e-12),
        "acceptance_rate": float(accepted / max(total * initial_flow.shape[0], 1)),
        "effective_sample_size": ess,
        "config": config.as_dict(),
        "steps": int(steps),
        "burn_in": int(burn_in),
        "thin": int(thin),
        "proposal_std": float(proposal_std),
    }


@torch.no_grad()
def metropolis_hastings_translation(
    initial_theta,
    image1,
    image2,
    config: BayesianPIVConfig,
    steps: int = 500,
    burn_in: int = 250,
    thin: int = 1,
    proposal_std: float = 0.05,
    seed: int = 3407,
):
    """Exact MH on a two-parameter constant-translation posterior.

    This low-dimensional parameterization is used for SBC. It avoids making
    a high-dimensional random-walk chain appear calibrated merely because it
    has not mixed; the full-field sampler remains available above.
    """
    config.validate()
    if initial_theta.ndim != 2 or initial_theta.shape[1] != 2:
        raise ValueError("initial_theta must have shape [B,2]")
    if image1.ndim != 4 or image2.shape != image1.shape:
        raise ValueError("images must have shape [B,C,H,W]")
    if image1.shape[0] != initial_theta.shape[0]:
        raise ValueError("batch dimensions must match")
    if int(steps) <= 0 or int(burn_in) < 0 or int(thin) <= 0 or float(proposal_std) <= 0.0:
        raise ValueError("invalid chain parameters")

    def log_theta_posterior(theta):
        flow = theta[:, :, None, None].expand(-1, -1, image1.shape[-2], image1.shape[-1])
        prior = _normal_log_prob(theta, float(config.prior_mean), config.prior_std).sum(dim=1)
        return prior + log_likelihood(image1, image2, flow, config)

    generator = torch.Generator(device=initial_theta.device).manual_seed(int(seed))
    current = initial_theta.detach().clone()
    current_logp = log_theta_posterior(current)
    collected = []
    accepted = 0
    total = int(burn_in) + int(steps) * int(thin)
    for iteration in range(total):
        proposal = current + torch.randn(current.shape, generator=generator, device=current.device, dtype=current.dtype) * float(proposal_std)
        proposal_logp = log_theta_posterior(proposal)
        log_accept = proposal_logp - current_logp
        uniform = torch.rand(log_accept.shape, generator=generator, device=current.device, dtype=current.dtype).clamp(min=1e-8).log()
        accept = uniform < log_accept
        current = torch.where(accept[:, None], proposal, current)
        current_logp = torch.where(accept, proposal_logp, current_logp)
        accepted += int(accept.sum().item())
        if iteration >= int(burn_in) and ((iteration - int(burn_in)) % int(thin) == 0):
            collected.append(current.clone())
    theta_chain = torch.stack(collected, dim=1)
    flow_chain = theta_chain[:, :, :, None, None].expand(-1, -1, -1, image1.shape[-2], image1.shape[-1]).contiguous()
    return {
        "parameter_samples": theta_chain,
        "samples": flow_chain,
        "posterior_mean": theta_chain.mean(dim=1),
        "posterior_std": theta_chain.std(dim=1, unbiased=False),
        "acceptance_rate": float(accepted / max(total * initial_theta.shape[0], 1)),
        "config": config.as_dict(),
        "steps": int(steps),
        "burn_in": int(burn_in),
        "thin": int(thin),
        "proposal_std": float(proposal_std),
        "parameterization": "constant_translation",
    }


@torch.no_grad()
def pcn_metropolis(
    initial_flow,
    image1,
    image2,
    config: BayesianPIVConfig,
    steps: int = 200,
    burn_in: int = 100,
    thin: int = 1,
    beta: float = 0.05,
    prior_center=None,
    seed: int = 3407,
):
    """Prior-preserving pCN MH for the explicit full-field posterior.

    With a Gaussian prior, proposal = center + sqrt(1-beta^2)*(x-center)
    + beta*prior_noise leaves the prior invariant, so the MH ratio contains
    only the normalized observation likelihood. This is exact in stationarity
    and remains viable in high-dimensional fields, unlike a pixelwise random
    walk. The finite chain is still approximate and reports ESS diagnostics.
    """
    config.validate()
    if initial_flow.ndim != 4 or initial_flow.shape[1] != 2:
        raise ValueError("initial_flow must have shape [B,2,H,W]")
    if image1.shape != image2.shape or image1.shape[0] != initial_flow.shape[0]:
        raise ValueError("images and initial flow have incompatible shapes")
    if not 0.0 < float(beta) <= 1.0:
        raise ValueError("beta must be in (0,1]")
    if int(steps) <= 0 or int(burn_in) < 0 or int(thin) <= 0:
        raise ValueError("invalid chain parameters")
    center = initial_flow.new_full(initial_flow.shape, float(config.prior_mean)) if prior_center is None else prior_center
    if center.shape != initial_flow.shape:
        raise ValueError("prior_center must match initial flow")

    def likelihood(flow):
        return log_likelihood(image1, image2, flow, config)

    generator = torch.Generator(device=initial_flow.device).manual_seed(int(seed))
    current = initial_flow.detach().clone()
    current_loglik = likelihood(current)
    collected = []
    accepted = 0
    total = int(burn_in) + int(steps) * int(thin)
    prior_scale = float(config.prior_std)
    coefficient = math.sqrt(max(1.0 - float(beta) ** 2, 0.0))
    for iteration in range(total):
        prior_noise = torch.randn(current.shape, generator=generator, device=current.device, dtype=current.dtype) * prior_scale
        proposal = center + coefficient * (current - center) + float(beta) * prior_noise
        proposal_loglik = likelihood(proposal)
        log_accept = proposal_loglik - current_loglik
        uniform = torch.rand(log_accept.shape, generator=generator, device=current.device, dtype=current.dtype).clamp(min=1e-8).log()
        accept = uniform < log_accept
        current = torch.where(accept.view(-1, 1, 1, 1), proposal, current)
        current_loglik = torch.where(accept, proposal_loglik, current_loglik)
        accepted += int(accept.sum().item())
        if iteration >= int(burn_in) and ((iteration - int(burn_in)) % int(thin) == 0):
            collected.append(current.clone())
    chain = torch.stack(collected, dim=1)
    mean = chain.mean(dim=1)
    centered = chain - mean.unsqueeze(1)
    variance = centered.square().mean(dim=1)
    autocorrelation_lag1 = (
        (centered[:, :-1] * centered[:, 1:]).mean(dim=(1, 2, 3, 4))
        / centered.square().mean(dim=(1, 2, 3, 4)).clamp(min=1e-12)
    ) if chain.shape[1] > 1 else torch.zeros(chain.shape[0], device=chain.device)
    ess = chain.shape[1] / (1.0 + 2.0 * autocorrelation_lag1.clamp(min=-0.99, max=0.99))
    return {
        "samples": chain,
        "posterior_mean": mean,
        "posterior_std": torch.sqrt(variance + 1e-12),
        "acceptance_rate": float(accepted / max(total * initial_flow.shape[0], 1)),
        "effective_sample_size": ess,
        "config": config.as_dict(),
        "steps": int(steps),
        "burn_in": int(burn_in),
        "thin": int(thin),
        "beta": float(beta),
        "sampler": "pCN_MH_prior_preserving",
    }


def mala_metropolis(
    initial_flow,
    image1,
    image2,
    config: BayesianPIVConfig,
    steps: int = 200,
    burn_in: int = 100,
    thin: int = 1,
    step_size: float = 0.005,
    prior_center=None,
    seed: int = 3407,
):
    """Metropolis-adjusted Langevin sampler for the explicit full-field target."""
    config.validate()
    if initial_flow.ndim != 4 or initial_flow.shape[1] != 2:
        raise ValueError("initial_flow must have shape [B,2,H,W]")
    if image1.shape != image2.shape or image1.shape[0] != initial_flow.shape[0]:
        raise ValueError("images and initial flow have incompatible shapes")
    if float(step_size) <= 0.0 or int(steps) <= 0 or int(burn_in) < 0 or int(thin) <= 0:
        raise ValueError("invalid MALA parameters")
    center = initial_flow.new_full(initial_flow.shape, float(config.prior_mean)) if prior_center is None else prior_center
    if center.shape != initial_flow.shape:
        raise ValueError("prior_center must match initial flow")
    generator = torch.Generator(device=initial_flow.device).manual_seed(int(seed))
    current = initial_flow.detach().clone().requires_grad_(True)

    def value_and_grad(flow):
        variable = flow.detach().requires_grad_(True)
        value = log_posterior(image1, image2, variable, config, prior_center=center)
        gradient = torch.autograd.grad(value.sum(), variable, only_inputs=True)[0]
        return value.detach(), gradient.detach()

    current_logp, current_grad = value_and_grad(current)
    collected = []
    accepted = 0
    total = int(burn_in) + int(steps) * int(thin)
    variance = float(step_size) ** 2
    for iteration in range(total):
        mean_forward = current.detach() + 0.5 * variance * current_grad
        proposal = mean_forward + torch.randn(current.shape, generator=generator, device=current.device, dtype=current.dtype) * float(step_size)
        proposal_logp, proposal_grad = value_and_grad(proposal)
        mean_reverse = proposal + 0.5 * variance * proposal_grad
        log_q_reverse = -((current.detach() - mean_reverse).square().flatten(1).sum(dim=1)) / (2.0 * variance)
        log_q_forward = -((proposal - mean_forward).square().flatten(1).sum(dim=1)) / (2.0 * variance)
        log_accept = proposal_logp - current_logp + log_q_reverse - log_q_forward
        uniform = torch.rand(log_accept.shape, generator=generator, device=current.device, dtype=current.dtype).clamp(min=1e-8).log()
        accept = uniform < log_accept
        current = torch.where(accept.view(-1, 1, 1, 1), proposal, current.detach()).requires_grad_(False)
        current_logp = torch.where(accept, proposal_logp, current_logp)
        current_grad = torch.where(accept.view(-1, 1, 1, 1), proposal_grad, current_grad)
        accepted += int(accept.sum().item())
        if iteration >= int(burn_in) and ((iteration - int(burn_in)) % int(thin) == 0):
            collected.append(current.clone())
    chain = torch.stack(collected, dim=1)
    mean = chain.mean(dim=1)
    centered = chain - mean.unsqueeze(1)
    variance_chain = centered.square().mean(dim=1)
    autocorrelation_lag1 = (
        (centered[:, :-1] * centered[:, 1:]).mean(dim=(1, 2, 3, 4))
        / centered.square().mean(dim=(1, 2, 3, 4)).clamp(min=1e-12)
    ) if chain.shape[1] > 1 else torch.zeros(chain.shape[0], device=chain.device)
    ess = chain.shape[1] / (1.0 + 2.0 * autocorrelation_lag1.clamp(min=-0.99, max=0.99))
    return {
        "samples": chain,
        "posterior_mean": mean,
        "posterior_std": torch.sqrt(variance_chain + 1e-12),
        "acceptance_rate": float(accepted / max(total * initial_flow.shape[0], 1)),
        "effective_sample_size": ess,
        "config": config.as_dict(),
        "steps": int(steps),
        "burn_in": int(burn_in),
        "thin": int(thin),
        "step_size": float(step_size),
        "sampler": "MALA",
    }


def simulate_observation(image2, true_flow, config: BayesianPIVConfig, generator=None):
    config.validate()
    prediction = particle_image_forward(image2, true_flow, particle_sigma=config.particle_sigma)
    noise = torch.randn(prediction.shape, generator=generator, device=prediction.device, dtype=prediction.dtype) * float(config.observation_std)
    return prediction + noise


def rank_statistic(samples, truth, component="mean"):
    if samples.ndim != 5 or truth.ndim != 4 or samples.shape[0] != truth.shape[0]:
        raise ValueError("samples must be [B,S,2,H,W], truth [B,2,H,W]")
    if component == "mean":
        sample_value = samples.mean(dim=(2, 3, 4))
        truth_value = truth.mean(dim=(1, 2, 3))
    elif component == "u":
        sample_value = samples[:, :, 0].mean(dim=(2, 3))
        truth_value = truth[:, 0].mean(dim=(1, 2))
    elif component == "v":
        sample_value = samples[:, :, 1].mean(dim=(2, 3))
        truth_value = truth[:, 1].mean(dim=(1, 2))
    else:
        raise ValueError("component must be mean, u or v")
    return (sample_value < truth_value.unsqueeze(1)).sum(dim=1)
