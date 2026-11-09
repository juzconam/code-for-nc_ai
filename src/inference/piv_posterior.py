"""Differentiable PIV observation model and physics-guided posterior energy."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from physics_utils import compute_divergence, compute_spatial_gradients, warp_tensor


@dataclass
class PosteriorGuidanceConfig:
    observation_weight: float = 0.0
    continuity_weight: float = 0.0
    smoothness_weight: float = 0.0
    particle_sigma: float = 1.0
    charbonnier_epsilon: float = 1e-3
    max_gradient_norm: float = 1.0
    gradient_step_size: float = 0.02
    start_fraction: float = 0.65

    @property
    def active(self):
        return self.observation_weight > 0 or self.continuity_weight > 0 or self.smoothness_weight > 0

    def as_dict(self):
        return asdict(self)


def _gaussian_kernel(sigma, device, dtype):
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma).square())
    return kernel / kernel.sum()


def gaussian_blur_psf(image, sigma):
    """Separable Gaussian PSF convolution shared by every particle-image forward model."""
    sigma = float(sigma)
    if sigma <= 0.0:
        return image
    kernel = _gaussian_kernel(sigma, image.device, image.dtype)
    channels = image.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padding = kernel.numel() // 2
    blurred = F.conv2d(image, horizontal, padding=(0, padding), groups=channels)
    return F.conv2d(blurred, vertical, padding=(padding, 0), groups=channels)


def particle_image_forward(image, flow, particle_sigma=1.0):
    """Advect a particle image and apply a differentiable Gaussian particle PSF."""
    advected = warp_tensor(image, flow)
    return gaussian_blur_psf(advected, particle_sigma)


def particle_observation_energy(image1, image2, flow, particle_sigma=1.0, epsilon=1e-3):
    """Robust heteroscedastic likelihood for particle-image advection.

    Local contrast normalization prevents bright/high-density interrogation regions
    from dominating the posterior guidance.
    """
    predicted = particle_image_forward(image2, flow, particle_sigma=particle_sigma)
    residual = image1 - predicted
    local_scale = F.avg_pool2d(residual.square(), 7, stride=1, padding=3).detach().sqrt().clamp(min=epsilon)
    return torch.sqrt((residual / local_scale).square() + float(epsilon) ** 2).mean()


def posterior_energy(image1, image2, flow, config, geometry=None):
    terms = {}
    total = flow.new_zeros(())
    if config.observation_weight > 0:
        terms["particle_observation"] = particle_observation_energy(
            image1, image2, flow, config.particle_sigma, config.charbonnier_epsilon
        )
        total = total + config.observation_weight * terms["particle_observation"]
    if config.continuity_weight > 0:
        terms["continuity"] = compute_divergence(flow, geometry=geometry).square().mean()
        total = total + config.continuity_weight * terms["continuity"]
    if config.smoothness_weight > 0:
        dx, dy = compute_spatial_gradients(flow)
        terms["smoothness"] = (dx.square() + dy.square()).mean()
        total = total + config.smoothness_weight * terms["smoothness"]
    return total, terms


def guided_clean_flow(predicted_clean, image1, image2, flow_scale, config, geometry=None, step_strength=1.0):
    """One normalized, bounded gradient step on the physical posterior energy."""
    if config is None or not config.active:
        return predicted_clean, {}
    with torch.enable_grad():
        clean = predicted_clean.detach().requires_grad_(True)
        physical_flow = clean * float(flow_scale)
        energy, terms = posterior_energy(image1, image2, physical_flow, config, geometry=geometry)
        gradient = torch.autograd.grad(energy, clean, only_inputs=True)[0]
        # Mean-reduced image energies produce gradients that shrink with image
        # resolution. Normalize per-sample RMS so guidance has a physical,
        # resolution-independent step, then retain a global safety bound.
        gradient_rms = gradient.flatten(1).square().mean(dim=1).sqrt().view(-1, 1, 1, 1).clamp(min=1e-8)
        update = gradient / gradient_rms * float(config.gradient_step_size) * float(step_strength)
        update_norm = update.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=1e-8)
        update = update * (float(config.max_gradient_norm) / update_norm).clamp(max=1.0)
        guided = clean - update
    return guided.detach(), {name: float(value.detach().item()) for name, value in terms.items()}


def posterior_risk(candidates, image1, image2, config, geometry=None):
    """Per-sample risk combining posterior ambiguity and observation mismatch."""
    # After reducing candidate dimension the tensor is [B,2,H,W]; vector
    # magnitude must therefore reduce channel dimension 1, not height.
    dispersion = candidates.std(dim=1, unbiased=False).norm(dim=1).mean(dim=(1, 2))
    mean_flow = candidates.mean(dim=1)
    observation = torch.stack([
        particle_observation_energy(image1[i:i+1], image2[i:i+1], mean_flow[i:i+1], config.particle_sigma)
        for i in range(mean_flow.shape[0])
    ])
    return dispersion + observation
