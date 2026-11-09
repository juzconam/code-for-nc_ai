"""Physics-constrained unrolled refinement for two-frame PIV."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics_utils import compute_divergence, compute_spatial_gradients, warp_tensor
from src.inference.piv_posterior import gaussian_blur_psf, particle_image_forward


PHYSICS_AGENT_SCHEMA_VERSION = 2
PHYSICS_AGENT_FAMILY = "physics_unrolled_piv_agent_v1"


def hodge_project(flow: torch.Tensor) -> torch.Tensor:
    """Periodic spectral Hodge projection with the spatial mean preserved."""
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [B, 2, H, W]")
    height, width = flow.shape[-2:]
    # The canonical kinematic operator uses central differences. Its Fourier
    # symbol is i*sin(k), not i*k; using the same symbol keeps the projection
    # consistent with the divergence reported by the evaluator.
    ky = torch.sin(2.0 * math.pi * torch.fft.fftfreq(height, device=flow.device, dtype=flow.dtype))
    kx = torch.sin(2.0 * math.pi * torch.fft.rfftfreq(width, device=flow.device, dtype=flow.dtype))
    ky = ky.view(1, height, 1)
    kx = kx.view(1, 1, -1)
    spectrum = torch.fft.rfft2(flow, dim=(-2, -1))
    u_hat, v_hat = spectrum[:, 0], spectrum[:, 1]
    k_squared = kx.square() + ky.square()
    safe_k_squared = torch.where(k_squared > 0, k_squared, torch.ones_like(k_squared))
    longitudinal = (kx * u_hat + ky * v_hat) / safe_k_squared
    u_projected = u_hat - kx * longitudinal
    v_projected = v_hat - ky * longitudinal
    projected = torch.stack([u_projected, v_projected], dim=1)
    return torch.fft.irfft2(projected, s=(height, width), dim=(-2, -1))


def _particle_energy_per_sample(image1, image2, flow, particle_sigma=1.0, epsilon=1e-3, symmetric_psf=True):
    """Symmetric particle-image advection energy.

    symmetric_psf passes the *reference* frame through the same PSF as the prediction.
    Blurring only the prediction biases the energy: at the true displacement the residual
    is then I1 - blur(I1) instead of zero, so the minimiser is offset from the true flow
    by an amount that grows with particle_sigma. Measured on an exact-translation pair,
    the residual at the true flow is 0.0142 with the biased form and 0.0017 with the
    symmetric form, and a Gauss-Newton descent diverges under the biased form.
    """
    if symmetric_psf:
        reference_1 = gaussian_blur_psf(image1, particle_sigma)
        reference_2 = gaussian_blur_psf(image2, particle_sigma)
    else:
        reference_1, reference_2 = image1, image2
    predicted_forward = particle_image_forward(image2, flow, particle_sigma=particle_sigma)
    predicted_reverse = particle_image_forward(image1, -flow, particle_sigma=particle_sigma)
    residual_forward = reference_1 - predicted_forward
    residual_reverse = reference_2 - predicted_reverse

    def robust_energy(residual):
        local_scale = F.avg_pool2d(residual.square(), 7, stride=1, padding=3).detach().sqrt().clamp(min=epsilon)
        robust = torch.sqrt((residual / local_scale).square() + float(epsilon) ** 2)
        return robust.flatten(1).mean(dim=1)
    energy = 0.5 * (robust_energy(residual_forward) + robust_energy(residual_reverse))
    return energy, residual_forward


def _robust_weights(residual, local_scale, epsilon=1e-3):
    """IRLS weight of the Charbonnier penalty rho(x)=sqrt(x^2+eps^2) applied to x=r/scale.

    The scalar 1/N implied by the mean-reduced energy cancels between the Gauss-Newton
    normal-equation sides, so the weights are used in their per-pixel form.
    """
    x = residual / local_scale
    return torch.rsqrt(x * x + float(epsilon) ** 2)


def _box_accumulate(tensor, window):
    if window is None or int(window) <= 1:
        return tensor
    size = int(window)
    if size % 2 == 0:
        size += 1
    return F.avg_pool2d(tensor, size, stride=1, padding=size // 2)


def particle_gauss_newton_step(
    image1,
    image2,
    flow,
    damping=1e-2,
    maximum_step=1.0,
    particle_sigma=1.0,
    epsilon=1e-3,
    window=5,
    damping_floor=1e-4,
    symmetric_psf=True,
):
    """Per-pixel Gauss-Newton step on the symmetric particle-image advection energy.

    Unlike an RMS-normalized gradient this returns a step with a *physical* magnitude:
    it is large where the photometric residual is large and vanishes where the estimate
    has already converged. The 2x2 normal equations are
        (J^T W J + lambda I) delta = -J^T W r
    with J built from the spatial gradients of the forward-model prediction and W the
    IRLS weight of the Charbonnier penalty. Forward and reverse advection terms are
    both included so the step stays symmetric under (I1, I2) exchange.
    """
    with torch.enable_grad():
        reference = flow.detach()
        if symmetric_psf:
            reference_1 = gaussian_blur_psf(image1, particle_sigma)
            reference_2 = gaussian_blur_psf(image2, particle_sigma)
        else:
            reference_1, reference_2 = image1, image2
        predicted_forward = particle_image_forward(image2, reference, particle_sigma=particle_sigma)
        predicted_reverse = particle_image_forward(image1, -reference, particle_sigma=particle_sigma)
        residual_forward = reference_1 - predicted_forward
        residual_reverse = reference_2 - predicted_reverse

        scale_forward = F.avg_pool2d(
            residual_forward.detach().square(), 7, stride=1, padding=3
        ).sqrt().clamp(min=epsilon)
        scale_reverse = F.avg_pool2d(
            residual_reverse.detach().square(), 7, stride=1, padding=3
        ).sqrt().clamp(min=epsilon)
        weight_forward = _robust_weights(residual_forward.detach(), scale_forward, epsilon)
        weight_reverse = _robust_weights(residual_reverse.detach(), scale_reverse, epsilon)

        # warp_tensor samples at (x + u), so d(pred_f)/d(u) = +grad(pred_f) and
        # d(pred_r)/d(u) = -grad(pred_r). Residuals carry the opposite sign.
        grad_forward_x, grad_forward_y = compute_spatial_gradients(predicted_forward, 1.0, 1.0)
        grad_reverse_x, grad_reverse_y = compute_spatial_gradients(predicted_reverse, 1.0, 1.0)

        residual_forward = residual_forward.detach()
        residual_reverse = residual_reverse.detach()

        hxx = weight_forward * grad_forward_x.square() + weight_reverse * grad_reverse_x.square()
        hxy = weight_forward * grad_forward_x * grad_forward_y + weight_reverse * grad_reverse_x * grad_reverse_y
        hyy = weight_forward * grad_forward_y.square() + weight_reverse * grad_reverse_y.square()
        bx = weight_forward * grad_forward_x * residual_forward - weight_reverse * grad_reverse_x * residual_reverse
        by = weight_forward * grad_forward_y * residual_forward - weight_reverse * grad_reverse_y * residual_reverse

        hxx = _box_accumulate(hxx, window)
        hxy = _box_accumulate(hxy, window)
        hyy = _box_accumulate(hyy, window)
        bx = _box_accumulate(bx, window)
        by = _box_accumulate(by, window)

        # Levenberg-Marquardt style damping relative to the local curvature keeps the
        # step scale invariant and well conditioned in low-texture regions.
        lam = float(damping) * 0.5 * (hxx + hyy) + float(damping_floor)
        hxx = hxx + lam
        hyy = hyy + lam
        determinant = (hxx * hyy - hxy.square()).clamp(min=1e-8)

        step_x = (hyy * bx - hxy * by) / determinant
        step_y = (hxx * by - hxy * bx) / determinant
        step = torch.cat([step_x, step_y], dim=1)

        step_norm = step.norm(p=2, dim=1, keepdim=True)
        limit = float(maximum_step)
        if limit > 0:
            step = step * (limit / step_norm.clamp(min=1e-12)).clamp(max=1.0)

    energy = _particle_energy_per_sample(
        image1, image2, reference, particle_sigma=particle_sigma, symmetric_psf=symmetric_psf
    )[0]
    return step.detach(), energy.detach()


def _bounded_norm(direction: torch.Tensor, maximum_step: float) -> torch.Tensor:
    """Clamp each pixel's step length instead of rescaling the whole field to a fixed RMS."""
    norm = direction.norm(p=2, dim=1, keepdim=True)
    return direction * (float(maximum_step) / norm.clamp(min=1e-12)).clamp(max=1.0)


def _bounded_rms(direction: torch.Tensor, maximum_rms: float) -> torch.Tensor:
    rms = direction.flatten(1).square().mean(dim=1).sqrt().view(-1, 1, 1, 1).clamp(min=1e-8)
    return direction * (float(maximum_rms) / rms).clamp(max=1.0)


def particle_descent_direction(image1, image2, flow, maximum_rms=0.05, particle_sigma=1.0, symmetric_psf=True):
    """Return a detached, resolution-independent descent direction for the PIV image equation."""
    with torch.enable_grad():
        probe = flow.detach().requires_grad_(True)
        energy, _ = _particle_energy_per_sample(
            image1, image2, probe, particle_sigma=particle_sigma, symmetric_psf=symmetric_psf
        )
        gradient = torch.autograd.grad(energy.sum(), probe, only_inputs=True)[0]
        gradient_rms = gradient.flatten(1).square().mean(dim=1).sqrt().view(-1, 1, 1, 1).clamp(min=1e-8)
        direction = -gradient / gradient_rms * float(maximum_rms)
    return direction.detach(), energy.detach()


def _boundary_taper(reference: torch.Tensor, width: int = 8) -> torch.Tensor:
    height, image_width = reference.shape[-2:]
    width = min(max(int(width), 0), height // 2, image_width // 2)
    if width == 0:
        return reference.new_ones((1, 1, height, image_width))
    y = torch.ones(height, device=reference.device, dtype=reference.dtype)
    x = torch.ones(image_width, device=reference.device, dtype=reference.dtype)
    ramp = torch.sin(torch.linspace(0.0, math.pi * 0.5, width + 2, device=reference.device, dtype=reference.dtype)[1:-1]).square()
    y[:width], y[-width:] = ramp, ramp.flip(0)
    x[:width], x[-width:] = ramp, ramp.flip(0)
    return (y[:, None] * x[None, :]).view(1, 1, height, image_width)


class PhysicsUnrolledPIVAgent(nn.Module):
    """Learn weights and trust regions; physics determines every correction direction."""

    model_family = PHYSICS_AGENT_FAMILY

    def __init__(
        self,
        unroll_steps: int = 3,
        observation_step_rms: float = 0.05,
        hodge_step_rms: float = 0.05,
        particle_sigma: float = 1.0,
        energy_temperature: float = 0.02,
        boundary_width: int = 8,
        observation_step_mode: str = "gauss_newton",
        observation_max_step: float = 1.0,
        observation_damping: float = 1e-2,
        observation_window: int = 5,
        hodge_max_step: float = 0.5,
        symmetric_psf: bool = True,
        hodge_step_mode: str = "norm",
    ):
        super().__init__()
        if hodge_step_mode not in {"norm", "rms"}:
            raise ValueError("hodge_step_mode must be norm or rms")
        if unroll_steps < 1 or observation_step_rms <= 0 or hodge_step_rms <= 0 or energy_temperature <= 0:
            raise ValueError("Invalid physics-unrolled agent configuration")
        if observation_step_mode not in {"gauss_newton", "gradient"}:
            raise ValueError("observation_step_mode must be gauss_newton or gradient")
        self.unroll_steps = int(unroll_steps)
        self.observation_step_rms = float(observation_step_rms)
        self.hodge_step_rms = float(hodge_step_rms)
        self.particle_sigma = float(particle_sigma)
        self.energy_temperature = float(energy_temperature)
        self.boundary_width = int(boundary_width)
        self.observation_step_mode = str(observation_step_mode)
        self.observation_max_step = float(observation_max_step)
        self.observation_damping = float(observation_damping)
        self.observation_window = int(observation_window)
        self.hodge_max_step = float(hodge_max_step)
        self.hodge_step_mode = str(hodge_step_mode)
        self.symmetric_psf = bool(symmetric_psf)
        # Runtime controls used by the constraint/tension study.  They are set
        # after checkpoint loading because they must not participate in the
        # agent_config compatibility check: they change the *strength* of an
        # existing direction, never the architecture.
        self.hodge_scale = 1.0
        self.observation_scale = 1.0
        # Images (4), flow/sigma (4), divergence/residual (2), physics directions (4).
        self.policy = nn.Sequential(
            nn.Conv2d(14, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 64, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(64, 48, 3, padding=1),
            nn.GELU(),
        )
        self.weight_head = nn.Conv2d(48, 3, 3, padding=1)
        nn.init.zeros_(self.weight_head.weight)
        # Start with a usable correction signal.  A very small initial gate
        # makes the no-harm objective easiest to satisfy by learning the
        # identity map before the physics directions can be optimized.
        nn.init.constant_(self.weight_head.bias, -0.5)

    @property
    def agent_config(self):
        return {
            "unroll_steps": self.unroll_steps,
            "observation_step_rms": self.observation_step_rms,
            "hodge_step_rms": self.hodge_step_rms,
            "particle_sigma": self.particle_sigma,
            "energy_temperature": self.energy_temperature,
            "boundary_width": self.boundary_width,
            "observation_step_mode": self.observation_step_mode,
            "observation_max_step": self.observation_max_step,
            "observation_damping": self.observation_damping,
            "observation_window": self.observation_window,
            "hodge_max_step": self.hodge_max_step,
            "hodge_step_mode": self.hodge_step_mode,
            "symmetric_psf": self.symmetric_psf,
        }

    @staticmethod
    def _fixed_scale(value, scale):
        return torch.tanh(value / float(scale))

    def set_hodge_scale(self, scale: float):
        """Scale the Hodge (divergence-free) direction without retraining.

        ``0`` disables the divergence-free direction exactly, which is the
        ``delta_h = 0`` end of the dose-response curve.  The published model is
        ``scale = 1``.
        """
        scale = float(scale)
        if scale < 0.0 or not math.isfinite(scale):
            raise ValueError("hodge scale must be a finite non-negative number")
        self.hodge_scale = scale
        return self

    def set_observation_scale(self, scale: float):
        scale = float(scale)
        if scale < 0.0 or not math.isfinite(scale):
            raise ValueError("observation scale must be a finite non-negative number")
        self.observation_scale = scale
        return self

    def _policy_inputs(self, image1, image2, flow, sigma, observation_direction, hodge_direction):
        warped = warp_tensor(image2, flow)
        residual = image1 - warped
        divergence = compute_divergence(flow)
        if sigma.shape[1] == 1:
            sigma = sigma.repeat(1, 2, 1, 1)
        observation_scale = (
            max(self.observation_max_step, 1e-6)
            if self.observation_step_mode == "gauss_newton"
            else self.observation_step_rms
        )
        return torch.cat([
            image1 * 2.0 - 1.0,
            image2 * 2.0 - 1.0,
            warped * 2.0 - 1.0,
            self._fixed_scale(residual, 0.25),
            self._fixed_scale(flow, 8.0),
            self._fixed_scale(sigma[:, :2], 0.05),
            self._fixed_scale(divergence, 0.05),
            residual.abs().clamp(max=1.0),
            self._fixed_scale(observation_direction, observation_scale),
            self._fixed_scale(hodge_direction, max(self.hodge_max_step, 1e-6)),
        ], dim=1)

    def _physics_step(
        self,
        image1,
        image2,
        flow,
        sigma,
        hard_acceptance=False,
        observation_enabled=True,
        hodge_enabled=True,
        proposal_scale=1.0,
    ):
        if self.observation_step_mode == "gauss_newton":
            observation_direction, observation_before = particle_gauss_newton_step(
                image1,
                image2,
                flow,
                damping=self.observation_damping,
                maximum_step=self.observation_max_step,
                particle_sigma=self.particle_sigma,
                window=self.observation_window,
                symmetric_psf=self.symmetric_psf,
            )
        else:
            observation_direction, observation_before = particle_descent_direction(
                image1,
                image2,
                flow,
                maximum_rms=self.observation_step_rms,
                particle_sigma=self.particle_sigma,
                symmetric_psf=self.symmetric_psf,
            )
        projected = hodge_project(flow.detach())
        hodge_raw = projected - flow.detach()
        if self.hodge_step_mode == "rms":
            hodge_direction = _bounded_rms(hodge_raw, self.hodge_step_rms)
        else:
            hodge_direction = _bounded_norm(hodge_raw, self.hodge_max_step)
        taper = _boundary_taper(flow, self.boundary_width)
        hodge_direction = hodge_direction * taper
        features = self.policy(self._policy_inputs(
            image1, image2, flow, sigma, observation_direction, hodge_direction
        ))
        weights = torch.sigmoid(self.weight_head(features))
        observation_weight, hodge_weight, learned_gate = weights.split(1, dim=1)
        if not observation_enabled:
            observation_weight = torch.zeros_like(observation_weight)
        if not hodge_enabled:
            hodge_weight = torch.zeros_like(hodge_weight)
        observation_direction = observation_direction * float(self.observation_scale)
        hodge_direction = hodge_direction * float(self.hodge_scale)
        proposed_delta = (
            observation_weight * observation_direction + hodge_weight * hodge_direction
        ) * float(proposal_scale)
        proposed = flow + proposed_delta
        observation_after, _ = _particle_energy_per_sample(
            image1, image2, proposed, particle_sigma=self.particle_sigma
        )
        divergence_before = compute_divergence(flow).square().flatten(1).mean(dim=1).sqrt()
        divergence_after = compute_divergence(proposed).square().flatten(1).mean(dim=1).sqrt()
        obs_gain = observation_before - observation_after
        # The update must not increase either the image likelihood energy or the
        # measured continuity residual. A small numerical tolerance is allowed,
        # but the default is conservative rather than an EPE-blind soft gate.
        div_tolerance = 0.01 * divergence_before.detach() + 1e-6
        div_gain = divergence_before + div_tolerance - divergence_after
        if hard_acceptance:
            physics_acceptance = ((obs_gain >= 0.0) & (div_gain >= 0.0)).to(flow.dtype)
        else:
            physics_acceptance = torch.sigmoid(obs_gain / self.energy_temperature) * torch.sigmoid(
                div_gain / self.energy_temperature
            )
        physics_acceptance = physics_acceptance.view(-1, 1, 1, 1)
        gate = learned_gate * physics_acceptance
        refined = flow + gate * proposed_delta
        return refined, {
            "input_flow": flow,
            "gate": gate,
            "learned_gate": learned_gate,
            "physics_acceptance": physics_acceptance,
            "observation_weight": observation_weight,
            "hodge_weight": hodge_weight,
            "observation_direction": observation_direction,
            "hodge_direction": hodge_direction,
            "proposed_flow": proposed,
            "observation_energy_before": observation_before,
            "observation_energy_after": observation_after,
            "divergence_before": divergence_before,
            "divergence_after": divergence_after,
            "hodge_direction_rms": hodge_direction.flatten(1).square().mean(dim=1).sqrt(),
            "observation_direction_rms": observation_direction.flatten(1).square().mean(dim=1).sqrt(),
            "proposal_delta_rms": proposed_delta.flatten(1).square().mean(dim=1).sqrt(),
            "proposal_scale": float(proposal_scale),
        }

    def refine(self, image1, image2, base_flow, sigma, hard_acceptance=None, intervention="full"):
        interventions = {
            "full": (True, True),
            "observation_only": (True, False),
            "hodge_only": (False, True),
            "zero_action_placebo": (False, False),
        }
        if intervention not in interventions:
            raise ValueError(f"Unknown physics intervention: {intervention}")
        observation_enabled, hodge_enabled = interventions[intervention]
        flow = base_flow
        step_aux = []
        use_hard_acceptance = (not self.training) if hard_acceptance is None else bool(hard_acceptance)
        for _ in range(self.unroll_steps):
            flow, aux = self._physics_step(
                image1, image2, flow, sigma,
                hard_acceptance=use_hard_acceptance,
                observation_enabled=observation_enabled,
                hodge_enabled=hodge_enabled,
            )
            step_aux.append(aux)
        combined = dict(step_aux[-1])
        combined["steps"] = step_aux
        combined["delta_flow"] = flow - base_flow
        combined["intervention"] = intervention
        combined["hodge_scale"] = float(self.hodge_scale)
        combined["observation_scale"] = float(self.observation_scale)
        combined["realized_delta_rms"] = combined["delta_flow"].flatten(1).square().mean(dim=1).sqrt()
        combined["acceptance_rate"] = torch.stack(
            [step["physics_acceptance"].reshape(step["physics_acceptance"].shape[0], -1).mean(dim=1) for step in step_aux]
        ).mean(dim=0)
        return flow, combined

    def forward(self, image1, image2, base_flow, sigma):
        return self.refine(image1, image2, base_flow, sigma)

    @torch.no_grad()
    def scan_step_scale(
        self,
        image1,
        image2,
        flow,
        sigma,
        scales=(1.0, 0.5, 0.25, 0.125, 0.0625),
        hard_acceptance=True,
        intervention="full",
    ):
        """Diagnostic for the 0 %-acceptance question: how acceptance varies with step size.

        Each scale re-proposes *one* physics step with the proposal shrunk by that
        factor.  If acceptance rises as the step shrinks, the no-harm gate is
        rejecting oversized steps; if it stays at zero, the direction itself is not
        a descent direction and the observation branch cannot work.
        """
        interventions = {
            "full": (True, True),
            "observation_only": (True, False),
            "hodge_only": (False, True),
        }
        if intervention not in interventions:
            raise ValueError(f"Unknown physics intervention: {intervention}")
        observation_enabled, hodge_enabled = interventions[intervention]
        rows = []
        for scale in scales:
            refined, aux = self._physics_step(
                image1,
                image2,
                flow,
                sigma,
                hard_acceptance=hard_acceptance,
                observation_enabled=observation_enabled,
                hodge_enabled=hodge_enabled,
                proposal_scale=float(scale),
            )
            observation_gain = (aux["observation_energy_before"] - aux["observation_energy_after"]).mean()
            divergence_change = (aux["divergence_before"] - aux["divergence_after"]).mean()
            rows.append({
                "intervention": intervention,
                "scale": float(scale),
                "observation_gain": float(observation_gain),
                "divergence_change": float(divergence_change),
                "acceptance_rate": float(aux["physics_acceptance"].mean()),
                "proposal_delta_rms": float(aux["proposal_delta_rms"].mean()),
                "realized_delta_rms": float((refined - flow).flatten(1).square().mean(dim=1).sqrt().mean()),
            })
        return rows


def make_physics_agent_checkpoint(agent: PhysicsUnrolledPIVAgent, **extra):
    return {
        "checkpoint_schema_version": PHYSICS_AGENT_SCHEMA_VERSION,
        "agent_model_family": PHYSICS_AGENT_FAMILY,
        "agent_config": agent.agent_config,
        "agent_state_dict": agent.state_dict(),
        **extra,
    }


def load_physics_agent_checkpoint(checkpoint, agent: PhysicsUnrolledPIVAgent):
    if not isinstance(checkpoint, dict):
        raise ValueError("Physics agent checkpoint must be a dictionary")
    if checkpoint.get("checkpoint_schema_version") != PHYSICS_AGENT_SCHEMA_VERSION:
        raise ValueError("Physics agent checkpoint schema is incompatible")
    if checkpoint.get("agent_model_family") != PHYSICS_AGENT_FAMILY:
        raise ValueError("Physics agent model family is incompatible")
    if checkpoint.get("agent_config") != agent.agent_config:
        raise ValueError("Physics agent configuration is incompatible")
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
    return agent
