"""Physics-side uncertainty machinery for the constraint/tension study.

The module supplies three things that the published pipeline did not have:

1. A **per-pixel** uncertainty estimator that does not depend on candidate
   diversity (a first-order photometric/gradient proxy of the Wieneke-class
   estimators), plus its conformal calibration constant.  This makes the
   "classical route + Monte-Carlo propagation" baseline (B4 in the revision
   plan) runnable on the same data as PIVID.

2. A **bounded-intervention certificate** (Method B of the revision plan).  The
   physics controller moves the field by a measurable amount ``Delta``; for an
   ``L_q``-Lipschitz derived quantity the induced bias is at most
   ``L_q * Delta``.  Adding that budget to a conformal interval on the
   pre-intervention field yields an interval whose coverage does not depend on
   the intervention being helpful.

3. An **intervention budget** decomposition (Method C) that separates the part
   of a derived-quantity error visible to candidate dispersion from the part
   carried by the common bias of all candidates.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from physics_utils import compute_kinematic_diagnostics, warp_tensor
from src.evaluation.conformal import conformal_quantile

GRADIENT_FLOOR = 1e-3


# ---------------------------------------------------------------------------
# 1. Per-pixel uncertainty proxy + Monte-Carlo propagation
# ---------------------------------------------------------------------------
def photometric_pixel_scale(image1, image2, flow, window: int = 9) -> torch.Tensor:
    """First-order displacement-noise proxy ``sigma(x) propto |residual| / |grad I|``.

    The photometric residual of a converged warping is dominated by image noise
    and interpolation error; dividing by the local image gradient converts it
    into an equivalent displacement error in pixels.  This is the classical
    first-order propagation behind correlation-statistics uncertainty
    (Wieneke 2015) and image-matching uncertainty (Sciacchitano 2013), reduced
    to its simplest scalar form.  It deliberately uses **no candidate spread**,
    which is what makes it a fair anti-thesis to the diffusion-dispersion route.
    """
    if image1.shape != image2.shape:
        raise ValueError("image1 and image2 must share a shape")
    residual = image1 - warp_tensor(image2, flow)
    kernel = int(max(window, 3))
    local_energy = F.avg_pool2d(residual.square(), kernel, stride=1, padding=kernel // 2).clamp(min=0.0)
    local_scale = torch.sqrt(local_energy)
    grad_x = torch.zeros_like(image1)
    grad_y = torch.zeros_like(image1)
    grad_x[..., :, 1:-1] = 0.5 * (image1[..., :, 2:] - image1[..., :, :-2])
    grad_y[..., 1:-1, :] = 0.5 * (image1[..., 2:, :] - image1[..., :-2, :])
    gradient = torch.sqrt(grad_x.square() + grad_y.square())
    gradient = F.avg_pool2d(gradient, kernel, stride=1, padding=kernel // 2)
    floor = GRADIENT_FLOOR * gradient.median().clamp(min=1e-6)
    base = local_scale / (gradient + floor)
    # One channel: the proxy predicts the *magnitude* of the displacement error.
    return base


def calibrate_pixel_scale(
    base: np.ndarray,
    error_vectors: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Conformal constant ``c`` such that ``|e| <= c * base`` at ``1 - alpha`` per pixel.

    ``alpha = 0.05`` reproduces a 95 % per-vector confidence level, which is the
    convention of the PIV uncertainty literature and the one used to convert the
    resulting constant into a standard deviation for Monte-Carlo propagation.
    """
    base = np.asarray(base, dtype=np.float64).ravel()
    magnitude = np.linalg.norm(np.asarray(error_vectors, dtype=np.float64), axis=-1).ravel()
    valid = np.isfinite(base) & np.isfinite(magnitude) & (base > 0)
    if not np.any(valid):
        raise ValueError("no valid pixels for pixel-scale calibration")
    return float(conformal_quantile(magnitude[valid] / base[valid], alpha))


def pixel_sigma_from_constant(base: torch.Tensor, constant: float, confidence: float = 0.95) -> torch.Tensor:
    """Turn a ``confidence``-level magnitude bound into a per-component standard deviation."""
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    z = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor((1.0 + float(confidence)) / 2.0)))
    magnitude = float(constant) * base / z
    return magnitude.repeat(1, 2, 1, 1) if base.shape[1] == 1 else magnitude


def monte_carlo_quantities(
    point_flow: torch.Tensor,
    sigma_components: torch.Tensor,
    quantity_fn,
    draws: int = 128,
    seed: int = 0,
) -> dict:
    """Propagate per-pixel Gaussian uncertainty through a derived-quantity functional.

    Returns ``{name: tensor[B, draws]}``.  The draw is seeded per call so the
    result is a deterministic function of the input, which is what conformal
    calibration requires of the scale.
    """
    if point_flow.shape != sigma_components.shape:
        raise ValueError("sigma_components must match the point field")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        (int(draws),) + tuple(point_flow.shape),
        generator=generator,
        dtype=torch.float32,
    ).to(point_flow.device)
    samples = point_flow.unsqueeze(0) + noise * sigma_components.unsqueeze(0)
    flat = samples.reshape(draws * point_flow.shape[0], *point_flow.shape[1:])
    values = quantity_fn(flat)
    return {name: value.reshape(int(draws), point_flow.shape[0]).transpose(0, 1) for name, value in values.items()}


# ---------------------------------------------------------------------------
# 2. Bounded-intervention certificate (Method B)
# ---------------------------------------------------------------------------
# q is L_q-Lipschitz in the displacement field with respect to the pixel-wise
# L2 norm, using the same central-difference operator as the evaluator:
#   mean_speed         L = 1
#   continuity_rms     L = sqrt(2)
#   mean_strain_rate   L = sqrt(2)
#   enstrophy          L = sqrt(2) * max|omega|
# (Derivation in docs/uncertainty_method_A_B_C.md; both derivatives gain a factor
# sqrt(2) from ||d_x f||_2 <= ||f||_2 for the central-difference symbol.)
LIPSCHITZ_CONSTANTS = {
    "mean_speed": 1.0,
    "continuity_rms": math.sqrt(2.0),
    "mean_strain_rate": math.sqrt(2.0),
    "kinetic_energy_proxy": None,  # data-dependent: max|v|
    "enstrophy": None,  # data-dependent: sqrt(2) * max|omega|
    "circulation": None,  # data-dependent: sqrt(2) * area / sqrt(N)
}


def lipschitz_constant(quantity: str, pre_flow: torch.Tensor, post_flow: torch.Tensor = None) -> float:
    """Explicit ``L_q`` for a quantity, using data-dependent maxima when needed."""
    base = LIPSCHITZ_CONSTANTS.get(quantity)
    if base is not None:
        return float(base)
    fields = [pre_flow] if post_flow is None else [pre_flow, post_flow]
    diagnostics = [compute_kinematic_diagnostics(field) for field in fields]
    if quantity == "enstrophy":
        peak = max(float(torch.abs(item["vorticity"]).max()) for item in diagnostics)
        return math.sqrt(2.0) * peak
    if quantity == "kinetic_energy_proxy":
        peak = max(float(item["velocity"].norm(dim=1).max()) for item in diagnostics)
        return peak
    if quantity == "circulation":
        vorticity = diagnostics[0]["vorticity"]
        cells = int(vorticity.shape[-2] * vorticity.shape[-1])
        return float(torch.abs(vorticity).max()) / math.sqrt(cells)
    raise KeyError(f"no Lipschitz constant registered for {quantity!r}")


def certified_interval(
    point_post,
    point_pre,
    scale_pre,
    q_statistic: float,
    epsilon: float,
    lipschitz: float,
    displacement_rms,
):
    """Method-B interval: conformal radius on the pre-intervention field plus the bias budget.

    ``|q* - q_post| <= |q* - q_pre| + L_q * Delta`` by the triangle inequality, so
    widening the pre-intervention conformal radius by ``L_q * Delta`` covers
    ``q*`` with probability at least ``1 - alpha`` **whatever direction the
    intervention moved in**.
    """
    point_post = np.asarray(point_post, dtype=np.float64)
    scale_pre = np.asarray(scale_pre, dtype=np.float64)
    displacement_rms = np.asarray(displacement_rms, dtype=np.float64)
    radius = q_statistic * (scale_pre + float(epsilon)) + float(lipschitz) * displacement_rms
    return point_post - radius, point_post + radius


def intervention_budget(pre_flow: torch.Tensor, post_flow: torch.Tensor) -> dict:
    """Measured size of the physics intervention, in displacement units.

    ``rms`` is the quantity that enters the Lipschitz certificate; ``max`` is the
    worst pixel and is reported because hard-acceptance gating is per pixel.
    """
    delta = (post_flow - pre_flow).detach()
    per_pixel = delta.norm(dim=1).flatten(1)
    return {
        "rms": per_pixel.square().mean(dim=1).sqrt(),
        "mean": per_pixel.mean(dim=1),
        "max": per_pixel.max(dim=1).values,
    }
