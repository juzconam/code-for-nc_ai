"""Strictly input-validated incompressible Navier-Stokes residuals."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from physics_utils import compute_spatial_gradients


@dataclass(frozen=True)
class NavierStokesMetadata:
    density: float
    kinematic_viscosity: float
    frame_interval: float
    pixel_size_x: float
    pixel_size_y: float
    pressure_units: str
    velocity_units: str
    boundary_condition: str
    body_force_declared: bool = False

    def validate(self):
        positive = (
            self.density,
            self.kinematic_viscosity,
            self.frame_interval,
            self.pixel_size_x,
            self.pixel_size_y,
        )
        if any(float(value) <= 0.0 for value in positive):
            raise ValueError("density, viscosity, time interval, and spatial scales must be positive")
        if not self.pressure_units or not self.velocity_units or not self.boundary_condition:
            raise ValueError("pressure/velocity units and boundary conditions must be explicitly declared")


def _laplacian(value, dx, dy):
    dx_value, dy_value = compute_spatial_gradients(value, dx, dy)
    dxx, _ = compute_spatial_gradients(dx_value, dx, dy)
    _, dyy = compute_spatial_gradients(dy_value, dx, dy)
    return dxx + dyy


def incompressible_navier_stokes_residual(
    velocity_previous: torch.Tensor,
    velocity_current: torch.Tensor,
    velocity_next: torch.Tensor,
    pressure_current: torch.Tensor,
    boundary_mask: torch.Tensor,
    boundary_velocity: torch.Tensor,
    metadata: NavierStokesMetadata,
    body_force: torch.Tensor | None = None,
):
    """Compute temporal momentum, continuity, and boundary residuals.

    Three calibrated velocity instants and pressure are mandatory. Missing
    variables are not silently estimated because that would change the claim
    from equation verification to an inverse closure problem.
    """
    metadata.validate()
    flows = (velocity_previous, velocity_current, velocity_next)
    if any(flow.ndim != 4 or flow.shape[1] != 2 for flow in flows):
        raise ValueError("three velocity fields with shape [B,2,H,W] are required")
    if not all(flow.shape == velocity_current.shape for flow in flows):
        raise ValueError("all velocity instants must have identical shape")
    expected_scalar = (velocity_current.shape[0], 1, *velocity_current.shape[-2:])
    if pressure_current.shape != expected_scalar or boundary_mask.shape != expected_scalar:
        raise ValueError("pressure and boundary mask must have shape [B,1,H,W]")
    if boundary_velocity.shape != velocity_current.shape:
        raise ValueError("boundary velocity must match velocity shape")
    if body_force is None:
        if not metadata.body_force_declared:
            raise ValueError("body force must be supplied or explicitly declared zero in metadata")
        body_force = torch.zeros_like(velocity_current)
    if body_force.shape != velocity_current.shape:
        raise ValueError("body force must match velocity shape")

    dt = float(metadata.frame_interval)
    dx, dy = float(metadata.pixel_size_x), float(metadata.pixel_size_y)
    temporal = (velocity_next - velocity_previous) / (2.0 * dt)
    u, v = velocity_current[:, 0:1], velocity_current[:, 1:2]
    du_dx, du_dy = compute_spatial_gradients(u, dx, dy)
    dv_dx, dv_dy = compute_spatial_gradients(v, dx, dy)
    advection = torch.cat([u * du_dx + v * du_dy, u * dv_dx + v * dv_dy], dim=1)
    dp_dx, dp_dy = compute_spatial_gradients(pressure_current, dx, dy)
    pressure_gradient = torch.cat([dp_dx, dp_dy], dim=1) / float(metadata.density)
    viscosity = float(metadata.kinematic_viscosity) * _laplacian(velocity_current, dx, dy)
    momentum = temporal + advection + pressure_gradient - viscosity - body_force
    continuity = du_dx + dv_dy
    boundary = boundary_mask.to(velocity_current.dtype) * (velocity_current - boundary_velocity)
    interior = 1.0 - boundary_mask.to(velocity_current.dtype)
    return {
        "momentum_residual": momentum,
        "continuity_residual": continuity,
        "boundary_residual": boundary,
        "momentum_rms_interior": (interior * momentum.square().mean(dim=1, keepdim=True)).mean().sqrt(),
        "continuity_rms_interior": (interior * continuity.square()).mean().sqrt(),
        "boundary_rms": boundary.square().mean().sqrt(),
    }
