"""Monte-Carlo propagation of flow posterior uncertainty to scientific quantities."""

from __future__ import annotations

import torch

from physics_utils import compute_divergence, compute_strain_rate, compute_vorticity, coerce_flow_geometry


def flow_scientific_quantities(flow, geometry=None):
    geometry = coerce_flow_geometry(geometry)
    speed = flow.norm(dim=1)
    vorticity = compute_vorticity(flow, geometry)[:, 0]
    divergence = compute_divergence(flow, geometry)[:, 0]
    strain = compute_strain_rate(flow, geometry)[:, 0]
    cell_area = geometry.pixel_size_x * geometry.pixel_size_y
    return {
        "mean_speed": speed.mean(dim=(1, 2)),
        "kinetic_energy_proxy": 0.5 * speed.square().mean(dim=(1, 2)),
        "circulation": vorticity.sum(dim=(1, 2)) * cell_area,
        "enstrophy": 0.5 * vorticity.square().mean(dim=(1, 2)),
        "continuity_rms": divergence.square().mean(dim=(1, 2)).sqrt(),
        "mean_strain_rate": strain.mean(dim=(1, 2)),
    }


def propagate_scientific_uncertainty(candidates, geometry=None, coverage=0.90):
    if candidates.ndim != 5:
        raise ValueError("candidates must have shape [B,K,2,H,W]")
    batch, count = candidates.shape[:2]
    quantities = flow_scientific_quantities(candidates.reshape(batch * count, *candidates.shape[2:]), geometry)
    alpha = (1.0 - float(coverage)) / 2.0
    report = {}
    for name, values in quantities.items():
        values = values.reshape(batch, count)
        report[name] = {
            "mean": values.mean(dim=1),
            "std": values.std(dim=1, unbiased=False),
            "lower": torch.quantile(values, alpha, dim=1),
            "upper": torch.quantile(values, 1.0 - alpha, dim=1),
        }
    return report
