"""Canonical flow geometry, kinematic diagnostics, warping, and candidate aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FlowGeometry:
    """Physical conversion for a two-component PIV displacement or velocity field."""

    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    frame_interval: float = 1.0
    flow_representation: str = "displacement_px"

    def __post_init__(self):
        if self.pixel_size_x <= 0.0 or self.pixel_size_y <= 0.0 or self.frame_interval <= 0.0:
            raise ValueError("pixel sizes and frame interval must be positive.")
        if self.flow_representation not in {"displacement_px", "velocity"}:
            raise ValueError("flow_representation must be 'displacement_px' or 'velocity'.")

    @property
    def is_coordinate_calibrated(self):
        return not (
            self.pixel_size_x == 1.0
            and self.pixel_size_y == 1.0
            and self.frame_interval == 1.0
        )

    def as_dict(self):
        return {
            "pixel_size_x": float(self.pixel_size_x),
            "pixel_size_y": float(self.pixel_size_y),
            "frame_interval": float(self.frame_interval),
            "flow_representation": self.flow_representation,
            "coordinate_calibrated": bool(self.is_coordinate_calibrated),
        }


def coerce_flow_geometry(geometry=None):
    if geometry is None:
        return FlowGeometry()
    if isinstance(geometry, FlowGeometry):
        return geometry
    if isinstance(geometry, dict):
        return FlowGeometry(
            pixel_size_x=float(geometry.get("pixel_size_x", 1.0)),
            pixel_size_y=float(geometry.get("pixel_size_y", 1.0)),
            frame_interval=float(geometry.get("frame_interval", 1.0)),
            flow_representation=str(geometry.get("flow_representation", "displacement_px")),
        )
    raise TypeError(f"Unsupported flow geometry: {type(geometry)!r}")


def _central_difference(tensor, dimension, spacing):
    if tensor.shape[dimension] <= 1:
        return torch.zeros_like(tensor)
    if dimension == -1 or dimension == tensor.ndim - 1:
        padded = F.pad(tensor, (1, 1, 0, 0), mode="replicate")
        return (padded[..., 2:] - padded[..., :-2]) / (2.0 * float(spacing))
    if dimension == -2 or dimension == tensor.ndim - 2:
        padded = F.pad(tensor, (0, 0, 1, 1), mode="replicate")
        return (padded[..., 2:, :] - padded[..., :-2, :]) / (2.0 * float(spacing))
    raise ValueError("Only the spatial height and width dimensions are supported.")


def compute_spatial_gradients(tensor, pixel_size_x=1.0, pixel_size_y=1.0):
    """Central finite differences with a single documented replicated-edge convention."""
    return (
        _central_difference(tensor, -1, pixel_size_x),
        _central_difference(tensor, -2, pixel_size_y),
    )


def flow_to_velocity(flow, geometry=None):
    geometry = coerce_flow_geometry(geometry)
    if geometry.flow_representation == "velocity":
        return flow
    scale = flow.new_tensor([geometry.pixel_size_x, geometry.pixel_size_y]).view(1, 2, 1, 1)
    return flow * scale / geometry.frame_interval


def compute_kinematic_diagnostics(flow, geometry=None):
    """Return dimensionally consistent velocity-gradient diagnostics from one canonical operator."""
    geometry = coerce_flow_geometry(geometry)
    velocity = flow_to_velocity(flow, geometry)
    u, v = velocity[:, 0:1], velocity[:, 1:2]
    du_dx, du_dy = compute_spatial_gradients(u, geometry.pixel_size_x, geometry.pixel_size_y)
    dv_dx, dv_dy = compute_spatial_gradients(v, geometry.pixel_size_x, geometry.pixel_size_y)
    divergence = du_dx + dv_dy
    vorticity = dv_dx - du_dy
    shear = du_dy + dv_dx
    strain_invariant = torch.sqrt(du_dx.square() + dv_dy.square() + 0.5 * shear.square() + 1e-12)
    return {
        "velocity": velocity,
        "du_dx": du_dx,
        "du_dy": du_dy,
        "dv_dx": dv_dx,
        "dv_dy": dv_dy,
        "divergence": divergence,
        "vorticity": vorticity,
        "shear": shear,
        "strain_invariant": strain_invariant,
    }


def compute_divergence(flow, geometry=None):
    return compute_kinematic_diagnostics(flow, geometry)["divergence"]


def compute_vorticity(flow, geometry=None):
    return compute_kinematic_diagnostics(flow, geometry)["vorticity"]


def compute_strain_rate(flow, geometry=None):
    return compute_kinematic_diagnostics(flow, geometry)["strain_invariant"]


def kinematic_risk_map(flow, geometry=None, max_risk=1.0):
    """Scale-free relative continuity irregularity for conservative gating, not a PDE residual."""
    divergence = torch.abs(compute_divergence(flow, geometry))
    scale = divergence.detach().mean(dim=[2, 3], keepdim=True).clamp(min=1e-8)
    return (divergence / (3.0 * scale)).clamp(min=0.0, max=float(max_risk))


def warp_tensor(tensor, flow):
    """The sole image/flow warp convention: bilinear, border padding, align_corners=True."""
    batch, _, height, width = tensor.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=tensor.device, dtype=tensor.dtype),
        torch.arange(width, device=tensor.device, dtype=tensor.dtype),
        indexing="ij",
    )
    base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, height, width, 2)
    sampling_grid = base_grid + flow.permute(0, 2, 3, 1)
    sampling_grid[..., 0] = 2.0 * sampling_grid[..., 0] / max(width - 1, 1) - 1.0
    sampling_grid[..., 1] = 2.0 * sampling_grid[..., 1] / max(height - 1, 1) - 1.0
    return F.grid_sample(tensor, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True)


class FlowAggregator:
    """Aggregate conditional-diffusion samples using matching and kinematic consistency diagnostics."""

    def __init__(
        self,
        ncc_window_size=9,
        kinematic_scale=2.0,
        matching_scale=2.0,
        fb_scale=1.5,
        geometry=None,
    ):
        self.eps = 1e-8
        self.window_size = int(ncc_window_size)
        self.kinematic_scale = float(kinematic_scale)
        self.matching_scale = float(matching_scale)
        self.fb_scale = float(fb_scale)
        self.geometry = coerce_flow_geometry(geometry)

    def compute_fb_error(self, flow_fw, flow_bw):
        warped_flow_bw = warp_tensor(flow_bw, flow_fw)
        return torch.norm(flow_fw + warped_flow_bw, dim=1, keepdim=True)

    def compute_local_ncc(self, image_a, image_b):
        channels = image_a.size(1)
        pad = self.window_size // 2
        kernel = torch.ones(
            channels,
            1,
            self.window_size,
            self.window_size,
            device=image_a.device,
            dtype=image_a.dtype,
        ) / float(self.window_size**2)
        mean_a = F.conv2d(image_a, kernel, padding=pad, groups=channels)
        mean_b = F.conv2d(image_b, kernel, padding=pad, groups=channels)
        zero_a = image_a - mean_a
        zero_b = image_b - mean_b
        var_a = F.conv2d(zero_a.square(), kernel, padding=pad, groups=channels)
        var_b = F.conv2d(zero_b.square(), kernel, padding=pad, groups=channels)
        covariance = F.conv2d(zero_a * zero_b, kernel, padding=pad, groups=channels)
        ncc = covariance / torch.sqrt(var_a * var_b + self.eps)
        return torch.clamp(ncc.mean(dim=1, keepdim=True), -1.0, 1.0)

    def filter_and_aggregate(self, candidates_fw, image_1, image_2, flow_bw_mean=None, return_stats=False):
        _, num_candidates, _, _, _ = candidates_fw.shape
        device = candidates_fw.device
        candidate_mean = candidates_fw.mean(dim=1, keepdim=True)
        candidate_std = candidates_fw.std(dim=1, keepdim=True, unbiased=False)
        valid_components = torch.abs(candidates_fw - candidate_mean) <= (2.0 * candidate_std + self.eps)
        valid_mask = (valid_components[:, :, 0:1] & valid_components[:, :, 1:2]).to(candidates_fw.dtype)

        weight_terms, ncc_terms, risk_terms, fb_terms = [], [], [], []
        for index in range(num_candidates):
            flow_i = candidates_fw[:, index]
            kinematic_risk = kinematic_risk_map(flow_i, self.geometry)
            weight_kinematic = torch.exp(-self.kinematic_scale * kinematic_risk)
            ncc_i = self.compute_local_ncc(image_1, warp_tensor(image_2, flow_i))
            weight_matching = torch.exp(self.matching_scale * ncc_i)
            if flow_bw_mean is None:
                fb_error = torch.zeros_like(kinematic_risk)
                weight_fb = torch.ones_like(kinematic_risk)
            else:
                fb_error = self.compute_fb_error(flow_i, flow_bw_mean)
                fb_scale = fb_error.detach().mean(dim=[2, 3], keepdim=True).clamp(min=self.eps)
                weight_fb = torch.exp(-self.fb_scale * fb_error / fb_scale)
            weight_terms.append(weight_kinematic * weight_matching * weight_fb * valid_mask[:, index])
            ncc_terms.append(ncc_i)
            risk_terms.append(kinematic_risk)
            fb_terms.append(fb_error)

        weight_stack = torch.stack(weight_terms, dim=1)
        weight_norm = weight_stack / weight_stack.sum(dim=1, keepdim=True).clamp(min=self.eps)
        aggregated_flow = torch.sum(weight_norm * candidates_fw, dim=1)
        variance = torch.sum(weight_norm * (candidates_fw - aggregated_flow.unsqueeze(1)).square(), dim=1)
        sigma_flow = torch.sqrt(variance + self.eps)
        if not return_stats:
            return aggregated_flow, sigma_flow

        ncc_stack = torch.stack(ncc_terms, dim=1)
        risk_stack = torch.stack(risk_terms, dim=1)
        fb_stack = torch.stack(fb_terms, dim=1)
        best_idx = torch.argmax(ncc_stack, dim=1, keepdim=True)
        best_flow = torch.gather(candidates_fw, 1, best_idx.expand(-1, -1, candidates_fw.size(2), -1, -1)).squeeze(1)
        best_ncc = torch.gather(ncc_stack, 1, best_idx).squeeze(1)
        topk = min(3, num_candidates)
        topk_values, topk_idx = torch.topk(ncc_stack, k=topk, dim=1)
        topk_weights = torch.softmax(topk_values * 10.0, dim=1)
        topk_flows = torch.gather(candidates_fw, 1, topk_idx.expand(-1, -1, candidates_fw.size(2), -1, -1))
        topk_match_flow = torch.sum(topk_weights * topk_flows, dim=1)
        if num_candidates > 1:
            top2 = torch.topk(ncc_stack, k=2, dim=1).values
            ncc_margin = top2[:, 0] - top2[:, 1]
        else:
            ncc_margin = torch.zeros_like(best_ncc)
        ncc_prob = torch.softmax(ncc_stack.squeeze(2) * 5.0, dim=1)
        ncc_entropy = -torch.sum(ncc_prob * torch.log(ncc_prob.clamp(min=self.eps)), dim=1, keepdim=True)
        if num_candidates > 1:
            ncc_entropy = ncc_entropy / torch.log(torch.tensor(float(num_candidates), device=device, dtype=ncc_entropy.dtype))
        return aggregated_flow, sigma_flow, {
            "candidate_mean": candidate_mean.squeeze(1),
            "candidate_std": candidate_std.squeeze(1),
            "best_match_flow": best_flow,
            "topk_match_flow": topk_match_flow,
            "topk_flows": topk_flows,
            "topk_weights": topk_weights,
            "ncc_stack": ncc_stack,
            "ncc_mean": ncc_stack.mean(dim=1),
            "ncc_std": ncc_stack.std(dim=1, unbiased=False),
            "ncc_best": best_ncc,
            "ncc_margin": ncc_margin,
            "ncc_entropy": ncc_entropy,
            "candidate_kinematic_risk": risk_stack.mean(dim=1),
            "candidate_fb_error": fb_stack.mean(dim=1),
            "forward_backward_consistency_active": flow_bw_mean is not None,
            "aggregator_weights": weight_norm,
        }
