from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from physics_utils import (
    coerce_flow_geometry,
    compute_divergence,
    compute_spatial_gradients,
    compute_strain_rate,
    compute_vorticity,
    warp_tensor,
)


def gradient_energy(tensor, geometry=None):
    geometry = coerce_flow_geometry(geometry)
    dx, dy = compute_spatial_gradients(tensor, geometry.pixel_size_x, geometry.pixel_size_y)
    return torch.mean(dx ** 2 + dy ** 2)


def vorticity_gradient_energy(flow, geometry=None):
    return gradient_energy(compute_vorticity(flow, geometry=geometry), geometry=geometry)


def unpack_flow_batch(batch, device):
    if len(batch) == 3:
        sample_batch, label_batch, domain_ids = batch
        return sample_batch.to(device), label_batch.to(device), domain_ids.to(device)
    sample_batch, label_batch = batch
    return sample_batch.to(device), label_batch.to(device), None


def flow_kinematic_metrics(flow, geometry=None):
    with torch.no_grad():
        div = compute_divergence(flow, geometry=geometry)
        vort = compute_vorticity(flow, geometry=geometry)
        return {
            "div_l2": torch.mean(div ** 2).item(),
            "div_l1": torch.mean(torch.abs(div)).item(),
            "vorticity_l2": torch.mean(vort ** 2).item(),
            "vorticity_grad_l2": vorticity_gradient_energy(flow, geometry=geometry).item(),
            "strain_rate_l2": torch.mean(compute_strain_rate(flow, geometry=geometry) ** 2).item(),
        }


def reset_diffusion_sampling_seed(seed):
    """Reset only the PyTorch RNGs used by DDIM candidate generation for paired evaluation."""
    if seed is None:
        return
    seed = int(seed)
    if seed < 0:
        raise ValueError("Diffusion sampling seed must be non-negative.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_iterative_agent(agent, I1, I2, F_bar, sigma_F, steps=1, domain_ids=None, candidate_stats=None):
    steps = max(1, int(steps))
    flow = F_bar
    total_delta = torch.zeros_like(F_bar)
    aux = None
    for _ in range(steps):
        delta_flow, aux = agent.forward_with_aux(
            I1,
            I2,
            flow,
            sigma_F,
            domain_ids=domain_ids,
            candidate_stats=candidate_stats,
        )
        flow = flow + delta_flow
        total_delta = total_delta + delta_flow
    if aux is None:
        aux = {}
    return flow, total_delta, aux


def aggregate_for_agent(
    aggregator,
    candidates_fw,
    I1,
    I2,
    use_candidate_reweighting=False,
    flow_bw_mean=None,
):
    if use_candidate_reweighting:
        return aggregator.filter_and_aggregate(
            candidates_fw,
            I1,
            I2,
            flow_bw_mean=flow_bw_mean,
            return_stats=True,
        )
    F_bar, sigma_F = aggregator.filter_and_aggregate(candidates_fw, I1, I2, flow_bw_mean=flow_bw_mean)
    return F_bar, sigma_F, None


def predict_and_aggregate(
    diffusion_model,
    feature_extractor,
    aggregator,
    I1,
    I2,
    use_candidate_reweighting=False,
    use_forward_backward_consistency=True,
):
    """Generate both directions before weighting forward hypotheses by consistency."""
    condition_fw = feature_extractor(I1, I2)
    candidates_fw = diffusion_model(condition_fw, I1, I2, use_ema=True)
    flow_bw_mean = None
    if use_forward_backward_consistency:
        condition_bw = feature_extractor(I2, I1)
        candidates_bw = diffusion_model(condition_bw, I2, I1, use_ema=True)
        flow_bw_mean, _ = aggregator.filter_and_aggregate(candidates_bw, I2, I1)
    return aggregate_for_agent(
        aggregator,
        candidates_fw,
        I1,
        I2,
        use_candidate_reweighting=use_candidate_reweighting,
        flow_bw_mean=flow_bw_mean,
    )


def fit_uncertainty_scale(error: torch.Tensor, dispersion: torch.Tensor, coverage: float = 0.90):
    """Fit a split-conformal scale for non-negative candidate dispersion."""
    if error.numel() == 0 or dispersion.numel() == 0 or error.numel() != dispersion.numel():
        raise ValueError("error and dispersion must be non-empty tensors with identical lengths.")
    coverage = float(coverage)
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between zero and one.")
    ratios = error.detach().flatten() / dispersion.detach().flatten().clamp(min=1e-6)
    ratios = ratios[torch.isfinite(ratios)]
    if ratios.numel() == 0:
        raise ValueError("No finite uncertainty calibration ratios are available.")
    rank = min(int(torch.ceil(torch.tensor((ratios.numel() + 1) * coverage)).item()), ratios.numel())
    scale = torch.kthvalue(ratios, k=max(rank, 1)).values
    return {"scale": float(scale.item()), "coverage": coverage, "count": int(ratios.numel())}


def calibrate_candidate_uncertainty(
    loader,
    diffusion_model,
    feature_extractor,
    aggregator,
    device,
    coverage=0.90,
    use_candidate_reweighting=True,
    use_forward_backward_consistency=True,
    max_batches=0,
):
    """Calibrate candidate dispersion against validation EPE without changing the flow estimate."""
    errors, dispersions = [], []
    diffusion_model.eval()
    feature_extractor.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            sample_batch, gt_flow, _ = unpack_flow_batch(batch, device)
            I1, I2 = sample_batch[:, 0:1], sample_batch[:, 1:2]
            F_bar, sigma_F, _ = predict_and_aggregate(
                diffusion_model,
                feature_extractor,
                aggregator,
                I1,
                I2,
                use_candidate_reweighting=use_candidate_reweighting,
                use_forward_backward_consistency=use_forward_backward_consistency,
            )
            errors.append(torch.norm(F_bar - gt_flow, p=2, dim=1).flatten(1).mean(dim=1).cpu())
            dispersions.append(torch.norm(sigma_F, p=2, dim=1).flatten(1).mean(dim=1).cpu())
    if not errors:
        raise ValueError("Uncertainty calibration loader produced no batches.")
    calibration = fit_uncertainty_scale(torch.cat(errors), torch.cat(dispersions), coverage=coverage)
    calibration["method"] = "split_conformal_scale"
    calibration["forward_backward_consistency"] = bool(use_forward_backward_consistency)
    return calibration


def apply_linear_calibration(F_bar, agent_flow, calibration):
    if calibration is None:
        return agent_flow
    weights = calibration["weights"].to(F_bar.device, dtype=F_bar.dtype)
    bias = calibration["bias"].to(F_bar.device, dtype=F_bar.dtype)
    corrected = torch.empty_like(F_bar)
    corrected[:, 0:1] = (
        weights[0, 0] * F_bar[:, 0:1]
        + weights[0, 1] * agent_flow[:, 0:1]
        + bias[0].view(1, 1, 1, 1)
    )
    corrected[:, 1:2] = (
        weights[1, 0] * F_bar[:, 1:2]
        + weights[1, 1] * agent_flow[:, 1:2]
        + bias[1].view(1, 1, 1, 1)
    )
    return corrected


def local_ncc_score(I1, I2_warped, window_size=9):
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=I1.device, dtype=I1.dtype) / (window_size ** 2)
    mu1 = F.conv2d(I1, kernel, padding=pad)
    mu2 = F.conv2d(I2_warped, kernel, padding=pad)
    z1 = I1 - mu1
    z2 = I2_warped - mu2
    var1 = F.conv2d(z1 ** 2, kernel, padding=pad)
    var2 = F.conv2d(z2 ** 2, kernel, padding=pad)
    cov = F.conv2d(z1 * z2, kernel, padding=pad)
    return torch.clamp(cov / torch.sqrt(var1 * var2 + 1e-6), -1.0, 1.0)


def build_selection_features(I1, I2, F_bar, agent_flow, geometry=None):
    base_warp = warp_tensor(I2, F_bar)
    agent_warp = warp_tensor(I2, agent_flow)
    base_photo = torch.mean(torch.abs(base_warp - I1), dim=[1, 2, 3])
    agent_photo = torch.mean(torch.abs(agent_warp - I1), dim=[1, 2, 3])
    base_ncc = local_ncc_score(I1, base_warp).mean(dim=[1, 2, 3])
    agent_ncc = local_ncc_score(I1, agent_warp).mean(dim=[1, 2, 3])
    base_div = torch.mean(torch.abs(compute_divergence(F_bar, geometry=geometry)), dim=[1, 2, 3])
    agent_div = torch.mean(torch.abs(compute_divergence(agent_flow, geometry=geometry)), dim=[1, 2, 3])
    delta_mean = torch.mean(torch.abs(agent_flow - F_bar), dim=[1, 2, 3])
    return torch.stack(
        [
            base_photo - agent_photo,
            agent_ncc - base_ncc,
            base_div - agent_div,
            -delta_mean,
        ],
        dim=1,
    )


def apply_selective_calibration(F_bar, agent_flow, selective_calibration, selection_features):
    if selective_calibration is None:
        select_mask = torch.ones(F_bar.shape[0], device=F_bar.device, dtype=torch.bool)
        return agent_flow, select_mask
    weights = selective_calibration["weights"].to(selection_features.device, dtype=selection_features.dtype)
    bias = selective_calibration["bias"].to(selection_features.device, dtype=selection_features.dtype)
    threshold = float(selective_calibration.get("threshold", 0.0))
    predicted_margin = selection_features @ weights + bias
    select_mask = predicted_margin > threshold
    blended_mask = select_mask.view(-1, 1, 1, 1)
    final_flow = torch.where(blended_mask, agent_flow, F_bar)
    return final_flow, select_mask


def evaluate_single_loader(
    loader,
    loader_name,
    diffusion_model,
    feature_extractor,
    aggregator,
    device,
    agent=None,
    selection_mode="agent",
    agent_scale=1.0,
    refine_steps=1,
    linear_calibration=None,
    selective_calibration=None,
    use_candidate_reweighting=False,
    use_forward_backward_consistency=True,
    uncertainty_calibration=None,
    use_domain_adapters=True,
    sample_uids=None,
    max_batches=0,
    collect_sample_metrics=False,
    candidate_sampling_seed=None,
):
    print(f"\n--- Evaluating {loader_name} ---")
    total_raw_epe = 0.0
    total_final_epe = 0.0
    total_delta_mean = 0.0
    total_delta_p95 = 0.0
    total_delta_max = 0.0
    total_improved_samples = 0
    total_delta_target_cosine = 0.0
    cosine_samples = 0
    total_div_raw = 0.0
    total_div_final = 0.0
    total_vort_grad_raw = 0.0
    total_vort_grad_final = 0.0
    total_strain_raw = 0.0
    total_strain_final = 0.0
    total_agent_selected = 0
    valid_samples = 0
    sample_rows = []

    reset_diffusion_sampling_seed(candidate_sampling_seed)
    diffusion_model.eval()
    feature_extractor.eval()
    if agent is not None:
        agent.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    evaluation_start = time.perf_counter()

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            sample_batch, gt_flow, domain_ids = unpack_flow_batch(batch, device)
            batch_size = sample_batch.size(0)
            I1 = sample_batch[:, 0:1, :, :]
            I2 = sample_batch[:, 1:2, :, :]

            F_bar, sigma_F, candidate_stats = predict_and_aggregate(
                diffusion_model,
                feature_extractor,
                aggregator,
                I1,
                I2,
                use_candidate_reweighting=use_candidate_reweighting,
                use_forward_backward_consistency=use_forward_backward_consistency,
            )

            raw_epe_map = torch.norm(F_bar - gt_flow, p=2, dim=1)
            raw_epe_per_sample = raw_epe_map.flatten(1).mean(dim=1)
            uncertainty_per_sample = torch.norm(sigma_F, p=2, dim=1).flatten(1).mean(dim=1)
            uncertainty_scale = float((uncertainty_calibration or {}).get("scale", 1.0))
            calibrated_uncertainty_per_sample = uncertainty_per_sample * uncertainty_scale
            base_flow_magnitude_per_sample = torch.norm(F_bar, p=2, dim=1).flatten(1).mean(dim=1)
            gt_flow_magnitude_per_sample = torch.norm(gt_flow, p=2, dim=1).flatten(1).mean(dim=1)
            gt_vorticity_per_sample = torch.abs(
                compute_vorticity(gt_flow, geometry=aggregator.geometry)
            ).flatten(1).mean(dim=1)
            gt_divergence_per_sample = torch.abs(
                compute_divergence(gt_flow, geometry=aggregator.geometry)
            ).flatten(1).mean(dim=1)
            base_ncc_per_sample = local_ncc_score(I1, warp_tensor(I2, F_bar)).flatten(1).mean(dim=1)
            raw_epe = raw_epe_per_sample.mean().item()
            delta_mean = 0.0
            delta_p95 = 0.0
            delta_max = 0.0
            delta_target_cosine = 0.0
            sample_delta_mean = torch.zeros_like(raw_epe_per_sample)
            sample_delta_p95 = torch.zeros_like(raw_epe_per_sample)
            sample_delta_max = torch.zeros_like(raw_epe_per_sample)
            sample_delta_cosine = torch.zeros_like(raw_epe_per_sample)
            sample_agent_selected = torch.zeros_like(raw_epe_per_sample, dtype=torch.bool)

            if agent is not None:
                agent_domain_ids = domain_ids if use_domain_adapters else None
                unscaled_agent_flow, _, _ = run_iterative_agent(
                    agent,
                    I1,
                    I2,
                    F_bar,
                    sigma_F,
                    steps=refine_steps,
                    domain_ids=agent_domain_ids,
                    candidate_stats=candidate_stats,
                )
                agent_flow = F_bar + float(agent_scale) * (unscaled_agent_flow - F_bar)
                if selection_mode in ("linear_calibrated", "quality_selective"):
                    agent_flow = apply_linear_calibration(F_bar, agent_flow, linear_calibration)

                if selection_mode == "base":
                    final_flow = F_bar
                elif selection_mode in ("agent", "linear_calibrated"):
                    final_flow = agent_flow
                elif selection_mode == "quality_selective":
                    selection_features = build_selection_features(I1, I2, F_bar, agent_flow, geometry=aggregator.geometry)
                    final_flow, sample_agent_selected = apply_selective_calibration(
                        F_bar,
                        agent_flow,
                        selective_calibration,
                        selection_features,
                    )
                else:
                    raise ValueError(f"Unsupported selection mode: {selection_mode}")

                applied_delta = final_flow - F_bar
                delta_abs_per_sample = applied_delta.abs().flatten(1)
                sample_delta_mean = delta_abs_per_sample.mean(dim=1)
                sample_delta_p95 = torch.quantile(delta_abs_per_sample, 0.95, dim=1)
                sample_delta_max = delta_abs_per_sample.max(dim=1).values
                delta_mean = sample_delta_mean.mean().item()
                delta_p95 = sample_delta_p95.mean().item()
                delta_max = sample_delta_max.mean().item()

                target_delta = gt_flow - F_bar
                delta_flat = applied_delta.flatten(1)
                target_delta_flat = target_delta.flatten(1)
                cosine = F.cosine_similarity(delta_flat, target_delta_flat, dim=1, eps=1e-8)
                sample_delta_cosine = cosine
                if selection_mode != "quality_selective":
                    sample_agent_selected = sample_delta_mean > 1e-12
                delta_target_cosine = cosine.mean().item()
                total_delta_target_cosine += cosine.sum().item()
                cosine_samples += batch_size
                total_agent_selected += sample_agent_selected.sum().item()
            else:
                final_flow = F_bar

            final_epe_map = torch.norm(final_flow - gt_flow, p=2, dim=1)
            final_epe_per_sample = final_epe_map.flatten(1).mean(dim=1)
            epe = final_epe_per_sample.mean().item()
            raw_kinematics = flow_kinematic_metrics(F_bar, geometry=aggregator.geometry)
            final_kinematics = flow_kinematic_metrics(final_flow, geometry=aggregator.geometry)
            total_raw_epe += raw_epe_per_sample.sum().item()
            total_final_epe += final_epe_per_sample.sum().item()
            total_delta_mean += delta_mean * batch_size
            total_delta_p95 += delta_p95 * batch_size
            total_delta_max += delta_max * batch_size
            total_improved_samples += (final_epe_per_sample < raw_epe_per_sample).sum().item()
            total_div_raw += raw_kinematics["div_l2"] * batch_size
            total_div_final += final_kinematics["div_l2"] * batch_size
            total_vort_grad_raw += raw_kinematics["vorticity_grad_l2"] * batch_size
            total_vort_grad_final += final_kinematics["vorticity_grad_l2"] * batch_size
            total_strain_raw += raw_kinematics["strain_rate_l2"] * batch_size
            total_strain_final += final_kinematics["strain_rate_l2"] * batch_size

            if collect_sample_metrics:
                sample_improved = final_epe_per_sample < raw_epe_per_sample
                for local_idx in range(batch_size):
                    global_sample_index = valid_samples + local_idx
                    row = {
                        "loader": loader_name,
                        "sample_index": global_sample_index,
                        "raw_epe": float(raw_epe_per_sample[local_idx].item()),
                        "final_epe": float(final_epe_per_sample[local_idx].item()),
                        "epe_improvement": float((raw_epe_per_sample[local_idx] - final_epe_per_sample[local_idx]).item()),
                        "epe_improvement_pct": float(
                            ((raw_epe_per_sample[local_idx] - final_epe_per_sample[local_idx]) / (raw_epe_per_sample[local_idx] + 1e-8) * 100.0).item()
                        ),
                        "delta_mean": float(sample_delta_mean[local_idx].item()),
                        "delta_p95": float(sample_delta_p95[local_idx].item()),
                        "delta_max": float(sample_delta_max[local_idx].item()),
                        "delta_target_cosine": float(sample_delta_cosine[local_idx].item()),
                        "agent_selected": int(sample_agent_selected[local_idx].item()),
                        "sample_improved": int(sample_improved[local_idx].item()),
                        "candidate_uncertainty_mean": float(uncertainty_per_sample[local_idx].item()),
                        "calibrated_uncertainty_mean": float(calibrated_uncertainty_per_sample[local_idx].item()),
                        "base_flow_magnitude_mean": float(base_flow_magnitude_per_sample[local_idx].item()),
                        "gt_flow_magnitude_mean": float(gt_flow_magnitude_per_sample[local_idx].item()),
                        "gt_vorticity_abs_mean": float(gt_vorticity_per_sample[local_idx].item()),
                        "gt_divergence_abs_mean": float(gt_divergence_per_sample[local_idx].item()),
                        "base_local_ncc_mean": float(base_ncc_per_sample[local_idx].item()),
                    }
                    if domain_ids is not None:
                        row["domain_id"] = int(domain_ids[local_idx].item())
                    if sample_uids is not None and global_sample_index < len(sample_uids):
                        row["sample_uid"] = sample_uids[global_sample_index]
                    sample_rows.append(row)
            valid_samples += batch_size

            if step % 20 == 0:
                if agent is not None:
                    print(
                        f"[{loader_name}] Step {step} | UPR-Flow EPE: "
                        f"{raw_epe:.4f} -> {epe:.4f} | "
                        f"scale/select: {agent_scale:.3f}/{selection_mode} | "
                        f"delta_mean/p95/max: {delta_mean:.6f}/{delta_p95:.6f}/{delta_max:.6f} | "
                        f"delta_target_cosine: {delta_target_cosine:.4f}"
                    )
                else:
                    print(f"[{loader_name}] Step {step} | base diffusion EPE: {raw_epe:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
    else:
        peak_cuda_memory_mb = 0.0
    evaluation_seconds = time.perf_counter() - evaluation_start
    parameter_count = sum(parameter.numel() for parameter in diffusion_model.parameters())
    parameter_count += sum(parameter.numel() for parameter in feature_extractor.parameters())
    if agent is not None:
        parameter_count += sum(parameter.numel() for parameter in agent.parameters())
    mean_raw_epe = total_raw_epe / valid_samples if valid_samples > 0 else 0.0
    mean_final_epe = total_final_epe / valid_samples if valid_samples > 0 else 0.0
    mean_delta = total_delta_mean / valid_samples if valid_samples > 0 else 0.0
    mean_delta_p95 = total_delta_p95 / valid_samples if valid_samples > 0 else 0.0
    mean_delta_max = total_delta_max / valid_samples if valid_samples > 0 else 0.0
    sample_improve_rate = total_improved_samples / valid_samples if valid_samples > 0 else 0.0
    mean_delta_target_cosine = total_delta_target_cosine / cosine_samples if cosine_samples > 0 else 0.0
    mean_div_raw = total_div_raw / valid_samples if valid_samples > 0 else 0.0
    mean_div_final = total_div_final / valid_samples if valid_samples > 0 else 0.0
    mean_vort_grad_raw = total_vort_grad_raw / valid_samples if valid_samples > 0 else 0.0
    mean_vort_grad_final = total_vort_grad_final / valid_samples if valid_samples > 0 else 0.0
    mean_strain_raw = total_strain_raw / valid_samples if valid_samples > 0 else 0.0
    mean_strain_final = total_strain_final / valid_samples if valid_samples > 0 else 0.0
    agent_selected_rate = total_agent_selected / valid_samples if valid_samples > 0 else 0.0
    metrics = {
        "loader": loader_name,
        "count": valid_samples,
        "selection_mode": selection_mode,
        "agent_scale": agent_scale,
        "raw_epe": mean_raw_epe,
        "final_epe": mean_final_epe,
        "epe_improvement": mean_raw_epe - mean_final_epe,
        "epe_improvement_pct": (mean_raw_epe - mean_final_epe) / (mean_raw_epe + 1e-8) * 100.0,
        "delta_mean": mean_delta,
        "delta_p95": mean_delta_p95,
        "delta_max": mean_delta_max,
        "agent_selected_rate": agent_selected_rate,
        "sample_improve_rate": sample_improve_rate,
        "delta_target_cosine": mean_delta_target_cosine,
        "raw_divergence_l2": mean_div_raw,
        "final_divergence_l2": mean_div_final,
        "raw_vorticity_gradient_l2": mean_vort_grad_raw,
        "final_vorticity_gradient_l2": mean_vort_grad_final,
        "raw_strain_rate_l2": mean_strain_raw,
        "final_strain_rate_l2": mean_strain_final,
        "candidate_sampling_seed": candidate_sampling_seed,
        "paired_candidate_replay": candidate_sampling_seed is not None,
        "evaluation_seconds": float(evaluation_seconds),
        "milliseconds_per_sample": float(1000.0 * evaluation_seconds / max(valid_samples, 1)),
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "parameter_count": int(parameter_count),
        "uncertainty_calibrated": bool(uncertainty_calibration),
        "uncertainty_scale": uncertainty_scale if valid_samples else None,
        "uncertainty_target_coverage": (uncertainty_calibration or {}).get("coverage"),
    }
    if uncertainty_calibration:
        calibration_hits = []
        for row in sample_rows:
            calibration_hits.append(float(row["raw_epe"]) <= float(row["calibrated_uncertainty_mean"]))
        if calibration_hits:
            metrics["uncertainty_empirical_coverage"] = float(sum(calibration_hits) / len(calibration_hits))
    print(
        f">>> [{loader_name}] Summary. EPE: {mean_raw_epe:.4f} -> {mean_final_epe:.4f} | "
        f"scale/select: {agent_scale:.3f}/{selection_mode} | agent-selected: {agent_selected_rate:.2%} | "
        f"delta mean/p95/max: {mean_delta:.6f}/{mean_delta_p95:.6f}/{mean_delta_max:.6f} | "
        f"sample improve: {sample_improve_rate:.2%} | cosine: {mean_delta_target_cosine:.4f} | "
        f"div: {mean_div_raw:.6f} -> {mean_div_final:.6f} | "
        f"vort-grad: {mean_vort_grad_raw:.6f} -> {mean_vort_grad_final:.6f}"
    )
    return metrics, sample_rows


def calibrate_linear_agent(
    loader,
    loader_name,
    diffusion_model,
    feature_extractor,
    aggregator,
    device,
    agent,
    ridge=1e-6,
    refine_steps=1,
    use_candidate_reweighting=False,
    use_forward_backward_consistency=True,
    use_domain_adapters=True,
    max_batches=0,
):
    print(f"\n--- Calibrating linear residual blend on {loader_name} ---")
    stats = [
        {"xtx": torch.zeros(3, 3, dtype=torch.float64), "xty": torch.zeros(3, dtype=torch.float64)}
        for _ in range(2)
    ]
    raw_sum = 0.0
    agent_sum = 0.0
    count = 0

    diffusion_model.eval()
    feature_extractor.eval()
    agent.eval()

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            sample_batch, gt_flow, domain_ids = unpack_flow_batch(batch, device)
            batch_size = sample_batch.size(0)
            I1 = sample_batch[:, 0:1, :, :]
            I2 = sample_batch[:, 1:2, :, :]

            F_bar, sigma_F, candidate_stats = predict_and_aggregate(
                diffusion_model,
                feature_extractor,
                aggregator,
                I1,
                I2,
                use_candidate_reweighting=use_candidate_reweighting,
                use_forward_backward_consistency=use_forward_backward_consistency,
            )
            agent_flow, _, _ = run_iterative_agent(
                agent,
                I1,
                I2,
                F_bar,
                sigma_F,
                steps=refine_steps,
                domain_ids=domain_ids if use_domain_adapters else None,
                candidate_stats=candidate_stats,
            )

            raw_epe = torch.norm(F_bar - gt_flow, p=2, dim=1).flatten(1).mean(dim=1)
            agent_epe = torch.norm(agent_flow - gt_flow, p=2, dim=1).flatten(1).mean(dim=1)
            raw_sum += raw_epe.sum().item()
            agent_sum += agent_epe.sum().item()
            count += batch_size

            ones = torch.ones_like(F_bar[:, 0:1])
            for channel in range(2):
                x = torch.stack(
                    [
                        F_bar[:, channel:channel + 1],
                        agent_flow[:, channel:channel + 1],
                        ones,
                    ],
                    dim=-1,
                ).reshape(-1, 3).double().cpu()
                y = gt_flow[:, channel:channel + 1].reshape(-1).double().cpu()
                stats[channel]["xtx"] += x.T @ x
                stats[channel]["xty"] += x.T @ y

    weights = torch.zeros(2, 2, dtype=torch.float32)
    bias = torch.zeros(2, dtype=torch.float32)
    eye = torch.eye(3, dtype=torch.float64)
    for channel in range(2):
        coeff = torch.linalg.solve(stats[channel]["xtx"] + float(ridge) * eye, stats[channel]["xty"])
        weights[channel, 0] = coeff[0].float()
        weights[channel, 1] = coeff[1].float()
        bias[channel] = coeff[2].float()

    calibration = {"weights": weights, "bias": bias}
    final_sum = 0.0
    final_count = 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            sample_batch, gt_flow, domain_ids = unpack_flow_batch(batch, device)
            batch_size = sample_batch.size(0)
            I1 = sample_batch[:, 0:1, :, :]
            I2 = sample_batch[:, 1:2, :, :]
            F_bar, sigma_F, candidate_stats = predict_and_aggregate(
                diffusion_model,
                feature_extractor,
                aggregator,
                I1,
                I2,
                use_candidate_reweighting=use_candidate_reweighting,
                use_forward_backward_consistency=use_forward_backward_consistency,
            )
            agent_flow, _, _ = run_iterative_agent(
                agent,
                I1,
                I2,
                F_bar,
                sigma_F,
                steps=refine_steps,
                domain_ids=domain_ids if use_domain_adapters else None,
                candidate_stats=candidate_stats,
            )
            corrected = apply_linear_calibration(F_bar, agent_flow, calibration)
            final_epe = torch.norm(corrected - gt_flow, p=2, dim=1).flatten(1).mean(dim=1)
            final_sum += final_epe.sum().item()
            final_count += batch_size

    raw_epe = raw_sum / max(count, 1)
    agent_epe = agent_sum / max(count, 1)
    final_epe = final_sum / max(final_count, 1)
    summary = {
        "count": count,
        "raw_epe": raw_epe,
        "agent_epe": agent_epe,
        "final_epe": final_epe,
        "improvement": raw_epe - final_epe,
        "improvement_pct": (raw_epe - final_epe) / (raw_epe + 1e-8) * 100.0,
        "u_base_weight": weights[0, 0].item(),
        "u_agent_weight": weights[0, 1].item(),
        "u_bias": bias[0].item(),
        "v_base_weight": weights[1, 0].item(),
        "v_agent_weight": weights[1, 1].item(),
        "v_bias": bias[1].item(),
    }
    print(
        f">>> Linear calibration {loader_name}: EPE {raw_epe:.4f} -> {final_epe:.4f} | "
        f"improvement {summary['improvement_pct']:.2f}%"
    )
    return calibration, summary


def calibrate_selective_agent(
    loader,
    loader_name,
    diffusion_model,
    feature_extractor,
    aggregator,
    device,
    agent,
    linear_calibration=None,
    ridge=1e-6,
    refine_steps=1,
    use_candidate_reweighting=False,
    use_forward_backward_consistency=True,
    use_domain_adapters=True,
    max_batches=0,
):
    print(f"\n--- Calibrating selective agent routing on {loader_name} ---")
    xtx = torch.zeros(5, 5, dtype=torch.float64)
    xty = torch.zeros(5, dtype=torch.float64)
    raw_sum = 0.0
    agent_sum = 0.0
    selected_sum = 0.0
    selected_count = 0
    count = 0

    diffusion_model.eval()
    feature_extractor.eval()
    agent.eval()

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            sample_batch, gt_flow, domain_ids = unpack_flow_batch(batch, device)
            batch_size = sample_batch.size(0)
            I1 = sample_batch[:, 0:1, :, :]
            I2 = sample_batch[:, 1:2, :, :]

            F_bar, sigma_F, candidate_stats = predict_and_aggregate(
                diffusion_model,
                feature_extractor,
                aggregator,
                I1,
                I2,
                use_candidate_reweighting=use_candidate_reweighting,
                use_forward_backward_consistency=use_forward_backward_consistency,
            )
            agent_flow, _, _ = run_iterative_agent(
                agent,
                I1,
                I2,
                F_bar,
                sigma_F,
                steps=refine_steps,
                domain_ids=domain_ids if use_domain_adapters else None,
                candidate_stats=candidate_stats,
            )
            agent_flow = apply_linear_calibration(F_bar, agent_flow, linear_calibration)

            raw_epe = torch.norm(F_bar - gt_flow, p=2, dim=1).flatten(1).mean(dim=1)
            agent_epe = torch.norm(agent_flow - gt_flow, p=2, dim=1).flatten(1).mean(dim=1)
            improvement_margin = raw_epe - agent_epe
            features = build_selection_features(I1, I2, F_bar, agent_flow, geometry=aggregator.geometry)
            ones = torch.ones((batch_size, 1), device=features.device, dtype=features.dtype)
            design = torch.cat([features, ones], dim=1).double().cpu()
            target = improvement_margin.double().cpu()
            xtx += design.T @ design
            xty += design.T @ target
            raw_sum += raw_epe.sum().item()
            agent_sum += agent_epe.sum().item()
            count += batch_size

    eye = torch.eye(5, dtype=torch.float64)
    coeff = torch.linalg.solve(xtx + float(ridge) * eye, xty)
    calibration = {
        "weights": coeff[:4].float(),
        "bias": coeff[4].float(),
        "threshold": 0.0,
    }

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            sample_batch, gt_flow, domain_ids = unpack_flow_batch(batch, device)
            batch_size = sample_batch.size(0)
            I1 = sample_batch[:, 0:1, :, :]
            I2 = sample_batch[:, 1:2, :, :]
            F_bar, sigma_F, candidate_stats = predict_and_aggregate(
                diffusion_model,
                feature_extractor,
                aggregator,
                I1,
                I2,
                use_candidate_reweighting=use_candidate_reweighting,
                use_forward_backward_consistency=use_forward_backward_consistency,
            )
            agent_flow, _, _ = run_iterative_agent(
                agent,
                I1,
                I2,
                F_bar,
                sigma_F,
                steps=refine_steps,
                domain_ids=domain_ids if use_domain_adapters else None,
                candidate_stats=candidate_stats,
            )
            agent_flow = apply_linear_calibration(F_bar, agent_flow, linear_calibration)
            features = build_selection_features(I1, I2, F_bar, agent_flow, geometry=aggregator.geometry)
            selected_flow, selected_mask = apply_selective_calibration(F_bar, agent_flow, calibration, features)
            selected_epe = torch.norm(selected_flow - gt_flow, p=2, dim=1).flatten(1).mean(dim=1)
            selected_sum += selected_epe.sum().item()
            selected_count += selected_mask.sum().item()

    raw_epe = raw_sum / max(count, 1)
    agent_epe = agent_sum / max(count, 1)
    final_epe = selected_sum / max(count, 1)
    summary = {
        "count": count,
        "raw_epe": raw_epe,
        "agent_epe": agent_epe,
        "final_epe": final_epe,
        "improvement": raw_epe - final_epe,
        "improvement_pct": (raw_epe - final_epe) / (raw_epe + 1e-8) * 100.0,
        "selected_rate": selected_count / max(count, 1),
        "feature_weights": calibration["weights"].tolist(),
        "bias": float(calibration["bias"].item()),
    }
    print(
        f">>> Selective routing {loader_name}: EPE {raw_epe:.4f} -> {final_epe:.4f} | "
        f"selected {summary['selected_rate']:.2%} | improvement {summary['improvement_pct']:.2f}%"
    )
    return calibration, summary
