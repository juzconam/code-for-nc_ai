import os
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
from PIL import Image

# ==========================================
# 导入自定义模块 (假设这些在你的项目中已存在)
# ==========================================
from src.data_processing.read_data import read_all, read_by_type, construct_dataset
from src.data_processing.custom_dataset import FlowDataset
from src.evaluation.core import (
    aggregate_for_agent,
    apply_linear_calibration,
    calibrate_linear_agent,
    calibrate_selective_agent,
    calibrate_candidate_uncertainty,
    evaluate_single_loader,
    flow_kinematic_metrics,
    predict_and_aggregate,
    run_iterative_agent,
)
from src.reporting.run_artifacts import (
    build_inference_run_summary,
    build_run_config,
    make_run_dir,
    save_json,
    write_metrics_csv,
)
from src.reporting.model_provenance import validate_backbone_holdout_provenance
from src.reporting.experimental_protocol import validate_experimental_protocol
from src.training.agent_state import restore_agent_training_state, save_agent_training_state
from models import (
    ConditionalDiffusionFlowGenerator,
    DualSourceFeatureExtractor,
    load_diffusion_checkpoint,
)
from physics_utils import (
    FlowAggregator,
    FlowGeometry,
    compute_kinematic_diagnostics,
    compute_spatial_gradients,
    kinematic_risk_map,
    warp_tensor,
)
from src.inference.piv_posterior import PosteriorGuidanceConfig

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ==========================================
# 可复现实验配置
# ==========================================
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = str(REPO_ROOT / "sample_data")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "piv_runs")
DEFAULT_AGENT_CHECKPOINT_NAME = "pea_agent_best.pth"
FLOW_TAG = 202021.25

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)

def apply_class2_noise(img_tensor, generator=None, reduce_intensity=True):
    """
    Approximate the RAFT-PIV Problem Class II degradation.
    """
    if reduce_intensity:
        scale_shape = img_tensor.shape[:-2] + (1, 1)
        scale_factor = 0.3 + 0.4 * torch.rand(
            scale_shape,
            device=img_tensor.device,
            dtype=img_tensor.dtype,
            generator=generator,
        )
        img_tensor = img_tensor * scale_factor

    mean = 0.01
    std_dev = math.sqrt(0.075)
    noise = torch.randn(
        img_tensor.shape,
        device=img_tensor.device,
        dtype=img_tensor.dtype,
        generator=generator,
    ) * std_dev + mean
    return torch.clamp(img_tensor + noise, 0.0, 1.0)


class Class2NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, seed=3407, reduce_intensity=True):
        self.base_dataset = base_dataset
        self.seed = int(seed)
        self.reduce_intensity = bool(reduce_intensity)
        for attr in (
            "data_index_list",
            "targets_index_list",
            "targets",
            "domain_ids",
            "sample_uid_by_index",
            "data",
            "train",
        ):
            if hasattr(base_dataset, attr):
                setattr(self, attr, getattr(base_dataset, attr))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if len(item) == 3:
            sample, label, domain_id = item
        else:
            sample, label = item
            domain_id = None
        sample = sample.clone()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(idx))
        sample[0:1] = apply_class2_noise(sample[0:1], generator=generator, reduce_intensity=self.reduce_intensity)
        sample[1:2] = apply_class2_noise(sample[1:2], generator=generator, reduce_intensity=self.reduce_intensity)
        if domain_id is not None:
            return sample, label, domain_id
        return sample, label

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="UPR-Flow: conditional diffusion with match-aware, kinematically conservative residual refinement."
    )
    parser.add_argument("--mode", choices=["train_agent", "inference"], default="train_agent")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--split-protocol",
        choices=["legacy_global", "stratified_domain", "blocked_domain", "manifest"],
        default="stratified_domain",
        help="Dataset split protocol. stratified_domain is the audit-safe default; manifest reuses an explicit split file.",
    )
    parser.add_argument("--split-seed", type=int, default=2, help="Seed used to create deterministic dataset splits.")
    parser.add_argument("--split-manifest", default="", help="Existing split_manifest.json used when --split-protocol manifest.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Per-stage validation/test fraction for generated splits.")
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=0,
        help="Samples omitted between contiguous train/validation/test blocks for blocked_domain splits.",
    )
    parser.add_argument(
        "--dataset-fingerprint-mode",
        choices=["triplet_sha256", "flow_sha256", "path_only"],
        default="triplet_sha256",
        help="Dataset identity mode stored in split provenance. triplet_sha256 is required for an auditable final run.",
    )
    parser.add_argument(
        "--dataset-kind",
        choices=["unknown", "simulation", "synthetic_degradation", "experimental"],
        default="unknown",
        help="Declared source category recorded in run provenance; it does not substitute for independent validation.",
    )
    parser.add_argument(
        "--dataset-manifest",
        default="",
        help="Optional JSON provenance manifest for the dataset, copied into the run record.",
    )
    parser.add_argument(
        "--write-data-index-cache",
        action="store_true",
        help="Allow dataset readers to write JSON index caches back into the data directory.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", required=True, help="Versioned conditional-DDPM checkpoint created by this release.")
    parser.add_argument(
        "--backbone-provenance-manifest",
        default="",
        help="JSON provenance for the diffusion checkpoint; required for strict held-out-domain backbone validation.",
    )
    parser.add_argument(
        "--require-backbone-holdout-provenance",
        action="store_true",
        help="Verify that the diffusion backbone checkpoint excludes held-out-domain supervision.",
    )
    parser.add_argument("--agent-checkpoint", default=DEFAULT_AGENT_CHECKPOINT_NAME)
    parser.add_argument("--agent-init-path", default=None)
    parser.add_argument(
        "--agent-resume-path",
        default=None,
        help="Epoch-boundary training-state file created by this repository; resumes agent and optimizer state exactly.",
    )
    parser.add_argument(
        "--agent-state-name",
        default="agent_training_state.pth",
        help="State filename saved after every completed agent-training epoch.",
    )
    parser.add_argument("--agent-path", default=None)
    parser.add_argument("--agent-run", default=None)
    parser.add_argument(
        "--require-agent-split-provenance",
        action="store_true",
        help="Require the loaded agent to originate from a train_agent run using the identical audited split.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--agent-epochs", type=int, default=50)
    parser.add_argument("--agent-lr", type=float, default=1e-3)
    parser.add_argument("--agent-max-delta", type=float, default=6.0)
    parser.add_argument("--agent-refine-steps", type=int, default=1)
    parser.add_argument("--residual-loss-weight", type=float, default=0.001)
    parser.add_argument(
        "--kinematic-regularization-weight",
        dest="kinematic_regularization_weight",
        type=float,
        default=0.05,
        help="Weight for conservative kinematic regularization; this is not a PDE or PINN loss.",
    )
    parser.add_argument("--delta-supervision-weight", type=float, default=2.0)
    parser.add_argument("--gate-loss-weight", type=float, default=0.2)
    parser.add_argument("--use-causal-policy", action="store_true",
                        help="Use diffusion-posterior potential-outcome heads to gate agent interventions.")
    parser.add_argument("--causal-risk-loss-weight", type=float, default=0.5)
    parser.add_argument("--causal-counterfactual-loss-weight", type=float, default=0.25)
    parser.add_argument("--causal-gate-temperature", type=float, default=0.25)
    parser.add_argument("--match-loss-weight", type=float, default=0.05)
    parser.add_argument("--no-harm-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-harm-margin", type=float, default=0.0)
    parser.add_argument("--candidate-loss-weight", type=float, default=0.25)
    parser.add_argument("--candidate-gate-loss-weight", type=float, default=0.1)
    parser.add_argument("--candidate-delta-loss-weight", type=float, default=0.25)
    parser.add_argument("--agent-heads-only", action="store_true")
    parser.add_argument("--hard-pixel-gamma", type=float, default=4.0)
    parser.add_argument("--agent-matching-radius", type=int, default=0)
    parser.add_argument(
        "--agent-input-mode", choices=["full", "observation_only", "posterior_only"], default="full",
        help="Controlled information bottleneck for diffusion-agent dependency ablations.",
    )
    parser.add_argument("--use-candidate-reweighting", dest="use_candidate_reweighting", action="store_true", default=True)
    parser.add_argument("--disable-candidate-reweighting", dest="use_candidate_reweighting", action="store_false")
    parser.add_argument("--candidate-topk", type=int, default=3)
    parser.add_argument("--use-domain-conditioning", action="store_true")
    parser.add_argument("--use-domain-heads", action="store_true")
    parser.add_argument("--domain-head-anchor-weight", type=float, default=0.0)
    parser.add_argument("--domain-head-residual-scale", type=float, default=0.25)
    parser.add_argument("--num-flow-domains", type=int, default=5)
    parser.add_argument("--disable-uncertainty-input", action="store_true")
    parser.add_argument(
        "--disable-kinematic-input",
        dest="disable_kinematic_input",
        action="store_true",
        help="Disable kinematic diagnostic inputs to the refinement agent.",
    )
    parser.add_argument("--diffusion-timesteps", type=int, default=100)
    parser.add_argument("--diffusion-sampling-steps", type=int, default=10)
    parser.add_argument("--diffusion-candidates", type=int, default=4)
    parser.add_argument("--diffusion-flow-scale", type=float, default=20.0)
    parser.add_argument("--diffusion-center-loss-weight", type=float, default=5.0)
    parser.add_argument("--diffusion-residual-reconstruction-weight", type=float, default=2.0)
    parser.add_argument("--no-diffusion-learn-variance", dest="diffusion_learn_variance", action="store_false")
    parser.set_defaults(diffusion_learn_variance=True)
    parser.add_argument("--diffusion-variance-injection-scale", type=float, default=0.0)
    parser.add_argument("--diffusion-variance-fit-weight", type=float, default=1.0)
    parser.add_argument(
        "--diffusion-denoised-clip",
        type=float,
        default=0.5,
        help="Smooth bound on normalized residual x0 used consistently during training and DDIM sampling.",
    )
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--posterior-observation-weight", type=float, default=0.0,
                        help="Particle-image likelihood guidance applied inside DDIM reverse steps.")
    parser.add_argument("--posterior-continuity-weight", type=float, default=0.0,
                        help="Coordinate-aware continuity guidance applied inside DDIM reverse steps.")
    parser.add_argument("--posterior-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--particle-psf-sigma", type=float, default=1.0)
    parser.add_argument("--posterior-guidance-start", type=float, default=0.65)
    parser.add_argument("--adaptive-posterior-sampling", action="store_true")
    parser.add_argument("--adaptive-min-candidates", type=int, default=2)
    parser.add_argument("--adaptive-min-steps", type=int, default=4)
    parser.add_argument("--adaptive-risk-threshold", type=float, default=0.25)
    parser.add_argument("--uncertainty-coverage", type=float, default=0.90)
    parser.add_argument(
        "--evaluation-sampling-seed",
        type=int,
        default=3407,
        help="Torch RNG seed replayed before base and agent evaluation so both use identical DDIM candidates and multi-seed runs share one evaluation draw.",
    )
    parser.add_argument(
        "--disable-uncertainty-calibration",
        dest="calibrate_uncertainty",
        action="store_false",
        default=True,
        help="Disable validation-set conformal scaling of candidate dispersion for a controlled ablation.",
    )
    parser.add_argument(
        "--disable-forward-backward-consistency",
        dest="use_forward_backward_consistency",
        action="store_false",
        default=True,
        help="Disable bidirectional candidate consistency only for a controlled ablation.",
    )
    parser.add_argument("--pixel-size-x", type=float, default=1.0)
    parser.add_argument("--pixel-size-y", type=float, default=1.0)
    parser.add_argument("--frame-interval", type=float, default=1.0)
    parser.add_argument(
        "--flow-representation",
        choices=["displacement_px", "velocity"],
        default="displacement_px",
        help="Coordinate semantics for kinematic diagnostics; provide calibration for physical-unit PIV analysis.",
    )
    parser.add_argument("--disable-domain-balanced-sampler", action="store_true")
    parser.add_argument(
        "--simulate-class2",
        action="store_true",
        help="Apply a Problem Class II-style noise wrapper during training/inference."
    )
    parser.add_argument(
        "--domain-boosts",
        default="",
        help="Optional comma-separated sampling multipliers, e.g. JHTDB_channel:2,backstep:1.5,SQG:1.5"
    )
    parser.add_argument(
        "--focus-domains",
        default="",
        help="Optional comma-separated training-domain subset for train_agent, e.g. JHTDB_channel,SQG"
    )
    parser.add_argument(
        "--exclude-domains",
        default="",
        help="Comma-separated domains excluded from train_agent, for example cylinder in leave-one-domain-out training."
    )
    parser.add_argument(
        "--eval-domains",
        default="",
        help="Optional comma-separated inference-domain subset, for example cylinder."
    )
    parser.add_argument(
        "--evaluation-split",
        choices=["train", "validation", "test"],
        default="test",
        help="Split evaluated during inference; train is a fit diagnostic, not generalization evidence.",
    )
    parser.add_argument(
        "--evaluation-protocol",
        choices=["in_domain", "zero_shot_holdout"],
        default="in_domain",
        help="Evaluation provenance. zero_shot_holdout forbids target-label calibration and evaluates the shared agent only."
    )
    parser.add_argument(
        "--disable-domain-adapters",
        action="store_true",
        help="Disable domain FiLM and domain-head adapters during inference while keeping the shared residual agent active."
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-calibration-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--selection-mode",
        choices=["base", "agent", "linear_calibrated", "quality_selective"],
        default="agent",
        help="Inference-time flow selection. Use linear_calibrated only as a separately labelled target-domain-calibrated analysis."
    )
    parser.add_argument("--agent-scale", type=float, default=1.0)
    parser.add_argument("--linear-calibration-ridge", type=float, default=1e-6)
    parser.add_argument(
        "--export-paper-artifacts",
        action="store_true",
        help="Save sample-level paper figures and intermediate tensors during inference."
    )
    parser.add_argument(
        "--export-paper-samples",
        type=int,
        default=6,
        help="Number of samples per dataset to export when paper artifacts are enabled."
    )
    parser.add_argument(
        "--run-kind",
        choices=["full", "smoke", "debug"],
        default="full",
        help="Tag the run intent so downstream scripts can distinguish paper-grade runs from smoke/debug runs.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow reusing an existing non-empty run directory.",
    )
    parser.add_argument("--run-name", default=None)
    return parser


def write_flow_flo(path, flow_hw2):
    flow_hw2 = np.asarray(flow_hw2, dtype=np.float32)
    height, width, channels = flow_hw2.shape
    if channels != 2:
        raise ValueError(f"Expected flow shape [H, W, 2], got {flow_hw2.shape}")
    with open(path, "wb") as f:
        f.write(struct.pack("<f", FLOW_TAG))
        f.write(struct.pack("<i", width))
        f.write(struct.pack("<i", height))
        f.write(flow_hw2.tobytes())


def save_gray_png(path, tensor_hw):
    arr = tensor_hw.detach().cpu().numpy()
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 0:
        arr = np.array([[float(arr)]], dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - arr.min()
    denom = arr.max() - arr.min()
    if denom > 1e-8:
        arr = arr / denom
    img = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def save_rgb_png(path, rgb_hw3):
    arr = np.asarray(rgb_hw3)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def flow_to_rgb_uint8(flow_hw2):
    import flowiz as fz

    rgb = fz.convert_from_flow(np.asarray(flow_hw2, dtype=np.float32))
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def tensor_flow_to_hw2(flow_tensor):
    return flow_tensor.detach().cpu().permute(1, 2, 0).contiguous().numpy()


def tensor_to_hw(tensor):
    return tensor.detach().cpu().squeeze().numpy()


def slice_batch_dict(batch_dict, index):
    if batch_dict is None:
        return None
    sliced = {}
    for key, value in batch_dict.items():
        if torch.is_tensor(value):
            sliced[key] = value[index:index + 1]
        else:
            sliced[key] = value
    return sliced


def export_sample_artifact(
    export_dir,
    dataset_name,
    sample_idx,
    sample_batch,
    gt_flow,
    base_flow,
    agent_flow,
    final_flow,
    refined_delta,
    candidate_stats,
    agent_aux,
):
    sample_dir = Path(export_dir) / dataset_name / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    I1 = sample_batch[0, 0]
    I2 = sample_batch[0, 1]
    base_epe = torch.norm(base_flow - gt_flow, p=2, dim=1)[0]
    agent_epe = torch.norm(agent_flow - gt_flow, p=2, dim=1)[0] if agent_flow is not None else None
    final_epe = torch.norm(final_flow - gt_flow, p=2, dim=1)[0]
    delta_mag = torch.norm(refined_delta, p=2, dim=1)[0]
    gate = agent_aux.get("gate")
    if gate is not None:
        gate = gate[0, 0]
    candidate_gate = agent_aux.get("candidate_gate")
    if candidate_gate is not None:
        candidate_gate = candidate_gate[0, 0]
    candidate_blend = agent_aux.get("candidate_blend")
    if candidate_blend is not None:
        candidate_blend = candidate_blend[0]

    flow_to_rgb = {
        "base_flow": base_flow[0],
        "agent_flow": agent_flow[0] if agent_flow is not None else None,
        "final_flow": final_flow[0],
        "gt_flow": gt_flow[0],
    }
    for name, flow_hw2 in flow_to_rgb.items():
        if flow_hw2 is None:
            continue
        save_rgb_png(sample_dir / f"{name}.png", flow_to_rgb_uint8(tensor_flow_to_hw2(flow_hw2)))
        write_flow_flo(sample_dir / f"{name}.flo", tensor_flow_to_hw2(flow_hw2))

    save_gray_png(sample_dir / "img1.png", I1)
    save_gray_png(sample_dir / "img2.png", I2)
    save_gray_png(sample_dir / "base_error.png", base_epe)
    save_gray_png(sample_dir / "final_error.png", final_epe)
    save_gray_png(sample_dir / "delta_mag.png", delta_mag)
    if agent_epe is not None:
        save_gray_png(sample_dir / "agent_error.png", agent_epe)

    if gate is not None:
        save_gray_png(sample_dir / "gate.png", gate)
        np.save(sample_dir / "gate.npy", gate.detach().cpu().numpy().astype(np.float32))
    if candidate_gate is not None:
        save_gray_png(sample_dir / "candidate_gate.png", candidate_gate)
        np.save(sample_dir / "candidate_gate.npy", candidate_gate.detach().cpu().numpy().astype(np.float32))
    if candidate_blend is not None:
        if candidate_blend.dim() == 2:
            save_gray_png(sample_dir / "candidate_blend.png", candidate_blend)
        elif candidate_blend.dim() == 3 and candidate_blend.shape[0] == 1:
            save_gray_png(sample_dir / "candidate_blend.png", candidate_blend[0])
        else:
            for idx in range(candidate_blend.shape[0]):
                save_gray_png(sample_dir / f"candidate_blend_{idx:02d}.png", candidate_blend[idx])
        np.save(sample_dir / "candidate_blend.npy", candidate_blend.detach().cpu().numpy().astype(np.float32))

    if candidate_stats is not None:
        candidate_mean = candidate_stats.get("candidate_mean")
        candidate_std = candidate_stats.get("candidate_std")
        best_match_flow = candidate_stats.get("best_match_flow")
        topk_match_flow = candidate_stats.get("topk_match_flow")
        topk_flows = candidate_stats.get("topk_flows")
        topk_weights = candidate_stats.get("topk_weights")
        aggregator_weights = candidate_stats.get("aggregator_weights")
        if candidate_mean is not None:
            save_rgb_png(sample_dir / "candidate_mean.png", flow_to_rgb_uint8(tensor_flow_to_hw2(candidate_mean[0])))
            np.save(sample_dir / "candidate_mean.npy", candidate_mean.detach().cpu().numpy().astype(np.float32))
        if candidate_std is not None:
            save_gray_png(sample_dir / "candidate_std_mag.png", torch.norm(candidate_std, p=2, dim=1)[0])
            np.save(sample_dir / "candidate_std.npy", candidate_std.detach().cpu().numpy().astype(np.float32))
        if best_match_flow is not None:
            save_rgb_png(sample_dir / "best_match_flow.png", flow_to_rgb_uint8(tensor_flow_to_hw2(best_match_flow[0])))
            np.save(sample_dir / "best_match_flow.npy", best_match_flow.detach().cpu().numpy().astype(np.float32))
        if topk_match_flow is not None:
            save_rgb_png(sample_dir / "topk_match_flow.png", flow_to_rgb_uint8(tensor_flow_to_hw2(topk_match_flow[0])))
            np.save(sample_dir / "topk_match_flow.npy", topk_match_flow.detach().cpu().numpy().astype(np.float32))
        if aggregator_weights is not None:
            agg_weight_np = aggregator_weights.detach().cpu().numpy()[0]
            np.save(sample_dir / "aggregator_weights.npy", agg_weight_np.astype(np.float32))
            for idx in range(agg_weight_np.shape[0]):
                save_gray_png(sample_dir / f"aggregator_weight_{idx:02d}.png", torch.from_numpy(agg_weight_np[idx].astype(np.float32)))
        ncc_stack = candidate_stats.get("ncc_stack")
        if ncc_stack is not None:
            ncc_stack_np = ncc_stack.detach().cpu().numpy()[0, :, 0]
            np.save(sample_dir / "ncc_stack.npy", ncc_stack_np.astype(np.float32))
            ncc_candidate_means = []
            for idx in range(ncc_stack_np.shape[0]):
                ncc_map = torch.from_numpy(ncc_stack_np[idx].astype(np.float32))
                save_gray_png(sample_dir / f"ncc_candidate_{idx:02d}.png", ncc_map)
                ncc_candidate_means.append(float(ncc_map.mean().item()))
        else:
            ncc_candidate_means = []

        if topk_flows is not None:
            topk_flow_means = []
            for idx in range(topk_flows.shape[1]):
                flow_map = topk_flows[0, idx]
                save_rgb_png(sample_dir / f"topk_flow_{idx:02d}.png", flow_to_rgb_uint8(tensor_flow_to_hw2(flow_map)))
                write_flow_flo(sample_dir / f"topk_flow_{idx:02d}.flo", tensor_flow_to_hw2(flow_map))
                topk_flow_means.append(float(torch.norm(flow_map, p=2, dim=0).mean().item()))
            np.save(sample_dir / "topk_flows.npy", topk_flows.detach().cpu().numpy().astype(np.float32))
        else:
            topk_flow_means = []

        if topk_weights is not None:
            topk_weights_np = topk_weights.detach().cpu().numpy()[0, :, 0]
            for idx in range(topk_weights_np.shape[0]):
                save_gray_png(sample_dir / f"topk_weight_{idx:02d}.png", torch.from_numpy(topk_weights_np[idx].astype(np.float32)))
            topk_weight_means = [float(topk_weights_np[idx].mean()) for idx in range(topk_weights_np.shape[0])]
            np.save(sample_dir / "topk_weights.npy", topk_weights.detach().cpu().numpy().astype(np.float32))
        else:
            topk_weight_means = []

        ncc_summary = {
            "ncc_candidate_means": ncc_candidate_means,
            "topk_flow_means": topk_flow_means,
            "topk_weight_means": topk_weight_means,
            "ncc_mean": float(candidate_stats["ncc_mean"][0].mean().item()) if candidate_stats.get("ncc_mean") is not None else None,
            "ncc_std": float(candidate_stats["ncc_std"][0].mean().item()) if candidate_stats.get("ncc_std") is not None else None,
            "ncc_best": float(candidate_stats["ncc_best"][0].mean().item()) if candidate_stats.get("ncc_best") is not None else None,
            "ncc_margin": float(candidate_stats["ncc_margin"][0].mean().item()) if candidate_stats.get("ncc_margin") is not None else None,
            "ncc_entropy": float(candidate_stats["ncc_entropy"][0].mean().item()) if candidate_stats.get("ncc_entropy") is not None else None,
            "candidate_count": int(candidate_stats["ncc_stack"].shape[1]) if candidate_stats.get("ncc_stack") is not None else None,
        }
        save_json(sample_dir / "candidate_summary.json", ncc_summary)

    sample_summary = {
        "dataset": dataset_name,
        "sample_index": sample_idx,
        "sample_dir": str(sample_dir),
        "base_epe": float(base_epe.mean().item()),
        "agent_epe": float(agent_epe.mean().item()) if agent_epe is not None else None,
        "final_epe": float(final_epe.mean().item()),
        "improvement": float(base_epe.mean().item() - final_epe.mean().item()),
        "improvement_pct": float((base_epe.mean().item() - final_epe.mean().item()) / (base_epe.mean().item() + 1e-8) * 100.0),
    }
    save_json(sample_dir / "sample_summary.json", sample_summary)
    return sample_summary

def parse_float_list(csv_text):
    values = []
    for item in str(csv_text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def parse_name_list(csv_text):
    values = []
    for item in str(csv_text or "").split(","):
        item = item.strip()
        if item:
            values.append(item)
    return values

def parse_domain_boosts(boost_text):
    boosts = {}
    for item in str(boost_text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid domain boost '{item}'. Expected NAME:WEIGHT.")
        name, value = item.split(":", 1)
        boosts[name.strip()] = float(value.strip())
    return boosts


def resolve_existing_path(path_value, *fallback_roots):
    if path_value is None:
        return None

    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    search_roots = []
    for root in (REPO_ROOT, *fallback_roots, Path.cwd()):
        if root is None:
            continue
        search_roots.append(Path(root))

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)

    return str((search_roots[0] / candidate).resolve())

def resolve_agent_path(args):
    if args.agent_path is not None:
        return resolve_existing_path(args.agent_path, args.output_dir)
    if args.agent_run is not None:
        return resolve_existing_path(
            os.path.join(args.agent_run, args.agent_checkpoint),
            args.output_dir,
        )
    candidates = []
    for root, _, files in os.walk(args.output_dir):
        if args.agent_checkpoint in files and "best_agent_metrics.json" in files:
            path = os.path.join(root, args.agent_checkpoint)
            candidates.append((os.path.getmtime(path), path))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return resolve_existing_path(args.agent_checkpoint, args.output_dir)

def print_agent_training_summary(agent_path):
    metrics_path = os.path.join(os.path.dirname(agent_path), "best_agent_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Warning: agent training summary not found: {metrics_path}")
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print(
        "Agent best metrics | "
        f"epoch: {metrics.get('epoch', 'N/A')} | "
        f"val_improvement: {metrics.get('val_improvement', 0.0):.6f} "
        f"({metrics.get('val_improvement_pct', 0.0):.2f}%) | "
        f"val_delta_mean: {metrics.get('val_delta_mean', 0.0):.6f}"
    )

def method_card():
    return {
        "method_name": "UPR-Flow",
        "full_name": "Conditional Diffusion with Match-Aware, Kinematically Conservative Residual Refinement",
        "core_claim": (
            "A conditional DDPM produces multiple flow hypotheses. A gated residual agent uses local matching "
            "evidence, candidate ambiguity, and coordinate-aware kinematic diagnostics to reweight hypotheses "
            "and apply conservative corrections."
        ),
        "claim_boundary": (
            "Kinematic diagnostics are finite-difference risk signals for conservative gating, not a PINN, "
            "Navier-Stokes residual, pressure solver, or calibrated uncertainty estimate."
        ),
        "required_evidence": [
            "multi-step denoising checkpoint metadata",
            "three-seed controlled component ablations",
            "uncertainty calibration and risk-coverage analysis",
            "frozen primary-study manifest and sensitivity-grid coverage",
            "runtime, peak-memory, and parameter-count reporting",
            "strict leave-one-domain-out retraining",
            "external RAFT32, OpenPIV, and PIV-specialized baseline reruns",
            "independent experimental PIV with sealed acquisition groups",
        ],
        "recommended_ablations": [
            "conditional_diffusion_base",
            "without_candidate_reweighting",
            "without_local_matching",
            "without_kinematic_diagnostics",
            "full_match_kinematic_refiner",
        ],
    }

# ==========================================
# 1. 核心数学与物理辅助函数
# ==========================================
def epe_loss(pred_flow, gt_flow):
    """计算端点误差均值 (End-Point Error)"""
    return torch.norm(pred_flow - gt_flow, p=2, dim=1).mean()

def edge_aware_residual_loss(delta_flow, image):
    """只弱约束 residual 的空间高频，避免直接抹平基础流场。"""
    delta_dx, delta_dy = compute_spatial_gradients(delta_flow)
    img_dx, img_dy = compute_spatial_gradients(image)
    edge_x = torch.exp(-10.0 * torch.mean(torch.abs(img_dx), dim=1, keepdim=True))
    edge_y = torch.exp(-10.0 * torch.mean(torch.abs(img_dy), dim=1, keepdim=True))
    return torch.mean(edge_x * delta_dx ** 2) + torch.mean(edge_y * delta_dy ** 2)

def kinematic_conservatism_loss(flow, reference_flow, delta_flow=None, image=None, risk_margin=0.10, geometry=None):
    """Penalize kinematic-risk increases beyond the base estimate plus edge-aware residual roughness."""
    refined_risk = kinematic_risk_map(flow, geometry=geometry)
    base_risk = kinematic_risk_map(reference_flow.detach(), geometry=geometry)
    excess_risk = torch.mean(F.relu(refined_risk - base_risk - float(risk_margin)) ** 2)
    loss_residual_smooth = flow.new_tensor(0.0)
    if delta_flow is not None and image is not None:
        loss_residual_smooth = edge_aware_residual_loss(delta_flow, image)
    total = excess_risk + 0.02 * loss_residual_smooth
    return total, {
        "excess_kinematic_risk": excess_risk,
        "residual_smoothness": loss_residual_smooth
    }

def weighted_residual_supervision_loss(delta_flow, target_delta, base_error, gamma=4.0):
    """直接监督 residual，并给基础模型误差大的像素更高权重。"""
    with torch.no_grad():
        err_scale = base_error.detach().mean(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        hard_weight = 1.0 + gamma * (base_error.detach() / err_scale).clamp(max=5.0)
    per_channel_weight = hard_weight.expand_as(delta_flow)
    return F.smooth_l1_loss(delta_flow * per_channel_weight, target_delta * per_channel_weight, beta=1e-3)

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

def photometric_match_loss(I1, I2, flow):
    I2_warped = warp_tensor(I2, flow)
    ncc = local_ncc_score(I1, I2_warped)
    photo = F.smooth_l1_loss(I2_warped, I1, beta=0.02)
    return photo + 0.1 * (1.0 - ncc).mean()

def gate_supervision_target(target_delta, base_error, match_confidence, kinematic_risk, candidate_ambiguity=None):
    target_mag = torch.norm(target_delta, p=2, dim=1, keepdim=True)
    scale = base_error.detach().mean(dim=[2, 3], keepdim=True).clamp(min=1e-6)
    correction_need = target_mag / (scale + target_mag + 1e-6)
    matching_evidence = match_confidence.detach().clamp(0.0, 1.0)
    conservative_factor = (1.0 - kinematic_risk.detach().clamp(0.0, 1.0)).sqrt()
    if candidate_ambiguity is not None:
        matching_evidence = matching_evidence * (1.0 - candidate_ambiguity.detach().clamp(0.0, 1.0))
    return torch.clamp(correction_need * matching_evidence * conservative_factor, 0.0, 1.0)

def candidate_blend_supervision_target(F_bar, gt_flow, candidate_stats):
    if candidate_stats is None:
        return None
    target_flow = candidate_stats.get("topk_match_flow", candidate_stats.get("best_match_flow"))
    if target_flow is None:
        return None
    target_flow = target_flow.detach()
    base_err = torch.norm(F_bar - gt_flow, p=2, dim=1, keepdim=True)
    target_err = torch.norm(target_flow - gt_flow, p=2, dim=1, keepdim=True)
    improvement = torch.clamp(base_err - target_err, min=0.0)
    scale = base_err.detach().mean(dim=[2, 3], keepdim=True).clamp(min=1e-6)
    return torch.clamp(improvement / (scale + improvement + 1e-6), 0.0, 1.0)

def candidate_gate_supervision_target(F_bar, gt_flow, candidate_stats):
    target = candidate_blend_supervision_target(F_bar, gt_flow, candidate_stats)
    if target is None:
        return None
    ncc_best = candidate_stats.get("ncc_best")
    risk = candidate_stats.get("candidate_kinematic_risk")
    ambiguity = candidate_stats.get("ncc_entropy")
    if ncc_best is None or risk is None:
        return target
    evidence = ((ncc_best.detach() + 1.0) * 0.5).clamp(0.0, 1.0)
    if ambiguity is not None:
        evidence = evidence * (1.0 - ambiguity.detach().clamp(0.0, 1.0))
    return target * evidence * (1.0 - risk.detach().clamp(0.0, 1.0)).sqrt()

def candidate_delta_supervision_target(F_bar, gt_flow, candidate_stats):
    if candidate_stats is None:
        return None
    gate_target = candidate_gate_supervision_target(F_bar, gt_flow, candidate_stats)
    if gate_target is None:
        return None
    if "topk_flows" in candidate_stats:
        topk_flows = candidate_stats["topk_flows"].detach()
        mix_target = candidate_mixture_supervision_target(gt_flow, candidate_stats)
        if mix_target is None:
            return None
        target_flow = torch.sum(mix_target.unsqueeze(2) * topk_flows, dim=1)
    else:
        target_flow = candidate_stats.get("topk_match_flow", candidate_stats.get("best_match_flow"))
        if target_flow is None:
            return None
        target_flow = target_flow.detach()
    return gate_target * (target_flow - F_bar.detach())

def candidate_mixture_supervision_target(gt_flow, candidate_stats):
    if candidate_stats is None or "topk_flows" not in candidate_stats:
        return None
    topk_flows = candidate_stats["topk_flows"].detach()
    target_err = torch.norm(topk_flows - gt_flow.unsqueeze(1), p=2, dim=2, keepdim=False)
    logits = -target_err
    return torch.softmax(logits * 20.0, dim=1)

def no_harm_refinement_loss(F_refined, F_bar, gt_flow, margin=0.0):
    refined_error = torch.norm(F_refined - gt_flow, p=2, dim=1, keepdim=True)
    base_error = torch.norm(F_bar - gt_flow, p=2, dim=1, keepdim=True)
    return F.relu(refined_error - base_error + margin).mean()


def intervene_on_posterior(sigma_F, candidate_stats):
    """Batch permutation implements do(S=s') while images/base flow stay fixed."""
    if sigma_F.shape[0] < 2:
        return sigma_F, candidate_stats
    permutation = torch.roll(torch.arange(sigma_F.shape[0], device=sigma_F.device), shifts=1)
    intervened_stats = None
    if candidate_stats is not None:
        intervened_stats = {
            key: value[permutation] if torch.is_tensor(value) and value.shape[0] == sigma_F.shape[0] else value
            for key, value in candidate_stats.items()
        }
    return sigma_F[permutation], intervened_stats


def potential_outcome_risk_loss(agent_aux, base_error, action_error):
    risk0 = agent_aux.get("potential_risk_no_action")
    risk1 = agent_aux.get("potential_risk_action")
    if risk0 is None or risk1 is None:
        return base_error.new_zeros(())
    scale = base_error.detach().mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    target0 = (base_error.detach() / scale).clamp(max=10.0)
    target1 = (action_error.detach() / scale).clamp(max=10.0)
    return F.smooth_l1_loss(risk0, target0) + F.smooth_l1_loss(risk1, target1)

def domain_head_anchor_loss(agent):
    if getattr(agent, "domain_delta_heads", None) is None or getattr(agent, "domain_gate_heads", None) is None:
        return None
    loss = None
    for delta_head, gate_head in zip(agent.domain_delta_heads, agent.domain_gate_heads):
        term = (
            torch.mean(delta_head.weight ** 2)
            + torch.mean(delta_head.bias ** 2)
            + torch.mean(gate_head.weight ** 2)
            + torch.mean(gate_head.bias ** 2)
        )
        loss = term if loss is None else loss + term
    return loss / max(len(agent.domain_delta_heads), 1)

def build_domain_balanced_sampler(dataset, domain_boosts=None):
    domain_boosts = domain_boosts or {}
    if not hasattr(dataset, "data_index_list") or not hasattr(dataset, "targets") or dataset.targets is None:
        return None, {}
    label_indices = dataset.targets_index_list if getattr(dataset, "targets_index_list", None) is not None else dataset.data_index_list
    domains = []
    for label_idx in label_indices:
        target_path = dataset.targets[int(label_idx)]
        domains.append(os.path.basename(os.path.dirname(target_path)))
    if len(set(domains)) <= 1:
        return None, dict(Counter(domains))
    counts = Counter(domains)
    weights = torch.as_tensor([domain_boosts.get(d, 1.0) / counts[d] for d in domains], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler, dict(counts)

def load_agent_weights(agent, path, device):
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_schema_version") != 2:
        raise ValueError(
            "Legacy agent checkpoint rejected: retrain the match-kinematic residual agent with this release."
        )
    expected_family = "causal_posterior_residual_agent_v1" if agent.use_causal_policy else "match_kinematic_residual_agent_v1"
    if checkpoint.get("agent_model_family") != expected_family:
        raise ValueError("Agent checkpoint model family is incompatible with the current method.")
    state_dict = checkpoint.get("agent_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Agent checkpoint is missing agent_state_dict.")
    expected_config = {
        "max_delta": float(agent.max_delta),
        "use_uncertainty": bool(agent.use_uncertainty),
        "use_kinematic_features": bool(agent.use_kinematic_features),
        "matching_radius": int(agent.matching_radius),
        "use_candidate_reweighting": bool(agent.use_candidate_reweighting),
        "candidate_topk": int(agent.candidate_topk),
        "use_domain_conditioning": bool(agent.use_domain_conditioning),
        "use_domain_heads": bool(agent.use_domain_heads),
        "domain_head_residual_scale": float(agent.domain_head_residual_scale),
        "num_domains": int(agent.num_domains),
        "flow_geometry": agent.geometry.as_dict() if agent.geometry is not None else FlowGeometry().as_dict(),
    }
    if agent.use_causal_policy:
        expected_config.update({
            "use_causal_policy": True,
            "causal_gate_temperature": float(agent.causal_gate_temperature),
        })
    if agent.agent_input_mode != "full":
        expected_config["agent_input_mode"] = agent.agent_input_mode
    if checkpoint.get("agent_config") != expected_config:
        raise ValueError(
            "Agent checkpoint configuration is incompatible with the requested refinement model. "
            f"checkpoint={checkpoint.get('agent_config')}, requested={expected_config}"
        )
    target_state = agent.state_dict()
    if set(state_dict) != set(target_state):
        raise ValueError("Agent checkpoint parameter names do not match the requested refinement architecture.")
    incompatible = [key for key in target_state if state_dict[key].shape != target_state[key].shape]
    if incompatible:
        raise ValueError(f"Agent checkpoint parameter shapes are incompatible: {incompatible[:3]}")
    agent.load_state_dict(state_dict, strict=True)
    return {"adapted": [], "skipped": [], "missing": [], "unexpected": []}

def build_flow_aggregator(args):
    geometry = FlowGeometry(
        pixel_size_x=args.pixel_size_x,
        pixel_size_y=args.pixel_size_y,
        frame_interval=args.frame_interval,
        flow_representation=args.flow_representation,
    )
    return FlowAggregator(geometry=geometry)


def configure_posterior_inference(diffusion_model, aggregator, args):
    guidance = PosteriorGuidanceConfig(
        observation_weight=args.posterior_observation_weight,
        continuity_weight=args.posterior_continuity_weight,
        smoothness_weight=args.posterior_smoothness_weight,
        particle_sigma=args.particle_psf_sigma,
        start_fraction=args.posterior_guidance_start,
    )
    diffusion_model.configure_posterior_guidance(guidance, geometry=aggregator.geometry)
    if args.adaptive_posterior_sampling:
        diffusion_model.configure_adaptive_sampling({
            "min_candidates": args.adaptive_min_candidates,
            "max_candidates": args.diffusion_candidates,
            "min_steps": args.adaptive_min_steps,
            "max_steps": args.diffusion_sampling_steps,
            "risk_threshold": args.adaptive_risk_threshold,
        })
    return {"guidance": guidance.as_dict(), "adaptive_policy": diffusion_model.adaptive_policy}

# ==========================================
# 2. Agent：监督式残差修正网络 (The Brain)
# ==========================================
class MatchKinematicResidualAgent(nn.Module):
    def __init__(
        self,
        max_delta=6.0,
        use_uncertainty=True,
        use_kinematic_features=True,
        matching_radius=0,
        use_candidate_reweighting=False,
        candidate_topk=3,
        use_domain_conditioning=False,
        use_domain_heads=False,
        domain_head_residual_scale=0.25,
        num_domains=5,
        geometry=None,
        use_causal_policy=False,
        causal_gate_temperature=0.25,
        agent_input_mode="full",
    ):
        super().__init__()
        # I1/I2/I2_warp/residual, flow, uncertainty, physics, and local matching features.
        self.matching_radius = max(0, int(matching_radius))
        self.use_candidate_reweighting = bool(use_candidate_reweighting)
        self.candidate_topk = max(1, int(candidate_topk))
        candidate_channels = 17 if self.use_candidate_reweighting else 0
        in_channels = 15 + (6 if self.matching_radius > 0 else 0) + candidate_channels
        self.max_delta = max_delta
        self.use_uncertainty = use_uncertainty
        self.use_kinematic_features = use_kinematic_features
        self.use_domain_conditioning = use_domain_conditioning
        self.num_domains = max(1, int(num_domains))
        self.use_domain_heads = bool(use_domain_heads)
        self.domain_head_residual_scale = float(domain_head_residual_scale)
        self.geometry = geometry
        self.use_causal_policy = bool(use_causal_policy)
        self.causal_gate_temperature = float(causal_gate_temperature)
        self.agent_input_mode = str(agent_input_mode)
        if self.agent_input_mode not in {"full", "observation_only", "posterior_only"}:
            raise ValueError("agent_input_mode must be full, observation_only, or posterior_only")
        if self.causal_gate_temperature <= 0:
            raise ValueError("causal_gate_temperature must be positive")

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.gate_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.potential_risk_head = nn.Conv2d(64, 2, kernel_size=3, padding=1) if self.use_causal_policy else None
        self.candidate_blend_head = nn.Conv2d(64, self.candidate_topk, kernel_size=3, padding=1) if self.use_candidate_reweighting else None
        self.candidate_gate_head = nn.Conv2d(64, 1, kernel_size=3, padding=1) if self.use_candidate_reweighting else None
        self.domain_delta_heads = nn.ModuleList([
            nn.Conv2d(64, 2, kernel_size=3, padding=1) for _ in range(self.num_domains)
        ]) if self.use_domain_heads else None
        self.domain_gate_heads = nn.ModuleList([
            nn.Conv2d(64, 1, kernel_size=3, padding=1) for _ in range(self.num_domains)
        ]) if self.use_domain_heads else None
        self.domain_film = nn.Embedding(self.num_domains, 128) if use_domain_conditioning else None
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.normal_(self.gate_head.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.gate_head.bias, -2.0)
        if self.candidate_blend_head is not None:
            nn.init.normal_(self.candidate_blend_head.weight, mean=0.0, std=1e-4)
            nn.init.constant_(self.candidate_blend_head.bias, -3.0)
        if self.candidate_gate_head is not None:
            nn.init.normal_(self.candidate_gate_head.weight, mean=0.0, std=1e-4)
            nn.init.constant_(self.candidate_gate_head.bias, -3.0)
        if self.domain_delta_heads is not None:
            for delta_head in self.domain_delta_heads:
                nn.init.zeros_(delta_head.weight)
                nn.init.zeros_(delta_head.bias)
        if self.domain_gate_heads is not None:
            for gate_head in self.domain_gate_heads:
                nn.init.zeros_(gate_head.weight)
                nn.init.zeros_(gate_head.bias)
        if self.domain_film is not None:
            nn.init.zeros_(self.domain_film.weight)
        if self.potential_risk_head is not None:
            nn.init.zeros_(self.potential_risk_head.weight)
            nn.init.constant_(self.potential_risk_head.bias, 0.5)

    def _normalize_per_sample(self, tensor):
        dims = tuple(range(1, tensor.dim()))
        scale = tensor.detach().abs().amax(dim=dims, keepdim=True).clamp(min=1e-6)
        return tensor / scale

    def _normalize_image(self, img):
        mean = img.mean(dim=[2, 3], keepdim=True)
        std = img.std(dim=[2, 3], keepdim=True).clamp(min=1e-5)
        return (img - mean) / std

    def _local_matching_features(self, I1, I2_warped):
        if self.matching_radius <= 0:
            return None
        scores = []
        offsets = []
        radius = self.matching_radius
        padded = F.pad(I2_warped, (radius, radius, radius, radius), mode="replicate")
        for dy in range(-self.matching_radius, self.matching_radius + 1):
            for dx in range(-self.matching_radius, self.matching_radius + 1):
                y0 = radius + dy
                x0 = radius + dx
                shifted = padded[:, :, y0:y0 + I1.shape[2], x0:x0 + I1.shape[3]]
                score = -torch.abs(I1 - shifted)
                scores.append(score)
                offsets.append((dx, dy))
        score_volume = torch.cat(scores, dim=1)
        best_idx = torch.argmax(score_volume, dim=1, keepdim=True)
        best_score = torch.gather(score_volume, 1, best_idx)
        if score_volume.shape[1] > 1:
            second_score = torch.topk(score_volume, k=2, dim=1).values[:, 1:2]
        else:
            second_score = best_score
        dx_values = torch.tensor([item[0] for item in offsets], device=I1.device, dtype=I1.dtype).view(1, -1, 1, 1)
        dy_values = torch.tensor([item[1] for item in offsets], device=I1.device, dtype=I1.dtype).view(1, -1, 1, 1)
        dx_grid = dx_values.expand_as(score_volume)
        dy_grid = dy_values.expand_as(score_volume)
        best_dx = torch.gather(dx_grid, 1, best_idx) / max(self.matching_radius, 1)
        best_dy = torch.gather(dy_grid, 1, best_idx) / max(self.matching_radius, 1)
        prob = F.softmax(score_volume * 20.0, dim=1)
        soft_dx = torch.sum(prob * dx_grid, dim=1, keepdim=True) / max(self.matching_radius, 1)
        soft_dy = torch.sum(prob * dy_grid, dim=1, keepdim=True) / max(self.matching_radius, 1)
        score_margin = best_score - second_score
        entropy = -torch.sum(prob * torch.log(prob.clamp(min=1e-6)), dim=1, keepdim=True)
        entropy = entropy / np.log(score_volume.shape[1])
        return torch.cat([
            best_dx,
            best_dy,
            soft_dx,
            soft_dy,
            self._normalize_per_sample(score_margin),
            1.0 - entropy,
        ], dim=1)

    def _candidate_features(self, F_bar, candidate_stats):
        if not self.use_candidate_reweighting:
            return []
        if candidate_stats is None:
            B, _, H, W = F_bar.shape
            return [F_bar.new_zeros((B, 17, H, W))]
        candidate_mean = candidate_stats.get("candidate_mean", F_bar)
        candidate_std = candidate_stats.get("candidate_std", torch.zeros_like(F_bar))
        best_match_flow = candidate_stats.get("topk_match_flow", candidate_stats.get("best_match_flow", F_bar))
        ncc_mean = candidate_stats.get("ncc_mean", torch.zeros_like(F_bar[:, 0:1]))
        ncc_std = candidate_stats.get("ncc_std", torch.zeros_like(F_bar[:, 0:1]))
        ncc_best = candidate_stats.get("ncc_best", torch.zeros_like(F_bar[:, 0:1]))
        ncc_margin = candidate_stats.get("ncc_margin", torch.zeros_like(F_bar[:, 0:1]))
        ncc_entropy = candidate_stats.get("ncc_entropy", torch.zeros_like(F_bar[:, 0:1]))
        candidate_risk = candidate_stats.get("candidate_kinematic_risk", torch.zeros_like(F_bar[:, 0:1]))
        candidate_fb_error = candidate_stats.get("candidate_fb_error", torch.zeros_like(F_bar[:, 0:1]))
        candidate_delta = best_match_flow - F_bar
        mean_delta = candidate_mean - F_bar
        return [
            self._normalize_per_sample(candidate_mean),
            self._normalize_per_sample(candidate_std),
            self._normalize_per_sample(best_match_flow),
            self._normalize_per_sample(candidate_delta),
            self._normalize_per_sample(mean_delta),
            ncc_mean,
            ncc_std,
            ncc_best,
            self._normalize_per_sample(ncc_margin),
            1.0 - ncc_entropy,
            self._normalize_per_sample(candidate_risk),
            self._normalize_per_sample(candidate_fb_error),
        ]

    def _build_inputs(self, I1, I2, F_bar, sigma_F, candidate_stats=None):
        if sigma_F.shape[1] == 1:
            sigma_F = sigma_F.repeat(1, 2, 1, 1)
        elif sigma_F.shape[1] > 2:
            sigma_F = sigma_F[:, :2, :, :]

        I2_warped = warp_tensor(I2, F_bar)
        residual = I1 - I2_warped
        photometric_residual = self._normalize_per_sample(residual)
        photometric_abs = self._normalize_per_sample(torch.abs(residual))
        ncc = local_ncc_score(I1, I2_warped)
        match_error = 1.0 - ncc
        match_confidence = 0.5 * (ncc + 1.0)

        diagnostics = compute_kinematic_diagnostics(F_bar, geometry=self.geometry)
        div_residual = self._normalize_per_sample(torch.abs(diagnostics["divergence"]))
        vort = self._normalize_per_sample(torch.abs(diagnostics["vorticity"]))
        strain_mag = self._normalize_per_sample(diagnostics["strain_invariant"])
        uncertainty_mag = self._normalize_per_sample(torch.norm(sigma_F, p=2, dim=1, keepdim=True))
        sigma_input = self._normalize_per_sample(sigma_F)

        if not self.use_uncertainty:
            sigma_input = torch.zeros_like(sigma_input)
            uncertainty_mag = torch.zeros_like(uncertainty_mag)
        if not self.use_kinematic_features:
            div_residual = torch.zeros_like(div_residual)
            vort = torch.zeros_like(vort)
            strain_mag = torch.zeros_like(strain_mag)

        if self.agent_input_mode == "posterior_only":
            image_1_input = torch.zeros_like(self._normalize_image(I1))
            image_2_input = torch.zeros_like(self._normalize_image(I2))
            warped_input = torch.zeros_like(self._normalize_image(I2_warped))
            photometric_residual = torch.zeros_like(photometric_residual)
            photometric_abs = torch.zeros_like(photometric_abs)
            match_error = torch.zeros_like(match_error)
            match_confidence = torch.zeros_like(match_confidence)
        else:
            image_1_input = self._normalize_image(I1)
            image_2_input = self._normalize_image(I2)
            warped_input = self._normalize_image(I2_warped)

        if self.agent_input_mode == "observation_only":
            sigma_input = torch.zeros_like(sigma_input)
            uncertainty_mag = torch.zeros_like(uncertainty_mag)
            div_residual = torch.zeros_like(div_residual)
            vort = torch.zeros_like(vort)
            strain_mag = torch.zeros_like(strain_mag)

        inputs = [
            image_1_input,
            image_2_input,
            warped_input,
            photometric_residual,
            photometric_abs,
            self._normalize_per_sample(F_bar),
            sigma_input,
            div_residual,
            vort,
            strain_mag,
            uncertainty_mag,
            match_error,
            match_confidence,
        ]
        local_match = self._local_matching_features(I1, I2_warped)
        if local_match is not None:
            if self.agent_input_mode == "posterior_only":
                local_match = torch.zeros_like(local_match)
            inputs.append(local_match)
        candidate_inputs = self._candidate_features(F_bar, candidate_stats)
        if self.agent_input_mode == "observation_only":
            candidate_inputs = [torch.zeros_like(value) for value in candidate_inputs]
        inputs.extend(candidate_inputs)
        return torch.cat(inputs, dim=1)

    def forward_with_aux(self, I1, I2, F_bar, sigma_F, domain_ids=None, candidate_stats=None):
        features = self.encoder(self._build_inputs(I1, I2, F_bar, sigma_F, candidate_stats=candidate_stats))
        if self.domain_film is not None and domain_ids is not None:
            film = self.domain_film(domain_ids.clamp(min=0, max=self.domain_film.num_embeddings - 1))
            gamma, beta = film.chunk(2, dim=1)
            gamma = gamma.view(-1, 64, 1, 1)
            beta = beta.view(-1, 64, 1, 1)
            features = features * (1.0 + gamma) + beta
        delta_logits = self.delta_head(features)
        gate_logits = self.gate_head(features)
        if self.domain_delta_heads is not None and self.domain_gate_heads is not None and domain_ids is not None:
            safe_domain_ids = domain_ids.clamp(min=0, max=self.num_domains - 1)
            for domain_idx in torch.unique(safe_domain_ids).tolist():
                mask = safe_domain_ids == int(domain_idx)
                delta_logits[mask] = delta_logits[mask] + self.domain_head_residual_scale * self.domain_delta_heads[int(domain_idx)](features[mask])
                gate_logits[mask] = gate_logits[mask] + self.domain_head_residual_scale * self.domain_gate_heads[int(domain_idx)](features[mask])
        raw_delta = self.max_delta * torch.tanh(delta_logits)
        gate = torch.sigmoid(gate_logits)
        potential_risks = None
        treatment_effect = None
        causal_gate = torch.ones_like(gate)
        if self.potential_risk_head is not None:
            potential_risks = F.softplus(self.potential_risk_head(features))
            treatment_effect = potential_risks[:, 0:1] - potential_risks[:, 1:2]
            causal_gate = torch.sigmoid(treatment_effect / self.causal_gate_temperature)
            gate = gate * causal_gate
        residual_delta = gate * raw_delta
        candidate_delta = torch.zeros_like(residual_delta)
        candidate_blend = torch.zeros_like(gate)
        candidate_gate = torch.zeros_like(gate)
        if self.candidate_blend_head is not None and candidate_stats is not None:
            candidate_target = candidate_stats.get("topk_match_flow", candidate_stats.get("best_match_flow", F_bar))
            topk_flows = candidate_stats.get("topk_flows")
            if topk_flows is not None:
                mix_logits = self.candidate_blend_head(features)
                mixture_weights = torch.softmax(mix_logits, dim=1)
                candidate_target = torch.sum(mixture_weights.unsqueeze(2) * topk_flows, dim=1)
                candidate_blend = mixture_weights[:, 0:1]
            else:
                candidate_blend = torch.sigmoid(self.candidate_blend_head(features))
            candidate_gate = torch.sigmoid(self.candidate_gate_head(features)) if self.candidate_gate_head is not None else torch.ones_like(gate)
            candidate_gate = candidate_gate * causal_gate
            candidate_delta = candidate_gate * (candidate_target - F_bar)
        delta_flow = residual_delta + candidate_delta
        return delta_flow, {
            "raw_delta": raw_delta,
            "gate": gate,
            "residual_delta": residual_delta,
            "candidate_delta": candidate_delta,
            "candidate_blend": candidate_blend,
            "candidate_gate": candidate_gate,
            "candidate_target_flow": F_bar + candidate_delta,
            "potential_risk_no_action": potential_risks[:, 0:1] if potential_risks is not None else None,
            "potential_risk_action": potential_risks[:, 1:2] if potential_risks is not None else None,
            "treatment_effect": treatment_effect,
            "causal_gate": causal_gate,
        }

    def forward(self, I1, I2, F_bar, sigma_F, domain_ids=None, candidate_stats=None):
        delta_flow, _ = self.forward_with_aux(
            I1,
            I2,
            F_bar,
            sigma_F,
            domain_ids=domain_ids,
            candidate_stats=candidate_stats
        )
        return delta_flow

    def refine(self, I1, I2, F_bar, sigma_F, domain_ids=None, candidate_stats=None):
        delta_flow, _ = self.forward_with_aux(
            I1,
            I2,
            F_bar,
            sigma_F,
            domain_ids=domain_ids,
            candidate_stats=candidate_stats
        )
        return F_bar + delta_flow

# ==========================================
# 3. 定向物理优化器 (The Scalpel)
# ==========================================
# ==========================================
# 4. 离线训练 Agent (教它直接修正流场)
# ==========================================
def train_agent_offline(
    train_loader,
    val_loader,
    diffusion_model,
    feature_extractor,
    aggregator,
    device,
    run_dir,
    num_epochs=50,
    lr=1e-3,
    max_delta=6.0,
    refine_steps=1,
    residual_loss_weight=0.01,
    kinematic_regularization_weight=0.05,
    delta_supervision_weight=2.0,
    gate_loss_weight=0.2,
    use_causal_policy=False,
    causal_risk_loss_weight=0.5,
    causal_counterfactual_loss_weight=0.25,
    causal_gate_temperature=0.25,
    match_loss_weight=0.05,
    no_harm_loss_weight=1.0,
    no_harm_margin=0.0,
    candidate_loss_weight=0.25,
    candidate_gate_loss_weight=0.1,
    candidate_delta_loss_weight=0.25,
    agent_init_path=None,
    heads_only=False,
    hard_pixel_gamma=4.0,
    use_uncertainty=True,
    use_kinematic_features=True,
    matching_radius=0,
    agent_input_mode="full",
    use_candidate_reweighting=True,
    use_forward_backward_consistency=True,
    candidate_topk=3,
    use_domain_conditioning=False,
    use_domain_heads=False,
    domain_head_anchor_weight=0.0,
    domain_head_residual_scale=0.25,
    num_domains=5,
    max_train_batches=0,
    max_val_batches=0,
    agent_checkpoint_name="pea_agent_best.pth",
    agent_resume_path=None,
    agent_state_name="agent_training_state.pth",
    training_identity=None,
    ):
    print("\n" + "="*60)
    print("Starting UPR-Flow match-aware, kinematically conservative residual refinement")
    print("="*60)
    agent_checkpoint_name = os.path.basename(agent_checkpoint_name) or DEFAULT_AGENT_CHECKPOINT_NAME

    agent = MatchKinematicResidualAgent(
        max_delta=max_delta,
        use_uncertainty=use_uncertainty,
        use_kinematic_features=use_kinematic_features,
        matching_radius=matching_radius,
        use_candidate_reweighting=use_candidate_reweighting,
        candidate_topk=candidate_topk,
        use_domain_conditioning=use_domain_conditioning,
        use_domain_heads=use_domain_heads,
        domain_head_residual_scale=domain_head_residual_scale,
        num_domains=num_domains,
        geometry=aggregator.geometry,
        use_causal_policy=use_causal_policy,
        causal_gate_temperature=causal_gate_temperature,
        agent_input_mode=agent_input_mode,
    ).to(device)
    if agent_init_path is not None and agent_resume_path is not None:
        raise ValueError("agent_init_path and agent_resume_path are mutually exclusive.")
    if agent_init_path is not None:
        load_report = load_agent_weights(agent, agent_init_path, device)
        print(f"Loaded initial agent weights from: {agent_init_path}")
        if load_report["adapted"]:
            print(f"Adapted agent weights: {load_report['adapted']}")
        if load_report["skipped"]:
            print(f"Skipped incompatible agent weights: {load_report['skipped'][:3]}")
    agent_optimizer = Adam(agent.parameters(), lr=lr)
    if heads_only:
        for param in agent.encoder.parameters():
            param.requires_grad = False
        frozen_params = [p for p in agent.parameters() if p.requires_grad]
        agent_optimizer = Adam(frozen_params, lr=lr)
        print("Heads-only fine-tuning enabled: encoder frozen, delta/gate heads trainable.")
    
    # 冻结基础扩散模型
    diffusion_model.eval()
    feature_extractor.eval()
    for p in diffusion_model.parameters(): p.requires_grad = False
    for p in feature_extractor.parameters(): p.requires_grad = False
        
    resume_signature = {
        "max_delta": float(max_delta),
        "refine_steps": int(refine_steps),
        "lr": float(lr),
        "heads_only": bool(heads_only),
        "use_uncertainty": bool(use_uncertainty),
        "use_kinematic_features": bool(use_kinematic_features),
        "matching_radius": int(matching_radius),
        "agent_input_mode": str(agent_input_mode),
        "use_candidate_reweighting": bool(use_candidate_reweighting),
        "use_forward_backward_consistency": bool(use_forward_backward_consistency),
        "candidate_topk": int(candidate_topk),
        "use_domain_conditioning": bool(use_domain_conditioning),
        "use_domain_heads": bool(use_domain_heads),
        "domain_head_residual_scale": float(domain_head_residual_scale),
        "num_domains": int(num_domains),
        "residual_loss_weight": float(residual_loss_weight),
        "kinematic_regularization_weight": float(kinematic_regularization_weight),
        "delta_supervision_weight": float(delta_supervision_weight),
        "gate_loss_weight": float(gate_loss_weight),
        "match_loss_weight": float(match_loss_weight),
        "no_harm_loss_weight": float(no_harm_loss_weight),
        "no_harm_margin": float(no_harm_margin),
        "candidate_loss_weight": float(candidate_loss_weight),
        "candidate_gate_loss_weight": float(candidate_gate_loss_weight),
        "candidate_delta_loss_weight": float(candidate_delta_loss_weight),
        "hard_pixel_gamma": float(hard_pixel_gamma),
        "use_causal_policy": bool(use_causal_policy),
        "causal_gate_temperature": float(causal_gate_temperature),
        "causal_risk_loss_weight": float(causal_risk_loss_weight),
        "causal_counterfactual_loss_weight": float(causal_counterfactual_loss_weight),
        "domain_head_anchor_weight": float(domain_head_anchor_weight),
        "training_identity": dict(training_identity or {}),
    }
    best_val_loss = float('inf')
    history = []
    start_epoch = 0
    history_path = os.path.join(run_dir, "agent_training_history.csv")
    history_json_path = os.path.join(run_dir, "agent_training_history.json")
    state_path = os.path.abspath(agent_resume_path) if agent_resume_path else os.path.join(
        run_dir, os.path.basename(agent_state_name) or "agent_training_state.pth"
    )
    if agent_resume_path is not None:
        restored = restore_agent_training_state(
            state_path,
            agent,
            agent_optimizer,
            device,
            expected_signature=resume_signature,
        )
        start_epoch = restored["completed_epochs"]
        best_val_loss = restored["best_val_loss"]
        history = restored["history"]
        if start_epoch > num_epochs:
            raise ValueError(
                f"Resume state has {start_epoch} completed epochs, exceeding requested agent_epochs={num_epochs}."
            )
        write_metrics_csv(history_path, history)
        save_json(history_json_path, history)
        print(f"Resumed agent training from epoch {start_epoch}/{num_epochs}: {state_path}")
    
    for epoch in range(start_epoch, num_epochs):
        agent.train()
        total_train_loss = 0.0
        total_train_raw_epe = 0.0
        total_train_refined_epe = 0.0
        total_train_delta = 0.0
        train_batches = 0
        active_refine_steps = max(1, int(refine_steps))
        
        for step, batch in enumerate(train_loader):
            if max_train_batches and step >= max_train_batches:
                break
            sample_batch, gt_flow, domain_ids = unpack_flow_batch(batch, device)
            I1, I2 = sample_batch[:, 0:1, :, :], sample_batch[:, 1:2, :, :]
            
            # 1. 获取扩散模型的初始预测场 (无梯度)
            with torch.no_grad():
                F_bar, sigma_F, candidate_stats = predict_and_aggregate(
                    diffusion_model,
                    feature_extractor,
                    aggregator,
                    I1,
                    I2,
                    use_candidate_reweighting=use_candidate_reweighting,
                    use_forward_backward_consistency=use_forward_backward_consistency,
                )

            # 2. Agent 直接学习“该往哪个方向修正”
            F_refined, delta_flow, agent_aux = run_iterative_agent(
                agent,
                I1,
                I2,
                F_bar.detach(),
                sigma_F.detach(),
                steps=active_refine_steps,
                domain_ids=domain_ids,
                candidate_stats=candidate_stats
            )
            target_delta = gt_flow - F_bar.detach()
            base_error = torch.norm(F_bar.detach() - gt_flow, p=2, dim=1, keepdim=True)
            
            # 3. 监督式残差学习：直接学习 residual，同时保留最终 EPE 与弱物理约束
            loss_epe = epe_loss(F_refined, gt_flow)
            loss_delta_sup = weighted_residual_supervision_loss(
                delta_flow,
                target_delta,
                base_error,
                gamma=hard_pixel_gamma
            )
            base_ncc = local_ncc_score(I1, warp_tensor(I2, F_bar.detach()))
            candidate_ambiguity = candidate_stats.get("ncc_entropy") if candidate_stats is not None else None
            gate_target = gate_supervision_target(
                target_delta,
                base_error,
                match_confidence=0.5 * (base_ncc + 1.0),
                kinematic_risk=kinematic_risk_map(F_bar.detach(), geometry=aggregator.geometry),
                candidate_ambiguity=candidate_ambiguity,
            )
            loss_gate = F.binary_cross_entropy(agent_aux["gate"].clamp(1e-4, 1.0 - 1e-4), gate_target)
            candidate_mix_target = candidate_mixture_supervision_target(gt_flow, candidate_stats)
            candidate_blend_target = candidate_blend_supervision_target(F_bar.detach(), gt_flow, candidate_stats)
            candidate_gate_target = candidate_gate_supervision_target(F_bar.detach(), gt_flow, candidate_stats)
            candidate_delta_target = candidate_delta_supervision_target(F_bar.detach(), gt_flow, candidate_stats)
            if candidate_mix_target is not None and "candidate_blend" in agent_aux and agent_aux["candidate_blend"].shape[1] > 1:
                loss_candidate = -(candidate_mix_target * torch.log(agent_aux["candidate_blend"].clamp(min=1e-6))).sum(dim=1).mean()
            elif candidate_blend_target is not None and "candidate_blend" in agent_aux:
                loss_candidate = F.binary_cross_entropy(
                    agent_aux["candidate_blend"].clamp(1e-4, 1.0 - 1e-4),
                    candidate_blend_target
                )
            else:
                loss_candidate = delta_flow.new_tensor(0.0)
            if candidate_gate_target is not None and "candidate_gate" in agent_aux:
                loss_candidate_gate = F.binary_cross_entropy(
                    agent_aux["candidate_gate"].clamp(1e-4, 1.0 - 1e-4),
                    candidate_gate_target
                )
            else:
                loss_candidate_gate = delta_flow.new_tensor(0.0)
            if candidate_delta_target is not None and "candidate_delta" in agent_aux:
                loss_candidate_delta = weighted_residual_supervision_loss(
                    agent_aux["candidate_delta"],
                    candidate_delta_target,
                    base_error,
                    gamma=max(hard_pixel_gamma * 0.5, 1.0)
                )
            else:
                loss_candidate_delta = delta_flow.new_tensor(0.0)
            loss_match = photometric_match_loss(I1, I2, F_refined)
            loss_no_harm = no_harm_refinement_loss(F_refined, F_bar.detach(), gt_flow, margin=no_harm_margin)
            refined_error = torch.norm(F_refined - gt_flow, p=2, dim=1, keepdim=True)
            loss_causal_risk = potential_outcome_risk_loss(agent_aux, base_error, refined_error)
            loss_causal_counterfactual = delta_flow.new_zeros(())
            if use_causal_policy and F_bar.shape[0] > 1:
                sigma_do, stats_do = intervene_on_posterior(sigma_F.detach(), candidate_stats)
                F_do, _, aux_do = run_iterative_agent(
                    agent, I1, I2, F_bar.detach(), sigma_do, steps=active_refine_steps,
                    domain_ids=domain_ids, candidate_stats=stats_do,
                )
                error_do = torch.norm(F_do - gt_flow, p=2, dim=1, keepdim=True)
                loss_causal_counterfactual = potential_outcome_risk_loss(aux_do, base_error, error_do)
            loss_residual = torch.mean(delta_flow ** 2)
            loss_kinematic, kinematic_terms = kinematic_conservatism_loss(
                F_refined,
                F_bar.detach(),
                delta_flow=delta_flow,
                image=I1,
                geometry=aggregator.geometry,
            )
            anchor_loss = domain_head_anchor_loss(agent)
            if anchor_loss is None:
                anchor_loss = delta_flow.new_tensor(0.0)
            loss = (
                loss_epe
                + delta_supervision_weight * loss_delta_sup
                + gate_loss_weight * loss_gate
                + candidate_loss_weight * loss_candidate
                + candidate_gate_loss_weight * loss_candidate_gate
                + candidate_delta_loss_weight * loss_candidate_delta
                + match_loss_weight * loss_match
                + no_harm_loss_weight * loss_no_harm
                + causal_risk_loss_weight * loss_causal_risk
                + causal_counterfactual_loss_weight * loss_causal_counterfactual
                + residual_loss_weight * loss_residual
                + kinematic_regularization_weight * loss_kinematic
                + domain_head_anchor_weight * anchor_loss
            )
            
            agent_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            agent_optimizer.step()
            
            total_train_loss += loss.item()
            total_train_raw_epe += epe_loss(F_bar, gt_flow).item()
            total_train_refined_epe += loss_epe.item()
            total_train_delta += delta_flow.detach().abs().mean().item()
            train_batches += 1
            
            if step % 20 == 0:
                raw_epe = epe_loss(F_bar, gt_flow).item()
                refined_epe = loss_epe.item()
                delta_mean = delta_flow.detach().abs().mean().item()
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] Step {step} | "
                    f"Loss: {loss.item():.6f} | EPE: {raw_epe:.4f} -> {refined_epe:.4f} | "
                    f"delta_mean: {delta_mean:.6f} | "
                    f"gate: {agent_aux['gate'].detach().mean().item():.4f} | "
                    f"cand_gate: {agent_aux.get('candidate_gate', agent_aux['gate']).detach().mean().item():.4f} | "
                    f"cand_delta: {agent_aux.get('candidate_delta', delta_flow).detach().abs().mean().item():.6f} | "
                    f"target_delta: {target_delta.detach().abs().mean().item():.6f} | "
                    f"delta_sup: {loss_delta_sup.item():.6f} | gate/cand/cand_gate/cand_delta: {loss_gate.item():.6f}/{loss_candidate.item():.6f}/{loss_candidate_gate.item():.6f}/{loss_candidate_delta.item():.6f} | "
                    f"match: {loss_match.item():.6f} | no_harm: {loss_no_harm.item():.6f} | "
                    f"anchor: {anchor_loss.item():.6f} | "
                    f"kinematic-risk: {kinematic_terms['excess_kinematic_risk'].item():.6f}"
                )
                
        # 验证阶段：直接看 residual agent 是否降低 EPE
        print("--- 验证阶段 (Residual 修正效果) ---")
        agent.eval()
        val_loss = 0.0
        val_raw_epe = 0.0
        val_refined_epe = 0.0
        val_delta_mean = 0.0
        val_gate_mean = 0.0
        val_candidate_gate_mean = 0.0
        val_candidate_delta_mean = 0.0
        val_div_l2 = 0.0
        val_vorticity_grad_l2 = 0.0
        val_strain_rate_l2 = 0.0
        val_batches = 0
        val_sample_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if max_val_batches and val_batches >= max_val_batches:
                    break
                s_batch, l_batch, domain_ids = unpack_flow_batch(batch, device)
                batch_size = s_batch.size(0)
                I1, I2 = s_batch[:, 0:1, :, :], s_batch[:, 1:2, :, :]
                
                F_bar, sigma_F, candidate_stats = predict_and_aggregate(
                    diffusion_model,
                    feature_extractor,
                    aggregator,
                    I1,
                    I2,
                    use_candidate_reweighting=use_candidate_reweighting,
                    use_forward_backward_consistency=use_forward_backward_consistency,
                )
                
                F_refined, delta_flow, agent_aux = run_iterative_agent(
                    agent,
                    I1,
                    I2,
                    F_bar,
                    sigma_F,
                    steps=refine_steps,
                    domain_ids=domain_ids,
                    candidate_stats=candidate_stats
                )
                batch_loss = epe_loss(F_refined, l_batch)
                val_loss += batch_loss.item()
                val_raw_epe += epe_loss(F_bar, l_batch).item()
                val_refined_epe += batch_loss.item()
                val_delta_mean += delta_flow.abs().mean().item()
                val_gate_mean += agent_aux["gate"].mean().item()
                val_candidate_gate_mean += agent_aux.get("candidate_gate", agent_aux["gate"]).mean().item()
                val_candidate_delta_mean += agent_aux.get("candidate_delta", delta_flow).abs().mean().item()
                refined_kinematics = flow_kinematic_metrics(F_refined, geometry=aggregator.geometry)
                val_div_l2 += refined_kinematics["div_l2"] * batch_size
                val_vorticity_grad_l2 += refined_kinematics["vorticity_grad_l2"] * batch_size
                val_strain_rate_l2 += refined_kinematics["strain_rate_l2"] * batch_size
                val_batches += 1
                val_sample_count += batch_size
        
        avg_val_loss = val_loss / max(val_batches, 1)
        avg_train_loss = total_train_loss / max(train_batches, 1)
        avg_train_raw_epe = total_train_raw_epe / max(train_batches, 1)
        avg_train_refined_epe = total_train_refined_epe / max(train_batches, 1)
        avg_train_delta = total_train_delta / max(train_batches, 1)
        avg_raw_epe = val_raw_epe / max(val_batches, 1)
        avg_refined_epe = val_refined_epe / max(val_batches, 1)
        avg_delta_mean = val_delta_mean / max(val_batches, 1)
        avg_gate_mean = val_gate_mean / max(val_batches, 1)
        avg_candidate_gate_mean = val_candidate_gate_mean / max(val_batches, 1)
        avg_candidate_delta_mean = val_candidate_delta_mean / max(val_batches, 1)
        val_sample_count = max(val_sample_count, 1)
        avg_val_div_l2 = val_div_l2 / val_sample_count
        avg_val_vorticity_grad_l2 = val_vorticity_grad_l2 / val_sample_count
        avg_val_strain_rate_l2 = val_strain_rate_l2 / val_sample_count
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_raw_epe": avg_train_raw_epe,
            "train_refined_epe": avg_train_refined_epe,
            "train_delta_mean": avg_train_delta,
            "delta_supervision_weight": delta_supervision_weight,
            "hard_pixel_gamma": hard_pixel_gamma,
            "val_loss": avg_val_loss,
            "val_raw_epe": avg_raw_epe,
            "val_refined_epe": avg_refined_epe,
            "val_delta_mean": avg_delta_mean,
            "val_gate_mean": avg_gate_mean,
            "val_candidate_gate_mean": avg_candidate_gate_mean,
            "val_candidate_delta_mean": avg_candidate_delta_mean,
            "val_improvement": avg_raw_epe - avg_refined_epe,
            "val_improvement_pct": (avg_raw_epe - avg_refined_epe) / (avg_raw_epe + 1e-8) * 100.0,
            "val_div_l2": avg_val_div_l2,
            "val_vorticity_grad_l2": avg_val_vorticity_grad_l2,
            "val_strain_rate_l2": avg_val_strain_rate_l2
        }
        history.append(epoch_metrics)
        print(
            f">>> 验证集平均 Loss: {avg_val_loss:.6f} | "
            f"EPE: {avg_raw_epe:.4f} -> {avg_refined_epe:.4f} | "
            f"delta_mean: {avg_delta_mean:.6f} | "
            f"cand_gate: {avg_candidate_gate_mean:.4f} | "
            f"cand_delta: {avg_candidate_delta_mean:.6f} | "
            f"div_l2: {avg_val_div_l2:.6f}"
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "checkpoint_schema_version": 2,
                    "agent_model_family": (
                        "causal_posterior_residual_agent_v1" if use_causal_policy
                        else "match_kinematic_residual_agent_v1"
                    ),
                    "agent_config": {
                        "max_delta": float(max_delta),
                        "use_uncertainty": bool(use_uncertainty),
                        "use_kinematic_features": bool(use_kinematic_features),
                        "matching_radius": int(matching_radius),
                        "use_candidate_reweighting": bool(use_candidate_reweighting),
                        "candidate_topk": int(candidate_topk),
                        "use_domain_conditioning": bool(use_domain_conditioning),
                        "use_domain_heads": bool(use_domain_heads),
                        "domain_head_residual_scale": float(domain_head_residual_scale),
                        "num_domains": int(num_domains),
                        **({
                            "use_causal_policy": True,
                            "causal_gate_temperature": float(causal_gate_temperature),
                        } if use_causal_policy else {}),
                        **({"agent_input_mode": str(agent_input_mode)} if agent_input_mode != "full" else {}),
                        "flow_geometry": aggregator.geometry.as_dict(),
                    },
                    "agent_state_dict": agent.state_dict(),
                    "training_signature": resume_signature,
                },
                os.path.join(run_dir, agent_checkpoint_name),
            )
            save_json(os.path.join(run_dir, "best_agent_metrics.json"), epoch_metrics)
            print("Saved improved agent weights.\n")

        write_metrics_csv(history_path, history)
        save_json(history_json_path, history)
        save_agent_training_state(
            state_path,
            agent,
            agent_optimizer,
            completed_epochs=epoch + 1,
            best_val_loss=best_val_loss,
            history=history,
            training_signature=resume_signature,
        )
        print(f"Saved resumable agent state after epoch {epoch + 1}: {state_path}")

    save_json(history_json_path, history)
    return agent

# ==========================================
# 5. 推理与评估核心逻辑
# ==========================================
def unpack_flow_batch(batch, device):
    if len(batch) == 3:
        sample_batch, label_batch, domain_ids = batch
        return sample_batch.to(device), label_batch.to(device), domain_ids.to(device)
    sample_batch, label_batch = batch
    return sample_batch.to(device), label_batch.to(device), None


def subset_dataset_by_domains(dataset, allowed_domain_ids):
    if not allowed_domain_ids:
        return dataset
    if not hasattr(dataset, "data_index_list") or getattr(dataset, "domain_ids", None) is None:
        return dataset
    kept_indices = []
    kept_target_indices = []
    target_indices = dataset.targets_index_list if getattr(dataset, "targets_index_list", None) is not None else dataset.data_index_list
    for sample_idx, target_idx in zip(dataset.data_index_list, target_indices):
        domain_id = int(dataset.domain_ids[int(sample_idx)])
        if domain_id in allowed_domain_ids:
            kept_indices.append(sample_idx)
            kept_target_indices.append(target_idx)
    return FlowDataset(
        data_index_list=kept_indices,
        data=dataset.data,
        targets_index_list=kept_target_indices if dataset.targets is not None else None,
        targets=dataset.targets,
        domain_ids=dataset.domain_ids,
        sample_uid_by_index=getattr(dataset, "sample_uid_by_index", None),
        train=dataset.train,
    )


def split_manifest_hash(path):
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Referenced split manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        return str(json.load(handle).get("split_sha256", "")), str(manifest_path)


def validate_zero_shot_agent_provenance(agent_path, eval_domains, inference_split_metadata=None):
    expected_domains = sorted(set(eval_domains))
    protocol_path = Path(agent_path).resolve().parent / "training_protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(
            "zero_shot_holdout requires an agent produced by this repository's tracked training workflow. "
            f"Missing: {protocol_path}"
        )
    with protocol_path.open("r", encoding="utf-8") as handle:
        training_protocol = json.load(handle)
    excluded_domains = sorted(set(training_protocol.get("excluded_domains", [])))
    if training_protocol.get("protocol") != "leave_one_domain_out":
        raise ValueError(
            f"Agent provenance is not leave-one-domain-out: {protocol_path}"
        )
    if excluded_domains != expected_domains:
        raise ValueError(
            "Zero-shot held-out domains do not match the agent training provenance: "
            f"expected={expected_domains}, trained_exclusions={excluded_domains}."
        )
    training_split_manifest = training_protocol.get("source_split_manifest")
    if not training_split_manifest:
        raise ValueError("Zero-shot agent provenance is missing the fixed source split manifest path.")
    training_source_hash, training_source_path = split_manifest_hash(training_split_manifest)
    if inference_split_metadata is not None:
        inference_source_manifest = inference_split_metadata.get("source_manifest")
        if not inference_source_manifest:
            raise ValueError("Zero-shot inference must use a fixed manifest rather than an ad hoc generated split.")
        inference_source_hash, inference_source_path = split_manifest_hash(inference_source_manifest)
        if training_source_hash != inference_source_hash:
            raise ValueError(
                "Zero-shot agent and inference reference different fixed split manifests: "
                f"training={training_source_path}, inference={inference_source_path}."
            )
    return {
        "agent_path": str(Path(agent_path).resolve()),
        "training_protocol_path": str(protocol_path),
        "training_protocol": training_protocol,
        "held_out_domains": expected_domains,
        "source_split_manifest": training_source_path,
        "source_split_sha256": training_source_hash,
    }


def validate_in_domain_agent_provenance(agent_path, split_metadata):
    training_run_dir = Path(agent_path).resolve().parent
    config_path = training_run_dir / "run_config.json"
    protocol_path = training_run_dir / "training_protocol.json"
    if not config_path.exists() or not protocol_path.exists():
        raise FileNotFoundError(
            "Audited in-domain inference requires an agent produced by a tracked train_agent run with "
            f"run_config.json and training_protocol.json: {training_run_dir}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        training_config = json.load(handle)
    with protocol_path.open("r", encoding="utf-8") as handle:
        training_protocol = json.load(handle)
    if training_config.get("mode") != "train_agent":
        raise ValueError(f"Agent provenance is not a train_agent run: {training_run_dir}")
    if training_protocol.get("protocol") != "in_domain_training":
        raise ValueError(
            "Audited in-domain inference requires an in_domain_training agent. "
            f"Found protocol={training_protocol.get('protocol')}"
        )
    expected_hash = split_metadata["split_sha256"]
    training_hash = training_config.get("split_sha256")
    is_derived_dataset = bool(split_metadata.get("source_manifest"))
    source_split_hash = None
    source_split_path = None
    if is_derived_dataset:
        source_split_hash, source_split_path = split_manifest_hash(split_metadata["source_manifest"])
    if training_hash != expected_hash and training_hash != source_split_hash:
        raise ValueError(
            "Agent training split does not match the inference split or its declared source split: "
            f"training={training_hash}, inference={expected_hash}, source={source_split_hash}."
        )
    if training_config.get("dataset_fingerprint_mode") != "triplet_sha256":
        raise ValueError("Agent training did not use the required triplet_sha256 dataset fingerprint.")
    return {
        "agent_path": str(Path(agent_path).resolve()),
        "training_run_dir": str(training_run_dir),
        "training_config_path": str(config_path),
        "training_protocol_path": str(protocol_path),
        "training_split_sha256": training_config["split_sha256"],
        "inference_split_sha256": expected_hash,
        "inference_source_split_path": source_split_path,
        "inference_source_split_sha256": source_split_hash,
    }


def dataset_manifest_metadata(dataset_manifest_path):
    if not dataset_manifest_path:
        return None
    manifest_path = Path(dataset_manifest_path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset provenance manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Dataset provenance manifest must be a JSON object.")
    return {"path": str(manifest_path), "payload": payload}


def sample_uids_for_dataset(dataset):
    uid_by_index = getattr(dataset, "sample_uid_by_index", {}) or {}
    return [uid_by_index.get(int(index), f"index_{int(index)}") for index in dataset.data_index_list]


def maybe_wrap_class2_dataset(dataset, enabled, seed):
    if not enabled:
        return dataset
    return Class2NoiseDataset(dataset, seed=seed)

# ==========================================
# 6. 主程序入口
# ==========================================
if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.mode == "train_agent" and args.eval_domains:
        raise ValueError("--eval-domains is only valid for inference.")
    if args.mode == "inference" and (args.focus_domains or args.exclude_domains):
        raise ValueError("--focus-domains and --exclude-domains are only valid for train_agent.")
    if args.mode == "train_agent" and args.focus_domains and args.exclude_domains:
        raise ValueError("Use either --focus-domains or --exclude-domains, not both.")
    if args.mode == "train_agent" and args.evaluation_protocol != "in_domain":
        raise ValueError("--evaluation-protocol is only valid for inference.")
    if args.candidate_topk > args.diffusion_candidates:
        raise ValueError("--candidate-topk cannot exceed --diffusion-candidates.")
    if args.diffusion_denoised_clip <= 0.0:
        raise ValueError("--diffusion-denoised-clip must be positive.")
    if not 0.0 < args.uncertainty_coverage < 1.0:
        raise ValueError("--uncertainty-coverage must be strictly between zero and one.")
    if args.evaluation_sampling_seed is not None and args.evaluation_sampling_seed < 0:
        raise ValueError("--evaluation-sampling-seed must be non-negative.")
    if args.evaluation_protocol == "zero_shot_holdout":
        if args.mode != "inference":
            raise ValueError("zero_shot_holdout is an inference protocol.")
        holdout_domains = parse_name_list(args.eval_domains)
        if not holdout_domains:
            raise ValueError("zero_shot_holdout requires --eval-domains with the held-out domain name.")
        if len(set(holdout_domains)) != 1:
            raise ValueError("zero_shot_holdout currently requires exactly one held-out domain per run.")
        if args.selection_mode not in ("base", "agent"):
            raise ValueError(
                "zero_shot_holdout forbids target-label calibration; use --selection-mode base or agent."
            )
        if args.max_calibration_batches:
            raise ValueError("zero_shot_holdout does not run calibration; remove --max-calibration-batches.")
        if args.require_backbone_holdout_provenance and not args.backbone_provenance_manifest:
            raise ValueError(
                "Strict zero-shot evaluation requires --backbone-provenance-manifest for the diffusion checkpoint."
            )
    if args.mode == "train_agent" and args.require_backbone_holdout_provenance:
        if not parse_name_list(args.exclude_domains):
            raise ValueError("Backbone holdout provenance validation requires --exclude-domains during train_agent.")
        if not args.backbone_provenance_manifest:
            raise ValueError("Backbone holdout provenance validation requires --backbone-provenance-manifest.")
    if args.mode == "train_agent" and args.require_agent_split_provenance:
        raise ValueError("--require-agent-split-provenance is only valid for inference.")
    if args.agent_resume_path and args.mode != "train_agent":
        raise ValueError("--agent-resume-path is only valid for train_agent.")
    if args.agent_resume_path and args.agent_init_path:
        raise ValueError("--agent-resume-path cannot be combined with --agent-init-path.")
    if args.agent_resume_path:
        if not args.allow_overwrite:
            raise ValueError("Resuming a run requires --allow-overwrite to acknowledge reuse of its run directory.")
        args.agent_resume_path = resolve_existing_path(args.agent_resume_path, args.output_dir)
        if not os.path.isfile(args.agent_resume_path):
            raise FileNotFoundError(f"Agent training state not found: {args.agent_resume_path}")
    if os.path.basename(args.agent_state_name) != args.agent_state_name:
        raise ValueError("--agent-state-name must be a filename, not a path.")
    set_seed(args.seed)
    run_dir = make_run_dir(args.output_dir, args.run_name, allow_overwrite=args.allow_overwrite)
    if args.agent_resume_path and Path(args.agent_resume_path).resolve().parent != Path(run_dir).resolve():
        raise ValueError("--agent-resume-path must belong to the run directory being resumed.")
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")
    save_json(os.path.join(run_dir, "run_config.json"), build_run_config(args, run_dir, device))
    save_json(os.path.join(run_dir, "method_card.json"), method_card())

    domain_id_list = None
    domain_to_id = {}
    active_training_domain_names = None
    if args.mode == "train_agent":
        img1_2d, img2_2d, gt_2d, dir_names = read_by_type(
            args.data_path,
            write_index_cache=args.write_data_index_cache,
        )
        domain_to_id = {name: idx for idx, name in enumerate(dir_names)}
        focus_domains = parse_name_list(args.focus_domains)
        exclude_domains = parse_name_list(args.exclude_domains)
        requested_domains = focus_domains or exclude_domains
        unknown_domains = sorted(set(requested_domains) - set(domain_to_id))
        if unknown_domains:
            raise ValueError(f"Unknown training domains: {unknown_domains}. Available: {sorted(domain_to_id)}")
        if focus_domains:
            active_training_domain_names = set(focus_domains)
        elif exclude_domains:
            active_training_domain_names = set(domain_to_id) - set(exclude_domains)
        else:
            active_training_domain_names = set(domain_to_id)
        if not active_training_domain_names:
            raise ValueError("No training domains remain after applying the requested domain filter.")
        img1_list, img2_list, gt_list, domain_id_list = [], [], [], []
        for i, ds_name in enumerate(dir_names):
            if ds_name not in active_training_domain_names:
                continue
            img1_list.extend(img1_2d[i])
            img2_list.extend(img2_2d[i])
            gt_list.extend(gt_2d[i])
            domain_id_list.extend([domain_to_id[ds_name]] * len(gt_2d[i]))
        print(f"Training data loaded from {len(dir_names)} domains: {dir_names}")
    else:
        img1_2d, img2_2d, gt_2d, dir_names = read_by_type(
            args.data_path,
            write_index_cache=args.write_data_index_cache,
        )
        domain_to_id = {name: idx for idx, name in enumerate(dir_names)}
        img1_list, img2_list, gt_list, domain_id_list = [], [], [], []
        for i, ds_name in enumerate(dir_names):
            img1_list.extend(img1_2d[i])
            img2_list.extend(img2_2d[i])
            gt_list.extend(gt_2d[i])
            domain_id_list.extend([domain_to_id[ds_name]] * len(gt_2d[i]))
        print(f"Inference data loaded from {len(dir_names)} domains: {dir_names}")

    train_ds, val_ds, test_ds, split_metadata = construct_dataset(
        img1_list,
        img2_list,
        gt_list,
        domain_id_list=domain_id_list,
        split_seed=args.split_seed,
        split_protocol=args.split_protocol,
        split_manifest_path=args.split_manifest,
        test_size=args.test_size,
        fingerprint_mode=args.dataset_fingerprint_mode,
        purge_gap=args.purge_gap,
        return_metadata=True,
    )
    dataset_provenance = dataset_manifest_metadata(args.dataset_manifest)
    if args.dataset_kind == "experimental" and dataset_provenance is None:
        raise ValueError("--dataset-kind experimental requires --dataset-manifest with acquisition and test-set provenance.")
    experimental_protocol_report = None
    if args.dataset_kind == "experimental":
        experimental_protocol_report = validate_experimental_protocol(dataset_provenance["payload"], split_metadata)
        save_json(os.path.join(run_dir, "experimental_protocol_validation.json"), experimental_protocol_report)
        if not experimental_protocol_report["available"]:
            raise ValueError(
                "Experimental PIV protocol validation failed: "
                + "; ".join(experimental_protocol_report["issues"])
            )
    save_json(os.path.join(run_dir, "split_manifest.json"), split_metadata)
    if dataset_provenance is not None:
        save_json(os.path.join(run_dir, "dataset_provenance.json"), dataset_provenance)
    run_config = build_run_config(args, run_dir, device)
    run_config.update(
        {
            "split_sha256": split_metadata["split_sha256"],
            "split_counts": split_metadata["split_counts"],
            "dataset_fingerprint_mode": split_metadata["dataset_fingerprint_mode"],
            "split_schema_version": split_metadata.get("schema_version"),
            "effective_split_seed": split_metadata.get("split_seed"),
            "effective_test_size": split_metadata.get("test_size"),
            "effective_purge_gap": split_metadata.get("purge_gap"),
            "total_source_sample_count": split_metadata.get("total_source_sample_count"),
            "purged_sample_count": split_metadata.get("purged_sample_count", 0),
            "source_split_protocol": split_metadata.get("source_split_protocol"),
            "source_split_sha256": split_metadata.get("source_split_sha256"),
            "dataset_provenance_path": dataset_provenance["path"] if dataset_provenance else None,
            "experimental_protocol_validated": bool(
                experimental_protocol_report and experimental_protocol_report["available"]
            ),
        }
    )
    save_json(os.path.join(run_dir, "run_config.json"), run_config)
    print(
        f"Split protocol: {split_metadata['split_protocol']} | sha256={split_metadata['split_sha256'][:12]} | "
        f"counts={split_metadata['split_counts']}"
    )

    if args.mode == "train_agent":
        print("\n>>> Starting UPR-Flow residual refiner offline training <<<")
        focus_domains = parse_name_list(args.focus_domains)
        exclude_domains = parse_name_list(args.exclude_domains)
        if focus_domains:
            active_domain_ids = {domain_to_id[name] for name in focus_domains}
            active_domain_names = sorted(focus_domains)
            training_filter = "focus"
        elif exclude_domains:
            active_domain_ids = set(domain_to_id.values()) - {domain_to_id[name] for name in exclude_domains}
            active_domain_names = sorted(name for name in domain_to_id if name not in set(exclude_domains))
            training_filter = "exclude"
        else:
            active_domain_ids = set(domain_to_id.values())
            active_domain_names = sorted(domain_to_id)
            training_filter = "all"

        if focus_domains or exclude_domains:
            train_ds = subset_dataset_by_domains(train_ds, active_domain_ids)
            val_ds = subset_dataset_by_domains(val_ds, active_domain_ids)
            print(
                f"Training domain filter={training_filter}: {active_domain_names} | "
                f"train={len(train_ds)} | val={len(val_ds)}"
            )
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise ValueError("Training or validation split is empty after applying the domain filter.")
        save_json(
            os.path.join(run_dir, "training_protocol.json"),
            {
                "protocol": "leave_one_domain_out" if exclude_domains else "in_domain_training",
                "domain_filter": training_filter,
                "training_domains": active_domain_names,
                "excluded_domains": sorted(exclude_domains),
                "focused_domains": sorted(focus_domains),
                "train_sample_count": len(train_ds),
                "validation_sample_count": len(val_ds),
                "domain_adapters_trained": bool(args.use_domain_heads or args.use_domain_conditioning),
                "split_protocol": split_metadata["split_protocol"],
                "split_sha256": split_metadata["split_sha256"],
                "split_schema_version": split_metadata.get("schema_version"),
                "effective_split_seed": split_metadata.get("split_seed"),
                "effective_test_size": split_metadata.get("test_size"),
                "effective_purge_gap": split_metadata.get("purge_gap"),
                "total_source_sample_count": split_metadata.get("total_source_sample_count"),
                "purged_sample_count": split_metadata.get("purged_sample_count", 0),
                "source_split_manifest": split_metadata.get("source_manifest"),
                "source_split_protocol": split_metadata.get("source_split_protocol"),
                "source_split_sha256": split_metadata.get("source_split_sha256"),
                "dataset_kind": args.dataset_kind,
                "dataset_fingerprint_mode": split_metadata["dataset_fingerprint_mode"],
                "dataset_provenance_manifest": dataset_provenance["path"] if dataset_provenance else None,
                "agent_state_path": os.path.abspath(args.agent_resume_path)
                if args.agent_resume_path
                else os.path.abspath(os.path.join(run_dir, args.agent_state_name)),
                "agent_resume_requested": bool(args.agent_resume_path),
            },
        )

        train_ds = maybe_wrap_class2_dataset(train_ds, args.simulate_class2, args.seed)
        val_ds = maybe_wrap_class2_dataset(val_ds, args.simulate_class2, args.seed)

        sampler = None
        domain_boosts = {}
        if not args.disable_domain_balanced_sampler:
            domain_boosts = parse_domain_boosts(args.domain_boosts)
            sampler, domain_counts = build_domain_balanced_sampler(train_ds, domain_boosts=domain_boosts)
            if sampler is not None:
                print(f"Domain-balanced sampler enabled: {domain_counts}")
                if domain_boosts:
                    print(f"Domain sampling boosts: {domain_boosts}")

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        feature_extractor = DualSourceFeatureExtractor(device).to(device)
        diffusion_model = ConditionalDiffusionFlowGenerator(
            num_candidates=args.diffusion_candidates,
            flow_scale=args.diffusion_flow_scale,
            num_train_timesteps=args.diffusion_timesteps,
            sampling_steps=args.diffusion_sampling_steps,
            ddim_eta=args.ddim_eta,
            denoised_clip=args.diffusion_denoised_clip,
            center_loss_weight=args.diffusion_center_loss_weight,
            residual_reconstruction_weight=args.diffusion_residual_reconstruction_weight,
            learn_variance=args.diffusion_learn_variance,
            variance_fit_weight=args.diffusion_variance_fit_weight,
            variance_injection_scale=args.diffusion_variance_injection_scale,
        ).to(device)
        aggregator = build_flow_aggregator(args)
        posterior_inference_config = configure_posterior_inference(diffusion_model, aggregator, args)
        save_json(os.path.join(run_dir, "posterior_inference_config.json"), posterior_inference_config)

        pretrained_model_path = resolve_existing_path(
            args.checkpoint,
            args.output_dir,
            REPO_ROOT / "checkpoints",
        )
        if os.path.exists(pretrained_model_path):
            if args.require_backbone_holdout_provenance:
                backbone_provenance = validate_backbone_holdout_provenance(
                    args.backbone_provenance_manifest,
                    pretrained_model_path,
                    parse_name_list(args.exclude_domains),
                )
                save_json(os.path.join(run_dir, "backbone_holdout_provenance.json"), backbone_provenance)
            checkpoint = torch.load(pretrained_model_path, map_location=device)
            load_diffusion_checkpoint(checkpoint, feature_extractor, diffusion_model)
            print("Loaded pretrained diffusion backbone.")
        else:
            raise FileNotFoundError("Pretrained diffusion checkpoint not found for train_agent.")

        train_agent_offline(
            train_loader,
            val_loader,
            diffusion_model,
            feature_extractor,
            aggregator,
            device,
            run_dir=run_dir,
            num_epochs=args.agent_epochs,
            lr=args.agent_lr,
            max_delta=args.agent_max_delta,
            refine_steps=args.agent_refine_steps,
            residual_loss_weight=args.residual_loss_weight,
            kinematic_regularization_weight=args.kinematic_regularization_weight,
            delta_supervision_weight=args.delta_supervision_weight,
            gate_loss_weight=args.gate_loss_weight,
            use_causal_policy=args.use_causal_policy,
            causal_risk_loss_weight=args.causal_risk_loss_weight,
            causal_counterfactual_loss_weight=args.causal_counterfactual_loss_weight,
            causal_gate_temperature=args.causal_gate_temperature,
            match_loss_weight=args.match_loss_weight,
            no_harm_loss_weight=args.no_harm_loss_weight,
            no_harm_margin=args.no_harm_margin,
            candidate_loss_weight=args.candidate_loss_weight,
            candidate_gate_loss_weight=args.candidate_gate_loss_weight,
            candidate_delta_loss_weight=args.candidate_delta_loss_weight,
            agent_init_path=args.agent_init_path,
            heads_only=args.agent_heads_only,
            hard_pixel_gamma=args.hard_pixel_gamma,
            use_uncertainty=not args.disable_uncertainty_input,
            use_kinematic_features=not args.disable_kinematic_input,
            matching_radius=args.agent_matching_radius,
            agent_input_mode=args.agent_input_mode,
            use_candidate_reweighting=args.use_candidate_reweighting,
            use_forward_backward_consistency=args.use_forward_backward_consistency,
            candidate_topk=args.candidate_topk,
            use_domain_conditioning=args.use_domain_conditioning,
            use_domain_heads=args.use_domain_heads,
            domain_head_anchor_weight=args.domain_head_anchor_weight,
            domain_head_residual_scale=args.domain_head_residual_scale,
            num_domains=args.num_flow_domains,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            agent_checkpoint_name=args.agent_checkpoint,
            agent_resume_path=args.agent_resume_path,
            agent_state_name=args.agent_state_name,
            training_identity={
                "seed": int(args.seed),
                "split_sha256": split_metadata["split_sha256"],
                "checkpoint": str(Path(pretrained_model_path).resolve()),
                "training_domains": sorted(active_domain_names),
            },
        )

    elif args.mode == "inference":
        print("\n>>> Running UPR-Flow inference evaluation...")

        img1_2d, img2_2d, gt_2d, dir_names = read_by_type(
            args.data_path,
            write_index_cache=args.write_data_index_cache,
        )
        domain_to_id = {name: idx for idx, name in enumerate(dir_names)}
        img1_list, img2_list, gt_list, type_list, domain_id_list = [], [], [], [], []
        for i, ds_name in enumerate(dir_names):
            img1_list.extend(img1_2d[i])
            img2_list.extend(img2_2d[i])
            gt_list.extend(gt_2d[i])
            type_list.extend([ds_name] * len(gt_2d[i]))
            domain_id_list.extend([domain_to_id[ds_name]] * len(gt_2d[i]))

        inference_split_metadata = split_metadata
        available_domains = sorted(set(type_list))
        eval_domains = parse_name_list(args.eval_domains)
        unknown_eval_domains = sorted(set(eval_domains) - set(available_domains))
        if unknown_eval_domains:
            raise ValueError(f"Unknown evaluation domains: {unknown_eval_domains}. Available: {available_domains}")
        subdataset_types = sorted(set(eval_domains)) if eval_domains else available_domains
        if not subdataset_types:
            raise ValueError("No evaluation domains were selected.")
        use_domain_adapters = not args.disable_domain_adapters and args.evaluation_protocol != "zero_shot_holdout"
        if args.evaluation_protocol == "zero_shot_holdout":
            print(
                "Zero-shot holdout protocol: target-label calibration and domain-specific adapters are disabled."
            )
        elif not use_domain_adapters:
            print("Inference will use the shared residual agent without domain-specific adapters.")
        print(
            f"Evaluation protocol: {args.evaluation_protocol} | domains: {subdataset_types} | "
            f"domain_adapters_active={use_domain_adapters}"
        )

        val_loaders_dict = {}
        if hasattr(val_ds, "data_index_list"):
            for ds_type in subdataset_types:
                type_global_indices = [i for i, t in enumerate(type_list) if t == ds_type]
                sub_val_indices = sorted(set(val_ds.data_index_list) & set(type_global_indices))
                if sub_val_indices:
                    sub_val_ds = FlowDataset(
                        data_index_list=sub_val_indices,
                        data=val_ds.data,
                        targets_index_list=sub_val_indices if val_ds.targets is not None else None,
                        targets=val_ds.targets,
                        domain_ids=val_ds.domain_ids,
                        sample_uid_by_index=getattr(val_ds, "sample_uid_by_index", None),
                        train=False,
                    )
                    sub_val_ds = maybe_wrap_class2_dataset(sub_val_ds, args.simulate_class2, args.seed)
                    val_loaders_dict[ds_type] = DataLoader(
                        sub_val_ds,
                        batch_size=args.eval_batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                    )

        evaluation_ds = {"train": train_ds, "validation": val_ds, "test": test_ds}[args.evaluation_split]
        print(f"Evaluation split: {args.evaluation_split} ({len(evaluation_ds)} samples)")
        test_loaders_dict = {}
        if hasattr(evaluation_ds, "data_index_list"):
            for ds_type in subdataset_types:
                type_global_indices = [i for i, t in enumerate(type_list) if t == ds_type]
                sub_test_indices = sorted(set(evaluation_ds.data_index_list) & set(type_global_indices))
                if sub_test_indices:
                    sub_test_ds = FlowDataset(
                        data_index_list=sub_test_indices,
                        data=evaluation_ds.data,
                        targets_index_list=sub_test_indices if evaluation_ds.targets is not None else None,
                        targets=evaluation_ds.targets,
                        domain_ids=evaluation_ds.domain_ids,
                        sample_uid_by_index=getattr(evaluation_ds, "sample_uid_by_index", None),
                        train=False,
                    )
                    sub_test_ds = maybe_wrap_class2_dataset(sub_test_ds, args.simulate_class2, args.seed)
                    test_loaders_dict[ds_type] = DataLoader(
                        sub_test_ds,
                        batch_size=args.eval_batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                    )

        feature_extractor = DualSourceFeatureExtractor(device).to(device)
        diffusion_model = ConditionalDiffusionFlowGenerator(
            num_candidates=args.diffusion_candidates,
            flow_scale=args.diffusion_flow_scale,
            num_train_timesteps=args.diffusion_timesteps,
            sampling_steps=args.diffusion_sampling_steps,
            ddim_eta=args.ddim_eta,
            denoised_clip=args.diffusion_denoised_clip,
            center_loss_weight=args.diffusion_center_loss_weight,
            residual_reconstruction_weight=args.diffusion_residual_reconstruction_weight,
            learn_variance=args.diffusion_learn_variance,
            variance_fit_weight=args.diffusion_variance_fit_weight,
            variance_injection_scale=args.diffusion_variance_injection_scale,
        ).to(device)
        aggregator = build_flow_aggregator(args)
        posterior_inference_config = configure_posterior_inference(diffusion_model, aggregator, args)
        save_json(os.path.join(run_dir, "posterior_inference_config.json"), posterior_inference_config)

        model_path = resolve_existing_path(
            args.checkpoint,
            args.output_dir,
            REPO_ROOT / "checkpoints",
        )
        if os.path.exists(model_path):
            if args.require_backbone_holdout_provenance:
                backbone_provenance = validate_backbone_holdout_provenance(
                    args.backbone_provenance_manifest,
                    model_path,
                    parse_name_list(args.eval_domains),
                )
                save_json(os.path.join(run_dir, "backbone_holdout_provenance.json"), backbone_provenance)
            checkpoint = torch.load(model_path, map_location=device)
            load_diffusion_checkpoint(checkpoint, feature_extractor, diffusion_model)
        else:
            raise FileNotFoundError(f"Diffusion checkpoint not found: {model_path}")

        agent = MatchKinematicResidualAgent(
            max_delta=args.agent_max_delta,
            use_uncertainty=not args.disable_uncertainty_input,
            use_kinematic_features=not args.disable_kinematic_input,
            matching_radius=args.agent_matching_radius,
            use_candidate_reweighting=args.use_candidate_reweighting,
            candidate_topk=args.candidate_topk,
            use_domain_conditioning=args.use_domain_conditioning,
            use_domain_heads=args.use_domain_heads,
            domain_head_residual_scale=args.domain_head_residual_scale,
            num_domains=args.num_flow_domains,
            use_causal_policy=args.use_causal_policy,
            causal_gate_temperature=args.causal_gate_temperature,
            agent_input_mode=args.agent_input_mode,
            geometry=aggregator.geometry,
        ).to(device)
        agent_path = resolve_agent_path(args)
        print(f"Residual Agent path: {os.path.abspath(agent_path)}")
        if not os.path.exists(agent_path):
            raise FileNotFoundError(f"Agent checkpoint not found: {agent_path}")
        load_report = load_agent_weights(agent, agent_path, device)
        if load_report["adapted"]:
            print(f"Adapted agent weights: {load_report['adapted']}")
        if load_report["skipped"]:
            print(f"Skipped incompatible agent weights: {load_report['skipped'][:3]}")
        print_agent_training_summary(agent_path)
        if args.evaluation_protocol == "zero_shot_holdout":
            zero_shot_provenance = validate_zero_shot_agent_provenance(
                agent_path,
                subdataset_types,
                inference_split_metadata=inference_split_metadata,
            )
            save_json(os.path.join(run_dir, "zero_shot_provenance.json"), zero_shot_provenance)
        elif args.require_agent_split_provenance:
            in_domain_agent_provenance = validate_in_domain_agent_provenance(agent_path, inference_split_metadata)
            save_json(os.path.join(run_dir, "in_domain_agent_provenance.json"), in_domain_agent_provenance)

        calibration_rows = []
        linear_calibrations = {}
        selective_calibrations = {}
        uncertainty_calibrations = {}
        selected_modes = {}
        if args.calibrate_uncertainty and args.evaluation_protocol != "zero_shot_holdout":
            for ds_type, val_loader in val_loaders_dict.items():
                uncertainty_calibration = calibrate_candidate_uncertainty(
                    val_loader,
                    diffusion_model,
                    feature_extractor,
                    aggregator,
                    device,
                    coverage=args.uncertainty_coverage,
                    use_candidate_reweighting=args.use_candidate_reweighting,
                    use_forward_backward_consistency=args.use_forward_backward_consistency,
                    max_batches=args.max_calibration_batches,
                )
                uncertainty_calibrations[ds_type] = uncertainty_calibration
                calibration_rows.append({"dataset": ds_type, "calibrator": "candidate_dispersion_conformal", **uncertainty_calibration})
        if args.selection_mode in ("linear_calibrated", "quality_selective"):
            for ds_type, val_loader in val_loaders_dict.items():
                linear_calibration, linear_summary = calibrate_linear_agent(
                    val_loader,
                    f"{ds_type}-validation",
                    diffusion_model,
                    feature_extractor,
                    aggregator,
                    device,
                    agent,
                    ridge=args.linear_calibration_ridge,
                    refine_steps=args.agent_refine_steps,
                    use_candidate_reweighting=args.use_candidate_reweighting,
                    use_forward_backward_consistency=args.use_forward_backward_consistency,
                    use_domain_adapters=use_domain_adapters,
                    max_batches=args.max_calibration_batches,
                )
                linear_calibrations[ds_type] = linear_calibration
                calibration_rows.append({"dataset": ds_type, "calibrator": "linear", **linear_summary})
                if args.selection_mode == "quality_selective":
                    selective_calibration, selective_summary = calibrate_selective_agent(
                        val_loader,
                        f"{ds_type}-validation",
                        diffusion_model,
                        feature_extractor,
                        aggregator,
                        device,
                        agent,
                        linear_calibration=linear_calibration,
                        ridge=args.linear_calibration_ridge,
                        refine_steps=args.agent_refine_steps,
                        use_candidate_reweighting=args.use_candidate_reweighting,
                        use_forward_backward_consistency=args.use_forward_backward_consistency,
                        use_domain_adapters=use_domain_adapters,
                        max_batches=args.max_calibration_batches,
                    )
                    selective_calibrations[ds_type] = selective_calibration
                    calibration_rows.append({"dataset": ds_type, "calibrator": "selective", **selective_summary})
                    selected_modes[ds_type] = "quality_selective"
                else:
                    selected_modes[ds_type] = "linear_calibrated"
            save_json(os.path.join(run_dir, "calibration_metrics.json"), calibration_rows)
        else:
            selected_modes = {ds_type: args.selection_mode for ds_type in test_loaders_dict}

        results_rows = []
        sample_metrics_rows = []
        for ds_type, loader in test_loaders_dict.items():
            selected_mode = selected_modes.get(ds_type, args.selection_mode)
            linear_calibration = linear_calibrations.get(ds_type)
            selective_calibration = selective_calibrations.get(ds_type)
            uncertainty_calibration = uncertainty_calibrations.get(ds_type)
            print(f"\n{'=' * 60}")
            print(f"Evaluating [{ds_type}]")
            evaluation_sampling_seed = int(args.evaluation_sampling_seed)

            base_metrics, base_sample_rows = evaluate_single_loader(
                loader,
                f"{ds_type}-base",
                diffusion_model,
                feature_extractor,
                aggregator,
                device,
                agent=None,
                selection_mode="base",
                use_candidate_reweighting=args.use_candidate_reweighting,
                use_forward_backward_consistency=args.use_forward_backward_consistency,
                uncertainty_calibration=uncertainty_calibration,
                sample_uids=sample_uids_for_dataset(loader.dataset),
                max_batches=args.max_eval_batches,
                collect_sample_metrics=True,
                candidate_sampling_seed=evaluation_sampling_seed,
            )
            results_rows.append({"dataset": ds_type, "method": "base_diffusion", **base_metrics})
            for row in base_sample_rows:
                sample_metrics_rows.append(
                    {
                        "dataset": ds_type,
                        "method": "base_diffusion",
                        "run_kind": args.run_kind,
                        "selection_mode": "base",
                        "evaluation_protocol": args.evaluation_protocol,
                        "domain_adapters_active": False,
                        **row,
                    }
                )

            agent_metrics, agent_sample_rows = evaluate_single_loader(
                loader,
                f"{ds_type}-UPR-Flow",
                diffusion_model,
                feature_extractor,
                aggregator,
                device,
                agent=agent,
                selection_mode=selected_mode,
                agent_scale=args.agent_scale,
                refine_steps=args.agent_refine_steps,
                linear_calibration=linear_calibration,
                selective_calibration=selective_calibration,
                use_candidate_reweighting=args.use_candidate_reweighting,
                use_forward_backward_consistency=args.use_forward_backward_consistency,
                uncertainty_calibration=uncertainty_calibration,
                use_domain_adapters=use_domain_adapters,
                sample_uids=sample_uids_for_dataset(loader.dataset),
                max_batches=args.max_eval_batches,
                collect_sample_metrics=True,
                candidate_sampling_seed=evaluation_sampling_seed,
            )
            method_name = "residual_agent" if selected_mode == "agent" else f"residual_agent_{selected_mode}"
            results_rows.append({"dataset": ds_type, "method": method_name, **agent_metrics})
            for row in agent_sample_rows:
                sample_metrics_rows.append(
                    {
                        "dataset": ds_type,
                        "method": method_name,
                        "run_kind": args.run_kind,
                        "selection_mode": selected_mode,
                        "evaluation_protocol": args.evaluation_protocol,
                        "domain_adapters_active": bool(use_domain_adapters),
                        **row,
                    }
                )

        print("\n\n" + "*" * 30)
        print("UPR-Flow Ablation Summary")
        print("*" * 30)
        for row in results_rows:
            print(
                f"- {row['dataset']:<15} | {row['method']:<32} | "
                f"final EPE: {row['final_epe']:.4f} | "
                f"improvement: {row['epe_improvement']:.4f} ({row['epe_improvement_pct']:.2f}%)"
            )

        save_json(os.path.join(run_dir, "inference_metrics.json"), results_rows)
        write_metrics_csv(os.path.join(run_dir, "inference_metrics.csv"), results_rows)
        if sample_metrics_rows:
            save_json(os.path.join(run_dir, "sample_metrics.json"), sample_metrics_rows)
            write_metrics_csv(os.path.join(run_dir, "sample_metrics.csv"), sample_metrics_rows)
        run_summary = build_inference_run_summary(
            args,
            run_dir,
            results_rows,
            calibration_rows=calibration_rows,
            sample_metrics_exported=bool(sample_metrics_rows),
        )
        save_json(os.path.join(run_dir, "run_summary.json"), run_summary)
        eligibility_state = "paper-eligible" if run_summary["paper_eligible"] else "not paper-eligible"
        smoke_reason_text = ", ".join(run_summary["smoke_reasons"]) if run_summary["smoke_reasons"] else "none"
        print(f"Run summary: {eligibility_state} | smoke reasons: {smoke_reason_text}")

        if args.export_paper_artifacts:
            export_root = Path(run_dir) / "paper_artifacts"
            export_root.mkdir(parents=True, exist_ok=True)
            export_index = []
            for ds_type, loader in test_loaders_dict.items():
                selected_mode = selected_modes.get(ds_type, args.selection_mode)
                linear_calibration = linear_calibrations.get(ds_type)
                sample_counter = 0
                with torch.no_grad():
                    for step, batch in enumerate(loader):
                        if args.max_eval_batches and step >= args.max_eval_batches:
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
                            use_candidate_reweighting=args.use_candidate_reweighting,
                            use_forward_backward_consistency=args.use_forward_backward_consistency,
                        )
                        agent_flow, delta_flow, agent_aux = run_iterative_agent(
                            agent,
                            I1,
                            I2,
                            F_bar,
                            sigma_F,
                            steps=args.agent_refine_steps,
                            domain_ids=domain_ids if use_domain_adapters else None,
                            candidate_stats=candidate_stats,
                        )
                        final_flow = F_bar + float(args.agent_scale) * (agent_flow - F_bar)
                        if selected_mode == "linear_calibrated" and linear_calibration is not None:
                            final_flow = apply_linear_calibration(F_bar, final_flow, linear_calibration)

                        for item_idx in range(batch_size):
                            if sample_counter >= args.export_paper_samples:
                                break
                            local_sample_batch = sample_batch[item_idx:item_idx + 1]
                            local_gt = gt_flow[item_idx:item_idx + 1]
                            local_base = F_bar[item_idx:item_idx + 1]
                            local_agent = agent_flow[item_idx:item_idx + 1]
                            local_final = final_flow[item_idx:item_idx + 1]
                            local_delta = delta_flow[item_idx:item_idx + 1]
                            local_candidate_stats = slice_batch_dict(candidate_stats, item_idx)
                            local_agent_aux = slice_batch_dict(agent_aux, item_idx)
                            sample_summary = export_sample_artifact(
                                export_root,
                                ds_type,
                                sample_counter,
                                local_sample_batch,
                                local_gt,
                                local_base,
                                local_agent,
                                local_final,
                                local_delta,
                                local_candidate_stats,
                                local_agent_aux,
                            )
                            export_index.append(sample_summary)
                            sample_counter += 1
                        if sample_counter >= args.export_paper_samples:
                            break
            save_json(export_root / "export_index.json", export_index)
            print(f"Saved paper artifacts to: {export_root}")
