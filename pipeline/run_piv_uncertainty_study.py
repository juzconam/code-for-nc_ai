"""Constraint/tension study: does a physical constraint blind candidate dispersion?

This script is the experimental replacement for the published "conformal" table.
For every image pair it

1. draws ``K`` DDIM candidates with **per-sample deterministic initial noise** so
   that the dispersion ``s(x)`` is a deterministic function of the input rather
   than of the batch composition (a split-conformal requirement the original
   ``sampling_seed + batch_index`` scheme did not satisfy);
2. refines the *same* candidates with the physics controller at several Hodge
   strengths, from ``0`` (no divergence-free direction) to ``4x`` the trained
   strength, plus the observation-only and Hodge-only ablations;
3. propagates every candidate set to the six scientific quantities and stores the
   per-sample quantities that the offline analysis needs.

Nothing is aggregated or calibrated here: the output is a per-sample record so
that the split-conformal fit and the coverage evaluation stay auditable and can
be redone at any ``alpha`` or ``K`` without re-running the network.

Shard files are written per batch and merged at the end, so an interrupted run
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import get_device, set_seed
from models import ConditionalDiffusionFlowGenerator, DualSourceFeatureExtractor, load_diffusion_checkpoint
from physics_utils import FlowAggregator, compute_kinematic_diagnostics
from src.agents.physics_unrolled import PhysicsUnrolledPIVAgent, load_physics_agent_checkpoint
from src.data_processing.custom_dataset import FlowDataset
from src.data_processing.read_data import read_by_type
from src.evaluation.conformal import uncertainty_normalized_metrics
from src.evaluation.physics_uncertainty import monte_carlo_quantities, photometric_pixel_scale
from src.evaluation.scientific_quantities import flow_scientific_quantities

QUANTITIES = (
    "mean_speed",
    "kinetic_energy_proxy",
    "circulation",
    "enstrophy",
    "continuity_rms",
    "mean_strain_rate",
)
PAPER_QUANTITIES = ("mean_speed", "enstrophy", "mean_strain_rate", "continuity_rms")

ARM_SPECS = (
    {"name": "pre", "agent": False, "intervention": "full", "hodge_scale": 1.0},
    {"name": "hodge0.00", "agent": True, "intervention": "full", "hodge_scale": 0.0},
    {"name": "hodge0.25", "agent": True, "intervention": "full", "hodge_scale": 0.25},
    {"name": "hodge0.50", "agent": True, "intervention": "full", "hodge_scale": 0.50},
    {"name": "hodge1.00", "agent": True, "intervention": "full", "hodge_scale": 1.00},
    {"name": "hodge2.00", "agent": True, "intervention": "full", "hodge_scale": 2.00},
    {"name": "hodge4.00", "agent": True, "intervention": "full", "hodge_scale": 4.00},
    {"name": "hodge_only", "agent": True, "intervention": "hodge_only", "hodge_scale": 1.00},
    {"name": "observation_only", "agent": True, "intervention": "observation_only", "hodge_scale": 1.00},
)
PRESETS = {
    "full": [spec["name"] for spec in ARM_SPECS],
    "dose": [spec["name"] for spec in ARM_SPECS[:7]],
    "calibration": ["pre", "hodge1.00"],
    "minimal": ["pre", "hodge1.00", "hodge_only", "observation_only"],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=r"D:\newpiv\sample_data")
    parser.add_argument("--split-manifest", default=str(ROOT / "splits" / "class1_primary_split_v2.json"))
    parser.add_argument(
        "--backbone-checkpoint",
        default=str(ROOT / "piv_runs" / "residual_ddpm_v6_x0_seed3407" / "diffusion_backbone_best.pth"),
    )
    parser.add_argument(
        "--agent-checkpoint",
        default=str(ROOT / "piv_runs" / "physics_unrolled_v3_warmup_scratch_20ep_seed3417" / "physics_agent_best.pth"),
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "purged"])
    parser.add_argument("--arms", default="full", help="preset name or comma-separated arm names")
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--k-grid", default="", help="comma-separated K prefixes; default 2,5,<candidates>")
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--particle-sigma", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Take every n-th sample.  Used for the K sweep, where a full pass at large K is too slow; "
             "a strided subset keeps the domain mixture of the split instead of taking a prefix.",
    )
    parser.add_argument("--mc-draws", type=int, default=32)
    parser.add_argument("--output-dir", default=str(ROOT / "piv_runs" / "uncertainty_study"))
    parser.add_argument("--force", action="store_true", help="Recompute shards that already exist")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Deterministic candidate noise
# ---------------------------------------------------------------------------
def stable_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def candidate_noise(uid: str, count: int, shape, direction: str, base_seed: int) -> torch.Tensor:
    """Per-sample, per-candidate DDIM initial noise.

    ``shape`` is ``(2, H, W)``.  Returns ``[count, 2, H, W]`` on CPU.  The draw
    depends only on ``(base_seed, uid, direction, candidate_index)``, so candidate
    ``k`` of sample ``i`` is identical regardless of batch size, batch order or how
    many candidates are requested in total.  That property is what makes the
    derived dispersion a fixed function of the image pair.
    """
    generator = torch.Generator(device="cpu").manual_seed(stable_seed(base_seed, uid, direction))
    return torch.stack([torch.randn(shape, generator=generator) for _ in range(int(count))], dim=0)


class IndexedSplit(Dataset):
    """Wrap ``FlowDataset`` so the sample uid survives collation."""

    def __init__(self, dataset: FlowDataset, positions=None):
        self.dataset = dataset
        self.positions = list(range(len(dataset))) if positions is None else list(positions)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        sample, label, domain = self.dataset[self.positions[index]]
        return sample, label, domain, self.dataset.data_index_list[self.positions[index]]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_datasets(data_path: str, split_manifest: str):
    manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    img1_by_domain, img2_by_domain, gt_by_domain, domain_names = read_by_type(data_path)
    img1, img2, gt, domain_ids = [], [], [], []
    for domain_id, _ in enumerate(domain_names):
        img1.extend(img1_by_domain[domain_id])
        img2.extend(img2_by_domain[domain_id])
        gt.extend(gt_by_domain[domain_id])
        domain_ids.extend([domain_id] * len(gt_by_domain[domain_id]))
    uid_by_index = {int(row["index"]): row["sample_uid"] for row in manifest["samples"]}
    domain_by_index = {int(row["index"]): row["domain"] for row in manifest["samples"]}
    split_indices = {name: [] for name in ("train", "validation", "test", "purged")}
    for row in manifest["samples"]:
        split_indices[str(row["split"])].append(int(row["index"]))
    datasets = {}
    for name, indices in split_indices.items():
        indices = sorted(indices)
        datasets[name] = FlowDataset(
            indices,
            [img1, img2],
            targets_index_list=indices,
            targets=gt,
            domain_ids=domain_ids,
            sample_uid_by_index=uid_by_index,
        )
    return datasets, manifest, domain_names, uid_by_index, domain_by_index


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def load_models(args, device):
    checkpoint = torch.load(args.backbone_checkpoint, map_location=device, weights_only=False)
    config = checkpoint["diffusion_config"]
    feature = DualSourceFeatureExtractor(device).to(device).eval()
    diffusion = ConditionalDiffusionFlowGenerator(
        num_candidates=config["num_candidates"],
        flow_scale=config["flow_scale"],
        num_train_timesteps=config["num_train_timesteps"],
        sampling_steps=config["sampling_steps"],
        ddim_eta=config["ddim_eta"],
        denoised_clip=config["denoised_clip"],
        center_loss_weight=config["center_loss_weight"],
        residual_reconstruction_weight=config["residual_reconstruction_weight"],
        learn_variance=config.get("learn_variance", False),
        variance_fit_weight=config.get("variance_fit_weight", 1.0),
        prior_channels=config.get("prior_channels", 0),
    ).to(device).eval()
    load_diffusion_checkpoint(checkpoint, feature, diffusion)
    for module in (feature, diffusion):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    agent_checkpoint = torch.load(args.agent_checkpoint, map_location=device, weights_only=False)
    agent_config = dict(agent_checkpoint.get("agent_config", {}))
    if agent_checkpoint.get("checkpoint_schema_version") == 1:
        agent_config.setdefault("observation_step_mode", "gradient")
        agent_config.setdefault("hodge_step_mode", "rms")
        agent_config.setdefault("symmetric_psf", True)
    supported = set(PhysicsUnrolledPIVAgent.__init__.__code__.co_varnames)
    agent_kwargs = {key: value for key, value in agent_config.items() if key in supported}
    agent_kwargs.setdefault("particle_sigma", args.particle_sigma)
    agent = PhysicsUnrolledPIVAgent(**agent_kwargs).to(device).eval()
    if agent_checkpoint.get("checkpoint_schema_version") == 1:
        agent.load_state_dict(agent_checkpoint["agent_state_dict"], strict=True)
    else:
        load_physics_agent_checkpoint(agent_checkpoint, agent)
    return feature, diffusion, agent, FlowAggregator(), agent_checkpoint


# ---------------------------------------------------------------------------
# Per-batch computation
# ---------------------------------------------------------------------------
def _aggregate(candidates, image1, image2, flow_bw_mean, aggregator):
    return aggregator.filter_and_aggregate(
        candidates, image1, image2, flow_bw_mean=flow_bw_mean, return_stats=True
    )


def _quantities(flow, geometry):
    values = flow_scientific_quantities(flow, geometry)
    return {name: values[name] for name in QUANTITIES}


def _refine(agent, image1, image2, candidates, sigma, intervention):
    batch, count = candidates.shape[:2]
    flat = candidates.reshape(batch * count, *candidates.shape[2:])
    flat_image1 = image1.repeat_interleave(count, dim=0)
    flat_image2 = image2.repeat_interleave(count, dim=0)
    flat_sigma = sigma.repeat_interleave(count, dim=0)
    with torch.no_grad():
        refined, aux = agent.refine(
            flat_image1, flat_image2, flat, flat_sigma, hard_acceptance=True, intervention=intervention
        )
    return refined.reshape(batch, count, *refined.shape[1:]), aux


def _mc_summary(field, sigma_components, geometry, draws, seed):
    quantities = monte_carlo_quantities(field, sigma_components, lambda flow: _quantities(flow, geometry), draws=draws, seed=seed)
    summary = {}
    for name in QUANTITIES:
        values = quantities[name]
        summary[name] = {
            "std": values.std(dim=1, unbiased=False),
            "q05": torch.quantile(values, 0.05, dim=1),
            "q95": torch.quantile(values, 0.95, dim=1),
            "mean": values.mean(dim=1),
        }
    return summary


def run(args):
    device = get_device("auto")
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    shard_dir = output_dir / "shards" / args.split
    shard_dir.mkdir(parents=True, exist_ok=True)

    arm_names = PRESETS[args.arms] if args.arms in PRESETS else [item.strip() for item in args.arms.split(",")]
    arms = [spec for spec in ARM_SPECS if spec["name"] in arm_names]
    if len(arms) != len(arm_names):
        missing = sorted(set(arm_names) - {spec["name"] for spec in arms})
        raise ValueError(f"unknown arms: {missing}")

    k_grid = sorted({int(item) for item in args.k_grid.split(",") if item.strip()}) if args.k_grid else None
    if k_grid is None:
        k_grid = sorted({min(5, args.candidates), args.candidates}) if args.candidates >= 2 else [1]
    if k_grid[-1] > args.candidates:
        raise ValueError("k-grid cannot exceed the number of candidates")
    k_grid = sorted(set(int(k) for k in k_grid) | {int(args.candidates), 1})

    datasets, manifest, domain_names, uid_by_index, domain_by_index = build_datasets(
        args.data_path, args.split_manifest
    )
    dataset = datasets[args.split]
    positions = list(range(0, len(dataset), max(1, int(args.stride))))
    wrapper = IndexedSplit(dataset, positions)
    loader = DataLoader(wrapper, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    feature, diffusion, agent, aggregator, agent_checkpoint = load_models(args, device)
    geometry = aggregator.geometry

    run_meta = {
        "program": "run_piv_uncertainty_study",
        "split": args.split,
        "split_sha256": manifest["split_sha256"],
        "data_path": args.data_path,
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "backbone_checkpoint": str(Path(args.backbone_checkpoint).resolve()),
        "agent_checkpoint": str(Path(args.agent_checkpoint).resolve()),
        "agent_config": agent_checkpoint.get("agent_config"),
        "candidates": int(args.candidates),
        "k_grid": k_grid,
        "sampling_steps": int(args.sampling_steps),
        "arms": arms,
        "quantities": list(QUANTITIES),
        "deterministic_initial_noise": True,
        "seed": int(args.seed),
        "mc_draws": int(args.mc_draws) if args.mc_draws > 0 else 0,
        "sample_count": len(wrapper),
        "stride": int(max(1, int(args.stride))),
        "domain_counts": {
            domain: sum(1 for position in positions if domain_by_index.get(dataset.data_index_list[position]) == domain)
            for domain in domain_names
        },
    }
    (output_dir / f"run_meta_{args.split}.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )

    batch_timings = []
    written = []
    total_samples = 0
    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        shard_path = shard_dir / f"batch_{batch_index:05d}.npz"
        if shard_path.exists() and not args.force:
            continue
        samples, target, domain_id, indices = batch
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        image1, image2 = samples[:, 0:1], samples[:, 1:2]
        batch_size = int(target.shape[0])
        uids = [uid_by_index[int(index)] for index in indices]
        domains = np.array([domain_by_index.get(int(index), "unknown") for index in indices])
        if args.max_samples and total_samples + batch_size > args.max_samples:
            keep = max(0, args.max_samples - total_samples)
            if keep == 0:
                break
            samples, target, image1, image2 = samples[:keep], target[:keep], image1[:keep], image2[:keep]
            uids, domains, indices = uids[:keep], domains[:keep], indices[:keep]
            batch_size = keep

        started = time.time()
        noise_shape = (2, image1.shape[-2], image1.shape[-1])
        fw_noise = torch.stack([
            candidate_noise(uid, args.candidates, noise_shape, "fw", args.seed) for uid in uids
        ], dim=1).to(device)  # [K, B, 2, H, W]
        bw_noise = torch.stack([
            candidate_noise(uid, args.candidates, noise_shape, "bw", args.seed) for uid in uids
        ], dim=1).to(device)
        with torch.no_grad():
            condition_fw = feature(image1, image2)
            candidates = diffusion(
                condition_fw, image1, image2, use_ema=True,
                num_candidates=args.candidates, sampling_steps=args.sampling_steps,
                initial_noise=fw_noise,
            )
            condition_bw = feature(image2, image1)
            candidates_bw = diffusion(
                condition_bw, image2, image1, use_ema=True,
                num_candidates=args.candidates, sampling_steps=args.sampling_steps,
                initial_noise=bw_noise,
            )
            flow_bw_mean = candidates_bw.mean(dim=1)
        sampling_seconds = time.time() - started

        truth = _quantities(target, geometry)
        records = {
            "uid": np.array(uids),
            "domain": domains,
            "index": np.asarray([int(index) for index in indices]),
            "q_truth": np.stack([truth[name].detach().cpu().numpy() for name in QUANTITIES], axis=1),
        }
        _, pre_sigma, _ = _aggregate(candidates, image1, image2, flow_bw_mean, aggregator)

        per_arm_seconds = {}
        aggregate_cache = {}
        for spec in arms:
            arm_started = time.time()
            name = spec["name"]
            if not spec["agent"]:
                refined = candidates
                aux = None
            else:
                agent.set_hodge_scale(float(spec["hodge_scale"]))
                refined, aux = _refine(agent, image1, image2, candidates, pre_sigma, spec["intervention"])
            q_candidate = _quantities(
                refined.reshape(-1, *refined.shape[2:]), geometry
            )
            q_candidate = {key: value.reshape(batch_size, args.candidates) for key, value in q_candidate.items()}
            q_point = np.zeros((batch_size, len(k_grid), len(QUANTITIES)), dtype=np.float32)
            sigma = np.zeros_like(q_point)
            for k_index, k_value in enumerate(k_grid):
                aggregate, sigma_fields, _ = _aggregate(
                    refined[:, :k_value], image1, image2, flow_bw_mean, aggregator
                )
                if k_value == args.candidates:
                    aggregate_cache[name] = (aggregate, sigma_fields)
                point = _quantities(aggregate, geometry)
                for q_index, quantity in enumerate(QUANTITIES):
                    values = q_candidate[quantity][:, :k_value]
                    q_point[:, k_index, q_index] = point[quantity].detach().cpu().numpy()
                    sigma[:, k_index, q_index] = values.std(dim=1, unbiased=False).detach().cpu().numpy()
            full = {key: value.detach() for key, value in q_candidate.items()}
            records[f"q_cand__{name}"] = np.stack(
                [full[quantity].cpu().numpy() for quantity in QUANTITIES], axis=2
            )
            records[f"q_point__{name}"] = q_point
            records[f"sigma__{name}"] = sigma
            records[f"q05__{name}"] = np.stack(
                [torch.quantile(full[quantity], 0.05, dim=1).cpu().numpy() for quantity in QUANTITIES], axis=1
            )
            records[f"q95__{name}"] = np.stack(
                [torch.quantile(full[quantity], 0.95, dim=1).cpu().numpy() for quantity in QUANTITIES], axis=1
            )
            aggregate, sigma_fields = aggregate_cache[name]
            error = aggregate - target
            pixel_metrics = uncertainty_normalized_metrics(
                error.permute(0, 2, 3, 1).detach().cpu().numpy(),
                sigma_fields.permute(0, 2, 3, 1).detach().cpu().numpy(),
            )
            error_magnitude = error.norm(dim=1).flatten(1)
            sigma_magnitude = sigma_fields.norm(dim=1).flatten(1)
            records[f"epe__{name}"] = error_magnitude.mean(dim=1).detach().cpu().numpy()
            if spec["name"] == "pre":
                records[f"delta__{name}"] = np.zeros(batch_size, dtype=np.float32)
                records["max_vort_pre"] = (
                    compute_kinematic_diagnostics(aggregate, geometry)["vorticity"]
                    .abs().flatten(1).max(dim=1).values.detach().cpu().numpy()
                )
            else:
                records[f"delta__{name}"] = (
                    (refined.mean(dim=1) - candidates.mean(dim=1))
                    .norm(dim=1).flatten(1).square().mean(dim=1).sqrt().detach().cpu().numpy()
                )
            records[f"accept__{name}"] = (
                np.ones(batch_size, dtype=np.float32) if aux is None else
                aux["acceptance_rate"].detach().cpu().numpy()
            )
            records[f"hodge_dir__{name}"] = (
                np.zeros(batch_size, dtype=np.float32) if aux is None else
                aux["hodge_direction_rms"].detach().cpu().numpy()
            )
            records[f"div__{name}"] = (
                compute_kinematic_diagnostics(aggregate, geometry)["divergence"]
                .square().flatten(1).mean(dim=1).sqrt().detach().cpu().numpy()
            )
            records[f"sigpix__{name}"] = sigma_magnitude.mean(dim=1).detach().cpu().numpy()
            records[f"unb__{name}"] = np.full(batch_size, pixel_metrics["unb"], dtype=np.float32)
            records[f"une__{name}"] = np.full(batch_size, pixel_metrics["une"], dtype=np.float32)
            records[f"unerms__{name}"] = np.full(batch_size, pixel_metrics["une_rms"], dtype=np.float32)

            if args.mc_draws > 0 and name in {"pre", "hodge1.00"}:
                photometric = photometric_pixel_scale(image1, image2, aggregate)
                sigma_photo = photometric.repeat(1, 2, 1, 1)
                for tag, components in (
                    ("photometric", sigma_photo),
                    ("candidate", sigma_fields),
                ):
                    seed = stable_seed(args.seed, "mc", args.split, name, tag) % (2 ** 31 - 1)
                    summary = _mc_summary(aggregate, components, geometry, args.mc_draws, seed)
                    records[f"mc_std__{name}__{tag}"] = np.stack(
                        [summary[key]["std"].detach().cpu().numpy() for key in QUANTITIES], axis=1
                    )
                    records[f"mc_q05__{name}__{tag}"] = np.stack(
                        [summary[key]["q05"].detach().cpu().numpy() for key in QUANTITIES], axis=1
                    )
                    records[f"mc_q95__{name}__{tag}"] = np.stack(
                        [summary[key]["q95"].detach().cpu().numpy() for key in QUANTITIES], axis=1
                    )
                    records[f"mc_mean__{name}__{tag}"] = np.stack(
                        [summary[key]["mean"].detach().cpu().numpy() for key in QUANTITIES], axis=1
                    )
            per_arm_seconds[name] = time.time() - arm_started

        elapsed = time.time() - started
        np.savez(shard_path, **records)
        (shard_dir / f"batch_{batch_index:05d}.timing.json").write_text(
            json.dumps({
                "batch": batch_index,
                "samples": batch_size,
                "sampling_seconds": sampling_seconds,
                "total_seconds": elapsed,
                "arms_seconds": per_arm_seconds,
            }),
            encoding="utf-8",
        )
        written.append(shard_path)
        total_samples += batch_size
        batch_timings.append({
            "batch": batch_index,
            "samples": batch_size,
            "sampling_seconds": sampling_seconds,
            "total_seconds": elapsed,
            "arms_seconds": per_arm_seconds,
        })
        if (batch_index + 1) % 5 == 0 or batch_index == 0:
            print(
                f"[{args.split}] batch {batch_index}: n={batch_size} "
                f"sample={sampling_seconds:.1f}s total={elapsed:.1f}s "
                f"({1000 * elapsed / max(batch_size, 1):.0f} ms/sample) "
                f"done={total_samples}/{len(wrapper)}",
                flush=True,
            )

    _merge_shards(shard_dir, output_dir, args, run_meta, batch_timings, k_grid, arm_names)
    print(f"finished {args.split}: {total_samples} samples", flush=True)


def _merge_shards(shard_dir, output_dir, args, run_meta, batch_timings, k_grid, arm_names):
    shards = sorted(shard_dir.glob("batch_*.npz"))
    if not shards:
        print("no shards to merge", flush=True)
        return
    timing_files = sorted(shard_dir.glob("batch_*.timing.json"))
    if timing_files:
        batch_timings = [json.loads(path.read_text(encoding="utf-8")) for path in timing_files]
    merged = {}
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            for key in data.files:
                merged.setdefault(key, []).append(data[key])
    payload = {key: np.concatenate(value, axis=0) for key, value in merged.items()}
    target_path = output_dir / f"study_{args.split}.npz"
    np.savez(target_path, **payload)
    timing = {
        "split": args.split,
        "samples": int(payload["uid"].shape[0]),
        "mean_sampling_seconds_per_batch": float(np.mean([row["sampling_seconds"] for row in batch_timings])) if batch_timings else None,
        "mean_total_seconds_per_batch": float(np.mean([row["total_seconds"] for row in batch_timings])) if batch_timings else None,
        "mean_ms_per_sample": float(np.mean([1000.0 * row["total_seconds"] / max(row["samples"], 1) for row in batch_timings])) if batch_timings else None,
        "arms_ms_per_sample": {
            name: float(np.mean([1000.0 * row["arms_seconds"].get(name, 0.0) / max(row["samples"], 1) for row in batch_timings]))
            for name in arm_names
        },
        "batches": batch_timings,
    }
    if batch_timings:
        (output_dir / f"timing_{args.split}.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    (output_dir / f"study_{args.split}_meta.json").write_text(
        json.dumps({**run_meta, "merged_shards": len(shards), "arrays": {key: list(value.shape) for key, value in payload.items()}}, indent=2),
        encoding="utf-8",
    )
    print(f"merged {len(shards)} shards -> {target_path}", flush=True)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
