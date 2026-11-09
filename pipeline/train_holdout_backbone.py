"""Train an auditable diffusion backbone with one PIV domain held out."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import get_device, parse_name_list, set_seed, subset_dataset_by_domains
from models import ConditionalDiffusionFlowGenerator, DualSourceFeatureExtractor, make_diffusion_checkpoint
from src.data_processing.read_data import construct_dataset, read_by_type
from src.reporting.backbone_training import write_backbone_provenance
from src.reporting.run_artifacts import build_run_config, make_run_dir, save_json
from src.training.backbone_state import restore_backbone_training_state, save_backbone_training_state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a versioned learned-center residual-DDPM PIV backbone with full-domain or strict held-out provenance."
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--holdout-domain",
        default="",
        help="Optional held-out domain. Omit for the all-domain backbone used by in-domain experiments.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--diffusion-timesteps", type=int, default=100)
    parser.add_argument("--diffusion-sampling-steps", type=int, default=10)
    parser.add_argument("--diffusion-candidates", type=int, default=4)
    parser.add_argument("--diffusion-flow-scale", type=float, default=20.0)
    parser.add_argument(
        "--diffusion-center-loss-weight",
        type=float,
        default=5.0,
        help="Weight of the normalized deterministic-center EPE in residual-DDPM training.",
    )
    parser.add_argument("--diffusion-residual-reconstruction-weight", type=float, default=2.0)
    parser.add_argument("--diffusion-denoised-clip", type=float, default=0.5)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--no-diffusion-learn-variance", dest="diffusion_learn_variance", action="store_false")
    parser.add_argument("--diffusion-variance-injection-scale", type=float, default=0.0)
    parser.add_argument("--diffusion-variance-fit-weight", type=float, default=1.0)
    parser.set_defaults(diffusion_learn_variance=True)
    parser.add_argument(
        "--prior-root",
        default="",
        help="Directory of precomputed external prior .npy files (<root>/<domain>/<stem>_prior.npy).",
    )
    parser.add_argument(
        "--prior-channels",
        type=int,
        default=0,
        help="Number of leading prior channels to condition the center head on (0 disables the prior).",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--posterior-diagnostic-batches", type=int, default=8)
    parser.add_argument("--posterior-diagnostic-candidates", type=int, default=4)
    parser.add_argument("--posterior-diagnostic-seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="diffusion_backbone_best.pth")
    parser.add_argument("--backbone-resume-path", default=None)
    parser.add_argument("--backbone-state-name", default="backbone_training_state.pth")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--external-pretraining-declaration",
        default="",
        help="Required declaration for the frozen torchvision RAFT feature weights and any other external pretraining.",
    )
    return parser.parse_args()


def flatten_source_domains(data_path, holdout_domain):
    image1_by_domain, image2_by_domain, flow_by_domain, domains = read_by_type(data_path)
    if holdout_domain and holdout_domain not in domains:
        raise ValueError(f"Unknown holdout domain '{holdout_domain}'. Available: {domains}")
    image1_paths, image2_paths, flow_paths, domain_ids = [], [], [], []
    domain_to_id = {domain: index for index, domain in enumerate(domains)}
    training_domains = [domain for domain in domains if domain != holdout_domain]
    for domain, image1, image2, flow in zip(domains, image1_by_domain, image2_by_domain, flow_by_domain):
        image1_paths.extend(image1)
        image2_paths.extend(image2)
        flow_paths.extend(flow)
        domain_ids.extend([domain_to_id[domain]] * len(flow))
    return image1_paths, image2_paths, flow_paths, domain_ids, training_domains, domain_to_id


def evaluate_backbone(
    feature_extractor,
    diffusion_model,
    loader,
    device,
    max_batches=0,
    posterior_diagnostic_batches=8,
    posterior_diagnostic_candidates=4,
    posterior_diagnostic_seed=3407,
):
    feature_extractor.eval()
    diffusion_model.eval()
    total_loss = 0.0
    total_reconstruction = 0.0
    total_epe = 0.0
    total_center_epe = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            samples, flow, _ = batch
            samples = samples.to(device)
            flow = flow.to(device)
            image1, image2 = samples[:, 0:1], samples[:, 1:2]
            prior = samples[:, 2:] if samples.shape[1] > 2 else None
            condition = feature_extractor(image1, image2)
            loss, reconstruction, epe, center_epe = diffusion_model.compute_loss(
                condition, image1, image2, flow, epoch=0, max_epochs=1, prior=prior
            )
            batch_size = int(samples.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_reconstruction += float(reconstruction.item()) * batch_size
            total_epe += float(epe.item()) * batch_size
            total_center_epe += float(center_epe.item()) * batch_size
            sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Backbone validation loader produced no batches.")
    metrics = {
        "validation_loss": total_loss / sample_count,
        "validation_reconstruction_loss": total_reconstruction / sample_count,
        "validation_epe": total_epe / sample_count,
        "validation_center_epe": total_center_epe / sample_count,
        "validation_sample_count": sample_count,
    }

    sampled_count = 0
    sampled_mean_epe = 0.0
    sampled_aggregate_epe = 0.0
    sampled_oracle_epe = 0.0
    sampled_center_epe = 0.0
    sampled_candidate_beats_center = 0
    sampled_aggregate_beats_center = 0
    cuda_devices = [] if device.type != "cuda" else [device.index or torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        torch.manual_seed(int(posterior_diagnostic_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(posterior_diagnostic_seed))
        for batch_index, batch in enumerate(loader):
            if batch_index >= int(posterior_diagnostic_batches):
                break
            samples, flow, _ = batch
            samples = samples.to(device)
            flow = flow.to(device)
            image1, image2 = samples[:, 0:1], samples[:, 1:2]
            prior = samples[:, 2:] if samples.shape[1] > 2 else None
            condition = feature_extractor(image1, image2)
            center = diffusion_model.predict_center(condition, image1, image2, prior=prior)
            candidates = diffusion_model(
                condition,
                image1,
                image2,
                use_ema=True,
                num_candidates=int(posterior_diagnostic_candidates),
                prior=prior,
            )
            candidate_epe = torch.norm(candidates - flow.unsqueeze(1), p=2, dim=2).flatten(2).mean(dim=2)
            aggregate_epe = torch.norm(candidates.mean(dim=1) - flow, p=2, dim=1).flatten(1).mean(dim=1)
            center_epe = torch.norm(center - flow, p=2, dim=1).flatten(1).mean(dim=1)
            oracle_epe = candidate_epe.min(dim=1).values
            batch_size = int(flow.shape[0])
            sampled_count += batch_size
            sampled_mean_epe += float(candidate_epe.mean(dim=1).sum().item())
            sampled_aggregate_epe += float(aggregate_epe.sum().item())
            sampled_oracle_epe += float(oracle_epe.sum().item())
            sampled_center_epe += float(center_epe.sum().item())
            sampled_candidate_beats_center += int((oracle_epe < center_epe).sum().item())
            sampled_aggregate_beats_center += int((aggregate_epe < center_epe).sum().item())
    if sampled_count == 0:
        raise ValueError("Posterior diagnostic loader produced no batches.")
    metrics.update({
        "validation_sampled_count": sampled_count,
        "validation_sampled_mean_candidate_epe": sampled_mean_epe / sampled_count,
        "validation_sampled_aggregate_epe": sampled_aggregate_epe / sampled_count,
        "validation_sampled_oracle_epe": sampled_oracle_epe / sampled_count,
        "validation_sampled_center_epe": sampled_center_epe / sampled_count,
        "validation_residual_candidate_beats_center_rate": sampled_candidate_beats_center / sampled_count,
        "validation_aggregate_beats_center_rate": sampled_aggregate_beats_center / sampled_count,
        "posterior_diagnostic_seed": int(posterior_diagnostic_seed),
        "posterior_diagnostic_candidates": int(posterior_diagnostic_candidates),
    })
    return metrics


def write_history(path, history):
    if not history:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1 or args.diffusion_denoised_clip <= 0.0:
        raise ValueError("epochs, batch-size, eval-batch-size, and diffusion-denoised-clip must be positive.")
    if not args.external_pretraining_declaration.strip():
        raise ValueError(
            "Backbone training requires --external-pretraining-declaration. "
            "Document the frozen torchvision RAFT weights and confirm they contain no held-out PIV supervision."
        )
    if args.backbone_resume_path and not args.allow_overwrite:
        raise ValueError("Resuming backbone training requires --allow-overwrite.")
    if Path(args.backbone_state_name).name != args.backbone_state_name:
        raise ValueError("--backbone-state-name must be a filename, not a path.")
    set_seed(args.seed)
    device = get_device(args.device)
    run_dir = Path(make_run_dir(args.output_dir, args.run_name, args.allow_overwrite))
    if args.backbone_resume_path:
        resume_path = Path(args.backbone_resume_path).expanduser().resolve()
        if resume_path.parent != run_dir.resolve():
            raise ValueError("--backbone-resume-path must belong to the run directory being resumed.")
    else:
        resume_path = run_dir / Path(args.backbone_state_name).name
    run_config = build_run_config(args, run_dir, device)
    run_config.update(
        {
            "mode": "train_backbone",
            "run_kind": "smoke" if args.max_train_batches or args.max_val_batches else "full",
        }
    )
    save_json(run_dir / "run_config.json", run_config)

    image1, image2, flow, domain_ids, training_domains, domain_to_id = flatten_source_domains(
        args.data_path, args.holdout_domain
    )
    train_ds, val_ds, _, split_metadata = construct_dataset(
        image1,
        image2,
        flow,
        domain_id_list=domain_ids,
        split_protocol="manifest",
        split_manifest_path=args.split_manifest,
        fingerprint_mode="triplet_sha256",
        return_metadata=True,
        prior_root=args.prior_root or None,
        prior_channels=args.prior_channels,
    )
    active_ids = {domain_to_id[domain] for domain in training_domains}
    train_ds = subset_dataset_by_domains(train_ds, active_ids)
    val_ds = subset_dataset_by_domains(val_ds, active_ids)
    if not len(train_ds) or not len(val_ds):
        raise ValueError("Source-domain train or validation subset is empty.")
    save_json(run_dir / "split_manifest.json", split_metadata)
    is_holdout = bool(args.holdout_domain)
    protocol = {
        "protocol": "leave_one_domain_out_backbone_training" if is_holdout else "all_domain_backbone_training",
        "run_kind": "smoke" if args.max_train_batches or args.max_val_batches else "full",
        "training_domains": sorted(training_domains),
        "excluded_domains": [args.holdout_domain] if is_holdout else [],
        "source_split_manifest": str(Path(args.split_manifest).expanduser().resolve()),
        "source_split_sha256": split_metadata.get("source_split_sha256") or split_metadata["split_sha256"],
        "effective_split_sha256": split_metadata["split_sha256"],
        "effective_purge_gap": split_metadata.get("purge_gap"),
        "train_sample_count": len(train_ds),
        "validation_sample_count": len(val_ds),
        "external_pretraining_declaration": args.external_pretraining_declaration.strip(),
    }
    save_json(run_dir / "backbone_training_protocol.json", protocol)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
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
        prior_channels=args.prior_channels,
    ).to(device)
    if args.prior_channels and not args.prior_root:
        raise ValueError("--prior-channels > 0 requires --prior-root.")
    if args.prior_channels:
        print(
            f"[prior] root={args.prior_root} channels={args.prior_channels}"
            f" missing={train_ds.prior_missing}",
            flush=True,
        )
    feature_extractor.raft.eval()
    for parameter in feature_extractor.raft.parameters():
        parameter.requires_grad = False
    optimizer = Adam(
        [parameter for parameter in list(feature_extractor.parameters()) + list(diffusion_model.parameters()) if parameter.requires_grad],
        lr=args.lr,
    )

    checkpoint_path = run_dir / Path(args.checkpoint_name).name
    history = []
    best_validation_score = float("inf")
    training_signature = {
        "seed": int(args.seed),
        "holdout_domain": args.holdout_domain or None,
        "training_domains": sorted(training_domains),
        "split_sha256": split_metadata["split_sha256"],
        "source_split_sha256": protocol["source_split_sha256"],
        "lr": float(args.lr),
        "diffusion_config": diffusion_model.diffusion_config,
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "external_pretraining_declaration": args.external_pretraining_declaration.strip(),
    }
    start_epoch = 0
    if args.backbone_resume_path:
        restored = restore_backbone_training_state(
            resume_path,
            feature_extractor,
            diffusion_model,
            optimizer,
            device,
            expected_signature=training_signature,
        )
        start_epoch = restored["completed_epochs"]
        best_validation_score = restored["best_validation_score"]
        history = restored["history"]
        if start_epoch > args.epochs:
            raise ValueError(f"Resume state has {start_epoch} epochs, exceeding requested epochs={args.epochs}.")
        print(f"Resumed held-out backbone from epoch {start_epoch}/{args.epochs}: {resume_path}")
    for epoch_index in range(start_epoch, args.epochs):
        feature_extractor.train()
        feature_extractor.raft.eval()
        diffusion_model.train()
        running_loss = 0.0
        running_epe = 0.0
        running_center_epe = 0.0
        train_samples = 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            samples, flow, _ = batch
            samples = samples.to(device)
            flow = flow.to(device)
            image1, image2 = samples[:, 0:1], samples[:, 1:2]
            prior = samples[:, 2:] if samples.shape[1] > 2 else None
            condition = feature_extractor(image1, image2)
            loss, _, epe, center_epe = diffusion_model.compute_loss(
                condition, image1, image2, flow, epoch=epoch_index, max_epochs=args.epochs, prior=prior
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(feature_extractor.parameters()) + list(diffusion_model.parameters()), max_norm=1.0
            )
            optimizer.step()
            batch_size = int(samples.shape[0])
            running_loss += float(loss.item()) * batch_size
            running_epe += float(epe.item()) * batch_size
            running_center_epe += float(center_epe.item()) * batch_size
            train_samples += batch_size
        if not train_samples:
            raise ValueError("Backbone training loader produced no batches.")
        diffusion_model.update_ema(epoch_index + 1, args.epochs)
        validation = evaluate_backbone(
            feature_extractor,
            diffusion_model,
            val_loader,
            device,
            max_batches=args.max_val_batches,
            posterior_diagnostic_batches=args.posterior_diagnostic_batches,
            posterior_diagnostic_candidates=args.posterior_diagnostic_candidates,
            posterior_diagnostic_seed=args.posterior_diagnostic_seed,
        )
        row = {
            "epoch": epoch_index + 1,
            "train_loss": running_loss / train_samples,
            "train_epe": running_epe / train_samples,
            "train_center_epe": running_center_epe / train_samples,
            **validation,
        }
        history.append(row)
        print(
            f"Epoch [{epoch_index + 1}/{args.epochs}] loss={row['train_loss']:.6f} "
            f"train_epe={row['train_epe']:.6f} val_loss={row['validation_loss']:.6f} "
            f"val_epe={row['validation_epe']:.6f} "
            f"center_epe={row['train_center_epe']:.6f}/{row['validation_center_epe']:.6f} "
            f"sampled center/aggregate/oracle={row['validation_sampled_center_epe']:.6f}/"
            f"{row['validation_sampled_aggregate_epe']:.6f}/{row['validation_sampled_oracle_epe']:.6f}"
        )
        validation_score = row["validation_sampled_aggregate_epe"]
        if validation_score < best_validation_score:
            best_validation_score = validation_score
            torch.save(
                make_diffusion_checkpoint(
                    feature_extractor,
                    diffusion_model,
                    optimizer_state_dict=optimizer.state_dict(),
                    epoch=epoch_index + 1,
                    validation_metrics=validation,
                ),
                checkpoint_path,
            )
        save_backbone_training_state(
            resume_path,
            feature_extractor,
            diffusion_model,
            optimizer,
            completed_epochs=epoch_index + 1,
            best_validation_score=best_validation_score,
            history=history,
            training_signature=training_signature,
        )
        write_history(run_dir / "backbone_training_history.csv", history)
        save_json(run_dir / "backbone_training_history.json", history)
        print(f"Saved resumable held-out backbone state after epoch {epoch_index + 1}: {resume_path}")

    if not history:
        raise ValueError("Backbone training produced no epoch history.")
    write_history(run_dir / "backbone_training_history.csv", history)
    save_json(run_dir / "backbone_training_history.json", history)
    provenance = write_backbone_provenance(
        run_dir / "backbone_provenance.json",
        checkpoint_path,
        training_domains=training_domains,
        excluded_domains=[args.holdout_domain] if is_holdout else [],
        training_data=(
            f"Supervised Class I source-domain samples only ({', '.join(sorted(training_domains))}); "
            f"external pretraining declaration: {args.external_pretraining_declaration.strip()}"
        ),
        training_protocol=protocol,
    )
    save_json(
        run_dir / "backbone_run_summary.json",
        {
            "mode": "train_backbone",
            "run_kind": "smoke" if args.max_train_batches or args.max_val_batches else "full",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "provenance_manifest": str((run_dir / "backbone_provenance.json").resolve()),
            "held_out_domain": args.holdout_domain or None,
            "training_domains": sorted(training_domains),
            "checkpoint_selection_metric": "validation_sampled_aggregate_epe",
            "best_validation_score": best_validation_score,
            "posterior_effective_on_last_epoch": bool(history[-1]["validation_aggregate_beats_center_rate"] > 0.5),
        },
    )
    print(f"Saved held-out backbone checkpoint: {checkpoint_path}")
    print(f"Saved backbone provenance: {run_dir / 'backbone_provenance.json'}")


if __name__ == "__main__":
    main()
