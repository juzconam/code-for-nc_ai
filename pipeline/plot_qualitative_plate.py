"""Qualitative plate: real PIV samples, field maps of the derived quantities.

Nature papers need one figure where the reader can see the phenomenon rather than
infer it from a metric.  For three test samples (three different flow regimes) this
script renders

  * the ground-truth speed field,
  * the reconstructed (aggregated) speed field,
  * the candidate spread sigma across the K samples,
  * the divergence of the ground truth,
  * the divergence of the reconstruction.

The point is visual: columns 1 and 2 look alike, columns 4 and 5 do not.  The numbers
in the row labels (divergence RMS of truth versus reconstruction, and their ratio)
connect the picture to the statistics in the paper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import get_device, set_seed
from physics_utils import compute_divergence, flow_to_velocity
from run_piv_uncertainty_study import IndexedSplit, build_datasets, candidate_noise, load_models

DOMAINS = ("DNS_turbulence", "backstep", "cylinder")
COLUMNS = ("真值速度 |u|", "重建速度 |u|", "候选离散度 σ", "真值散度", "重建散度（与左栏同色标）")


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
    parser.add_argument("--split", default="test")
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--particle-sigma", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(ROOT / "paper_figures" / "uncertainty_study"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device("auto")
    set_seed(args.seed)
    output = Path(args.output_dir)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    datasets, _manifest, domain_names, uid_by_index, domain_by_index = build_datasets(
        args.data_path, args.split_manifest
    )
    dataset = datasets[args.split]

    # one sample per requested domain, taken through the same index wrapper the
    # quantitative study uses so that uids and domains line up
    positions = list(range(len(dataset)))
    wrapper = IndexedSplit(dataset, positions)
    loader = DataLoader(wrapper, batch_size=1, shuffle=False, num_workers=0)
    picks = {}
    for samples, target, domain_id, indices in loader:
        label = domain_names[int(domain_id[0])] if domain_names else str(domain_id)
        if label in DOMAINS and label not in picks:
            picks[label] = (samples.clone(), target.clone(), str(uid_by_index[int(indices[0])]))
    missing = [name for name in DOMAINS if name not in picks]
    if missing:
        raise SystemExit(f"domains not found: {missing}")
    print(f"[plate] picked {[(k, v[2]) for k, v in picks.items()]}", flush=True)

    feature, diffusion, _agent, aggregator, _checkpoint = load_models(args, device)
    geometry = aggregator.geometry

    records = []
    for domain in DOMAINS:
        samples, target_tensor, uid = picks[domain]
        image1 = samples[0, 0:1].unsqueeze(0).to(device)
        image2 = samples[0, 1:2].unsqueeze(0).to(device)
        target = target_tensor.to(device)
        noise = candidate_noise(uid, args.candidates,
                                (2, image1.shape[-2], image1.shape[-1]), "plate", args.seed)
        noise = noise.unsqueeze(1).to(device)   # [K, B, 2, H, W]
        with torch.no_grad():
            condition = feature(image1, image2)
            candidates = diffusion(
                condition, image1, image2, use_ema=True, num_candidates=args.candidates,
                sampling_steps=args.sampling_steps, initial_noise=noise,
            )
            aggregated, _sigma = aggregator.filter_and_aggregate(candidates, image1, image2)
            speed_truth = flow_to_velocity(target, geometry).norm(dim=1)[0].cpu().numpy()
            speed_recon = flow_to_velocity(aggregated, geometry).norm(dim=1)[0].cpu().numpy()
            spread = candidates.std(dim=1).norm(dim=1)[0].cpu().numpy()
            div_truth = compute_divergence(target, geometry=geometry)[0, 0].cpu().numpy()
            div_recon = compute_divergence(aggregated, geometry=geometry)[0, 0].cpu().numpy()
        truth_rms = float(np.sqrt((div_truth ** 2).mean()))
        recon_rms = float(np.sqrt((div_recon ** 2).mean()))
        records.append({
            "domain": domain,
            "speed_truth": speed_truth,
            "speed_recon": speed_recon,
            "spread": spread,
            "div_truth": div_truth,
            "div_recon": div_recon,
            "truth_rms": truth_rms,
            "recon_rms": recon_rms,
        })
        print(f"[plate] {domain:16s} div truth {truth_rms:.5f}  recon {recon_rms:.5f}  "
              f"ratio {recon_rms / max(truth_rms, 1e-12):.1f}", flush=True)

    np.savez(output / "tables" / "qualitative_plate.npz", **{
        f"{key}_{index}": value for index, item in enumerate(records)
        for key, value in item.items() if isinstance(value, np.ndarray)
    })

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "savefig.dpi": 300, "font.size": 7,
        "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
    })

    rows = len(records)
    figure, axes = plt.subplots(rows, len(COLUMNS), figsize=(11.0, 2.55 * rows))
    if rows == 1:
        axes = axes[None, :]

    images = []
    for row, item in enumerate(records):
        panels = (
            (item["speed_truth"], "viridis", "速度"),
            (item["speed_recon"], "viridis", "速度"),
            (item["spread"], "magma", "离散度"),
            (item["div_truth"], "RdBu_r", "散度"),
            (item["div_recon"], "RdBu_r", "散度"),
        )
        # the two divergence panels of a row share one colour scale, taken from the
        # reconstruction's 99th percentile.  The ground truth then genuinely looks
        # blank because it is 5-200x smaller, which is the point of the figure.
        div_limit = max(float(np.quantile(np.abs(item["div_recon"]), 0.99)), 1e-9)
        for column, (field, cmap, kind) in enumerate(panels):
            ax = axes[row, column]
            if kind == "散度":
                image = ax.imshow(field, cmap=cmap, vmin=-div_limit, vmax=div_limit)
            else:
                image = ax.imshow(field, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if row == 0:
                ax.set_title(COLUMNS[column], pad=3)
            images.append(image)
        axes[row, 0].set_ylabel(item["domain"].replace("_", " "), fontsize=7)
        axes[row, 0].text(
            0.02, -0.06,
            f"div RMS: 真值 {item['truth_rms']:.4f} ｜ 重建 {item['recon_rms']:.3f}"
            f"（{item['recon_rms'] / max(item['truth_rms'], 1e-12):.0f}×）",
            transform=axes[row, 0].transAxes, fontsize=5.8, va="top",
        )

    # one colour bar per column, placed under the last row
    for column, image in enumerate(images[: len(COLUMNS)]):
        box = axes[rows - 1, column].get_position()
        bar = figure.add_axes([box.x0 + 0.06 * box.width, 0.035,
                               0.88 * box.width, 0.008])
        figure.colorbar(image, cax=bar, orientation="horizontal")

    figure.subplots_adjust(left=0.09, right=0.995, top=0.955, bottom=0.10,
                           wspace=0.06, hspace=0.08)
    path = output / "figures" / "fig19_qualitative_plate.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
