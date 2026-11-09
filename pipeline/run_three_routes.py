"""Three routes for physics, one grid.

The claim under test is that physics changes the candidate spread's relation to the error
whichever route it takes in -- through the architecture, through the training loss, or through an
operator applied at inference -- and that the three routes differ in magnitude rather than in kind.

Configurations (same architecture, same 1 065 fields, same seed unless stated):

  baseline      no physics
  loss0.5       divergence penalty with weight 0.5   (tier 2, weak)
  loss5         divergence penalty with weight 5     (tier 2, medium)
  loss50        divergence penalty with weight 50    (tier 2, strong)
  arch          stream-function output, divergence free by construction  (tier 1)
  arch_seed1    the same architecture constraint, different training seed

For each configuration the script saves per-field point, spread and truth for four derived
quantities, so the ratio R, the coverage, the spread magnitude and the rank correlation between
spread and error can all be computed from one run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures" / "uncertainty_study" / "tables"

CONFIGS = [
    ("baseline", {"mode": "plain", "weight": 0.0, "seed": 0}),
    ("loss0.5", {"mode": "loss", "weight": 0.5, "seed": 0}),
    ("loss5", {"mode": "loss", "weight": 5.0, "seed": 0}),
    ("loss50", {"mode": "loss", "weight": 50.0, "seed": 0}),
    ("arch", {"mode": "arch", "weight": 0.0, "seed": 0}),
    ("arch_seed1", {"mode": "arch", "weight": 0.0, "seed": 1}),
]


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
    )


class TinyUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, base=48, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, base), nn.SiLU(), nn.Linear(base, base))
        self.time_dim = time_dim
        self.inc = conv_block(in_ch, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.block1 = conv_block(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.block2 = conv_block(base * 4, base * 4)
        self.mid = conv_block(base * 4, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.upblock2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.upblock1 = conv_block(base * 2, base)
        self.outc = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x, t, cond):
        emb = self.time_mlp(self._embed(t, x.device))
        h = self.inc(torch.cat([x, cond], dim=1))
        h = h + emb[:, :, None, None]
        d1 = self.block1(self.down1(h))
        d2 = self.block2(self.down2(d1))
        m = self.mid(d2)
        u2 = self.upblock2(torch.cat([self.up2(m), d1], dim=1))
        u1 = self.upblock1(torch.cat([self.up1(u2), h], dim=1))
        return self.outc(u1)

    def _embed(self, t, device):
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def spectral_div(u, v):
    ny, nx = u.shape[-2:]
    kx = 2 * math.pi * torch.fft.fftfreq(nx, device=u.device)[None, :]
    ky = 2 * math.pi * torch.fft.fftfreq(ny, device=u.device)[:, None]
    return torch.real(torch.fft.ifft2(1j * kx * torch.fft.fft2(u) + 1j * ky * torch.fft.fft2(v)))


def stream_to_velocity(psi):
    ny, nx = psi.shape[-2:]
    kx = 2 * math.pi * torch.fft.fftfreq(nx, device=psi.device)[None, :]
    ky = 2 * math.pi * torch.fft.fftfreq(ny, device=psi.device)[:, None]
    psi_hat = torch.fft.fft2(psi)
    return (torch.real(torch.fft.ifft2(1j * ky * psi_hat)),
            torch.real(torch.fft.ifft2(-1j * kx * psi_hat)))


def velocity_to_stream(u, v):
    ny, nx = u.shape[-2:]
    kx = 2 * math.pi * torch.fft.fftfreq(nx, device=u.device)[None, :]
    ky = 2 * math.pi * torch.fft.fftfreq(ny, device=u.device)[:, None]
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0
    psi_hat = (1j * ky * torch.fft.fft2(u) - 1j * kx * torch.fft.fft2(v)) / k2
    psi_hat[..., 0, 0] = 0
    return torch.real(torch.fft.ifft2(psi_hat))


class Diffusion(nn.Module):
    def __init__(self, steps=200):
        super().__init__()
        self.steps = steps
        betas = torch.linspace(1e-4, 0.02, steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, dim=0))

    def q_sample(self, x0, t, noise):
        ab = self.alpha_bar[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def train(mode, weight, seed, train_data, device, steps, batch=32, lr=2e-4):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    diffusion = Diffusion().to(device)
    out_ch = 1 if mode == "arch" else 2
    model = TinyUNet(in_ch=out_ch + 2, out_ch=out_ch).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    data = torch.from_numpy(train_data).to(device)
    n = data.shape[0]
    started = time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        x0 = data[idx].permute(0, 3, 1, 2)
        cond = F.interpolate(F.avg_pool2d(x0, 2), scale_factor=2, mode="nearest")
        t = torch.randint(0, diffusion.steps, (batch,), device=device)
        noise = torch.randn_like(x0)

        if mode == "arch":
            target = velocity_to_stream(x0[:, 0], x0[:, 1]).unsqueeze(1)
            xq = diffusion.q_sample(target, t, noise[:, :1])
            loss = F.mse_loss(model(xq, t, cond), noise[:, :1])
        else:
            xq = diffusion.q_sample(x0, t, noise)
            pred = model(xq, t, cond)
            loss = F.mse_loss(pred, noise)
            if mode == "loss" and weight > 0:
                ab = diffusion.alpha_bar[t][:, None, None, None]
                x0_hat = (xq - (1 - ab).sqrt() * pred) / ab.sqrt()
                loss = loss + weight * (spectral_div(x0_hat[:, 0], x0_hat[:, 1]) ** 2).mean()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 500 == 0:
            print(f"    step {step} loss {loss.item():.4f} {time.time()-started:.0f}s", flush=True)
    return model, diffusion


@torch.no_grad()
def sample(model, diffusion, cond, mode, k, steps, device):
    cond = cond.repeat(k, 1, 1, 1)
    ch = 1 if mode == "arch" else 2
    x = torch.randn(k, ch, cond.shape[-2], cond.shape[-1], device=device)
    times = torch.linspace(diffusion.steps - 1, 0, steps, device=device).long()
    for i in range(len(times) - 1):
        t_cur, t_next = times[i], times[i + 1]
        t_batch = torch.full((k,), int(t_cur), device=device, dtype=torch.long)
        eps = model(x, t_batch, cond)
        ab_cur, ab_next = diffusion.alpha_bar[t_cur], diffusion.alpha_bar[t_next]
        x0 = (x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
    if mode == "arch":
        u, v = stream_to_velocity(x[:, 0])
        return torch.stack([u, v], dim=1)
    return x


def derived(u, v):
    uy, ux = torch.gradient(u)
    vy, vx = torch.gradient(v)
    speed = torch.sqrt(u ** 2 + v ** 2).mean()
    vort = vx - uy
    strain = torch.sqrt(ux ** 2 + vy ** 2 + 0.5 * (uy + vx) ** 2).mean()
    return {"mean_speed": speed.item(),
            "enstrophy": (0.5 * (vort ** 2).mean()).item(),
            "mean_strain_rate": strain.item(),
            "divergence_rms": torch.sqrt(((ux + vy) ** 2).mean()).item()}


def build_data(size=32, seed=0):
    data = np.load(ROOT / "external" / "fundiff_data" / "piv_fields.npz", allow_pickle=False)
    fields = data["uvw"].astype(np.float32)
    scale = float(np.std(fields))
    fields = fields / scale
    factor = 64 // size
    fields = fields.reshape(fields.shape[0], size, factor, size, factor, 2).mean(axis=(2, 4))
    order = np.random.default_rng(seed).permutation(fields.shape[0])
    return fields[order[:900]], fields[order[900:1000]], scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--out", default=str(OUT / "three_routes.npz"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    train_data, test_data, scale = build_data()
    test_low = test_data.reshape(len(test_data), 16, 2, 16, 2, 2).mean(axis=(2, 4))
    test_cond = np.repeat(np.repeat(test_low, 2, axis=1), 2, axis=2)

    store = {}
    for name, config in CONFIGS:
        print(f"[{name}]", flush=True)
        model, diffusion = train(config["mode"], config["weight"], config["seed"],
                                 train_data, device, args.steps)
        points, spreads, truths = [], [], []
        for i in range(min(args.frames, test_data.shape[0])):
            cond = torch.from_numpy(test_cond[i:i + 1]).permute(0, 3, 1, 2).to(device)
            cands = sample(model, diffusion, cond, config["mode"], args.candidates,
                           args.sampling_steps, device)
            values = [derived(cands[c, 0], cands[c, 1]) for c in range(args.candidates)]
            truth_t = torch.from_numpy(test_data[i]).permute(2, 0, 1).to(device)
            truth = derived(truth_t[0], truth_t[1])
            points.append({k: np.mean([v[k] for v in values]) for k in truth})
            spreads.append({k: np.std([v[k] for v in values]) for k in truth})
            truths.append(truth)
        for key in truths[0]:
            store[f"{name}__point__{key}"] = np.array([p[key] for p in points]) * scale
            store[f"{name}__spread__{key}"] = np.array([s[key] for s in spreads]) * scale
            store[f"{name}__truth__{key}"] = np.array([t[key] for t in truths]) * scale
        print(f"  done {name}", flush=True)

    Path(args.out).write_text("", encoding="utf-8")
    np.savez(args.out.replace(".json", ".npz"), **store)
    print(f"wrote {args.out.replace('.json', '.npz')}")


if __name__ == "__main__":
    main()
