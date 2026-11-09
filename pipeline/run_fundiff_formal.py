"""FunDiff cross-implementation check, formal version.

Upgrades over the feasibility run (scripts/run_fundiff_check.py):
  * a larger model and far longer training, so the conditional samples are actually
    informative rather than dominated by underfitting;
  * three observation resolutions (2x, 4x, 8x downsampling), i.e. three constraint
    strengths -- this turns the check into an ablation of "does strength matter";
  * several random train/test splits and calibration halves, reported as mean +- sd;
  * coverage of both interval types, so that the width comparison is at matched coverage.

The data are held fixed (our PIV fields): the point of the check is to swap the
*implementation* -- FunDiff's Encoder / Decoder / DiT and its rectified-flow objective --
while keeping everything else identical.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1] / "external" / "fundiff-main"
sys.path.insert(0, str(REPO))

import jax
import jax.numpy as jnp
from jax import random
import optax

from function_diffusion.models import Encoder, Decoder, DiT

from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=160)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--latents", type=int, default=96)
    parser.add_argument("--fae-steps", type=int, default=2500)
    parser.add_argument("--dit-steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--flow-steps", type=int, default=25)
    parser.add_argument("--n-eval", type=int, default=80)
    parser.add_argument("--obs-factors", default="2,4,8")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "external" / "fundiff_data" / "piv_fields.npz"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper_figures" / "uncertainty_study" / "tables" / "fundiff_check_formal.json"))
    return parser.parse_args()


def grid_coords(size):
    x = (np.arange(size) + 0.5) / size
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)


def scalars(field):
    u, v = field[..., 0], field[..., 1]
    speed = np.sqrt(u ** 2 + v ** 2)
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    vort = dv_dx - du_dy
    strain = np.sqrt(du_dx ** 2 + dv_dy ** 2 + 0.5 * (du_dy + dv_dx) ** 2)
    div = du_dx + dv_dy
    return {
        "mean_speed": float(speed.mean()),
        "enstrophy": float(0.5 * (vort ** 2).mean()),
        "mean_strain_rate": float(strain.mean()),
        "divergence_rms": float(np.sqrt((div ** 2).mean())),
    }


def main():
    args = parse_args()
    data = np.load(args.data, allow_pickle=False)
    fields = data["uvw"].astype(np.float32)
    fields = fields / float(np.std(fields))
    side = fields.shape[1]
    coords_full = jnp.asarray(grid_coords(side))
    eval_size = 32
    coords_eval = jnp.asarray(grid_coords(eval_size))

    enc_cfg = dict(patch_size=(8, 8), emb_dim=args.dim, num_latents=args.latents,
                   grid_size=(side, side), depth=args.depth, num_heads=4, mlp_ratio=2,
                   layer_norm_eps=1e-5)
    dec_cfg = dict(period=False, fourier_freq=5.0, dec_emb_dim=args.dim, dec_depth=2,
                   dec_num_heads=4, mlp_ratio=2, num_mlp_layers=2, out_dim=2, layer_norm_eps=1e-5)
    dit_cfg = dict(model_name="DiT", emb_dim=args.dim, depth=args.depth, num_heads=4,
                   mlp_ratio=2, out_dim=args.dim)

    factors = [int(f) for f in args.obs_factors.split(",")]
    results = {"config": vars(args), "by_factor": {}}

    for split in range(args.splits):
        rng = np.random.default_rng(100 + split)
        order = rng.permutation(fields.shape[0])
        test_idx = order[:args.n_eval * 2]
        train_idx = order[args.n_eval * 2: args.n_eval * 2 + 700]

        encoder = Encoder(**enc_cfg)
        decoder = Decoder(**dec_cfg)
        enc_params = encoder.init(random.PRNGKey(split), jnp.asarray(fields[:2]))
        dec_params = decoder.init(random.PRNGKey(10 + split),
                                  encoder.apply(enc_params, jnp.asarray(fields[:2])),
                                  coords_full[0])
        fae_params = (enc_params, dec_params)
        fae_opt = optax.adam(2e-4)
        fae_state = fae_opt.init(fae_params)
        dit = DiT(**dit_cfg)
        dit_opt = optax.adam(2e-4)

        def gather_values(batch, idx):
            row, col = idx // side, idx % side
            return batch[:, row, col, :]

        def fae_step(params, opt_state, batch, coords, idx):
            def loss_fn(p):
                enc_p, dec_p = p
                z = encoder.apply(enc_p, batch)
                pred = jax.vmap(lambda c: decoder.apply(dec_p, z, c))(coords)
                pred = pred[..., 0, :].transpose(1, 0, 2)
                return jnp.mean((pred - gather_values(batch, idx)) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = fae_opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        fae_step_jit = jax.jit(fae_step)
        key = random.PRNGKey(split)
        steps = 2 if args.smoke else args.fae_steps
        start = time.time()
        for step in range(steps):
            batch = jnp.asarray(fields[np.random.choice(train_idx, args.batch, replace=False)])
            key, k1 = random.split(key)
            idx = random.choice(k1, side * side, (args.queries,), replace=False)
            fae_params, fae_state, loss = fae_step_jit(fae_params, fae_state, batch, coords_full[idx], idx)
            if step % 500 == 0 or step == steps - 1:
                print(f"[split {split}][fae] {step:5d} loss {float(loss):.5f} {time.time()-start:.0f}s", flush=True)

        def cond_latent(batch, factor):
            low = batch[:, ::factor, ::factor, :]
            if factor > 1:
                low = jnp.repeat(jnp.repeat(low, factor, axis=1), factor, axis=2)
            return encoder.apply(fae_params[0], low)

        z_full = encoder.apply(fae_params[0], jnp.asarray(fields[train_idx[:2]]))
        z_cond = cond_latent(jnp.asarray(fields[train_idx[:2]]), factors[0])
        dit_params = dit.init(random.PRNGKey(20 + split), z_full, jnp.zeros((2,)), z_cond)
        dit_state = dit_opt.init(dit_params)

        def dit_step(params, opt_state, batch, key):
            z1 = encoder.apply(fae_params[0], batch)
            c = cond_latent(batch, factors[0])
            key, k1, k2 = random.split(key, 3)
            z0 = random.normal(k1, z1.shape)
            t = random.uniform(k2, (z1.shape[0], 1, 1))
            z_t = t * z1 + (1.0 - t) * z0
            target = z1 - z0

            def loss_fn(p):
                return jnp.mean((dit.apply(p, z_t, t.reshape(-1), c) - target) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = dit_opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        dit_step_jit = jax.jit(dit_step)
        start = time.time()
        for step in range(steps if args.smoke else args.dit_steps):
            batch = jnp.asarray(fields[np.random.choice(train_idx, args.batch, replace=False)])
            key, k1 = random.split(key)
            dit_params, dit_state, loss = dit_step_jit(dit_params, dit_state, batch, k1)
            if step % 1000 == 0 or step == args.dit_steps - 1:
                print(f"[split {split}][dit] {step:5d} loss {float(loss):.5f} {time.time()-start:.0f}s", flush=True)

        latent_shape = z_full.shape[1:]

        def sample_fields(obs, k, key):
            z_cond = encoder.apply(fae_params[0], obs[None])
            z_cond = jnp.repeat(z_cond, k, axis=0)
            key, k1 = random.split(key)
            z = random.normal(k1, (k,) + latent_shape)
            dt = 1.0 / args.flow_steps
            for i in range(args.flow_steps):
                t = jnp.ones((k,)) * i / args.flow_steps
                z = z + dit.apply(dit_params, z, t, z_cond) * dt

            def decode_one(zi):
                out = jax.vmap(lambda c: decoder.apply(fae_params[1], zi[None], c))(coords_eval)
                return out[..., 0, :][:, 0, :]

            decoded = jax.vmap(decode_one)(z)
            return decoded.reshape(k, eval_size, eval_size, 2)

        sample_jit = jax.jit(sample_fields, static_argnums=(1,))

        for factor in factors:
            records = []
            for index in test_idx[:args.n_eval]:
                field = fields[index]
                low = field[::factor, ::factor, :]
                if factor > 1:
                    low = np.repeat(np.repeat(low, factor, axis=0), factor, axis=1)
                samples = np.asarray(sample_jit(jnp.asarray(low), args.candidates, key))
                key, _ = random.split(key)
                truth = scalars(field)
                cand = [scalars(samples[c]) for c in range(args.candidates)]
                records.append({
                    "truth": truth,
                    "mean": {n: float(np.mean([c[n] for c in cand])) for n in truth},
                    "spread": {n: float(np.std([c[n] for c in cand])) for n in truth},
                })
            print(f"[split {split}][factor {factor}] evaluated {len(records)} samples", flush=True)

            bucket = results["by_factor"].setdefault(str(factor), {})
            for name in records[0]["truth"]:
                s = np.array([r["spread"][name] for r in records])
                e = np.array([abs(r["mean"][name] - r["truth"][name]) for r in records])
                ok = np.isfinite(s) & np.isfinite(e)
                s, e = s[ok], e[ok]
                if len(s) < 20:
                    continue
                rho = float(spearmanr(s, e).statistic)
                dynamic_range = float(np.percentile(s, 95) / max(np.percentile(s, 5), 1e-12))
                entry = bucket.setdefault(name, {"R": [], "range": [], "rho": [], "cover_scaled": [], "cover_const": []})
                entry["range"].append(dynamic_range)
                entry["rho"].append(rho)
                alpha = 0.1
                # repeatability comes from resampling the calibration half
                for rep in range(4):
                    perm = np.random.default_rng(500 + rep).permutation(len(s))
                    half = len(s) // 2
                    cal, ev = perm[:half], perm[half:]
                    cal_s, cal_e, ev_s, ev_e = s[cal], e[cal], s[ev], e[ev]
                    k_q = int(np.ceil((len(cal_s) + 1) * (1 - alpha))) - 1
                    q = float(np.sort(cal_e / (cal_s + 1e-12))[min(k_q, len(cal_s) - 1)])
                    q_e = float(np.sort(cal_e)[min(k_q, len(cal_e) - 1)])
                    width_scaled = float(np.mean(2 * ev_s * q))
                    width_const = float(2 * q_e)
                    entry["R"].append(width_const / max(width_scaled, 1e-12))
                    entry["cover_scaled"].append(float(np.mean(ev_e <= ev_s * q + 1e-12)))
                    entry["cover_const"].append(float(np.mean(ev_e <= q_e)))

    summary = {}
    for factor, bucket in results["by_factor"].items():
        summary[factor] = {}
        for name, entry in bucket.items():
            summary[factor][name] = {
                key: {"mean": float(np.mean(value)), "sd": float(np.std(value))}
                for key, value in entry.items()
            }
    results["summary"] = summary

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for factor in sorted(summary, key=int):
        print(f"\n=== observation downsampled {factor}x ===")
        for name, stats in summary[factor].items():
            print(f"{name:20s} R {stats['R']['mean']:.3f} +- {stats['R']['sd']:.3f} | "
                  f"range {stats['range']['mean']:7.1f} | rho {stats['rho']['mean']:+.3f} | "
                  f"cover {stats['cover_scaled']['mean']:.3f}/{stats['cover_const']['mean']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
