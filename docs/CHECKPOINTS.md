# Checkpoints

The checkpoints are **not** included in this repository. They are archived at `<ZENODO_DOI>` and should be placed at the paths below before running `pipeline/`.

| Role | Expected path | Size | SHA-256 |
| --- | --- | --- | --- |
| Backbone (conditional residual DDPM, seed 3407) | `piv_runs/residual_ddpm_v6_x0_seed3407/diffusion_backbone_best.pth` | 141.3 MB | `DEF2F70960F6E85B79311893DE205297092D43DC4A0651144A05113C2DB0DC74` |
| Main gate (agent, seed 3417) | `piv_runs/physics_unrolled_v3_warmup_scratch_20ep_seed3417/physics_agent_best.pth` | 0.2 MB | `D7C45CB78CEFCB667A913A3A5ED9435C2C6702A070CB718339B801C75301B46E` |
| Second gate (independent checkpoint, seed 3415) | `piv_runs/physics_unrolled_v2_gate_full_seed3415/physics_agent_best.pth` | 0.2 MB | `DE79FB6F4C0846D6EEB0E1137FC763B50059C308484FFF18F6A1C1C304F15B4B` |

The backbone is shared by the main instance, the held-out domains and the replication check. The two agents differ only by training seed/initialisation; the replication in `piv_runs/replicate_agent_v2/` uses the second gate.

## Verify after download

```powershell
Get-FileHash .\piv_runs\residual_ddpm_v6_x0_seed3407\diffusion_backbone_best.pth -Algorithm SHA256
Get-FileHash .\piv_runs\physics_unrolled_v3_warmup_scratch_20ep_seed3417\physics_agent_best.pth -Algorithm SHA256
Get-FileHash .\piv_runs\physics_unrolled_v2_gate_full_seed3415\physics_agent_best.pth -Algorithm SHA256
```

If a hash differs, do not use the checkpoint: the reported numbers assume the exact weights above.

## What is reproducible without checkpoints

Everything in `scripts/` runs from `piv_runs/study_*.npz` and the derived tables. No checkpoint is needed for the main-text numbers, Fig. 2, Fig. 3 or Table 1.
