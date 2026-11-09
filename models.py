"""Conditional diffusion backbone and frozen RAFT feature conditioning for UPR-Flow."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.ops import DeformConv2d

from src.inference.piv_posterior import PosteriorGuidanceConfig, guided_clean_flow, posterior_risk

DIFFUSION_CHECKPOINT_SCHEMA_VERSION = 8
DIFFUSION_MODEL_FAMILY = "conditional_residual_ddpm_flow_v4"

LOG_VAR_MIN = -16.0
LOG_VAR_MAX = 4.0
# log var of the normalized residual: (gt - center) / flow_scale. At initialization the
# center error is ~1.6 px, i.e. a residual of ~0.08, so log(0.08^2) ~= -5. Starting the
# head there keeps its gradient near zero instead of dominating the first updates.
LOG_VAR_INIT = -5.0


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        attention_channels = max(in_channels // 8, 1)
        self.query = nn.Conv2d(in_channels, attention_channels, 1)
        self.key = nn.Conv2d(in_channels, attention_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        query = self.query(x).reshape(batch, -1, height * width).transpose(1, 2)
        key = self.key(x).reshape(batch, -1, height * width)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).reshape(batch, channels, height * width)
        attended = torch.bmm(value, attention.transpose(1, 2)).reshape(batch, channels, height, width)
        return x + self.gamma * attended


class SinkhornOptimalTransport(nn.Module):
    def __init__(self, max_iters: int = 10, epsilon: float = 0.05):
        super().__init__()
        self.max_iters = int(max_iters)
        self.epsilon = float(epsilon)

    def forward(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        kernel = torch.exp(-cost_matrix / self.epsilon)
        batch, rows, columns = kernel.shape
        row_mass = torch.full((batch, rows, 1), 1.0 / rows, device=kernel.device, dtype=kernel.dtype)
        column_mass = torch.full((batch, columns, 1), 1.0 / columns, device=kernel.device, dtype=kernel.dtype)
        u = row_mass
        for _ in range(self.max_iters):
            v = column_mass / (torch.bmm(kernel.transpose(1, 2), u) + 1e-8)
            u = row_mass / (torch.bmm(kernel, v) + 1e-8)
        return u * kernel * v.transpose(1, 2)


class DualSourceFeatureExtractor(nn.Module):
    """Frozen RAFT features with trainable local transport conditioning."""

    def __init__(self, device: torch.device, hidden_dim: int = 128):
        super().__init__()
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
        self.raft.eval()
        self.self_attention = SpatialSelfAttention(in_channels=256)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 18, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)
        self.deform_conv = DeformConv2d(256, 256, kernel_size=3, padding=1)
        self.sinkhorn = SinkhornOptimalTransport(max_iters=10, epsilon=0.05)
        self.ot_feat_proj = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, hidden_dim, kernel_size=3, padding=1),
        )

    def compute_deformable_ot_cost(self, fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = fmap1.shape
        offset = self.offset_conv(torch.cat([fmap1, fmap2], dim=1))
        fmap2_deformed = self.deform_conv(fmap2, offset)
        fmap1_normalized = F.normalize(fmap1, dim=1)
        fmap2_normalized = F.normalize(fmap2_deformed, dim=1)
        local_candidates = F.unfold(fmap2_normalized, kernel_size=3, padding=1).reshape(batch, channels, 9, height * width)
        query = fmap1_normalized.reshape(batch, channels, 1, height * width)
        cost = 1.0 - F.cosine_similarity(query, local_candidates, dim=1)
        plan = self.sinkhorn(cost.permute(0, 2, 1)).permute(0, 2, 1).reshape(batch, 9, height, width)
        return self.ot_feat_proj(plan)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        image1_normalized = image1 * 2.0 - 1.0
        image2_normalized = image2 * 2.0 - 1.0
        if image1_normalized.shape[1] == 1:
            image1_normalized = image1_normalized.repeat(1, 3, 1, 1)
            image2_normalized = image2_normalized.repeat(1, 3, 1, 1)
        with torch.no_grad():
            fmap1 = self.raft.feature_encoder(image1_normalized)
            fmap2 = self.raft.feature_encoder(image2_normalized)
            context = self.raft.context_encoder(image1_normalized)
        attended = self.self_attention(fmap1)
        transport = self.compute_deformable_ot_cost(attended, fmap2)
        features = self.conv_out(torch.cat([context, attended, transport], dim=1))
        return F.interpolate(features, size=image1.shape[-2:], mode="bilinear", align_corners=True)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        frequency = math.log(10000.0) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device, dtype=time.dtype) * -frequency)
        phase = time[:, None] * embeddings[None, :]
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


class ConditionalPixelUNet(nn.Module):
    """Bounded x0 predictor with an optional heteroscedastic variance head."""

    def __init__(self, flow_channels: int = 2, condition_channels: int = 128, learn_variance: bool = True):
        super().__init__()
        time_dim = 256
        self.flow_channels = int(flow_channels)
        self.learn_variance = bool(learn_variance)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        input_channels = flow_channels + condition_channels + 2
        self.down1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.time_embed_1 = nn.Linear(time_dim, 128)
        self.time_embed_2 = nn.Linear(time_dim, 256)
        self.time_embed_bottleneck = nn.Linear(time_dim, 256)
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        out_channels = flow_channels * (2 if self.learn_variance else 1)
        self.output = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if self.learn_variance:
            # Start the variance head at a physically sensible residual scale instead of
            # unit variance, so the NLL term does not dominate the first updates.
            with torch.no_grad():
                self.output.bias[flow_channels:].fill_(LOG_VAR_INIT)

    def split_head(self, head_output: torch.Tensor):
        """Return (mu_logits, log_var) where log_var is None in the homoscedastic case."""
        if not self.learn_variance:
            return head_output, None
        mu_logits, log_var = head_output.split(self.flow_channels, dim=1)
        return mu_logits, log_var.clamp(min=LOG_VAR_MIN, max=LOG_VAR_MAX)

    def forward(
        self,
        noisy_flow: torch.Tensor,
        normalized_timestep: torch.Tensor,
        condition_features: torch.Tensor,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_mlp(normalized_timestep)
        x = torch.cat([noisy_flow, condition_features, image1, image2], dim=1)
        down1 = self.down1(x) + self.time_embed_1(time_embedding).unsqueeze(-1).unsqueeze(-1)
        down2 = self.down2(F.max_pool2d(down1, kernel_size=2))
        down2 = down2 + self.time_embed_2(time_embedding).unsqueeze(-1).unsqueeze(-1)
        bottleneck = self.bottleneck(F.max_pool2d(down2, kernel_size=2))
        bottleneck = bottleneck + self.time_embed_bottleneck(time_embedding).unsqueeze(-1).unsqueeze(-1)
        up2 = F.interpolate(bottleneck, size=down2.shape[-2:], mode="bilinear", align_corners=True)
        up2 = self.up_conv2(torch.cat([up2, down2], dim=1))
        up1 = F.interpolate(up2, size=down1.shape[-2:], mode="bilinear", align_corners=True)
        return self.split_head(self.output(self.up_conv1(torch.cat([up1, down1], dim=1))))


class DeterministicFlowCenter(nn.Module):
    """Predict the conditional center around which the residual posterior is sampled."""

    def __init__(self, condition_channels: int = 128, prior_channels: int = 0):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.prior_channels = int(prior_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(condition_channels + 2 + self.prior_channels, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.output.bias)
        if self.prior_channels:
            # Zero-init the external-prior input columns.  At initialisation the
            # prior therefore contributes exactly nothing and the head is
            # identical to the no-prior model; the fusion is learned, never
            # imposed, so the worst case cannot be worse than the baseline.
            base = condition_channels + 2
            with torch.no_grad():
                self.encoder[0].weight[:, base : base + self.prior_channels, :, :].zero_()

    def forward(
        self,
        condition_features: torch.Tensor,
        image1: torch.Tensor,
        image2: torch.Tensor,
        flow_limit: float,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [condition_features, image1, image2]
        if self.prior_channels:
            if prior is None:
                prior = torch.zeros(
                    condition_features.shape[0],
                    self.prior_channels,
                    *condition_features.shape[-2:],
                    device=condition_features.device,
                    dtype=condition_features.dtype,
                )
            parts.append(prior)
        logits = self.output(self.encoder(torch.cat(parts, dim=1)))
        limit = max(float(flow_limit), 1e-6)
        return limit * torch.tanh(logits / limit)


def _cosine_beta_schedule(num_timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = int(num_timesteps) + 1
    grid = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    alpha_bar = torch.cos(((grid / num_timesteps) + offset) / (1.0 + offset) * math.pi * 0.5).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    return (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999)


class ConditionalDiffusionFlowGenerator(nn.Module):
    """A learned-center residual DDPM sampled through multi-step DDIM."""

    model_family = DIFFUSION_MODEL_FAMILY

    def __init__(
        self,
        num_candidates: int = 8,
        flow_scale: float = 20.0,
        num_train_timesteps: int = 100,
        sampling_steps: int = 20,
        ddim_eta: float = 0.0,
        denoised_clip: float = 0.5,
        center_loss_weight: float = 5.0,
        residual_reconstruction_weight: float = 2.0,
        learn_variance: bool = True,
        variance_fit_weight: float = 1.0,
        variance_injection_scale: float = 0.0,
        prior_channels: int = 0,
    ):
        super().__init__()
        if (
            num_candidates < 1
            or flow_scale <= 0.0
            or num_train_timesteps < 2
            or sampling_steps < 1
            or denoised_clip <= 0.0
            or center_loss_weight <= 0.0
            or residual_reconstruction_weight <= 0.0
            or variance_fit_weight < 0.0
        ):
            raise ValueError("Invalid conditional diffusion configuration.")
        self.num_candidates = int(num_candidates)
        self.flow_scale = float(flow_scale)
        self.num_train_timesteps = int(num_train_timesteps)
        self.sampling_steps = min(int(sampling_steps), self.num_train_timesteps)
        self.ddim_eta = float(ddim_eta)
        self.denoised_clip = float(denoised_clip)
        self.center_loss_weight = float(center_loss_weight)
        self.residual_reconstruction_weight = float(residual_reconstruction_weight)
        self.learn_variance = bool(learn_variance)
        self.variance_fit_weight = float(variance_fit_weight)
        # Non-zero injection samples x0 ~ N(mu, var) instead of returning mu. It widens the
        # candidate ensemble for uncertainty studies but strictly increases single-sample
        # EPE, so it stays off for accuracy-critical runs.
        self.variance_injection_scale = float(variance_injection_scale)
        self.prior_channels = int(prior_channels)
        self.last_aleatoric_std = None
        # Runtime inference policies are deliberately excluded from diffusion_config
        # so a new posterior sampler can be used with a version-compatible checkpoint.
        self.posterior_guidance = PosteriorGuidanceConfig()
        self.posterior_geometry = None
        self.last_sampling_metadata = {}
        self.adaptive_policy = None
        self.center_head = DeterministicFlowCenter(condition_channels=128, prior_channels=self.prior_channels)
        self.unet = ConditionalPixelUNet(flow_channels=2, condition_channels=128, learn_variance=self.learn_variance)
        self.ema_unet = copy.deepcopy(self.unet)
        self.ema_unet.eval()
        for parameter in self.ema_unet.parameters():
            parameter.requires_grad_(False)
        betas = _cosine_beta_schedule(self.num_train_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))

    @property
    def diffusion_config(self) -> Dict[str, Any]:
        return {
            "model_family": self.model_family,
            "prediction_type": "x0",
            "parameterization": "residual_around_learned_center",
            "noise_schedule": "cosine",
            "num_train_timesteps": self.num_train_timesteps,
            "sampling_method": "ddim",
            "sampling_steps": self.sampling_steps,
            "ddim_eta": self.ddim_eta,
            "num_candidates": self.num_candidates,
            "flow_scale": self.flow_scale,
            "denoised_clip": self.denoised_clip,
            "center_loss_weight": self.center_loss_weight,
            "residual_reconstruction_weight": self.residual_reconstruction_weight,
            "learn_variance": self.learn_variance,
            "variance_fit_weight": self.variance_fit_weight,
            "prediction_head": "mu_log_var_mse_mean_stopgrad_variance" if self.learn_variance else "mu_mse",
            "log_var_range": [LOG_VAR_MIN, LOG_VAR_MAX] if self.learn_variance else None,
            "prior_channels": self.prior_channels,
        }

    def predict_center(
        self,
        condition_features: torch.Tensor,
        image1: torch.Tensor,
        image2: torch.Tensor,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.center_head(
            condition_features,
            image1,
            image2,
            flow_limit=self.flow_scale,
            prior=prior,
        )

    def _extract(self, values: torch.Tensor, timestep: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return values.gather(0, timestep).reshape(reference.shape[0], 1, 1, 1).to(dtype=reference.dtype)

    def _normalized_timestep(self, timestep: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return timestep.to(device=reference.device, dtype=reference.dtype) / float(self.num_train_timesteps - 1)

    def q_sample(self, clean_flow: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._extract(self.sqrt_alpha_bar, timestep, clean_flow) * clean_flow + self._extract(
            self.sqrt_one_minus_alpha_bar, timestep, clean_flow
        ) * noise

    def predict_x0_from_noise(self, noisy_flow: torch.Tensor, timestep: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        return (noisy_flow - self._extract(self.sqrt_one_minus_alpha_bar, timestep, noisy_flow) * predicted_noise) / self._extract(
            self.sqrt_alpha_bar, timestep, noisy_flow
        ).clamp(min=1e-8)

    def predict_noise_from_x0(self, noisy_flow: torch.Tensor, timestep: torch.Tensor, predicted_clean: torch.Tensor) -> torch.Tensor:
        return (
            noisy_flow - self._extract(self.sqrt_alpha_bar, timestep, noisy_flow) * predicted_clean
        ) / self._extract(self.sqrt_one_minus_alpha_bar, timestep, noisy_flow).clamp(min=1e-8)

    def bound_clean_residual(self, predicted_clean: torch.Tensor) -> torch.Tensor:
        return self.denoised_clip * torch.tanh(predicted_clean / self.denoised_clip)

    @torch.no_grad()
    def update_ema(self, *_: Any) -> None:
        self.ema_updates.add_(1)
        updates = int(self.ema_updates.item())
        decay = min(0.9999, 1.0 - 1.0 / float(updates + 1))
        for ema_parameter, parameter in zip(self.ema_unet.parameters(), self.unet.parameters()):
            ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
        for ema_buffer, buffer in zip(self.ema_unet.buffers(), self.unet.buffers()):
            ema_buffer.copy_(buffer)

    def compute_loss(
        self,
        condition_features: torch.Tensor,
        image1: torch.Tensor,
        image2: torch.Tensor,
        gt_flow: torch.Tensor,
        epoch: int | None = None,
        max_epochs: int | None = None,
        prior: torch.Tensor | None = None,
    ):
        del epoch, max_epochs
        batch_size = gt_flow.shape[0]
        center_flow = self.predict_center(condition_features, image1, image2, prior=prior)
        clean_flow = (gt_flow - center_flow.detach()) / self.flow_scale
        timestep = torch.randint(0, self.num_train_timesteps, (batch_size,), device=gt_flow.device, dtype=torch.long)
        noise = torch.randn_like(clean_flow)
        noisy_flow = self.q_sample(clean_flow, timestep, noise)
        predicted_clean_logits, predicted_log_var = self.unet(
            noisy_flow,
            self._normalized_timestep(timestep, noisy_flow),
            condition_features,
            image1,
            image2,
        )
        predicted_clean = self.bound_clean_residual(predicted_clean_logits)
        if predicted_log_var is None:
            denoising_loss = F.mse_loss(predicted_clean, clean_flow)
        else:
            # Stop-gradient variance fitting. A plain heteroscedastic NLL scales the mean
            # gradient by 1/sigma^2; once sigma^2 tracks the squared residual (as it must at
            # the NLL optimum) that factor reaches ~1e3 in the sub-pixel residual regime, and
            # combined with global gradient clipping it starves every other parameter. Measured
            # on the full PIV split: the deterministic center stalled at 1.5 px while the
            # homoscedastic MSE baseline had already reached 0.42 px by epoch 5.
            # Keeping the mean head on an un-damped MSE and fitting sigma against the
            # detached squared residual recovers the baseline dynamics and still yields a
            # per-pixel aleatoric sigma, which is what the diagnostics actually need.
            variance = predicted_log_var.exp().clamp(min=1e-7)
            variance_target = (clean_flow - predicted_clean).square().detach()
            variance_term = 0.5 * (predicted_log_var + variance_target / variance).mean()
            denoising_loss = (
                F.mse_loss(predicted_clean, clean_flow)
                + self.variance_fit_weight * variance_term
            )
        center_epe = torch.norm(center_flow - gt_flow, p=2, dim=1).mean()
        center_loss = center_epe / self.flow_scale
        denoised_flow = center_flow + predicted_clean * self.flow_scale
        residual_reconstruction_epe = torch.norm(denoised_flow - gt_flow, p=2, dim=1).mean() / self.flow_scale
        total_loss = (
            denoising_loss
            + self.center_loss_weight * center_loss
            + self.residual_reconstruction_weight * residual_reconstruction_epe
        )
        epe = torch.norm(denoised_flow.detach() - gt_flow, p=2, dim=1).mean()
        return total_loss, denoising_loss.detach(), epe, center_epe.detach()

    @torch.no_grad()
    def ddim_sample(
        self,
        condition_features: torch.Tensor,
        image1: torch.Tensor,
        image2: torch.Tensor,
        model: nn.Module,
        sampling_steps: int,
        guidance_config: PosteriorGuidanceConfig | None = None,
        center_flow: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, _, height, width = condition_features.shape
        if center_flow is None:
            center_flow = self.predict_center(condition_features, image1, image2, prior=prior)
        if initial_noise is None:
            flow = torch.randn(batch, 2, height, width, device=condition_features.device, dtype=condition_features.dtype)
        else:
            # A caller-supplied initial noise makes each candidate a deterministic
            # function of (image pair, candidate index, seed) instead of a function
            # of the batch composition.  Split-conformal calibration requires the
            # scale s(x) to be a deterministic function of the input, so candidate
            # sampling must not depend on how samples are batched.
            if float(self.ddim_eta) > 0.0:
                raise ValueError(
                    "initial_noise is only supported with ddim_eta == 0; stochastic DDIM "
                    "would still depend on the ambient RNG state."
                )
            flow = initial_noise.to(device=condition_features.device, dtype=condition_features.dtype)
            if flow.shape != (batch, 2, height, width):
                raise ValueError(
                    f"initial_noise must have shape {(batch, 2, height, width)}, got {tuple(flow.shape)}"
                )
        steps = min(max(1, int(sampling_steps)), self.num_train_timesteps)
        timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            steps,
            device=condition_features.device,
        ).round().long().unique_consecutive()
        for index, timestep_value in enumerate(timesteps):
            timestep = torch.full((batch,), int(timestep_value.item()), device=flow.device, dtype=torch.long)
            head = model(
                flow,
                self._normalized_timestep(timestep, flow),
                condition_features,
                image1,
                image2,
            )
            # Homoscedastic or stub heads return a bare tensor; tolerate both contracts.
            if isinstance(head, tuple):
                head_output, predicted_log_var = head
            else:
                head_output, predicted_log_var = head, None
            predicted_clean = self.bound_clean_residual(head_output)
            sampled_std = None
            if predicted_log_var is not None:
                sampled_std = predicted_log_var.exp().sqrt()
                self.last_aleatoric_std = sampled_std.detach() * self.flow_scale
            predicted_noise = self.predict_noise_from_x0(flow, timestep, predicted_clean)
            guidance = guidance_config or self.posterior_guidance
            progress = float(index + 1) / float(len(timesteps))
            if guidance.active and progress >= float(guidance.start_fraction):
                predicted_full_flow = center_flow + predicted_clean * self.flow_scale
                guided_full_flow, _ = guided_clean_flow(
                    predicted_full_flow / self.flow_scale,
                    image1,
                    image2,
                    self.flow_scale,
                    guidance,
                    geometry=self.posterior_geometry,
                    step_strength=progress / max(len(timesteps), 1),
                )
                predicted_clean = (guided_full_flow * self.flow_scale - center_flow) / self.flow_scale
                predicted_clean = self.bound_clean_residual(predicted_clean)
            if index == len(timesteps) - 1:
                flow = predicted_clean
                if sampled_std is not None and self.variance_injection_scale > 0.0:
                    flow = flow + self.variance_injection_scale * sampled_std * torch.randn_like(flow)
                continue
            previous_value = int(timesteps[index + 1].item())
            alpha_t = self.alpha_bar[int(timestep_value.item())].to(dtype=flow.dtype)
            alpha_previous = self.alpha_bar[previous_value].to(dtype=flow.dtype)
            sigma = self.ddim_eta * torch.sqrt(
                ((1.0 - alpha_previous) / (1.0 - alpha_t)).clamp(min=0.0)
                * (1.0 - alpha_t / alpha_previous).clamp(min=0.0)
            )
            direction = torch.sqrt((1.0 - alpha_previous - sigma.square()).clamp(min=0.0)) * predicted_noise
            stochastic = sigma * torch.randn_like(flow) if self.ddim_eta > 0.0 else 0.0
            flow = torch.sqrt(alpha_previous) * predicted_clean + direction + stochastic
        return center_flow + flow * self.flow_scale

    @torch.no_grad()
    def forward(
        self,
        condition_features: torch.Tensor,
        image1: torch.Tensor,
        image2: torch.Tensor,
        use_ema: bool = True,
        num_candidates: int | None = None,
        sampling_steps: int | None = None,
        guidance_config: PosteriorGuidanceConfig | None = None,
        prior: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.adaptive_policy and num_candidates is None and sampling_steps is None:
            return self.adaptive_sample(
                condition_features, image1, image2, use_ema=use_ema, prior=prior, **self.adaptive_policy
            )
        model = self.ema_unet if use_ema else self.unet
        candidate_count = self.num_candidates if num_candidates is None else int(num_candidates)
        steps = self.sampling_steps if sampling_steps is None else int(sampling_steps)
        center_flow = self.predict_center(condition_features, image1, image2, prior=prior)
        self.last_sampling_metadata = {
            "adaptive": False,
            "candidates_used": candidate_count,
            "sampling_steps_used": steps,
            "parameterization": "residual_around_learned_center",
            "deterministic_initial_noise": initial_noise is not None,
        }
        if initial_noise is not None and int(initial_noise.shape[0]) != candidate_count:
            raise ValueError(
                f"initial_noise carries {int(initial_noise.shape[0])} candidates but "
                f"num_candidates={candidate_count}"
            )
        samples = [
            self.ddim_sample(
                condition_features, image1, image2, model=model, sampling_steps=steps,
                guidance_config=guidance_config,
                center_flow=center_flow,
                initial_noise=None if initial_noise is None else initial_noise[index],
            ).unsqueeze(1)
            for index in range(candidate_count)
        ]
        return torch.cat(samples, dim=1)

    def configure_posterior_guidance(self, config=None, geometry=None):
        self.posterior_guidance = config or PosteriorGuidanceConfig()
        self.posterior_geometry = geometry

    def configure_adaptive_sampling(self, policy=None):
        self.adaptive_policy = dict(policy) if policy else None

    @torch.no_grad()
    def adaptive_sample(
        self,
        condition_features,
        image1,
        image2,
        min_candidates=2,
        max_candidates=None,
        min_steps=4,
        max_steps=None,
        risk_threshold=0.25,
        use_ema=True,
        prior: torch.Tensor | None = None,
    ):
        """Risk-controlled sampling with early exit and a bounded compute budget."""
        maximum_candidates = self.num_candidates if max_candidates is None else int(max_candidates)
        maximum_steps = self.sampling_steps if max_steps is None else int(max_steps)
        count = min(max(2, int(min_candidates)), maximum_candidates)
        steps = min(max(1, int(min_steps)), maximum_steps)
        model = self.ema_unet if use_ema else self.unet
        center_flow = self.predict_center(condition_features, image1, image2, prior=prior)
        candidates = []
        risk = None
        while True:
            while len(candidates) < count:
                candidates.append(self.ddim_sample(
                    condition_features, image1, image2, model=model, sampling_steps=steps,
                    guidance_config=self.posterior_guidance,
                    center_flow=center_flow,
                ).unsqueeze(1))
            stacked = torch.cat(candidates, dim=1)
            risk = posterior_risk(stacked, image1, image2, self.posterior_guidance, self.posterior_geometry)
            if risk.mean().item() <= float(risk_threshold) or (count >= maximum_candidates and steps >= maximum_steps):
                break
            if steps < maximum_steps:
                steps = min(maximum_steps, max(steps + 1, steps * 2))
                candidates = []  # previous paths used a coarser discretization
            else:
                count = min(maximum_candidates, max(count + 1, count * 2))
        self.last_sampling_metadata = {
            "adaptive": True,
            "candidates_used": len(candidates),
            "sampling_steps_used": steps,
            "mean_posterior_risk": float(risk.mean().item()),
            "risk_threshold": float(risk_threshold),
        }
        return stacked


def make_diffusion_checkpoint(
    feature_extractor: nn.Module,
    diffusion_model: ConditionalDiffusionFlowGenerator,
    **extra: Any,
) -> Dict[str, Any]:
    """Serialize a versioned, self-describing backbone checkpoint."""
    return {
        "checkpoint_schema_version": DIFFUSION_CHECKPOINT_SCHEMA_VERSION,
        "model_family": DIFFUSION_MODEL_FAMILY,
        "diffusion_config": diffusion_model.diffusion_config,
        "feature_extractor_state_dict": feature_extractor.state_dict(),
        "diffusion_model_state_dict": diffusion_model.state_dict(),
        **extra,
    }


def _upgrade_legacy_diffusion_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in config keys added after schema 6 for checkpoints whose tensors are unchanged.

    Schema 7 added the optional heteroscedastic head; schema 8 added the optional
    external-prior conditioning channels.  A schema-6 checkpoint was by construction
    homoscedastic (a 2-channel mean head) and a schema-7 checkpoint had no prior
    input, so their implied values are exact rather than guessed: the state dict
    loads with zero missing and zero unexpected keys.
    """
    if checkpoint.get("checkpoint_schema_version") == DIFFUSION_CHECKPOINT_SCHEMA_VERSION:
        return checkpoint
    if checkpoint.get("checkpoint_schema_version") not in (6, 7):
        return checkpoint
    if checkpoint.get("model_family") != DIFFUSION_MODEL_FAMILY:
        return checkpoint
    config = checkpoint.get("diffusion_config")
    if not isinstance(config, dict):
        return checkpoint
    added: Dict[str, Any] = {}
    if "learn_variance" not in config:
        added.update(
            {
                "learn_variance": False,
                "variance_fit_weight": 1.0,
                "prediction_head": "mu_mse",
                "log_var_range": None,
            }
        )
    if "prior_channels" not in config:
        added["prior_channels"] = 0
    if not added:
        return checkpoint
    upgraded = dict(checkpoint)
    upgraded["diffusion_config"] = {**config, **added}
    upgraded["checkpoint_schema_version"] = DIFFUSION_CHECKPOINT_SCHEMA_VERSION
    upgraded["upgraded_from_schema"] = checkpoint.get("checkpoint_schema_version")
    return upgraded


def load_diffusion_checkpoint(
    checkpoint: Dict[str, Any],
    feature_extractor: nn.Module,
    diffusion_model: ConditionalDiffusionFlowGenerator,
) -> Dict[str, Any]:
    """Reject legacy one-step checkpoints instead of silently relabeling them as diffusion models."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Backbone checkpoint must be a dictionary with residual-DDPM metadata.")
    checkpoint = _upgrade_legacy_diffusion_checkpoint(checkpoint)
    schema = checkpoint.get("checkpoint_schema_version")
    family = checkpoint.get("model_family")
    config = checkpoint.get("diffusion_config")
    if schema != DIFFUSION_CHECKPOINT_SCHEMA_VERSION or family != DIFFUSION_MODEL_FAMILY or not isinstance(config, dict):
        raise ValueError(
            "Legacy or unversioned backbone checkpoint rejected: it is not a verified learned-center residual DDPM. "
            "Retrain the backbone with this release before producing paper-facing results."
        )
    if config != diffusion_model.diffusion_config:
        raise ValueError(
            "Diffusion checkpoint configuration does not match the requested model. "
            f"checkpoint={config}, requested={diffusion_model.diffusion_config}"
        )
    required = {"feature_extractor_state_dict", "diffusion_model_state_dict"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Conditional DDPM checkpoint is missing required keys: {sorted(missing)}")
    feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
    diffusion_model.load_state_dict(checkpoint["diffusion_model_state_dict"])
    return {
        "checkpoint_schema_version": schema,
        "model_family": family,
        "diffusion_config": config,
    }
