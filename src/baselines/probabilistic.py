"""Matched-capacity point, ensemble, and VAE baselines for flow posterior studies."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FlowDecoder(nn.Module):
    def __init__(self, input_channels, width=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv2d(width, 2, 3, padding=1),
        )

    def forward(self, value):
        return self.network(value)


class SinglePointFlowRegressor(nn.Module):
    """Deterministic matched-input baseline; repeated samples are intentionally identical."""
    def __init__(self, condition_channels=128):
        super().__init__()
        self.decoder = _FlowDecoder(condition_channels + 2)

    def forward(self, condition, image1, image2, num_candidates=None):
        flow = self.decoder(torch.cat([condition, image1, image2], dim=1))
        if num_candidates is None:
            return flow
        return flow.unsqueeze(1).expand(-1, int(num_candidates), -1, -1, -1)


class DeepFlowEnsemble(nn.Module):
    """Independent deterministic predictors separating ensemble from diffusion gains."""
    def __init__(self, members=4, condition_channels=128):
        super().__init__()
        if members < 2:
            raise ValueError("DeepFlowEnsemble requires at least two members")
        self.members = nn.ModuleList([SinglePointFlowRegressor(condition_channels) for _ in range(int(members))])

    def forward(self, condition, image1, image2):
        return torch.stack([member(condition, image1, image2) for member in self.members], dim=1)


class ConditionalFlowVAE(nn.Module):
    """Conditional VAE baseline with a spatial latent field for multimodal flow hypotheses."""
    def __init__(self, condition_channels=128, latent_channels=8):
        super().__init__()
        self.latent_channels = int(latent_channels)
        encoder_channels = condition_channels + 4
        self.encoder = nn.Sequential(
            nn.Conv2d(encoder_channels, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 2 * self.latent_channels, 3, padding=1),
        )
        self.decoder = _FlowDecoder(condition_channels + 2 + self.latent_channels)

    def encode(self, condition, image1, image2, flow):
        moments = self.encoder(torch.cat([condition, image1, image2, flow], dim=1))
        return moments.chunk(2, dim=1)

    def decode(self, condition, image1, image2, latent):
        return self.decoder(torch.cat([condition, image1, image2, latent], dim=1))

    def loss(self, condition, image1, image2, target, beta=1e-3):
        mean, logvar = self.encode(condition, image1, image2, target)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        reconstruction = self.decode(condition, image1, image2, latent)
        kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).mean()
        return F.smooth_l1_loss(reconstruction, target) + float(beta) * kl, {"kl": kl, "reconstruction": reconstruction}

    def forward(self, condition, image1, image2, num_candidates=4):
        shape = (condition.shape[0], self.latent_channels, *condition.shape[-2:])
        return torch.stack([
            self.decode(condition, image1, image2, torch.randn(shape, device=condition.device, dtype=condition.dtype))
            for _ in range(int(num_candidates))
        ], dim=1)
