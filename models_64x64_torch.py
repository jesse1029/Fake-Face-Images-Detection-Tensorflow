"""PyTorch implementation of the 64x64 generator and discriminator networks.

This module mirrors the TensorFlow version in :mod:`models_64x64` but uses
PyTorch's ``nn.Module`` classes so it can be integrated with a PyTorch
training pipeline.  The architecture follows the standard DCGAN design that
is used throughout the original project.  It intentionally keeps the API
minimal so the modules can be reused in standalone scripts or notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DCGANConfig:
    """Configuration describing the DCGAN model layout."""

    latent_dim: int = 100
    base_channels: int = 64
    image_channels: int = 3


class Generator(nn.Module):
    """Generator network that maps latent vectors to 64x64 RGB images."""

    def __init__(self, config: DCGANConfig = DCGANConfig()) -> None:
        super().__init__()
        self.config = config

        self.project = nn.Sequential(
            nn.Linear(config.latent_dim, 4 * 4 * config.base_channels * 8, bias=False),
            nn.BatchNorm1d(config.base_channels * 8 * 4 * 4),
            nn.ReLU(True),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(config.base_channels * 8, config.base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(config.base_channels * 4, config.base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(config.base_channels * 2, config.base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(config.base_channels, config.image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.size(1) != self.config.latent_dim:
            raise ValueError(
                f"Expected latent vectors of shape (batch, {self.config.latent_dim}), "
                f"got {tuple(z.shape)}"
            )
        x = self.project(z)
        x = x.view(z.size(0), self.config.base_channels * 8, 4, 4)
        return self.net(x)


class Discriminator(nn.Module):
    """Discriminator network that scores 64x64 RGB images."""

    def __init__(self, config: DCGANConfig = DCGANConfig()) -> None:
        super().__init__()
        self.config = config

        def block(in_channels: int, out_channels: int, normalize: bool = True) -> nn.Sequential:
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(config.image_channels, config.base_channels, normalize=False),
            block(config.base_channels, config.base_channels * 2),
            block(config.base_channels * 2, config.base_channels * 4),
            block(config.base_channels * 4, config.base_channels * 8),
        )

        self.head = nn.Conv2d(config.base_channels * 8, 1, 4, 1, 0, bias=False)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() != 4 or img.size(1) != self.config.image_channels:
            raise ValueError(
                f"Expected images of shape (batch, {self.config.image_channels}, 64, 64), "
                f"got {tuple(img.shape)}"
            )
        features = self.net(img)
        logits = self.head(features).view(img.size(0))
        return logits


def weights_init(module: nn.Module) -> None:
    """Initialize model weights following the original DCGAN recipe."""

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


__all__ = [
    "DCGANConfig",
    "Generator",
    "Discriminator",
    "weights_init",
]
