"""PyTorch implementation of the 64x64 generator and discriminator networks.

This module mirrors the TensorFlow version in :mod:`models_64x64` but uses
PyTorch's ``nn.Module`` classes so it can be integrated with a PyTorch
training pipeline.  The architecture follows the standard DCGAN design that
is used throughout the original project.  It intentionally keeps the API
minimal so the modules can be reused in standalone scripts or notebooks.
"""
from __future__ import annotations

from collections import OrderedDict
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
            OrderedDict(
                [
                    (
                        "linear",
                        nn.Linear(
                            config.latent_dim,
                            4 * 4 * config.base_channels * 8,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm1d(config.base_channels * 8 * 4 * 4)),
                    ("relu", nn.ReLU(True)),
                ]
            )
        )

        self.upsamplers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "deconv1",
                        nn.ConvTranspose2d(
                            config.base_channels * 8,
                            config.base_channels * 4,
                            kernel_size=5,
                            stride=2,
                            padding=2,
                            output_padding=1,
                            bias=False,
                        ),
                    ),
                    ("bn1", nn.BatchNorm2d(config.base_channels * 4)),
                    ("relu1", nn.ReLU(True)),
                    (
                        "deconv2",
                        nn.ConvTranspose2d(
                            config.base_channels * 4,
                            config.base_channels * 2,
                            kernel_size=5,
                            stride=2,
                            padding=2,
                            output_padding=1,
                            bias=False,
                        ),
                    ),
                    ("bn2", nn.BatchNorm2d(config.base_channels * 2)),
                    ("relu2", nn.ReLU(True)),
                    (
                        "deconv3",
                        nn.ConvTranspose2d(
                            config.base_channels * 2,
                            config.base_channels,
                            kernel_size=5,
                            stride=2,
                            padding=2,
                            output_padding=1,
                            bias=False,
                        ),
                    ),
                    ("bn3", nn.BatchNorm2d(config.base_channels)),
                    ("relu3", nn.ReLU(True)),
                    (
                        "deconv4",
                        nn.ConvTranspose2d(
                            config.base_channels,
                            config.image_channels,
                            kernel_size=5,
                            stride=2,
                            padding=2,
                            output_padding=1,
                            bias=True,
                        ),
                    ),
                    ("tanh", nn.Tanh()),
                ]
            )
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.size(1) != self.config.latent_dim:
            raise ValueError(
                f"Expected latent vectors of shape (batch, {self.config.latent_dim}), "
                f"got {tuple(z.shape)}"
            )
        x = self.project(z)
        x = x.view(z.size(0), self.config.base_channels * 8, 4, 4)
        return self.upsamplers(x)


class Discriminator(nn.Module):
    """Discriminator network that scores 64x64 RGB images."""

    def __init__(self, config: DCGANConfig = DCGANConfig()) -> None:
        super().__init__()
        self.config = config

        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            config.image_channels,
                            config.base_channels,
                            kernel_size=5,
                            stride=2,
                            padding=2,
                            bias=True,
                        ),
                    ),
                    ("lrelu1", nn.LeakyReLU(0.2, inplace=True)),
                    (
                        "conv2",
                        nn.Conv2d(
                            config.base_channels,
                            config.base_channels * 2,
                            kernel_size=5,
                            stride=2,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    ("bn2", nn.BatchNorm2d(config.base_channels * 2)),
                    ("lrelu2", nn.LeakyReLU(0.2, inplace=True)),
                    (
                        "conv3",
                        nn.Conv2d(
                            config.base_channels * 2,
                            config.base_channels * 4,
                            kernel_size=5,
                            stride=2,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    ("bn3", nn.BatchNorm2d(config.base_channels * 4)),
                    ("lrelu3", nn.LeakyReLU(0.2, inplace=True)),
                    (
                        "conv4",
                        nn.Conv2d(
                            config.base_channels * 4,
                            config.base_channels * 8,
                            kernel_size=5,
                            stride=2,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    ("bn4", nn.BatchNorm2d(config.base_channels * 8)),
                    ("lrelu4", nn.LeakyReLU(0.2, inplace=True)),
                ]
            )
        )

        self.head = nn.Linear(config.base_channels * 8 * 4 * 4, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() != 4 or img.size(1) != self.config.image_channels:
            raise ValueError(
                f"Expected images of shape (batch, {self.config.image_channels}, 64, 64), "
                f"got {tuple(img.shape)}"
            )
        features = self.features(img)
        logits = self.head(features.view(img.size(0), -1)).view(img.size(0))
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
