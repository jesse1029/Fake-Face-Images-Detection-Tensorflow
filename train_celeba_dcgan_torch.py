"""PyTorch training script for the CelebA DCGAN baseline.

This script mirrors the behaviour of :mod:`train_celeba_dcgan` but uses the
PyTorch models defined in :mod:`models_64x64_torch`.  The defaults are chosen
so that existing datasets can be reused without additional preprocessing.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils as vutils

from models_64x64_torch import DCGANConfig, Discriminator, Generator, weights_init


class ImageFolderFlat(Dataset):
    """Load images from a directory without requiring class sub-folders."""

    def __init__(self, root: str, extensions: Tuple[str, ...] = (".jpg", ".png"), transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.paths = sorted(
            path
            for ext in extensions
            for path in self.root.glob(f"**/*{ext}")
            if path.is_file()
        )
        if not self.paths:
            raise RuntimeError(
                f"No images with extensions {extensions} were found under {self.root}."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_dataloader(data_root: str, batch_size: int, workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.CenterCrop(108),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = ImageFolderFlat(data_root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    loader = build_dataloader(args.data_root, args.batch_size, args.workers)
    config = DCGANConfig(latent_dim=args.latent_dim, base_channels=args.channels)

    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    optim_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(args.sample_grid ** 2, args.latent_dim, device=device)

    global_step = 0
    for epoch in range(args.epochs):
        for batch_idx, real in enumerate(loader):
            real = real.to(device)
            bsz = real.size(0)

            # --- Train discriminator ---
            optim_d.zero_grad(set_to_none=True)
            noise = torch.randn(bsz, args.latent_dim, device=device)
            fake = generator(noise).detach()

            logits_real = discriminator(real)
            logits_fake = discriminator(fake)

            labels_real = torch.ones_like(logits_real)
            labels_fake = torch.zeros_like(logits_fake)

            loss_real = criterion(logits_real, labels_real)
            loss_fake = criterion(logits_fake, labels_fake)
            loss_d = (loss_real + loss_fake) * 0.5
            loss_d.backward()
            optim_d.step()

            # --- Train generator ---
            optim_g.zero_grad(set_to_none=True)
            noise = torch.randn(bsz, args.latent_dim, device=device)
            fake = generator(noise)
            logits_fake = discriminator(fake)
            labels_for_g = torch.ones_like(logits_fake)
            loss_g = criterion(logits_fake, labels_for_g)
            loss_g.backward()
            optim_g.step()

            if global_step % args.log_interval == 0:
                print(
                    f"Epoch {epoch + 1:03d}/{args.epochs} | Batch {batch_idx + 1:04d}/{len(loader):04d} "
                    f"| D loss: {loss_d.item():.4f} | G loss: {loss_g.item():.4f}"
                )

            if global_step % args.sample_interval == 0:
                save_samples(generator, fixed_noise, args.sample_dir, args.sample_grid, global_step)

            if global_step % args.checkpoint_interval == 0:
                save_checkpoint(
                    generator,
                    discriminator,
                    optim_g,
                    optim_d,
                    args.checkpoint_dir,
                    epoch,
                    global_step,
                )

            global_step += 1

    print("Training finished.")


def save_samples(
    generator: Generator, noise: torch.Tensor, directory: str, grid_size: int, step: int
) -> None:
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        samples = generator(noise).cpu()
    generator.train()
    grid = vutils.make_grid(samples, nrow=grid_size, normalize=True, value_range=(-1, 1))
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = directory_path / f"step_{step:07d}_{timestamp}.png"
    vutils.save_image(grid, out_path)


def save_checkpoint(
    generator: Generator,
    discriminator: Discriminator,
    optim_g: optim.Optimizer,
    optim_d: optim.Optimizer,
    directory: str,
    epoch: int,
    step: int,
) -> None:
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
        },
        directory_path / f"dcgan_{step:07d}.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch DCGAN trainer for CelebA")
    parser.add_argument("--data-root", default="./data/img_align_celeba/img_align_celeba", help="Path to the CelebA image directory.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--latent-dim", type=int, default=100, help="Dimension of the latent vector z.")
    parser.add_argument("--channels", type=int, default=64, help="Base number of convolution channels.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for both optimizers.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader worker processes.")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available.")
    parser.add_argument("--log-interval", type=int, default=50, help="Steps between logging training metrics.")
    parser.add_argument("--sample-interval", type=int, default=500, help="Steps between writing sample grids.")
    parser.add_argument("--sample-grid", type=int, default=8, help="Number of images per row/column when saving samples.")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Steps between writing checkpoints.")
    parser.add_argument("--sample-dir", default="./sample_images_while_training/celeba_dcgan_torch", help="Directory to store generated samples.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/celeba_dcgan_torch", help="Directory to store checkpoints.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
