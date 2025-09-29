"""Utility to convert TensorFlow DCGAN checkpoints into PyTorch format.

The original project ships TensorFlow checkpoints trained with the
``models_64x64`` DCGAN architecture.  This tool rebuilds that graph,
loads a provided checkpoint, and translates the weights into the
PyTorch modules defined in :mod:`models_64x64_torch` so that the weights
can be reused with the new training and sampling scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import numpy as np

try:
    import tensorflow as tf  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard for optional dep
    raise SystemExit(
        "TensorFlow is required to run this conversion script. Install tensorflow>=1.10."
    ) from exc

import models_64x64 as tf_models
from models_64x64_torch import DCGANConfig, Discriminator, Generator


if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
    tf1 = tf.compat.v1
    tf1.disable_eager_execution()
else:  # pragma: no cover - TensorFlow 1.x fallback
    tf1 = tf


def _load_tensorflow_checkpoint(
    checkpoint: str, latent_dim: int, base_channels: int
) -> Dict[str, np.ndarray]:
    """Restore the TensorFlow DCGAN graph and collect variable values."""

    tf1.reset_default_graph()
    z = tf1.placeholder(tf.float32, shape=[None, latent_dim], name="z")
    x = tf1.placeholder(tf.float32, shape=[None, 64, 64, 3], name="x")

    with tf1.variable_scope(tf1.get_variable_scope()):
        tf_models.generator(z, dim=base_channels, reuse=False, training=False)
        tf_models.discriminator(x, dim=base_channels, reuse=False, training=False)

    saver = tf1.train.Saver()
    with tf1.Session() as sess:
        saver.restore(sess, checkpoint)
        values = {var.name: sess.run(var) for var in tf1.global_variables()}
    return values


def _get_var(values: Dict[str, np.ndarray], name: str) -> np.ndarray:
    try:
        return values[name]
    except KeyError as exc:
        raise KeyError(f"TensorFlow variable '{name}' was not found in the checkpoint") from exc


def _assign_batch_norm(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    values: Dict[str, np.ndarray],
    tf_prefix: str,
) -> None:
    state_dict[f"{prefix}.weight"] = torch.from_numpy(_get_var(values, f"{tf_prefix}/gamma:0")).float()
    state_dict[f"{prefix}.bias"] = torch.from_numpy(_get_var(values, f"{tf_prefix}/beta:0")).float()
    state_dict[f"{prefix}.running_mean"] = torch.from_numpy(
        _get_var(values, f"{tf_prefix}/moving_mean:0")
    ).float()
    state_dict[f"{prefix}.running_var"] = torch.from_numpy(
        _get_var(values, f"{tf_prefix}/moving_variance:0")
    ).float()


def _transpose_filter(weights: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(weights.transpose(3, 2, 0, 1)).float()


def _load_generator_from_tf(generator: Generator, values: Dict[str, np.ndarray]) -> None:
    state_dict = generator.state_dict()

    fc_weights = _get_var(values, "generator/flatten_fully_connected/fully_connected/weights:0")
    state_dict["project.linear.weight"] = torch.from_numpy(fc_weights.T).float()
    _assign_batch_norm(
        state_dict,
        "project.bn",
        values,
        "generator/flatten_fully_connected/BatchNorm",
    )

    for idx, layer_name in enumerate(["", "_1", "_2"]):
        weight = _get_var(values, f"generator/conv2d_transpose{layer_name}/weights:0")
        state_dict[f"upsamplers.deconv{idx + 1}.weight"] = _transpose_filter(weight)
        _assign_batch_norm(
            state_dict,
            f"upsamplers.bn{idx + 1}",
            values,
            f"generator/conv2d_transpose{layer_name}/BatchNorm",
        )

    weight = _get_var(values, "generator/conv2d_transpose_3/weights:0")
    state_dict["upsamplers.deconv4.weight"] = _transpose_filter(weight)
    state_dict["upsamplers.deconv4.bias"] = torch.from_numpy(
        _get_var(values, "generator/conv2d_transpose_3/biases:0")
    ).float()

    generator.load_state_dict(state_dict)


def _load_discriminator_from_tf(
    discriminator: Discriminator, values: Dict[str, np.ndarray]
) -> None:
    state_dict = discriminator.state_dict()

    weight = _get_var(values, "discriminator/conv2d/weights:0")
    state_dict["features.conv1.weight"] = _transpose_filter(weight)
    state_dict["features.conv1.bias"] = torch.from_numpy(
        _get_var(values, "discriminator/conv2d/biases:0")
    ).float()

    for idx, layer_name in enumerate(["_1", "_2", "_3"], start=2):
        weight = _get_var(values, f"discriminator/conv2d{layer_name}/weights:0")
        state_dict[f"features.conv{idx}.weight"] = _transpose_filter(weight)
        _assign_batch_norm(
            state_dict,
            f"features.bn{idx}",
            values,
            f"discriminator/conv2d{layer_name}/BatchNorm",
        )

    fc_weights = _get_var(values, "discriminator/flatten_fully_connected/fully_connected/weights:0")
    state_dict["head.weight"] = torch.from_numpy(fc_weights.T).float()
    state_dict["head.bias"] = torch.from_numpy(
        _get_var(values, "discriminator/flatten_fully_connected/fully_connected/biases:0")
    ).float()

    discriminator.load_state_dict(state_dict)


def convert_checkpoint(args: argparse.Namespace) -> None:
    values = _load_tensorflow_checkpoint(args.tf_checkpoint, args.latent_dim, args.channels)
    config = DCGANConfig(latent_dim=args.latent_dim, base_channels=args.channels, image_channels=args.image_channels)

    generator = Generator(config)
    discriminator = Discriminator(config)

    _load_generator_from_tf(generator, values)
    _load_discriminator_from_tf(discriminator, values)

    payload = {
        "config": config.__dict__,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved PyTorch checkpoint to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TensorFlow CelebA DCGAN checkpoint to PyTorch format")
    parser.add_argument("tf_checkpoint", help="Path to the TensorFlow checkpoint (e.g. checkpoints/model-100000)")
    parser.add_argument("output", help="Destination .pt file for the converted checkpoint")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent dimension used during training")
    parser.add_argument("--channels", type=int, default=64, help="Base channel width used in the model")
    parser.add_argument("--image-channels", type=int, default=3, help="Number of image colour channels")
    return parser.parse_args()


if __name__ == "__main__":
    convert_checkpoint(parse_args())
