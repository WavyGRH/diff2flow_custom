"""
Depth Dataset for Monocular Depth Estimation.

Supports Hypersim and Virtual KITTI v2 datasets for training Diff2Flow
on image-to-depth tasks. The depth maps are log-normalized as per the
paper's methodology.

Expected directory structure (Hypersim-style):
    root_dir/
    ├── images/
    │   ├── scene_001_frame_0001.png
    │   └── ...
    └── depths/
        ├── scene_001_frame_0001.npy
        └── ...
"""

from __future__ import annotations

from pathlib import Path  
from typing import Optional

import torch

from .base_dataset import BaseDataset


class DepthDataset(BaseDataset):
    """Dataset for image-depth pairs (Hypersim / Virtual KITTI v2 style).

    Each sample contains an RGB image and its corresponding depth map.
    The depth is log-normalized as described in the paper.

    Args:
        root_dir: Root directory containing images/ and depths/.
        resolution: Target resolution (H, W). Paper uses (384, 512).
        log_normalize: Whether to apply log normalization to depth.
    """

    def __init__(
        self,
        root_dir: str,
        resolution: tuple[int, int] = (384, 512),
        log_normalize: bool = True,
        **kwargs,
    ):
        self.log_normalize = log_normalize
        super().__init__(root_dir, resolution)

    def _load_samples(self) -> list:
        """Load paired image-depth samples."""
        image_dir = self.root_dir / "images"
        depth_dir = self.root_dir / "depths"

        if not image_dir.exists() or not depth_dir.exists():
            return []

        samples = []
        for img_path in sorted(image_dir.glob("*")):
            if img_path.suffix in (".png", ".jpg", ".jpeg"):
                # Find corresponding depth
                depth_path = depth_dir / (img_path.stem + ".npy")
                if not depth_path.exists():
                    depth_path = depth_dir / (img_path.stem + ".png")
                if depth_path.exists():
                    samples.append({
                        "image": str(img_path),
                        "depth": str(depth_path),
                    })

        return samples

    def __getitem__(self, idx: int) -> dict:
        sample_info = self.samples[idx]

        image = self.load_image(sample_info["image"])
        depth = self.load_depth(sample_info["depth"], log_normalize=self.log_normalize)

        # For image-to-depth: image is the context (concat), depth is the target
        # The model input is 8 channels: [noise_latent, image_latent]
        # Target: depth latent
        return {
            "latent": depth.repeat(3, 1, 1),  # Expand depth to 3 channels for VAE
            "context": image,                   # Image as conditioning context
        }


class MixedDepthDataset(torch.utils.data.ConcatDataset):
    """Mixed dataset combining Hypersim and Virtual KITTI v2.

    As described in the paper, the depth model is trained on a mixture
    of synthetic datasets.

    Args:
        hypersim_dir: Path to Hypersim dataset.
        vkitti_dir: Path to Virtual KITTI v2 dataset.
        resolution: Target resolution.
    """

    def __init__(
        self,
        hypersim_dir: str,
        vkitti_dir: str,
        resolution: tuple[int, int] = (384, 512),
    ):
        datasets = []

        if Path(hypersim_dir).exists():
            datasets.append(DepthDataset(hypersim_dir, resolution))

        if Path(vkitti_dir).exists():
            datasets.append(DepthDataset(vkitti_dir, resolution))

        if not datasets:
            raise FileNotFoundError(
                f"Neither Hypersim ({hypersim_dir}) nor "
                f"Virtual KITTI ({vkitti_dir}) datasets found."
            )

        super().__init__(datasets)
