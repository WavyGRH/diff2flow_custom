"""
Base Dataset Class.

Abstract dataset interface used by all task-specific datasets.
Handles common operations like image loading, resizing, normalization,
and latent encoding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class BaseDataset(Dataset, ABC):
    """Abstract base class for Diff2Flow datasets.

    All datasets return dictionaries with at least:
    - 'latent': Image latent code, shape [C, H, W]

    Optional keys depending on the task:
    - 'encoder_hidden_states': Text embedding, shape [seq_len, dim]
    - 'context': Additional context (e.g., depth, image), shape [C, H, W]
    - 'text': Raw text caption (string)

    Args:
        root_dir: Root directory of the dataset.
        resolution: Target image resolution (H, W).
        transform: Optional additional transforms.
    """

    def __init__(
        self,
        root_dir: str,
        resolution: tuple[int, int] = (512, 512),
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.resolution = resolution
        self.transform = transform
        self.samples = self._load_samples()

    @abstractmethod
    def _load_samples(self) -> list:
        """Load and return the list of sample paths/metadata."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample as a dictionary."""
        ...

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, path: str | Path) -> torch.Tensor:
        """Load and preprocess an image.

        Returns:
            Tensor of shape [3, H, W], values in [-1, 1].
        """
        img = Image.open(path).convert("RGB")
        img = img.resize((self.resolution[1], self.resolution[0]), Image.LANCZOS)

        # Convert to tensor in [-1, 1]
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        tensor = tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        return tensor

    def load_depth(self, path: str | Path, log_normalize: bool = True) -> torch.Tensor:
        """Load and preprocess a depth map.

        Args:
            path: Path to depth map (NPY, PNG, or EXR).
            log_normalize: If True, apply log normalization as per paper.

        Returns:
            Tensor of shape [1, H, W], values in [-1, 1] (after normalization).
        """
        path = Path(path)

        if path.suffix == ".npy":
            depth = np.load(str(path)).astype(np.float32)
        elif path.suffix in (".png", ".jpg"):
            depth = np.array(Image.open(path).convert("L"), dtype=np.float32)
        else:
            depth = np.load(str(path)).astype(np.float32)

        # Resize
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize(
            (self.resolution[1], self.resolution[0]), Image.NEAREST
        )
        depth = np.array(depth_img, dtype=np.float32)

        if log_normalize:
            # Log normalization as used in DepthFM / Diff2Flow
            depth = np.log(depth.clip(min=1e-8))
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = depth * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        else:
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = depth * 2.0 - 1.0

        tensor = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        return tensor


class DummyDataset(BaseDataset):
    """Dummy dataset for testing and development.

    Generates random latent codes without requiring any actual data files.

    Args:
        num_samples: Number of random samples.
        latent_channels: Number of latent channels (4 for SD).
        latent_size: Spatial size of latents (e.g., 64 for 512px images).
        has_context: Whether to include a context tensor (for depth/img2img).
        embed_dim: Dimension of text embeddings.
    """

    def __init__(
        self,
        num_samples: int = 100,
        latent_channels: int = 4,
        latent_size: int = 64,
        has_context: bool = False,
        embed_dim: int = 1024,
        **kwargs,
    ):
        self.num_samples = num_samples
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.has_context = has_context
        self.embed_dim = embed_dim
        self.samples = list(range(num_samples))

    def _load_samples(self) -> list:
        return list(range(self.num_samples))

    def __getitem__(self, idx: int) -> dict:
        # Use fixed seed for reproducibility within each sample
        rng = torch.Generator().manual_seed(idx)

        sample = {
            "latent": torch.randn(
                self.latent_channels, self.latent_size, self.latent_size,
                generator=rng,
            ),
            "encoder_hidden_states": torch.randn(77, self.embed_dim, generator=rng),
        }

        if self.has_context:
            sample["context"] = torch.randn(
                self.latent_channels, self.latent_size, self.latent_size,
                generator=rng,
            )

        return sample
