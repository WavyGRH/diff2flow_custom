"""
Text-Image Dataset for Text-to-Image Training.

Supports loading text-image pairs from filesystem or HuggingFace datasets.
Used for text-to-image finetuning and resolution adaptation experiments.

Expected directory structure:
    root_dir/
    ├── images/
    │   ├── 000001.png
    │   ├── 000002.png
    │   └── ...
    └── captions.json  (or captions.csv)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from .base_dataset import BaseDataset


class TextImageDataset(BaseDataset):
    """Dataset for text-image pairs (LAION-Aesthetics style).

    Args:
        root_dir: Root directory containing images/ and captions.
        resolution: Target image resolution.
        caption_file: Name of caption file ('captions.json' or 'captions.csv').
    """

    def __init__(
        self,
        root_dir: str,
        resolution: tuple[int, int] = (512, 512),
        caption_file: str = "captions.json",
        **kwargs,
    ):
        self.caption_file = caption_file
        super().__init__(root_dir, resolution)

    def _load_samples(self) -> list:
        """Load image paths and their captions."""
        caption_path = self.root_dir / self.caption_file

        if caption_path.exists():
            if caption_path.suffix == ".json":
                with open(caption_path) as f:
                    data = json.load(f)
                # Expect list of {"image": "path", "caption": "text"}
                return data
            else:
                # Simple format: one caption per line, image name derived from line number
                lines = caption_path.read_text().splitlines()
                return [
                    {"image": f"{i:06d}.png", "caption": line}
                    for i, line in enumerate(lines)
                ]
        else:
            # No captions — use empty strings
            image_dir = self.root_dir / "images"
            if image_dir.exists():
                images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
                return [{"image": str(p.name), "caption": ""} for p in images]
            return []

    def __getitem__(self, idx: int) -> dict:
        sample_info = self.samples[idx]
        image_path = self.root_dir / "images" / sample_info["image"]
        caption = sample_info.get("caption", "")

        # Load and preprocess image
        image = self.load_image(image_path)

        return {
            "latent": image,  # Will be encoded to latent by the training pipeline
            "text": caption,
        }


class HFTextImageDataset(Dataset):
    """Text-Image dataset loaded from HuggingFace datasets.

    Wraps a HuggingFace dataset (e.g., 'laion/laion2B-en-aesthetic')
    for use with the training pipeline.

    Args:
        dataset_name: HuggingFace dataset name.
        split: Dataset split ('train', 'validation', etc.).
        resolution: Target image resolution.
        max_samples: Maximum number of samples to use (None for all).
        image_column: Column name for images.
        caption_column: Column name for captions.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        resolution: tuple[int, int] = (512, 512),
        max_samples: Optional[int] = None,
        image_column: str = "image",
        caption_column: str = "text",
    ):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        self.resolution = resolution
        self.image_column = image_column
        self.caption_column = caption_column

        dataset = load_dataset(dataset_name, split=split, streaming=False)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        import numpy as np
        from PIL import Image

        item = self.dataset[idx]
        img = item[self.image_column]

        if isinstance(img, str):
            img = Image.open(img)

        img = img.convert("RGB").resize(
            (self.resolution[1], self.resolution[0]), Image.LANCZOS
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0

        caption = item.get(self.caption_column, "")

        return {
            "latent": tensor,
            "text": caption if isinstance(caption, str) else "",
        }
