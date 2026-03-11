"""
Depth Estimation Evaluation Script.

Evaluates a trained Diff2Flow depth model on standard benchmarks
(NYUv2, KITTI, DIODE, ScanNet, ETH3D).

Usage:
    python scripts/evaluate_depth.py --checkpoint path/to/checkpoint --dataset nyuv2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None):
    """Compute standard depth estimation metrics.

    Args:
        pred: Predicted depth, shape [H, W].
        gt: Ground truth depth, shape [H, W].
        mask: Optional valid pixel mask.

    Returns:
        Dictionary of metrics: AbsRel, delta1, RMSE.
    """
    if mask is None:
        mask = (gt > 0) & np.isfinite(gt)

    pred = pred[mask]
    gt = gt[mask]

    if len(pred) == 0:
        return {"abs_rel": float("nan"), "delta1": float("nan"), "rmse": float("nan")}

    # Affine-invariant alignment (least squares)
    # Solve: pred_aligned = scale * pred + shift
    # Minimize ||scale * pred + shift - gt||^2
    A = np.stack([pred, np.ones_like(pred)], axis=1)
    result = np.linalg.lstsq(A, gt, rcond=None)
    scale, shift = result[0]
    pred_aligned = scale * pred + shift

    # AbsRel
    abs_rel = np.mean(np.abs(pred_aligned - gt) / gt.clip(min=1e-8))

    # delta1: % pixels where max(pred/gt, gt/pred) < 1.25
    ratio = np.maximum(pred_aligned / gt.clip(min=1e-8), gt / pred_aligned.clip(min=1e-8))
    delta1 = np.mean(ratio < 1.25)

    # RMSE
    rmse = np.sqrt(np.mean((pred_aligned - gt) ** 2))

    return {"abs_rel": abs_rel, "delta1": delta1, "rmse": rmse}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Depth Estimation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="nyuv2",
                        choices=["nyuv2", "kitti", "diode", "scannet", "eth3d"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--ensemble_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Evaluating on {args.dataset} with {args.num_steps} steps, ensemble={args.ensemble_size}")

    # This is a template — full implementation requires dataset-specific loading
    logger.info(
        "Note: This script provides the evaluation framework. "
        "Dataset-specific loading must be configured for your local setup. "
        "See README.md for dataset download instructions."
    )


if __name__ == "__main__":
    main()
