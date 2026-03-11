"""
Diff2Flow Training Script.

Entry point for training Diff2Flow models. Supports all three objectives
(diffusion, FM, Diff2Flow) and multiple tasks (text2img, img2depth, reflow).

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --task text2img --model sd21_diff2flow --lora lora_base
    python scripts/train.py --task img2depth --data hypersim
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diff2flow.converter import Diff2FlowConverter
from diff2flow.trainer import Diff2FlowTrainer, TrainerConfig
from diff2flow.data.base_dataset import DummyDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Diff2Flow Training")

    # Config
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Task
    parser.add_argument(
        "--task", type=str, default="text2img",
        choices=["text2img", "img2depth", "reflow"],
        help="Task type",
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="sd21_diff2flow",
        choices=["sd21_diffusion", "sd21_fm", "sd21_diff2flow"],
        help="Model configuration",
    )
    parser.add_argument(
        "--model_id", type=str, default="stabilityai/stable-diffusion-2-1",
        help="HuggingFace model ID",
    )

    # LoRA
    parser.add_argument(
        "--lora", type=str, default=None,
        choices=["lora_base", "lora_small", None],
        help="LoRA configuration",
    )

    # Data
    parser.add_argument("--data", type=str, default=None, help="Data config or path")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Training
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=20000, help="Training iterations")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")

    # Misc
    parser.add_argument("--dummy_data", action="store_true", help="Use dummy dataset for testing")
    parser.add_argument("--dry_run", action="store_true", help="Run one step and exit")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")

    return parser.parse_args()


def get_objective(model_name: str) -> str:
    """Map model config name to objective type."""
    mapping = {
        "sd21_diffusion": "diffusion",
        "sd21_fm": "fm",
        "sd21_diff2flow": "diff2flow",
    }
    return mapping.get(model_name, "diff2flow")


def setup_model(args):
    """Initialize the model based on configuration."""
    from diff2flow.model import Diff2FlowModel

    in_channels = 8 if args.task == "img2depth" else 4
    parameterization = "v"  # SD 2.1 uses v-prediction

    model = Diff2FlowModel(
        model_id=args.model_id,
        in_channels=in_channels,
        parameterization=parameterization,
        use_fp16=args.fp16,
        device=args.device if args.device != "auto" else "cuda",
    )

    logger.info(f"Loading pre-trained model: {args.model_id}")
    model.load_pretrained()

    # Apply LoRA if requested
    if args.lora:
        from diff2flow.lora import apply_lora
        rank = "base" if args.lora == "lora_base" else "small"
        model = apply_lora(model, rank=rank)
        logger.info(f"Applied LoRA with rank preset: {rank}")

    # Move to device
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, device


def setup_data(args):
    """Initialize the dataset based on configuration."""
    if args.dummy_data or args.data is None:
        logger.info("Using dummy dataset for testing")
        has_context = args.task == "img2depth"
        dataset = DummyDataset(
            num_samples=100,
            has_context=has_context,
        )
    elif args.task == "img2depth":
        from diff2flow.data.depth_dataset import DepthDataset
        dataset = DepthDataset(root_dir=args.data)
    elif args.task == "reflow":
        from diff2flow.data.reflow_dataset import ReflowDataset
        dataset = ReflowDataset(root_dir=args.data)
    else:
        from diff2flow.data.text_image_dataset import TextImageDataset
        dataset = TextImageDataset(root_dir=args.data)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(f"Dataset loaded: {len(dataset)} samples, batch_size={args.batch_size}")
    return dataloader


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Diff2Flow Training")
    logger.info(f"  Task:       {args.task}")
    logger.info(f"  Model:      {args.model}")
    logger.info(f"  Objective:  {get_objective(args.model)}")
    logger.info(f"  LoRA:       {args.lora or 'None (full finetune)'}")
    logger.info(f"  Iterations: {args.iterations}")
    logger.info("=" * 60)

    # Setup
    model, device = setup_model(args)
    dataloader = setup_data(args)

    # Training config
    config = TrainerConfig(
        objective=get_objective(args.model),
        learning_rate=args.lr,
        num_iterations=1 if args.dry_run else args.iterations,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        use_fp16=args.fp16,
        output_dir=args.output_dir,
    )

    # Converter
    converter = Diff2FlowConverter(parameterization="v") if config.objective == "diff2flow" else None

    # Trainer
    trainer = Diff2FlowTrainer(model=model, config=config, converter=converter)

    # Train
    trainer.train(dataloader)

    if args.dry_run:
        logger.info("Dry run complete — one training step executed successfully.")
    else:
        logger.info(f"Training complete. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
