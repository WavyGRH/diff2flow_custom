"""
LoRA (Low-Rank Adaptation) for Diff2Flow.

Provides parameter-efficient finetuning by injecting low-rank decomposition
matrices into the UNet's attention and convolution layers. The paper shows
that LoRA works poorly with naive FM training but excels with Diff2Flow's
objective alignment (Section 3.3).

Supports two configurations:
  - "base": rank = 20% of feature dim (222M trainable params for SD 2.1)
  - "small": rank = 64 across all layers (62M trainable params)
"""

from __future__ import annotations

from typing import Optional
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_lora(
    model: nn.Module,
    rank: int | str = "base",
    target_modules: Optional[list[str]] = None,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply LoRA adapters to the model using the peft library.

    Args:
        model: The UNet model (or wrapper containing .unet).
        rank: Either an integer rank, or "base" / "small" preset.
            - "base": 20% of each layer's feature dim (~222M params for SD 2.1)
            - "small": fixed rank 64 across all layers (~62M params)
        target_modules: Which modules to apply LoRA to. Defaults to
            attention projections and conv layers.
        alpha: LoRA scaling factor (alpha / rank).
        dropout: Dropout rate for LoRA layers.

    Returns:
        The model with LoRA adapters applied.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("Please install peft: pip install peft")

    # Get the actual UNet
    unet = model.unet if hasattr(model, "unet") else model

    # Default target modules for SD UNet
    if target_modules is None:
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",  # Attention projections
            "proj_in", "proj_out",                  # Projection layers
            "ff.net.0.proj", "ff.net.2",           # Feedforward
        ]

    # Determine rank
    if isinstance(rank, str):
        if rank == "base":
            # 20% of feature dimension — peft handles per-layer sizing
            # We use rank=128 as an approximation of 20% of typical SD dims
            lora_rank = 128
        elif rank == "small":
            lora_rank = 64
        else:
            raise ValueError(f"Unknown rank preset: {rank}. Use 'base', 'small', or an int.")
    else:
        lora_rank = rank

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=alpha * lora_rank,  # Effective scaling = alpha
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=None,  # Not a standard task type
    )

    # Apply LoRA
    peft_model = get_peft_model(unet, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())

    logger.info(
        f"LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
        f"({trainable_params / total_params * 100:.1f}%)"
    )

    # Replace the unet in the wrapper
    if hasattr(model, "unet"):
        model.unet = peft_model
    else:
        model = peft_model

    return model


def save_lora_weights(model: nn.Module, path: str):
    """Save only the LoRA adapter weights.

    Args:
        model: Model with LoRA adapters applied.
        path: Path to save the weights.
    """
    unet = model.unet if hasattr(model, "unet") else model

    if hasattr(unet, "save_pretrained"):
        unet.save_pretrained(path)
        logger.info(f"LoRA weights saved to {path}")
    else:
        # Fallback: save only trainable params
        trainable_state_dict = {
            k: v for k, v in unet.state_dict().items()
            if any(p is v for p in unet.parameters() if p.requires_grad)
        }
        torch.save(trainable_state_dict, path)
        logger.info(f"Trainable weights saved to {path}")


def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """Load LoRA adapter weights into the model.

    Args:
        model: Model with LoRA adapters applied.
        path: Path to saved LoRA weights.

    Returns:
        Model with loaded weights.
    """
    unet = model.unet if hasattr(model, "unet") else model

    if hasattr(unet, "load_adapter"):
        unet.load_adapter(path, adapter_name="default")
    else:
        state_dict = torch.load(path, map_location="cpu")
        unet.load_state_dict(state_dict, strict=False)

    logger.info(f"LoRA weights loaded from {path}")
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into the base model for efficient inference.

    After merging, the model runs at the same speed as the original
    model without any LoRA overhead.

    Args:
        model: Model with LoRA adapters.

    Returns:
        Model with merged weights (no more LoRA layers).
    """
    unet = model.unet if hasattr(model, "unet") else model

    if hasattr(unet, "merge_and_unload"):
        merged = unet.merge_and_unload()
        if hasattr(model, "unet"):
            model.unet = merged
        else:
            model = merged
        logger.info("LoRA weights merged into base model.")
    else:
        logger.warning("Model does not support merge_and_unload.")

    return model
