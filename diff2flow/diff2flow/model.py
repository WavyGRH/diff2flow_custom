"""
Stable Diffusion Model Wrapper.

Wraps HuggingFace diffusers UNet2DConditionModel for use with
Diff2Flow training and inference. Handles model loading, conditioning,
and parameterization modes.

Supports:
- Stable Diffusion 1.5 / 2.1
- v-prediction and epsilon-prediction modes
- 4-channel (text2img) and 8-channel (image-conditioned) inputs
"""

from __future__ import annotations

from typing import Optional
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Diff2FlowModel(nn.Module):
    """Wrapper around a UNet backbone for Diff2Flow training.

    Provides a unified interface for loading pre-trained SD models,
    running forward passes, and handling different input configurations.

    Args:
        model_id: HuggingFace model ID (e.g., 'stabilityai/stable-diffusion-2-1').
        in_channels: Number of input channels (4 for text2img, 8 for img2depth).
        parameterization: Model prediction type ('v' or 'epsilon').
        use_fp16: Whether to use half-precision.
        device: Target device.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        in_channels: int = 4,
        parameterization: str = "v",
        use_fp16: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.model_id = model_id
        self.in_channels = in_channels
        self.parameterization = parameterization
        self.device_name = device
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self._loaded = False

    def load_pretrained(self):
        """Load the pre-trained model components from HuggingFace.

        Downloads and initializes:
        - UNet2DConditionModel (the main denoising network)
        - AutoencoderKL (VAE for latent encoding/decoding)
        - Text encoder and tokenizer (for conditioning)
        """
        try:
            from diffusers import UNet2DConditionModel, AutoencoderKL
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError:
            raise ImportError(
                "Please install diffusers and transformers: "
                "pip install diffusers transformers"
            )

        logger.info(f"Loading pre-trained model from {self.model_id}...")

        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet", torch_dtype=self.dtype
        )

        # If we need 8 input channels (e.g., for depth), modify first conv
        if self.in_channels != self.unet.config.in_channels:
            self._modify_input_channels(self.in_channels)

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=self.dtype
        )
        self.vae.requires_grad_(False)  # VAE stays frozen

        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder", torch_dtype=self.dtype
        )
        self.text_encoder.requires_grad_(False)  # Text encoder stays frozen

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer"
        )

        self._loaded = True
        logger.info("Model loaded successfully.")

    def _modify_input_channels(self, target_channels: int):
        """Modify UNet input channels (e.g., from 4 to 8 for depth).

        Duplicates the first conv layer's weights to extend input channels,
        with proper weight initialization for the new channels.
        """
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            target_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        ).to(dtype=self.dtype)

        # Copy existing weights for original channels
        with torch.no_grad():
            new_conv.weight[:, :old_conv.in_channels] = old_conv.weight
            # Initialize new channels to zero
            if target_channels > old_conv.in_channels:
                new_conv.weight[:, old_conv.in_channels:] = 0.0
            new_conv.bias = old_conv.bias

        self.unet.conv_in = new_conv
        logger.info(
            f"Modified UNet input channels: {old_conv.in_channels} -> {target_channels}"
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the UNet.

        Args:
            x: Input latent, shape [B, C, H, W].
                For text2img: C=4.
                For img2depth: C=8 (concatenated image + noise latents).
            timestep: Timestep(s), shape [B].
            encoder_hidden_states: Text embeddings, shape [B, seq_len, dim].
            context: Optional additional context to concatenate to input.

        Returns:
            Model prediction (v, epsilon, or x0), shape [B, 4, H, W].
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")

        # Concatenate context if provided (e.g., image latent for depth)
        if context is not None:
            x = torch.cat([x, context], dim=1)

        # Use null text embedding if no conditioning provided
        if encoder_hidden_states is None:
            encoder_hidden_states = self._get_null_embedding(x.shape[0], x.device)

        # Forward through UNet
        output = self.unet(
            x, timestep, encoder_hidden_states=encoder_hidden_states
        )

        return output.sample

    def _get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate null text embedding for unconditional generation."""
        if self.tokenizer is None or self.text_encoder is None:
            # Return a zero embedding if text components not loaded
            return torch.zeros(batch_size, 77, 1024, device=device, dtype=self.dtype)

        tokens = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            embeddings = self.text_encoder(tokens).last_hidden_state

        return embeddings

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space image to latent space using VAE.

        Args:
            x: Image tensor, shape [B, 3, H, W], values in [-1, 1].

        Returns:
            Latent, shape [B, 4, H/8, W/8].
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded.")
        with torch.no_grad():
            posterior = self.vae.encode(x).latent_dist
            latent = posterior.sample() * self.vae.config.scaling_factor
        return latent

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to pixel-space image using VAE.

        Args:
            z: Latent, shape [B, 4, H/8, W/8].

        Returns:
            Image tensor, shape [B, 3, H*8, W*8], values in [-1, 1].
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded.")
        with torch.no_grad():
            z = z / self.vae.config.scaling_factor
            image = self.vae.decode(z).sample
        return image

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """Encode text prompts to embeddings.

        Args:
            prompts: List of text strings.

        Returns:
            Text embeddings, shape [B, 77, dim].
        """
        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError("Text components not loaded.")

        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.text_encoder.device)

        with torch.no_grad():
            embeddings = self.text_encoder(tokens).last_hidden_state

        return embeddings
