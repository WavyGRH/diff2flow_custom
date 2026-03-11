"""
Generate Noise-Image Pairs for Reflow Training.

Samples from a pre-trained diffusion model to create paired
(noise, image) data used for trajectory straightening.

Usage:
    python scripts/generate_reflow_pairs.py \
        --model_id stabilityai/stable-diffusion-2-1 \
        --num_samples 10000 \
        --output_dir data/reflow_pairs/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Reflow Pairs")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="data/reflow_pairs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    noise_dir = output_dir / "noise"
    image_dir = output_dir / "images"
    noise_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Load the pipeline for generation
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        raise ImportError("Please install diffusers: pip install diffusers")

    logger.info(f"Loading model: {args.model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    ).to(device)

    # Generate pairs
    torch.manual_seed(args.seed)
    latent_h = args.resolution[0] // 8
    latent_w = args.resolution[1] // 8

    num_generated = 0
    pbar = tqdm(total=args.num_samples, desc="Generating pairs")

    while num_generated < args.num_samples:
        current_batch = min(args.batch_size, args.num_samples - num_generated)

        # Generate initial noise (this is what we save)
        noise = torch.randn(
            current_batch, 4, latent_h, latent_w,
            device=device, dtype=torch.float16,
        )

        # Run inference with the noise to get the generated image latent
        with torch.no_grad():
            # Encode null text
            text_inputs = pipe.tokenizer(
                [""] * current_batch,
                padding="max_length", max_length=77,
                truncation=True, return_tensors="pt",
            ).input_ids.to(device)
            text_embeddings = pipe.text_encoder(text_inputs).last_hidden_state

            # DDIM inference to get image from noise
            latents = noise.clone()
            pipe.scheduler.set_timesteps(args.num_steps, device=device)

            for t in pipe.scheduler.timesteps:
                model_input = pipe.scheduler.scale_model_input(latents, t)
                noise_pred = pipe.unet(
                    model_input, t, encoder_hidden_states=text_embeddings
                ).sample
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Save pairs
        for i in range(current_batch):
            idx = num_generated + i
            torch.save(noise[i].cpu(), str(noise_dir / f"{idx:06d}.pt"))
            torch.save(latents[i].cpu(), str(image_dir / f"{idx:06d}.pt"))

        num_generated += current_batch
        pbar.update(current_batch)

    pbar.close()
    logger.info(f"Generated {num_generated} pairs in {output_dir}")


if __name__ == "__main__":
    main()
