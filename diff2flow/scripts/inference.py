"""
Diff2Flow Inference Script.

Generate images or depth maps from a trained Diff2Flow model.

Usage:
    python scripts/inference.py --task text2img --prompt "A sunset over mountains" --num_steps 25
    python scripts/inference.py --task img2depth --input image.png --num_steps 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diff2flow.converter import Diff2FlowConverter
from diff2flow.sampler import EulerSampler
from diff2flow.timestep_mapping import TimestepMapper
from diff2flow.interpolant_align import InterpolantAligner
from diff2flow.schedules import NoiseScheduleVP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Diff2Flow Inference")

    parser.add_argument(
        "--task", type=str, default="text2img",
        choices=["text2img", "img2depth"],
        help="Inference task",
    )
    parser.add_argument("--prompt", type=str, default="A beautiful landscape", help="Text prompt")
    parser.add_argument("--input", type=str, default=None, help="Input image (for img2depth)")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--num_steps", type=int, default=25, help="Number of sampling steps (NFE)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 512], help="Output resolution H W")

    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to Diff2Flow checkpoint")
    parser.add_argument("--lora_weights", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def text_to_image(args):
    """Generate an image from a text prompt."""
    from diff2flow.model import Diff2FlowModel
    from diff2flow.lora import load_lora_weights, merge_lora_weights

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info(f"Loading model: {args.model_id}")
    model = Diff2FlowModel(model_id=args.model_id, device=device)
    model.load_pretrained()

    # Load checkpoint / LoRA weights if provided
    if args.lora_weights:
        from diff2flow.lora import apply_lora
        model = apply_lora(model, rank="base")
        model = load_lora_weights(model, args.lora_weights)
        model = merge_lora_weights(model)

    model = model.to(device)
    model.eval()

    # Setup Diff2Flow sampler
    schedule = NoiseScheduleVP()
    mapper = TimestepMapper(schedule)
    aligner = InterpolantAligner(schedule, mapper)

    sampler = EulerSampler(
        num_steps=args.num_steps,
        mapper=mapper,
        aligner=aligner,
        use_diff2flow=True,
    )

    # Text conditioning
    text_embeddings = model.encode_text([args.prompt])
    text_embeddings = text_embeddings.to(device)

    # Generate
    torch.manual_seed(args.seed)
    latent_h = args.resolution[0] // 8
    latent_w = args.resolution[1] // 8
    x_init = torch.randn(1, 4, latent_h, latent_w, device=device, dtype=model.dtype)

    logger.info(f"Generating image with {args.num_steps} steps...")

    def model_fn(x, t, enc_states):
        return model(x, t, encoder_hidden_states=enc_states)

    x_generated = sampler.sample(
        model_fn=model_fn,
        x_init=x_init,
        encoder_hidden_states=text_embeddings,
        guidance_scale=args.guidance_scale,
    )

    # Decode latent to image
    image = model.decode_latent(x_generated)
    image = ((image.clamp(-1, 1) + 1) / 2 * 255).byte()
    image = image[0].permute(1, 2, 0).cpu().numpy()

    Image.fromarray(image).save(args.output)
    logger.info(f"Image saved to {args.output}")


def image_to_depth(args):
    """Estimate depth from an input image."""
    from diff2flow.model import Diff2FlowModel

    if args.input is None:
        raise ValueError("--input is required for img2depth task")

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = Diff2FlowModel(
        model_id=args.model_id,
        in_channels=8,
        device=device,
    )
    model.load_pretrained()
    model = model.to(device)
    model.eval()

    # Load input image
    img = Image.open(args.input).convert("RGB")
    img = img.resize((args.resolution[1], args.resolution[0]), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    img_tensor = img_tensor.to(device, dtype=model.dtype)

    # Encode to latent
    img_latent = model.encode_latent(img_tensor)

    # Setup sampler
    schedule = NoiseScheduleVP()
    mapper = TimestepMapper(schedule)
    aligner = InterpolantAligner(schedule, mapper)
    sampler = EulerSampler(num_steps=args.num_steps, mapper=mapper, aligner=aligner)

    # Generate depth
    torch.manual_seed(args.seed)
    x_init = torch.randn_like(img_latent)

    def model_fn(x, t, enc_states):
        return model(x, t, encoder_hidden_states=enc_states, context=img_latent)

    depth_latent = sampler.sample(
        model_fn=model_fn,
        x_init=x_init,
        guidance_scale=1.0,
    )

    # Decode depth
    depth_image = model.decode_latent(depth_latent)
    depth = depth_image[0, 0].cpu().numpy()  # Take first channel
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = (depth * 255).astype(np.uint8)

    Image.fromarray(depth, mode="L").save(args.output)
    logger.info(f"Depth map saved to {args.output}")


def main():
    args = parse_args()

    if args.task == "text2img":
        text_to_image(args)
    elif args.task == "img2depth":
        image_to_depth(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
