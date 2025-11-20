from __future__ import annotations

import argparse
from pathlib import Path

from rldiff2cad import ConditionedDiffusionPipeline, DiffusionConfig, load_condition_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vector-conditioned CAD diffusion")
    parser.add_argument("prompt", help="Text prompt for generation")
    parser.add_argument("image", type=Path, help="Path to CAD conditioning image")
    parser.add_argument("vector", type=Path, help="Path to vector embedding (.npy)")
    parser.add_argument("--output", type=Path, default=Path("output.png"), help="Where to save the generated image")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5", help="Diffusion model id")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--strength", type=float, default=0.85, help="Noise strength for img2img")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image, vector = load_condition_inputs(args.image, args.vector)

    config = DiffusionConfig(
        model_id=args.model_id,
        device=args.device,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        strength=args.strength,
        image_path=args.image,
        vector_path=args.vector,
        seed=args.seed,
    )

    pipeline = ConditionedDiffusionPipeline(config, vector_dim=vector.shape[0])
    result = pipeline(args.prompt, init_image=image, vector=vector, output_path=str(args.output))
    print(f"Saved conditioned generation to {args.output}")
    result.show()


if __name__ == "__main__":
    main()
