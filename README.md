# RL-DIFF2CAD

Vector-conditioned Stable Diffusion pipeline for CAD imagery. The project pairs a CAD render with a learned vector embedding and injects that representation into a Stable Diffusion img2img pipeline to steer generation.

## Features
- Img2img Stable Diffusion conditioning on both CAD imagery and a companion vector embedding.
- Lightweight projection network to align vector features with text encoder dimensions.
- CLI script for running inference end-to-end.

## Quickstart
1. Install dependencies (PyTorch + diffusers + Pillow + numpy). Example:
   ```bash
   pip install torch diffusers pillow numpy
   ```
2. Prepare inputs:
   - `cad.png`: CAD render or sketch used as img2img conditioning image.
   - `vec.npy`: 1D numpy array containing the companion vector embedding.
3. Run inference:
   ```bash
   python scripts/run_inference.py "a clean industrial CAD render" cad.png vec.npy --output result.png
   ```

## Pipeline overview
The core implementation lives in `src/rldiff2cad/pipeline.py`:
- `VectorConditioner` projects the input vector to the Stable Diffusion text embedding size.
- `ConditionedDiffusionPipeline` wraps `StableDiffusionImg2ImgPipeline`, merges prompt embeddings with the projected vector, and runs denoising with configurable guidance scale, step count, and noise strength.

`DiffusionConfig` centralizes tunable parameters such as model id, device, guidance scale, and optional seed. Input loading and validation for images and vectors is handled in `src/rldiff2cad/data.py`.

## Notes
- The projection layer is intentionally small to keep conditioning lightweight; adjust hidden dimensions in `VectorConditioner` if your vector space is richer.
- The script uses classifier-free guidance with an empty negative prompt; extend `scripts/run_inference.py` if you need richer negative prompting.
- For CPU-only environments set `--device cpu` (inference will be slow).
