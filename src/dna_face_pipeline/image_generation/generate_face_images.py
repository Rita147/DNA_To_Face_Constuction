import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LORA_DIR = PROJECT_ROOT / "assets" / "lora"

# Ordered preference list — first file found on disk is used.
# The HQ v2 step-8000 checkpoint (output_hq_10k_v2) produced the best results
# across anatomy accuracy, skin tone fidelity, and eye symmetry.
# Download it from Google Drive and place it at assets/lora/hq_v2/ to activate.
LORA_PREFERENCE_ORDER = [
    LORA_DIR / "hq_v2" / "celeba_hq_lora-step00008000.safetensors",  # best — HQ 10k v2, step 8000
    LORA_DIR / "hq" / "celeba_hq_lora-step00006000.safetensors",     # second best — HQ 10k, step 6000
    LORA_DIR / "celeba_lora_v2.safetensors",                          # fallback — original v2
]

DEFAULT_INPUT_CSV = (
    PROJECT_ROOT / "data" / "sequencing_outputs" / "pipeline_outputs"
    / "face_generation_prompts_from_reads.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "generated_images"

# Inference defaults tuned during HQ v2 evaluation runs.
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
# LoRA scale: 0.8 prevents over-fitting the LoRA style while retaining
# the base model's natural face diversity. Lower if outputs look too uniform.
DEFAULT_LORA_SCALE = 0.8


def resolve_lora_path(explicit_path: Optional[Path]) -> Path:
    """Return the best available LoRA checkpoint.

    If the caller supplies an explicit path it is used directly.
    Otherwise the LORA_PREFERENCE_ORDER list is scanned and the first
    file that exists on disk is returned.
    """
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Specified LoRA not found: {explicit_path}")
        return explicit_path

    for candidate in LORA_PREFERENCE_ORDER:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No LoRA checkpoint found. Expected one of:\n"
        + "\n".join(f"  {p}" for p in LORA_PREFERENCE_ORDER)
        + "\nDownload the best checkpoint (HQ v2 step-8000) from Google Drive "
        "and place it at assets/lora/hq_v2/celeba_hq_lora-step00008000.safetensors"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate face images from a prompt CSV using Stable Diffusion v1.5 "
            "with a CelebA LoRA fine-tune.\n\n"
            "Best results use the HQ v2 LoRA (step-8000). Place the checkpoint at\n"
            "  assets/lora/hq_v2/celeba_hq_lora-step00008000.safetensors\n"
            "to activate it automatically."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_INPUT_CSV,
        help="CSV file with positive_prompt and negative_prompt columns.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated images and metadata are saved.",
    )
    parser.add_argument(
        "--model-id", type=str, default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID for the base diffusion model.",
    )
    parser.add_argument(
        "--lora-path", type=Path, default=None,
        help=(
            "Explicit path to a LoRA safetensors file. "
            "When omitted the best available checkpoint is selected automatically."
        ),
    )
    parser.add_argument(
        "--lora-scale", type=float, default=DEFAULT_LORA_SCALE,
        help=(
            f"LoRA influence weight (default: {DEFAULT_LORA_SCALE}). "
            "Lower values let the base model contribute more diversity."
        ),
    )
    parser.add_argument(
        "--rows", type=int, nargs="*", default=None,
        help="Specific row indices to generate (0-based). Generates all rows up to --limit when omitted.",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Maximum rows to process when --rows is not given.",
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS,
        help=f"Diffusion inference steps (default: {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE,
        help=f"Classifier-free guidance scale (default: {DEFAULT_GUIDANCE_SCALE}).",
    )
    parser.add_argument(
        "--width", type=int, default=DEFAULT_WIDTH,
        help=f"Output image width in pixels (default: {DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_HEIGHT,
        help=f"Output image height in pixels (default: {DEFAULT_HEIGHT}).",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="Added to each row's seed for reproducible, varied outputs.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def require_runtime() -> None:
    """Raise a clear error if GPU inference dependencies are not installed."""
    try:
        import torch  # noqa: F401
        from diffusers import StableDiffusionPipeline  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Image generation dependencies are missing. "
            "Install torch, diffusers, transformers, accelerate, and safetensors."
        ) from exc


def load_prompt_rows(
    csv_path: Path,
    selected_rows: Optional[List[int]],
    limit: int,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"positive_prompt", "negative_prompt"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prompt CSV is missing required columns: {sorted(missing)}")

    if selected_rows:
        return (
            df.iloc[selected_rows]
            .copy()
            .reset_index(drop=False)
            .rename(columns={"index": "source_row"})
        )

    subset = df.head(limit).copy()
    subset.insert(0, "source_row", subset.index)
    return subset.reset_index(drop=True)


def build_pipeline(model_id: str, lora_path: Path, device: str):
    """Load the base SD model and attach LoRA weights."""
    require_runtime()

    import torch
    from diffusers import StableDiffusionPipeline

    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.load_lora_weights(str(lora_path))

    print(f"  Base model : {model_id}")
    print(f"  LoRA       : {lora_path.name}")
    print(f"  Device     : {device}")

    return pipe, torch


def generate_images(args: argparse.Namespace) -> List[dict]:
    if not args.csv.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {args.csv}")

    lora_path = resolve_lora_path(args.lora_path)
    print(f"Using LoRA checkpoint: {lora_path}")

    df = load_prompt_rows(args.csv, args.rows, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe, torch = build_pipeline(args.model_id, lora_path, args.device)

    # Inference parameters logged into every metadata file for reproducibility.
    inference_params = {
        "model_id": args.model_id,
        "lora_path": str(lora_path),
        "lora_scale": args.lora_scale,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "width": args.width,
        "height": args.height,
        "seed_offset": args.seed_offset,
        "device": args.device,
    }

    results = []

    for i, row in df.iterrows():
        source_row = int(row["source_row"])
        seed = args.seed_offset + source_row
        sample_id = (
            row["sample_id"]
            if "sample_id" in row and pd.notna(row["sample_id"])
            else f"row_{source_row:04d}"
        )

        print(f"[{i + 1}/{len(df)}] {sample_id}  seed={seed}")

        generator = torch.Generator(device=args.device).manual_seed(seed)

        image = pipe(
            prompt=row["positive_prompt"],
            negative_prompt=row["negative_prompt"],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            generator=generator,
            # Apply LoRA at the tuned scale rather than full weight.
            # This preserves base-model face diversity while using LoRA
            # to steer toward photorealistic CelebA-style portraits.
            cross_attention_kwargs={"scale": args.lora_scale},
        ).images[0]

        image_path = args.output_dir / f"{sample_id}.png"
        metadata_path = args.output_dir / f"{sample_id}.json"

        image.save(image_path)

        metadata = {
            "sample_id": sample_id,
            "source_row": source_row,
            "seed": seed,
            "positive_prompt": row["positive_prompt"],
            "negative_prompt": row["negative_prompt"],
            "image_path": str(image_path),
            **inference_params,
        }

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        results.append(metadata)
        print(f"  Saved: {image_path.name}")

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nFinished. {len(results)} image(s) written to {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    return results


def main() -> None:
    args = parse_args()
    generate_images(args)


if __name__ == "__main__":
    main()
