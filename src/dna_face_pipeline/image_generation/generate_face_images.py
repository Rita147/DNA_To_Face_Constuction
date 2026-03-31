import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LORA_PATH = PROJECT_ROOT / "assets" / "lora" / "celeba_lora_v2.safetensors"
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "sequencing_outputs" / "pipeline_outputs" / "face_generation_prompts_from_reads.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "generated_images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate face images from prompt CSV files using Stable Diffusion v1.5 plus a CelebA LoRA."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT_CSV, help="CSV file containing prompt columns.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where generated images will be saved.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Base diffusion model identifier.")
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_PATH, help="Path to the LoRA safetensors file.")
    parser.add_argument("--rows", type=int, nargs="*", default=None, help="Specific row indices to generate.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of rows to generate when --rows is not provided.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--width", type=int, default=512, help="Output image width.")
    parser.add_argument("--height", type=int, default=512, help="Output image height.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Value added to the seed for reproducible outputs.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use, for example 'cuda' or 'cpu'.")
    return parser.parse_args()


def require_runtime():
    try:
        import torch  # noqa: F401
        from diffusers import StableDiffusionPipeline  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Image generation dependencies are missing. Install at least 'torch', 'diffusers', "
            "'transformers', 'accelerate', and 'safetensors' in the Python environment used to run this script."
        ) from exc


def load_prompt_rows(csv_path: Path, selected_rows: Optional[List[int]], limit: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"positive_prompt", "negative_prompt"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prompt CSV is missing required columns: {sorted(missing)}")

    if selected_rows:
        return df.iloc[selected_rows].copy().reset_index(drop=False).rename(columns={"index": "source_row"})

    subset = df.head(limit).copy()
    subset.insert(0, "source_row", subset.index)
    return subset.reset_index(drop=True)


def build_pipeline(model_id: str, lora_path: Path, device: str):
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
    return pipe, torch


def generate_images(args: argparse.Namespace) -> List[dict]:
    if not args.csv.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {args.csv}")
    if not args.lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {args.lora_path}")

    df = load_prompt_rows(args.csv, args.rows, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe, torch = build_pipeline(args.model_id, args.lora_path, args.device)
    results = []

    for i, row in df.iterrows():
        source_row = int(row["source_row"])
        seed = args.seed_offset + source_row
        print(f"[{i + 1}/{len(df)}] Generating row {source_row} with seed {seed}")

        generator = torch.Generator(device=args.device).manual_seed(seed)
        image = pipe(
            prompt=row["positive_prompt"],
            negative_prompt=row["negative_prompt"],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            generator=generator,
        ).images[0]

        sample_id = row["sample_id"] if "sample_id" in row and pd.notna(row["sample_id"]) else f"row_{source_row:04d}"
        image_path = args.output_dir / f"{sample_id}.png"
        metadata_path = args.output_dir / f"{sample_id}.json"

        image.save(image_path)
        metadata = {
            "sample_id": sample_id,
            "source_row": source_row,
            "seed": seed,
            "model_id": args.model_id,
            "lora_path": str(args.lora_path),
            "positive_prompt": row["positive_prompt"],
            "negative_prompt": row["negative_prompt"],
            "image_path": str(image_path),
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        results.append(metadata)
        print(f"Saved: {image_path}")

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Finished generating {len(results)} image(s). Manifest: {manifest_path}")
    return results


def main() -> None:
    args = parse_args()
    generate_images(args)


if __name__ == "__main__":
    main()
