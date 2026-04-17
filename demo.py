"""
demo.py — Side-by-side 2D image + 3D appearance model for the same DNA sample.

Usage:
    python demo.py                        # uses SYNTH_000000 by default
    python demo.py --sample SYNTH_000002  # any sample in the dataset
    python demo.py --sample SYNTH_000000 --save-dir demo_outputs
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless renderer — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch

# Make local package importable when run from the project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dna_face_pipeline.image_generation.generate_face_images import (
    build_pipeline,
    resolve_lora_path,
)
from dna_face_pipeline.dataset_builder import PhenotypeToPromptConverter


# ---------------------------------------------------------------------------
# Colour helpers for the 3D appearance panel
# ---------------------------------------------------------------------------

# Skin tone RGB values taken from appearance_mappings.py / appearance_scene.py
SKIN_TONE_RGB = {
    "very_light": (241 / 255, 194 / 255, 125 / 255),
    "light":      (224 / 255, 172 / 255, 105 / 255),
    "medium":     (198 / 255, 134 / 255,  66 / 255),
    "dark":       (141 / 255,  85 / 255,  36 / 255),
    "very_dark":  ( 80 / 255,  45 / 255,  15 / 255),
}

EYE_COLOUR_RGB = {
    "green": (92 / 255, 122 / 255,  84 / 255),
    "blue":  (88 / 255, 125 / 255, 168 / 255),
    "brown": (92 / 255,  61 / 255,  38 / 255),
    "dark_brown": (55 / 255, 35 / 255, 20 / 255),
    "hazel": (120 / 255, 100 / 255,  60 / 255),
    "gray":  (140 / 255, 140 / 255, 145 / 255),
}

HAIR_COLOUR_RGB = {
    "black":  ( 20 / 255,  15 / 255,  10 / 255),
    "brown":  ( 92 / 255,  62 / 255,  42 / 255),
    "auburn": (130 / 255,  60 / 255,  30 / 255),
    "red":    (150 / 255,  75 / 255,  40 / 255),
    "blonde": (196 / 255, 170 / 255,  96 / 255),
    "gray":   (160 / 255, 160 / 255, 160 / 255),
    "white":  (230 / 255, 230 / 255, 230 / 255),
}


def _colour(mapping: dict, key: str, default=(0.6, 0.6, 0.6)):
    """Case-insensitive colour lookup with graceful fallback."""
    if key is None:
        return default
    return mapping.get(str(key).strip().lower(), default)


# ---------------------------------------------------------------------------
# 2D image generation
# ---------------------------------------------------------------------------

def generate_2d_image(positive_prompt: str, negative_prompt: str, seed: int, device: str):
    """Run Stable Diffusion with the best available LoRA and return a PIL image."""
    lora_path = resolve_lora_path(None)
    print(f"  Loading SD pipeline ({lora_path.name})...")
    pipe, torch_mod = build_pipeline("runwayml/stable-diffusion-v1-5", lora_path, device)

    generator = torch_mod.Generator(device=device).manual_seed(seed)
    print("  Generating 2D image...")
    image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=512,
        height=512,
        generator=generator,
        cross_attention_kwargs={"scale": 0.8},
    ).images[0]
    return image


# ---------------------------------------------------------------------------
# 3D appearance schematic
# ---------------------------------------------------------------------------

def draw_3d_appearance_panel(ax, row: pd.Series):
    """
    Draw a schematic face diagram coloured with the predicted appearance
    parameters — skin tone, eye colour, hair colour, and key geometry traits.

    This is the appearance render plan that would be passed to the FLAME 3D
    renderer (appearance_scene.py) when pyrender meshes are generated.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    skin_colour   = _colour(SKIN_TONE_RGB,   row.get("skin_tone"))
    eye_colour    = _colour(EYE_COLOUR_RGB,  row.get("eye_color"))
    hair_colour   = _colour(HAIR_COLOUR_RGB, row.get("hair_color"))

    # ── Face oval ──────────────────────────────────────────────────────────
    face_width_label = str(row.get("face_width", "medium")).lower()
    face_rx = {"wide": 3.2, "narrow": 2.2}.get(face_width_label, 2.7)

    face_height_label = str(row.get("face_height", "medium")).lower()
    face_ry = {"long": 4.2, "short": 3.0}.get(face_height_label, 3.6)

    face = mpatches.Ellipse(
        (5, 6.5), face_rx * 2, face_ry * 2,
        facecolor=skin_colour, edgecolor="#555555", linewidth=1.5, zorder=2,
    )
    ax.add_patch(face)

    # ── Hair cap ───────────────────────────────────────────────────────────
    hair_texture = str(row.get("hair_texture", "straight")).lower()
    hair_thickness = str(row.get("hair_thickness", "medium")).lower()
    hair_cap_ry = face_ry + {"fine": 0.4, "thick": 0.9}.get(hair_thickness, 0.6)

    hair = mpatches.Ellipse(
        (5, 6.5 + face_ry * 0.55), face_rx * 2.1, hair_cap_ry,
        facecolor=hair_colour, edgecolor="none", zorder=1,
    )
    ax.add_patch(hair)

    # ── Eyes ──────────────────────────────────────────────────────────────
    eye_dist = str(row.get("eye_distance", "normal")).lower()
    eye_spacing = {"close": 0.9, "wide": 1.4}.get(eye_dist, 1.1)
    eye_size_label = str(row.get("eye_size", "medium")).lower()
    eye_rx = {"small": 0.28, "large": 0.45}.get(eye_size_label, 0.36)
    eye_ry = eye_rx * 0.55

    eye_y = 6.5 + face_ry * 0.15
    for ex in [5 - eye_spacing, 5 + eye_spacing]:
        # white
        ax.add_patch(mpatches.Ellipse(
            (ex, eye_y), eye_rx * 2.5, eye_ry * 2.5,
            facecolor="white", edgecolor="#444", linewidth=0.8, zorder=3,
        ))
        # iris
        ax.add_patch(mpatches.Ellipse(
            (ex, eye_y), eye_rx * 1.6, eye_ry * 1.6,
            facecolor=eye_colour, edgecolor="none", zorder=4,
        ))
        # pupil
        ax.add_patch(mpatches.Ellipse(
            (ex, eye_y), eye_rx * 0.7, eye_ry * 0.7,
            facecolor="black", edgecolor="none", zorder=5,
        ))

    # ── Nose ──────────────────────────────────────────────────────────────
    nose_w_label = str(row.get("nose_width", "medium")).lower()
    nose_w = {"narrow": 0.18, "wide": 0.38}.get(nose_w_label, 0.26)
    nose_y = 6.5 - face_ry * 0.12
    nose_pts = np.array([
        [5 - nose_w, nose_y - 0.55],
        [5,          nose_y + 0.55],
        [5 + nose_w, nose_y - 0.55],
    ])
    ax.plot(nose_pts[:, 0], nose_pts[:, 1],
            color="#555", linewidth=1.2, solid_capstyle="round", zorder=6)

    # ── Mouth ─────────────────────────────────────────────────────────────
    mouth_y = 6.5 - face_ry * 0.42
    mouth_w_label = str(row.get("mouth_width", "medium")).lower()
    mouth_w = {"narrow": 0.45, "wide": 0.80}.get(mouth_w_label, 0.60)
    lip_label = str(row.get("lip_thickness", "medium")).lower()
    lip_h = {"thin": 0.06, "full": 0.18, "very_full": 0.24}.get(lip_label, 0.12)

    lip_colour = tuple(max(0, c - 0.12) for c in skin_colour)
    ax.add_patch(mpatches.Ellipse(
        (5, mouth_y), mouth_w * 2, lip_h * 2,
        facecolor=lip_colour, edgecolor="#555", linewidth=0.8, zorder=6,
    ))

    # ── Jaw indicator ─────────────────────────────────────────────────────
    jaw_label = str(row.get("jaw_shape", "oval")).lower()
    jaw_text = {"square": "square jaw", "narrow": "narrow jaw", "oval": "oval jaw"}.get(
        jaw_label, f"{jaw_label} jaw"
    )
    ax.text(5, 6.5 - face_ry - 0.5, jaw_text,
            ha="center", va="top", fontsize=7.5, color="#555", style="italic", zorder=7)

    # ── Title and trait legend ─────────────────────────────────────────────
    sample_id = row.get("sample_id", "")
    ax.set_title(f"3D Appearance Model\n{sample_id}", fontsize=9, fontweight="bold", pad=6)

    legend_items = [
        mpatches.Patch(facecolor=skin_colour,  edgecolor="#888", label=f"Skin: {row.get('skin_tone','?')}"),
        mpatches.Patch(facecolor=eye_colour,   edgecolor="#888", label=f"Eyes: {row.get('eye_color','?')}"),
        mpatches.Patch(facecolor=hair_colour,  edgecolor="#888", label=f"Hair: {row.get('hair_color','?')} / {hair_texture}"),
    ]
    ax.legend(
        handles=legend_items, loc="lower center",
        fontsize=7, framealpha=0.85, edgecolor="#ccc",
        bbox_to_anchor=(0.5, -0.08),
    )

    # ── FLAME parameter summary panel ─────────────────────────────────────
    flame_traits = {
        "Face width":    row.get("face_width", "—"),
        "Face height":   row.get("face_height", "—"),
        "Jaw shape":     row.get("jaw_shape", "—"),
        "Chin":          row.get("chin_prominence", "—"),
        "Cheekbones":    row.get("cheekbone_height", "—"),
        "Nose width":    row.get("nose_width", "—"),
        "Eye size":      row.get("eye_size", "—"),
        "Eye shape":     row.get("eye_shape", "—"),
        "Eye distance":  row.get("eye_distance", "—"),
        "Brow arch":     row.get("eyebrow_arch", "—"),
        "Lip thickness": row.get("lip_thickness", "—"),
        "Mouth width":   row.get("mouth_width", "—"),
    }
    lines = [f"{k}: {v}" for k, v in flame_traits.items()]
    ax.text(
        0.02, 0.02, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=6.5, va="bottom", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#ddd"),
    )


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(sample_id: str, save_dir: Path, device: str):
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load sample data ───────────────────────────────────────────────────
    dataset_path = PROJECT_ROOT / "data" / "synthetic_dataset" / "synthetic_dataset_complete.csv"
    prompts_path = PROJECT_ROOT / "data" / "synthetic_dataset" / "face_generation_prompts.csv"

    df_dataset = pd.read_csv(dataset_path)
    df_prompts = pd.read_csv(prompts_path)

    sample_data = df_dataset[df_dataset["sample_id"] == sample_id]
    sample_prompt = df_prompts[df_prompts["sample_id"] == sample_id]

    if sample_data.empty:
        raise ValueError(f"Sample '{sample_id}' not found in dataset.")
    if sample_prompt.empty:
        raise ValueError(f"Sample '{sample_id}' not found in prompts CSV.")

    row = sample_data.iloc[0]
    prompt_row = sample_prompt.iloc[0]

    positive_prompt = prompt_row["positive_prompt"]
    negative_prompt = prompt_row["negative_prompt"]

    print(f"\nSample      : {sample_id}")
    print(f"Sex         : {row.get('sex', '?')}")
    print(f"Eye colour  : {row.get('eye_color', '?')}")
    print(f"Hair colour : {row.get('hair_color', '?')}")
    print(f"Skin tone   : {row.get('skin_tone', '?')}")
    print(f"\nPrompt preview:\n  {positive_prompt[:120]}...\n")

    # ── Stage 1: 2D image generation ──────────────────────────────────────
    print("[ Stage 1 ] Generating 2D portrait via Stable Diffusion + HQ v2 LoRA")
    seed = int(sample_id.replace("SYNTH_", "")) % (2 ** 31)
    image_2d = generate_2d_image(positive_prompt, negative_prompt, seed, device)
    image_path = save_dir / f"{sample_id}_2d.png"
    image_2d.save(image_path)
    print(f"  Saved 2D image: {image_path.name}")

    # ── Stage 2: 3D appearance model ──────────────────────────────────────
    print("\n[ Stage 2 ] Building 3D appearance model from phenotype parameters")
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 6.5),
        facecolor="#f8f8f6",
        gridspec_kw={"wspace": 0.08},
    )
    fig.suptitle(
        f"DNA → Face Pipeline  ·  {sample_id}",
        fontsize=13, fontweight="bold", y=1.01,
        color="#1a1917",
    )

    # Left: 2D generated portrait
    ax_2d = axes[0]
    ax_2d.imshow(image_2d)
    ax_2d.set_title("2D Portrait\nStable Diffusion v1.5 + CelebA HQ LoRA (step-8000)",
                    fontsize=9, fontweight="bold", pad=8)
    ax_2d.axis("off")

    # Right: 3D appearance schematic
    ax_3d = axes[1]
    draw_3d_appearance_panel(ax_3d, row)

    plt.tight_layout()
    output_path = save_dir / f"{sample_id}_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved comparison figure: {output_path.name}")

    # ── Stage 3: print FLAME control scores ───────────────────────────────
    print("\n[ Stage 3 ] FLAME geometry control scores (3D model parameters)")
    print(f"  {'Trait':<22} {'Value':<18} {'FLAME control'}")
    print(f"  {'-'*22} {'-'*18} {'-'*20}")
    geometry_map = [
        ("face_width",       "global_face / jaw"),
        ("face_height",      "global_face"),
        ("jaw_shape",        "jaw"),
        ("chin_prominence",  "chin"),
        ("cheekbone_height", "cheekbones"),
        ("nose_width",       "nose_width_only"),
        ("nose_size",        "nose_width + height"),
        ("eye_size",         "eye_size"),
        ("eye_shape",        "eyes_shape"),
        ("eye_distance",     "eyes_spacing"),
        ("eyebrow_arch",     "brows"),
        ("lip_thickness",    "lip_thickness"),
        ("mouth_width",      "mouth_width_only"),
    ]
    for trait, control in geometry_map:
        value = row.get(trait, "—")
        print(f"  {trait:<22} {str(value):<18} {control}")

    print(f"\nDone. Output saved to: {save_dir.resolve()}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D image + 3D model from one DNA sample.")
    parser.add_argument("--sample", default="SYNTH_000000", help="Sample ID from the dataset (default: SYNTH_000000)")
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "data" / "demo_outputs",
                        help="Directory to save outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device")
    args = parser.parse_args()

    run_demo(sample_id=args.sample, save_dir=args.save_dir, device=args.device)
