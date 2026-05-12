#!/usr/bin/env python3
"""Regenerate spin GIFs for all 8 website samples and copy to website/images/."""
import subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "src/dna_face_pipeline/3d_generation/FLAME_PyTorch/generate_from_parameters.py"
CSV    = ROOT / "data/synthetic_dataset/synthetic_dataset_complete.csv"
OUT    = ROOT / "website/images"
IMG    = ROOT / "website/images"

SAMPLES = [
    "SYNTH_000016",
    "SYNTH_000029",
    "SYNTH_000052",
    "SYNTH_000066",
    "SYNTH_000270",
    "SYNTH_000601",
    "SYNTH_001977",
    "SYNTH_002246",
]

for sid in SAMPLES:
    portrait = IMG / f"{sid}.png"
    cmd = [
        sys.executable, str(SCRIPT),
        "--sample-id", sid,
        "--dataset-csv", str(CSV),
        "--output-dir", str(OUT),
        "--image", str(portrait),
    ]
    print(f"\n{'='*50}")
    print(f"  {sid}")
    print(f"{'='*50}")
    result = subprocess.run(cmd, cwd=str(ROOT / "src/dna_face_pipeline/3d_generation/FLAME_PyTorch"))
    if result.returncode != 0:
        print(f"  [ERROR] {sid} failed with code {result.returncode}")
        continue

    # DECA output: website/images/deca_synth_000016/deca_synth_000016_spin.gif
    sid_low = sid.lower()
    src_gif = OUT / f"deca_{sid_low}" / f"deca_{sid_low}_spin.gif"
    dst_gif = OUT / f"{sid}.gif"
    if src_gif.exists():
        shutil.copy2(str(src_gif), str(dst_gif))
        print(f"  Copied {src_gif.name} -> {dst_gif.name}")
    else:
        print(f"  [WARN] GIF not found at {src_gif}")

print("\nAll samples processed.")
