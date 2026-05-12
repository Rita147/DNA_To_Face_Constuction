# DNA To Face Construction

This repository contains a synthetic DNA-to-face pipeline for a graduation project.
It turns simulated SNP genotypes into phenotype traits, text prompts, 2D portrait
images, and optional 3D FLAME render outputs.

## Project Layout

- `src/dna_face_pipeline/dataset_builder.py` builds the synthetic genotype,
  phenotype, and prompt CSV files.
- `src/dna_face_pipeline/sequencing/` simulates targeted reads and calls SNP
  genotypes from those reads.
- `src/dna_face_pipeline/image_generation/` generates 2D face images from prompt
  rows with Stable Diffusion and a CelebA LoRA checkpoint.
- `src/dna_face_pipeline/3d_generation/` maps phenotype rows to FLAME
  measurements and renders 3D outputs.
- `src/dna_face_pipeline/run_pipeline.py` ties the sequencing, phenotype, 2D,
  and 3D stages together.
- `data/` stores generated datasets and run outputs.
- `assets/lora/` stores local LoRA model weights.
- `website/` contains the static demo page and example images.

## Environment

Create the conda environment from the checked-in file:

```powershell
conda env create -f environment.yml
conda activate dnaface
```

The 2D image stage needs PyTorch, Diffusers, and a local LoRA checkpoint. The 3D
stage needs the FLAME assets expected by the `FLAME_PyTorch` code. Keep large
model files local unless they are intentionally tracked.

## Common Commands

Generate or refresh the synthetic dataset:

```powershell
python src\dna_face_pipeline\dataset_builder.py --samples 10000
```

Run the fast, non-GPU parts of the pipeline:

```powershell
python src\dna_face_pipeline\run_pipeline.py --skip-2d --skip-3d
```

Run the full pipeline for a small sample set:

```powershell
python src\dna_face_pipeline\run_pipeline.py --sample-count 5 --3d-samples SYNTH_000000
```

Generate 2D images directly from an existing prompt CSV:

```powershell
python src\dna_face_pipeline\image_generation\generate_face_images.py --csv data\sequencing_outputs\pipeline_outputs\face_generation_prompts_from_reads.csv --limit 5
```

Generate one 3D FLAME output directly:

```powershell
python src\dna_face_pipeline\3d_generation\generate_from_parameters.py --sample-id SYNTH_000000
```

## Output Policy

Generated run outputs are ignored by default:

- `data/generated_images/`
- `data/3d_outputs/`
- `data/demo_outputs/`
- `data/sequencing_outputs/`

The checked-in synthetic dataset and website example assets remain available as
stable demo inputs.
