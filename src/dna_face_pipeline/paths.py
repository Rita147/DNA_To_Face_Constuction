"""Shared filesystem paths for the DNA-to-face pipeline."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

SYNTHETIC_DATASET_DIR = DATA_DIR / "synthetic_dataset"
SEQUENCING_OUTPUT_DIR = DATA_DIR / "sequencing_outputs"
PIPELINE_OUTPUT_DIR = SEQUENCING_OUTPUT_DIR / "pipeline_outputs"
GENERATED_IMAGE_DIR = DATA_DIR / "generated_images"
GENERATED_3D_DIR = DATA_DIR / "3d_outputs"

SYNTHETIC_GENOTYPES_CSV = SYNTHETIC_DATASET_DIR / "synthetic_genotypes.csv"
SYNTHETIC_DATASET_CSV = SYNTHETIC_DATASET_DIR / "synthetic_dataset_complete.csv"
PREDICTED_PHENOTYPES_CSV = SYNTHETIC_DATASET_DIR / "predicted_phenotypes.csv"
