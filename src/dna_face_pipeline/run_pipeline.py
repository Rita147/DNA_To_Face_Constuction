import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dna_face_pipeline.dataset_builder import (
        DEFAULT_OUTPUT_DIR as DATASET_OUTPUT_DIR,
        PhenotypeToPromptConverter,
        ImprovedPhenotypePredictor,
        get_comprehensive_snp_mappings,
    )
else:
    from dna_face_pipeline.dataset_builder import (
        DEFAULT_OUTPUT_DIR as DATASET_OUTPUT_DIR,
        PhenotypeToPromptConverter,
        ImprovedPhenotypePredictor,
        get_comprehensive_snp_mappings,
    )


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
SEQUENCING_DIR = PROJECT_ROOT / "data" / "sequencing_outputs"
PHENOTYPES_PATH = DATASET_OUTPUT_DIR / "predicted_phenotypes.csv"
DATASET_CSV_PATH = DATASET_OUTPUT_DIR / "synthetic_dataset_complete.csv"


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def build_outputs(called_genotypes_path: Path, phenotypes_path: Path, output_dir: Path) -> None:
    called_df = pd.read_csv(called_genotypes_path)
    sex_df = pd.read_csv(phenotypes_path)[["sample_id", "sex"]]
    merged_df = called_df.merge(sex_df, on="sample_id", how="left")

    snp_mappings = get_comprehensive_snp_mappings()
    predictor = ImprovedPhenotypePredictor(snp_mappings)
    prompt_converter = PhenotypeToPromptConverter()
    negative_prompt = prompt_converter.create_negative_prompt()

    snp_columns = [col for col in called_df.columns if col != "sample_id"]
    phenotype_rows = []
    prompt_rows = []

    for _, row in merged_df.iterrows():
        genotype_dict = {snp: int(row[snp]) for snp in snp_columns}
        sex = row["sex"] if row["sex"] in {"male", "female"} else "female"
        phenotype = predictor.predict_all_traits(genotype_dict, sex=sex)
        phenotype_dict = phenotype.__dict__.copy()
        phenotype_dict["sample_id"] = row["sample_id"]
        phenotype_rows.append(phenotype_dict)

        prompt_rows.append(
            {
                "sample_id": row["sample_id"],
                "positive_prompt": prompt_converter.create_generation_prompt(pd.Series(phenotype_dict)),
                "negative_prompt": negative_prompt,
            }
        )

    phenotype_df = pd.DataFrame(phenotype_rows)
    prompt_df = pd.DataFrame(prompt_rows)
    combined_df = called_df.merge(phenotype_df, on="sample_id").merge(prompt_df, on="sample_id")

    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_phenotypes_path = output_dir / "predicted_phenotypes_from_reads.csv"
    generated_prompts_path = output_dir / "face_generation_prompts_from_reads.csv"
    combined_path = output_dir / "complete_pipeline_output_from_reads.csv"
    summary_path = output_dir / "pipeline_summary.json"

    phenotype_df.to_csv(predicted_phenotypes_path, index=False)
    prompt_df.to_csv(generated_prompts_path, index=False)
    combined_df.to_csv(combined_path, index=False)

    summary = {
        "samples": int(len(combined_df)),
        "snps": int(len(snp_columns)),
        "predicted_phenotypes_path": str(predicted_phenotypes_path),
        "generated_prompts_path": str(generated_prompts_path),
        "combined_output_path": str(combined_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return combined_path


def run_2d_image_generation(prompts_csv: Path, output_dir: Path) -> None:
    run_command(
        [
            sys.executable,
            str(MODULE_DIR / "image_generation" / "generate_face_images.py"),
            "--prompts-csv", str(prompts_csv),
            "--output-dir", str(output_dir),
        ]
    )


def run_3d_generation(sample_ids: list[str], dataset_csv: Path, output_dir: Path) -> None:
    run_command(
        [
            sys.executable,
            str(MODULE_DIR / "3d_generation" / "generate_from_parameters.py"),
            "--sample-ids", *sample_ids,
            "--dataset-csv", str(dataset_csv),
            "--output-dir", str(output_dir),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full synthetic DNA → sequencing → genotype → phenotype → 2D image → 3D model pipeline.\n\n"
            "Stages:\n"
            "  1  Genotype simulation (dataset_builder)\n"
            "  2  Sequencing read simulation\n"
            "  3  Genotype calling from reads → VCF\n"
            "  4  Phenotype prediction from called genotypes\n"
            "  5  2D face image generation (Stable Diffusion + LoRA)\n"
            "  6  3D FLAME model fitting from phenotypes"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sample-count", type=int, default=25,
                        help="Number of synthetic samples to generate (default: 25)")
    parser.add_argument("--coverage", type=int, default=20,
                        help="Target sequencing read depth per locus (default: 20)")
    parser.add_argument("--read-length", type=int, default=31,
                        help="Simulated read length in base pairs (default: 31)")
    parser.add_argument("--error-rate", type=float, default=0.01,
                        help="Per-base sequencing error rate (default: 0.01)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for reproducibility (default: 7)")
    parser.add_argument("--skip-2d", action="store_true",
                        help="Skip Stage 5: 2D Stable Diffusion image generation")
    parser.add_argument("--skip-3d", action="store_true",
                        help="Skip Stage 6: 3D FLAME model fitting")
    parser.add_argument("--3d-samples", nargs="+", metavar="SAMPLE_ID",
                        help="Specific sample IDs to run through Stage 6 (default: all samples)")
    args = parser.parse_args()

    # ── Stage 1: Genotype simulation ──────────────────────────────────────────
    print("\n[Stage 1/6] Simulating synthetic genotypes...")
    run_command(
        [
            sys.executable,
            str(MODULE_DIR / "sequencing" / "simulate_targeted_reads.py"),
            "--sample-count", str(args.sample_count),
            "--coverage", str(args.coverage),
            "--read-length", str(args.read_length),
            "--error-rate", str(args.error_rate),
            "--seed", str(args.seed),
        ]
    )

    # ── Stage 2 → 3: Sequencing simulation + genotype calling ─────────────────
    print("\n[Stage 2/6] Simulating sequencing reads...")
    print("\n[Stage 3/6] Calling genotypes from reads → VCF...")
    run_command([sys.executable, str(MODULE_DIR / "sequencing" / "call_genotypes_from_reads.py")])

    # ── Stage 4: Phenotype prediction ─────────────────────────────────────────
    print("\n[Stage 4/6] Predicting phenotypes from called genotypes...")
    pipeline_dir = SEQUENCING_DIR / "pipeline_outputs"
    combined_path = build_outputs(
        called_genotypes_path=SEQUENCING_DIR / "called_snp_genotypes.csv",
        phenotypes_path=PHENOTYPES_PATH,
        output_dir=pipeline_dir,
    )

    # ── Stage 5: 2D image generation ──────────────────────────────────────────
    if not args.skip_2d:
        print("\n[Stage 5/6] Generating 2D face images via Stable Diffusion + CelebA LoRA...")
        images_dir = PROJECT_ROOT / "data" / "generated_images"
        prompts_csv = pipeline_dir / "face_generation_prompts_from_reads.csv"
        run_2d_image_generation(prompts_csv=prompts_csv, output_dir=images_dir)
    else:
        print("\n[Stage 5/6] Skipped (--skip-2d)")

    # ── Stage 6: 3D FLAME model fitting ───────────────────────────────────────
    if not args.skip_3d:
        print("\n[Stage 6/6] Fitting 3D FLAME models from phenotype measurements...")
        output_3d_dir = PROJECT_ROOT / "data" / "3d_outputs"

        if args.__dict__["3d_samples"]:
            sample_ids = args.__dict__["3d_samples"]
        else:
            # default: use all sample IDs from the combined pipeline output
            if combined_path.exists():
                df = pd.read_csv(combined_path)
                sample_ids = df["sample_id"].tolist()
            else:
                sample_ids = []

        if sample_ids:
            run_3d_generation(
                sample_ids=sample_ids,
                dataset_csv=DATASET_CSV_PATH,
                output_dir=output_3d_dir,
            )
        else:
            print("  No sample IDs found — skipping 3D generation.")
    else:
        print("\n[Stage 6/6] Skipped (--skip-3d)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n✓ Pipeline complete.")
    print(f"  Sequencing artifacts : {SEQUENCING_DIR}")
    print(f"  Phenotype outputs    : {pipeline_dir}")
    if not args.skip_2d:
        print(f"  2D face images       : {PROJECT_ROOT / 'data' / 'generated_images'}")
    if not args.skip_3d:
        print(f"  3D model outputs     : {PROJECT_ROOT / 'data' / '3d_outputs'}")


if __name__ == "__main__":
    main()
