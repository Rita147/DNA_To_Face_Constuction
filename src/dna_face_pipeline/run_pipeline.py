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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full synthetic DNA -> genotype -> phenotype -> prompt pipeline.")
    parser.add_argument("--sample-count", type=int, default=25)
    parser.add_argument("--coverage", type=int, default=20)
    parser.add_argument("--read-length", type=int, default=31)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    run_command(
        [
            sys.executable,
            str(MODULE_DIR / "sequencing" / "simulate_targeted_reads.py"),
            "--sample-count",
            str(args.sample_count),
            "--coverage",
            str(args.coverage),
            "--read-length",
            str(args.read_length),
            "--error-rate",
            str(args.error_rate),
            "--seed",
            str(args.seed),
        ]
    )
    run_command([sys.executable, str(MODULE_DIR / "sequencing" / "call_genotypes_from_reads.py")])

    pipeline_dir = SEQUENCING_DIR / "pipeline_outputs"
    build_outputs(
        called_genotypes_path=SEQUENCING_DIR / "called_snp_genotypes.csv",
        phenotypes_path=PHENOTYPES_PATH,
        output_dir=pipeline_dir,
    )

    print("Completed full synthetic pipeline")
    print(f"- Sequencing artifacts: {SEQUENCING_DIR}")
    print(f"- Final pipeline outputs: {pipeline_dir}")


if __name__ == "__main__":
    main()
