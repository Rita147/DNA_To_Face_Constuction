import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_ROOT / "data" / "sequencing_outputs"


def call_genotype(ref_count: int, alt_count: int) -> int:
    total = ref_count + alt_count
    if total == 0:
        return -1

    alt_fraction = alt_count / total
    if alt_fraction < 0.2:
        return 0
    if alt_fraction < 0.8:
        return 1
    return 2


def build_vcf_rows(calls_df: pd.DataFrame, catalog_df: pd.DataFrame) -> List[str]:
    header = [
        "##fileformat=VCFv4.2",
        "##source=sequence_to_genotype.py",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT",
    ]
    rows = []
    merged = calls_df.merge(catalog_df, left_on="snp_id", right_on="snp_id", how="left")
    for _, row in merged.iterrows():
        gt_map = {0: "0/0", 1: "0/1", 2: "1/1", -1: "./."}
        fmt = f"GT:AD\t{gt_map[row['called_genotype']]}:{row['ref_count']},{row['alt_count']}"
        rows.append(
            f"{row['chrom']}\t{row['position']}\t{row['snp_id']}\t{row['ref']}\t{row['alt']}\t.\tPASS\tSAMPLE={row['sample_id']}\t{fmt}"
        )
    return header + rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Call SNP genotypes from simulated sequencing reads.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    args = parser.parse_args()

    reads_path = args.input_dir / "simulated_targeted_reads.csv"
    catalog_path = args.input_dir / "reference_snp_catalog.csv"
    truth_path = args.input_dir / "reference_genotypes_for_simulation.csv"

    reads_df = pd.read_csv(reads_path)
    catalog_df = pd.read_csv(catalog_path)
    truth_df = pd.read_csv(truth_path)

    allele_lookup: Dict[str, Dict[str, str]] = {
        row["snp_id"]: {"ref": row["ref"], "alt": row["alt"]}
        for _, row in catalog_df.iterrows()
    }

    calls = []
    for (sample_id, snp_id), group in reads_df.groupby(["sample_id", "snp_id"], sort=False):
        bases = group.apply(lambda row: row["sequence"][int(row["local_snp_index"])], axis=1)
        ref = allele_lookup[snp_id]["ref"]
        alt = allele_lookup[snp_id]["alt"]
        ref_count = int((bases == ref).sum())
        alt_count = int((bases == alt).sum())
        other_count = int(len(bases) - ref_count - alt_count)
        called = call_genotype(ref_count, alt_count)
        calls.append(
            {
                "sample_id": sample_id,
                "snp_id": snp_id,
                "ref_count": ref_count,
                "alt_count": alt_count,
                "other_count": other_count,
                "called_genotype": called,
                "depth": int(len(group)),
            }
        )

    calls_df = pd.DataFrame(calls)
    wide_calls = calls_df.pivot(index="sample_id", columns="snp_id", values="called_genotype").reset_index()
    wide_calls.columns.name = None

    truth_long = truth_df.melt(id_vars="sample_id", var_name="snp_id", value_name="true_genotype")
    comparison = calls_df.merge(truth_long, on=["sample_id", "snp_id"], how="left")
    comparison["correct"] = comparison["called_genotype"] == comparison["true_genotype"]

    summary = {
        "samples": int(comparison["sample_id"].nunique()),
        "snps": int(comparison["snp_id"].nunique()),
        "calls": int(len(comparison)),
        "accuracy": float(comparison["correct"].mean()) if len(comparison) else 0.0,
        "missing_calls": int((comparison["called_genotype"] < 0).sum()),
        "mean_depth": float(comparison["depth"].mean()) if len(comparison) else 0.0,
    }

    calls_path = args.input_dir / "called_snp_genotypes.csv"
    comparison_path = args.input_dir / "genotype_call_comparison.csv"
    summary_path = args.input_dir / "genotype_call_summary.json"
    vcf_path = args.input_dir / "variant_calls.vcf"

    wide_calls.to_csv(calls_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with vcf_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(build_vcf_rows(calls_df, catalog_df)))
        f.write("\n")

    print("Completed genotype calling from synthetic reads")
    print(f"- Called genotypes: {calls_path}")
    print(f"- VCF: {vcf_path}")
    print(f"- Accuracy: {summary['accuracy']:.3f}")


if __name__ == "__main__":
    main()
