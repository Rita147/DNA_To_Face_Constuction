import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


INPUT_GENOMES = Path(__file__).parent / "extended_facial_data" / "genomes_extended.csv"
OUTPUT_DIR = Path(__file__).parent / "extended_facial_data" / "sequencing_sim"


BASES = np.array(list("ACGT"))


def build_snp_catalog(snp_columns: List[str], seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    catalog_rows = []

    used_positions = set()
    for idx, snp in enumerate(snp_columns):
        chrom = (idx % 22) + 1
        position = 100000 + idx * 1000
        while (chrom, position) in used_positions:
            position += 1
        used_positions.add((chrom, position))

        ref = rng.choice(BASES)
        alt_choices = BASES[BASES != ref]
        alt = rng.choice(alt_choices)

        left_context = "".join(rng.choice(BASES, size=25))
        right_context = "".join(rng.choice(BASES, size=25))
        reference_window = f"{left_context}{ref}{right_context}"

        catalog_rows.append(
            {
                "snp_id": snp,
                "chrom": f"chr{chrom}",
                "position": position,
                "ref": ref,
                "alt": alt,
                "left_context": left_context,
                "right_context": right_context,
                "reference_window": reference_window,
            }
        )

    return pd.DataFrame(catalog_rows)


def dosage_to_alleles(dosage: int, ref: str, alt: str) -> Tuple[str, str]:
    dosage = int(dosage)
    if dosage <= 0:
        return ref, ref
    if dosage == 1:
        return ref, alt
    return alt, alt


def introduce_errors(seq: List[str], rng: np.random.Generator, error_rate: float) -> List[str]:
    if error_rate <= 0:
        return seq
    out = seq.copy()
    for i, base in enumerate(out):
        if rng.random() < error_rate:
            alternatives = BASES[BASES != base]
            out[i] = str(rng.choice(alternatives))
    return out


def simulate_reads(
    genomes_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    sample_count: int,
    coverage: int,
    read_length: int,
    error_rate: float,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rng = np.random.default_rng(seed)
    snp_columns = catalog_df["snp_id"].tolist()

    selected = genomes_df.sample(n=min(sample_count, len(genomes_df)), random_state=seed).copy()
    selected = selected.reset_index(drop=True)

    half_len = read_length // 2
    reads = []

    for _, sample_row in selected.iterrows():
        sample_id = sample_row["sample_id"]
        for _, snp_row in catalog_df.iterrows():
            snp_id = snp_row["snp_id"]
            ref = snp_row["ref"]
            alt = snp_row["alt"]
            left_context = snp_row["left_context"]
            right_context = snp_row["right_context"]
            allele_a, allele_b = dosage_to_alleles(sample_row[snp_id], ref, alt)

            for read_idx in range(coverage):
                haplotype = 0 if read_idx < coverage // 2 else 1
                allele = allele_a if haplotype == 0 else allele_b
                full_window = list(f"{left_context}{allele}{right_context}")

                max_start = len(full_window) - read_length
                start = max(0, min(max_start, 25 - half_len + rng.integers(-2, 3)))
                read_bases = full_window[start:start + read_length]
                local_snp_index = 25 - start
                read_bases = introduce_errors(read_bases, rng, error_rate)

                quality = "I" * len(read_bases)
                read_id = f"@{sample_id}|{snp_id}|read{read_idx:03d}|hap{haplotype}|pos{local_snp_index}"
                reads.append(
                    {
                        "read_id": read_id,
                        "sample_id": sample_id,
                        "snp_id": snp_id,
                        "chrom": snp_row["chrom"],
                        "position": int(snp_row["position"]),
                        "local_snp_index": int(local_snp_index),
                        "true_allele": allele,
                        "sequence": "".join(read_bases),
                        "quality": quality,
                    }
                )

    metadata = {
        "sample_count": int(len(selected)),
        "snp_count": int(len(snp_columns)),
        "coverage_per_snp": int(coverage),
        "read_length": int(read_length),
        "error_rate": float(error_rate),
        "seed": int(seed),
    }
    return pd.DataFrame(reads), metadata


def write_fastq(reads_df: pd.DataFrame, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for _, row in reads_df.iterrows():
            f.write(f"{row['read_id']}\n")
            f.write(f"{row['sequence']}\n")
            f.write("+\n")
            f.write(f"{row['quality']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate targeted SNP sequencing reads from synthetic genotypes.")
    parser.add_argument("--input", type=Path, default=INPUT_GENOMES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=25)
    parser.add_argument("--coverage", type=int, default=20)
    parser.add_argument("--read-length", type=int, default=31)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    genomes_df = pd.read_csv(args.input)
    snp_columns = [col for col in genomes_df.columns if col != "sample_id"]
    catalog_df = build_snp_catalog(snp_columns, seed=args.seed)

    reads_df, metadata = simulate_reads(
        genomes_df=genomes_df,
        catalog_df=catalog_df,
        sample_count=args.sample_count,
        coverage=args.coverage,
        read_length=args.read_length,
        error_rate=args.error_rate,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output_dir / "reference_snp_catalog.csv"
    reads_csv_path = args.output_dir / "simulated_reads.csv"
    reads_fastq_path = args.output_dir / "simulated_reads.fastq"
    selected_genotypes_path = args.output_dir / "true_genotypes.csv"
    metadata_path = args.output_dir / "simulation_metadata.json"

    selected_sample_ids = reads_df["sample_id"].drop_duplicates().tolist()
    selected_genotypes = genomes_df[genomes_df["sample_id"].isin(selected_sample_ids)].copy()

    catalog_df.to_csv(catalog_path, index=False)
    reads_df.to_csv(reads_csv_path, index=False)
    write_fastq(reads_df, reads_fastq_path)
    selected_genotypes.to_csv(selected_genotypes_path, index=False)
    metadata["files"] = {
        "catalog": str(catalog_path),
        "reads_csv": str(reads_csv_path),
        "reads_fastq": str(reads_fastq_path),
        "true_genotypes": str(selected_genotypes_path),
    }
    metadata["samples"] = selected_sample_ids
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Simulated sequencing reads for {metadata['sample_count']} samples")
    print(f"- Reads CSV: {reads_csv_path}")
    print(f"- FASTQ: {reads_fastq_path}")
    print(f"- SNP catalog: {catalog_path}")
    print(f"- True genotypes: {selected_genotypes_path}")


if __name__ == "__main__":
    main()
