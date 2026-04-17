import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "synthetic_dataset"
DEFAULT_PHENOTYPES_PATH = DEFAULT_DATASET_DIR / "predicted_phenotypes.csv"
DEFAULT_CATALOG_PATH = DEFAULT_DATASET_DIR / "hair_asset_catalog_template.csv"
DEFAULT_ASSIGNMENTS_PATH = DEFAULT_DATASET_DIR / "hair_asset_assignments.csv"
DEFAULT_COMBINATIONS_PATH = DEFAULT_DATASET_DIR / "hair_combination_summary.csv"

HAIR_COLORS = {
    "auburn": {
        "material_id": "mat_hair_auburn",
        "base_color_hex": "#7B3F24",
        "melanin": 0.42,
        "redness": 0.72,
    },
    "black": {
        "material_id": "mat_hair_black",
        "base_color_hex": "#11100E",
        "melanin": 0.95,
        "redness": 0.08,
    },
    "blonde": {
        "material_id": "mat_hair_blonde",
        "base_color_hex": "#D8B56D",
        "melanin": 0.18,
        "redness": 0.18,
    },
    "brown": {
        "material_id": "mat_hair_brown",
        "base_color_hex": "#4A2C1A",
        "melanin": 0.68,
        "redness": 0.22,
    },
    "red": {
        "material_id": "mat_hair_red",
        "base_color_hex": "#A83A20",
        "melanin": 0.25,
        "redness": 0.95,
    },
}

THICKNESS_SETTINGS = {
    "fine": {
        "density_multiplier": 0.75,
        "strand_width_multiplier": 0.75,
        "volume_multiplier": 0.82,
    },
    "medium": {
        "density_multiplier": 1.0,
        "strand_width_multiplier": 1.0,
        "volume_multiplier": 1.0,
    },
    "thick": {
        "density_multiplier": 1.25,
        "strand_width_multiplier": 1.18,
        "volume_multiplier": 1.22,
    },
}

TEXTURE_SETTINGS = {
    "straight": {
        "curl_multiplier": 0.0,
        "wave_multiplier": 0.05,
        "frizz_multiplier": 0.08,
    },
    "wavy": {
        "curl_multiplier": 0.35,
        "wave_multiplier": 0.65,
        "frizz_multiplier": 0.18,
    },
    "curly": {
        "curl_multiplier": 0.9,
        "wave_multiplier": 0.35,
        "frizz_multiplier": 0.32,
    },
}


def build_template_catalog() -> pd.DataFrame:
    """Create placeholder hair assets for every geometry-level combination."""
    rows = []
    for sex in ["female", "male"]:
        for texture in ["curly", "straight", "wavy"]:
            for hairline in ["rounded", "straight", "widow_peak"]:
                asset_id = f"hair_{sex}_{texture}_{hairline}"
                rows.append(
                    {
                        "asset_id": asset_id,
                        "asset_name": asset_id.replace("_", " "),
                        "source_library": "TODO",
                        "asset_path": f"assets/hair/{asset_id}.fbx",
                        "sex_compatibility": sex,
                        "hair_texture": texture,
                        "hairline_shape": hairline,
                        "supports_color_materials": True,
                        "supports_thickness_parameters": True,
                        "file_format": "fbx",
                        "license": "TODO",
                        "notes": "Replace this placeholder with a real hair asset path.",
                    }
                )
    return pd.DataFrame(rows)


def load_or_create_catalog(catalog_path: Path) -> pd.DataFrame:
    if catalog_path.exists():
        return pd.read_csv(catalog_path)

    catalog_df = build_template_catalog()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_df.to_csv(catalog_path, index=False)
    return catalog_df


def validate_phenotype_columns(phenotypes_df: pd.DataFrame) -> None:
    required_columns = {
        "sample_id",
        "sex",
        "hair_color",
        "hair_texture",
        "hair_thickness",
        "hairline_shape",
    }
    missing = required_columns.difference(phenotypes_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required phenotype columns: {missing_text}")


def find_best_asset(phenotype_row: pd.Series, catalog_df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Prefer exact geometry match, then relax sex, then relax hairline."""
    exact = catalog_df[
        (catalog_df["sex_compatibility"].isin([phenotype_row["sex"], "both"]))
        & (catalog_df["hair_texture"] == phenotype_row["hair_texture"])
        & (catalog_df["hairline_shape"] == phenotype_row["hairline_shape"])
    ]
    if not exact.empty:
        return exact.iloc[0], "exact"

    texture_match = catalog_df[
        (catalog_df["hair_texture"] == phenotype_row["hair_texture"])
        & (catalog_df["hairline_shape"] == phenotype_row["hairline_shape"])
    ]
    if not texture_match.empty:
        return texture_match.iloc[0], "sex_relaxed"

    hairline_relaxed = catalog_df[
        (catalog_df["sex_compatibility"].isin([phenotype_row["sex"], "both"]))
        & (catalog_df["hair_texture"] == phenotype_row["hair_texture"])
    ]
    if not hairline_relaxed.empty:
        return hairline_relaxed.iloc[0], "hairline_relaxed"

    return catalog_df.iloc[0], "fallback"


def build_hair_assignments(phenotypes_df: pd.DataFrame, catalog_df: pd.DataFrame) -> pd.DataFrame:
    validate_phenotype_columns(phenotypes_df)

    rows = []
    for _, phenotype in phenotypes_df.iterrows():
        color = HAIR_COLORS.get(phenotype["hair_color"], HAIR_COLORS["brown"])
        thickness = THICKNESS_SETTINGS.get(phenotype["hair_thickness"], THICKNESS_SETTINGS["medium"])
        texture = TEXTURE_SETTINGS.get(phenotype["hair_texture"], TEXTURE_SETTINGS["wavy"])
        asset, match_type = find_best_asset(phenotype, catalog_df)

        rows.append(
            {
                "sample_id": phenotype["sample_id"],
                "sex": phenotype["sex"],
                "hair_color": phenotype["hair_color"],
                "hair_texture": phenotype["hair_texture"],
                "hair_thickness": phenotype["hair_thickness"],
                "hairline_shape": phenotype["hairline_shape"],
                "asset_id": asset["asset_id"],
                "asset_name": asset["asset_name"],
                "asset_path": asset["asset_path"],
                "source_library": asset["source_library"],
                "asset_match_type": match_type,
                "material_id": color["material_id"],
                "base_color_hex": color["base_color_hex"],
                "melanin": color["melanin"],
                "redness": color["redness"],
                "density_multiplier": thickness["density_multiplier"],
                "strand_width_multiplier": thickness["strand_width_multiplier"],
                "volume_multiplier": thickness["volume_multiplier"],
                "curl_multiplier": texture["curl_multiplier"],
                "wave_multiplier": texture["wave_multiplier"],
                "frizz_multiplier": texture["frizz_multiplier"],
            }
        )

    return pd.DataFrame(rows)


def build_combination_summary(assignments_df: pd.DataFrame) -> pd.DataFrame:
    summary_columns = [
        "sex",
        "hair_color",
        "hair_texture",
        "hair_thickness",
        "hairline_shape",
        "asset_id",
        "material_id",
        "asset_match_type",
    ]
    return (
        assignments_df.groupby(summary_columns, dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values(summary_columns)
    )


def generate_hair_asset_mapping(
    phenotypes_path: Path = DEFAULT_PHENOTYPES_PATH,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    assignments_path: Path = DEFAULT_ASSIGNMENTS_PATH,
    combinations_path: Optional[Path] = DEFAULT_COMBINATIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phenotypes_df = pd.read_csv(phenotypes_path)
    catalog_df = load_or_create_catalog(catalog_path)
    assignments_df = build_hair_assignments(phenotypes_df, catalog_df)
    summary_df = build_combination_summary(assignments_df)

    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_df.to_csv(assignments_path, index=False)
    if combinations_path is not None:
        combinations_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(combinations_path, index=False)

    return assignments_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map predicted hair phenotypes to reusable 3D hair asset placeholders and material parameters."
    )
    parser.add_argument("--phenotypes", type=Path, default=DEFAULT_PHENOTYPES_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS_PATH)
    parser.add_argument("--combinations", type=Path, default=DEFAULT_COMBINATIONS_PATH)
    args = parser.parse_args()

    assignments_df, summary_df = generate_hair_asset_mapping(
        phenotypes_path=args.phenotypes,
        catalog_path=args.catalog,
        assignments_path=args.assignments,
        combinations_path=args.combinations,
    )

    print("Generated hair asset mapping")
    print(f"- Catalog: {args.catalog}")
    print(f"- Assignments: {args.assignments} ({len(assignments_df)} samples)")
    print(f"- Combination summary: {args.combinations} ({len(summary_df)} observed combinations)")
    print("- Replace placeholder asset_path values in the catalog with real 3D hair assets.")


if __name__ == "__main__":
    main()
