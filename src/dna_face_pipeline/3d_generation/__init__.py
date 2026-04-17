# 3D generation pipeline (Zharasbek Bimagambetov)
# Requires: flame_pytorch, pyrender, trimesh, torch
#
# generate_from_parameters.py — FLAME model fitting + rendering (main entry point)
# appearance_scene.py         — pyrender scene construction from FLAME vertices
# appearance_mappings.py      — phenotype trait → RGB colour tables
# measurement_definitions.py  — 68-landmark measurement definitions (TypedDicts)
# measurement_utils.py        — landmark-based measurement computation (PyTorch)

from .appearance_mappings import (
    extract_appearance_traits_from_row,
    build_appearance_render_plan,
)

__all__ = [
    "extract_appearance_traits_from_row",
    "build_appearance_render_plan",
]
