from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyrender
import trimesh
import trimesh.transformations as transformations

from pathlib import Path

HAIR_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "hair"
SUPPORTED_HAIR_ASSET_SUFFIXES = {".glb", ".gltf"}

EXACT_SKIN_TONE_RGB: Dict[str, np.ndarray] = {
    "very_light": np.array([241, 194, 125], dtype=np.float32) / 255.0,  # #f1c27d
    "light": np.array([224, 172, 105], dtype=np.float32) / 255.0,       # #e0ac69
    "dark": np.array([141, 85, 36], dtype=np.float32) / 255.0,          # #8d5524
}

EXACT_FRECKLE_RGB: np.ndarray = (
    np.array([0x6C, 0x34, 0x28], dtype=np.float32) / 255.0  # #6C3428
)


def resolve_face_rgba(
    appearance_render_plan: Optional[Dict[str, Any]],
    neutral_rgb: Sequence[float] = (0.70, 0.72, 0.78),
) -> np.ndarray:
    """
    Resolve the base face color.

    Exact overrides:
    - very_light -> #f1c27d
    - light      -> #e0ac69
    - dark       -> #8d5524

    For other labels, fall back to the appearance_render_plan RGB if present.
    """
    neutral_rgb_arr = np.array(neutral_rgb, dtype=np.float32)

    if appearance_render_plan is None:
        rgba = np.concatenate([neutral_rgb_arr, np.array([1.0], dtype=np.float32)])
        return rgba

    skin_tone_label = str(
        appearance_render_plan.get("skin_tone_label", "")
    ).strip().lower()

    if skin_tone_label in EXACT_SKIN_TONE_RGB:
        base_rgb = EXACT_SKIN_TONE_RGB[skin_tone_label]
        rgba = np.concatenate([base_rgb, np.array([1.0], dtype=np.float32)])
        return rgba

    skin_tone_rgb = appearance_render_plan.get("skin_tone_rgb")
    if skin_tone_rgb is not None:
        base_rgb = np.array(skin_tone_rgb, dtype=np.float32) / 255.0
        rgba = np.concatenate([np.clip(base_rgb, 0.0, 1.0), np.array([1.0], dtype=np.float32)])
        return rgba

    rgba = np.concatenate([neutral_rgb_arr, np.array([1.0], dtype=np.float32)])
    return rgba


def rgba_float_to_uint8(rgba: np.ndarray) -> np.ndarray:
    rgba = np.asarray(rgba, dtype=np.float32)
    rgba = np.clip(rgba, 0.0, 1.0)
    return np.round(rgba * 255.0).astype(np.uint8)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _normalize_face_vertex_coordinates(
    vertices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize face vertices into simple coordinate bands for region-aware coloring.

    Returns:
        x_norm: roughly [-1, +1], left-right across the face
        y_norm: [0, 1], bottom-to-top
        z_norm: [0, 1], back-to-front
    """
    verts = np.asarray(vertices, dtype=np.float32)

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = 0.5 * (mins + maxs)
    extents = np.maximum(maxs - mins, 1e-6)

    x_norm = (verts[:, 0] - center[0]) / (0.5 * extents[0] + 1e-6)
    y_norm = (verts[:, 1] - mins[1]) / (extents[1] + 1e-6)
    z_norm = (verts[:, 2] - mins[2]) / (extents[2] + 1e-6)

    return x_norm, y_norm, z_norm


def _vertex_hash_noise(
    vertices: np.ndarray,
    scale: float,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Deterministic pseudo-random noise per vertex from 3D coordinates.
    """
    verts = np.asarray(vertices, dtype=np.float32) * float(scale)
    offs = np.asarray(offset, dtype=np.float32)

    p = verts + offs[None, :]
    s = np.sin(
        p[:, 0] * 12.9898
        + p[:, 1] * 78.233
        + p[:, 2] * 37.719
    ) * 43758.5453

    return s - np.floor(s)


def _resolve_freckle_level(
    appearance_render_plan: Optional[Dict[str, Any]],
) -> str:
    if appearance_render_plan is None:
        return "none"

    raw = appearance_render_plan.get("freckling_label", "none")
    label = str(raw).strip().lower()

    if label in {"none", "no", "absent", "unknown"}:
        return "none"
    if label in {"some", "light", "mild"}:
        return "some"
    if label in {"extensive", "many", "heavy"}:
        return "extensive"

    return "none"


def resolve_freckle_rgb() -> np.ndarray:
    """
    Return the exact freckle color.

    This stays fixed so freckles always use the hardcoded tone rather than
    being adapted from the current skin color.
    """
    return EXACT_FRECKLE_RGB.copy()


def _blend_rgb_toward_color(
    vertex_rgb: np.ndarray,
    target_rgb: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float32)
    return (
        (1.0 - alpha[:, None]) * vertex_rgb
        + alpha[:, None] * target_rgb[None, :]
    )


def build_uniform_vertex_colors(
    num_vertices: int,
    rgb_uint8: list[int] | np.ndarray,
    alpha_uint8: int = 255,
) -> np.ndarray:
    rgb = np.asarray(rgb_uint8, dtype=np.uint8)
    rgba = np.concatenate([rgb, np.array([alpha_uint8], dtype=np.uint8)])
    return np.tile(rgba[None, :], (num_vertices, 1))


def concatenate_mesh_parts_without_visuals(
    mesh_parts: List[trimesh.Trimesh],
) -> Optional[trimesh.Trimesh]:
    """
    Concatenate mesh geometry only, ignoring original visuals/materials/textures.
    This avoids TextureVisuals concatenate errors.
    """
    vertices_list: List[np.ndarray] = []
    faces_list: List[np.ndarray] = []
    vertex_offset = 0

    for part in mesh_parts:
        if not isinstance(part, trimesh.Trimesh):
            continue

        verts = np.asarray(part.vertices, dtype=np.float64)
        faces = np.asarray(part.faces, dtype=np.int64)

        if len(verts) == 0 or len(faces) == 0:
            continue

        vertices_list.append(verts)
        faces_list.append(faces + vertex_offset)
        vertex_offset += len(verts)

    if not vertices_list or not faces_list:
        return None

    merged = trimesh.Trimesh(
        vertices=np.vstack(vertices_list),
        faces=np.vstack(faces_list),
        process=False,
    )
    return merged


def load_asset_as_single_trimesh(asset_path: Path) -> Optional[trimesh.Trimesh]:
    """
    Load an asset that may be either:
    - a single Trimesh
    - a Scene containing one or more mesh nodes

    Returns one merged Trimesh using geometry only, ignoring original visuals.
    """
    try:
        loaded = trimesh.load(asset_path, force="scene", process=False)
    except Exception as e:
        print(f"[appearance_scene] Failed to load hair asset: {e}")
        return None

    # Case 1: already a single mesh
    if isinstance(loaded, trimesh.Trimesh):
        if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
            print("[appearance_scene] Hair asset mesh is empty.")
            return None

        return trimesh.Trimesh(
            vertices=loaded.vertices.copy(),
            faces=loaded.faces.copy(),
            process=False,
        )

    # Case 2: scene with geometry nodes
    if isinstance(loaded, trimesh.Scene):
        mesh_parts: List[trimesh.Trimesh] = []

        # Preferred path: walk scene graph nodes so we preserve node transforms
        for node_name in loaded.graph.nodes_geometry:
            try:
                transform, geom_name = loaded.graph[node_name]
            except Exception as e:
                print(f"[appearance_scene] Failed to read scene node '{node_name}': {e}")
                continue

            geom = loaded.geometry.get(geom_name)
            if not isinstance(geom, trimesh.Trimesh):
                continue
            if len(geom.vertices) == 0 or len(geom.faces) == 0:
                continue

            part = trimesh.Trimesh(
                vertices=geom.vertices.copy(),
                faces=geom.faces.copy(),
                process=False,
            )
            part.apply_transform(transform)
            mesh_parts.append(part)

        # Fallback: use raw geometries directly if graph path yielded nothing
        if not mesh_parts:
            for geom in loaded.geometry.values():
                if not isinstance(geom, trimesh.Trimesh):
                    continue
                if len(geom.vertices) == 0 or len(geom.faces) == 0:
                    continue

                part = trimesh.Trimesh(
                    vertices=geom.vertices.copy(),
                    faces=geom.faces.copy(),
                    process=False,
                )
                mesh_parts.append(part)

        if not mesh_parts:
            print("[appearance_scene] Scene contained no usable mesh parts.")
            return None

        merged_mesh = concatenate_mesh_parts_without_visuals(mesh_parts)
        if merged_mesh is None:
            print("[appearance_scene] Failed to merge mesh parts.")
            return None

        return merged_mesh

    print(f"[appearance_scene] Unsupported loaded asset type: {type(loaded)}")
    return None


@lru_cache(maxsize=1)
def discover_hair_asset_catalog() -> Dict[str, Path]:
    catalog: Dict[str, Path] = {}

    if not HAIR_ASSET_DIR.exists():
        return catalog

    for asset_path in sorted(HAIR_ASSET_DIR.iterdir()):
        if not asset_path.is_file():
            continue
        if asset_path.suffix.lower() not in SUPPORTED_HAIR_ASSET_SUFFIXES:
            continue

        asset_id = asset_path.stem.strip().lower()
        if asset_id == "" or asset_id in catalog:
            continue

        catalog[asset_id] = asset_path

    return catalog


@lru_cache(maxsize=None)
def _resolve_hair_asset_path_from_id(hair_asset_id: str) -> Optional[Path]:
    asset_id = str(hair_asset_id).strip().lower()
    if asset_id == "":
        return None

    asset_path = discover_hair_asset_catalog().get(asset_id)
    if asset_path is None:
        print(f"[appearance_scene] No exact hair asset found for '{asset_id}'. Skipping hair.")
        return None

    return asset_path


def resolve_hair_asset_path(
    appearance_render_plan: Optional[Dict[str, Any]],
) -> Optional[Path]:
    if appearance_render_plan is None:
        return None

    raw_hair_asset_id = appearance_render_plan.get("hair_asset_id")
    if raw_hair_asset_id is None:
        return None

    hair_asset_id = str(raw_hair_asset_id).strip().lower()
    if hair_asset_id in {"", "none", "unknown"}:
        return None

    return _resolve_hair_asset_path_from_id(hair_asset_id)


@lru_cache(maxsize=None)
def _load_hair_asset_template(asset_path_str: str) -> Optional[trimesh.Trimesh]:
    return load_asset_as_single_trimesh(Path(asset_path_str))


def load_hair_asset_template(asset_path: Path) -> Optional[trimesh.Trimesh]:
    template_mesh = _load_hair_asset_template(str(asset_path))
    if template_mesh is None:
        return None

    return trimesh.Trimesh(
        vertices=template_mesh.vertices.copy(),
        faces=template_mesh.faces.copy(),
        process=False,
    )


def _resolve_hair_fit_adjustments(
    appearance_render_plan: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    hair_asset_id = str(
        (appearance_render_plan or {}).get("hair_asset_id", "")
    ).strip().lower()
    hair_thickness = str(
        (appearance_render_plan or {}).get("hair_thickness_label", "medium")
    ).strip().lower()
    hairline_shape = str(
        (appearance_render_plan or {}).get("hairline_shape_label", "straight")
    ).strip().lower()
    gender = str(
        (appearance_render_plan or {}).get("gender_label", "unknown")
    ).strip().lower()

    width_scale = 1.0
    height_scale = 1.0
    depth_scale = 1.0
    y_offset_ratio = 0.0
    z_offset_ratio = 0.0

    if hair_thickness == "fine":
        width_scale *= 1.02
        height_scale *= 1.03
        depth_scale *= 0.98
        y_offset_ratio += 0.008
        z_offset_ratio -= 0.012
    elif hair_thickness == "thick":
        width_scale *= 1.10
        height_scale *= 1.06
        depth_scale *= 1.14
        y_offset_ratio -= 0.002
        z_offset_ratio += 0.010

    if hairline_shape == "rounded":
        width_scale *= 1.01
        height_scale *= 1.01
        y_offset_ratio += 0.010
        z_offset_ratio -= 0.010
    elif hairline_shape == "widow_peak":
        y_offset_ratio -= 0.002
        z_offset_ratio += 0.004

    if gender == "female":
        height_scale *= 1.03
    elif gender == "male":
        height_scale *= 1.00

    # We are tuning hair assets one combination at a time. Keep these
    # adjustments narrowly scoped so other combinations remain unchanged.
    if hair_asset_id == "female_straight_rounded":
        width_scale *= 1.09
        height_scale *= 1.10
        depth_scale *= 1.08
        y_offset_ratio -= 0.050
        z_offset_ratio += 0.052
    elif hair_asset_id == "male_wavy_widow_peak":
        width_scale *= 0.84
        height_scale *= 0.80
        depth_scale *= 0.84
        y_offset_ratio += 0.055
        z_offset_ratio -= 0.055
    elif hair_asset_id == "male_curly_rounded":
        z_offset_ratio -= 0.075
    elif hair_asset_id == "male_curly_straight":
        z_offset_ratio -= 0.110
    elif hair_asset_id == "male_straight_widow_peak":
        width_scale *= 1.10
        height_scale *= 1.12
        depth_scale *= 1.08
        z_offset_ratio -= 0.110

    return {
        "width_scale": width_scale,
        "height_scale": height_scale,
        "depth_scale": depth_scale,
        "y_offset_ratio": y_offset_ratio,
        "z_offset_ratio": z_offset_ratio,
    }


def _apply_combo_specific_hair_orientation(
    hair_mesh: trimesh.Trimesh,
    appearance_render_plan: Optional[Dict[str, Any]],
) -> None:
    hair_asset_id = str(
        (appearance_render_plan or {}).get("hair_asset_id", "")
    ).strip().lower()

    # This asset imports facing backward relative to the FLAME head.
    if hair_asset_id == "male_wavy_widow_peak":
        rot = transformations.rotation_matrix(np.radians(180.0), [0, 1, 0])
        hair_mesh.apply_transform(rot)


def build_hair_mesh(
    vertices: np.ndarray,
    appearance_render_plan: Optional[Dict[str, Any]],
) -> Optional[trimesh.Trimesh]:
    """
    Load the exact hair asset selected by hair_asset_id, scale/translate it
    to the FLAME head, and tint it using hair_color_rgb.
    """

    if appearance_render_plan is None:
        return None

    hair_asset_path = resolve_hair_asset_path(appearance_render_plan)
    if hair_asset_path is None:
        return None

    hair_mesh = load_hair_asset_template(hair_asset_path)
    if hair_mesh is None:
        return None

    # ------------------------------------------------------------
    # Tint the hair using the dataset hair color
    # ------------------------------------------------------------
    hair_rgb = appearance_render_plan.get("hair_color_rgb", [92, 62, 42])
    hair_mesh.visual.vertex_colors = build_uniform_vertex_colors(
        len(hair_mesh.vertices),
        hair_rgb,
        alpha_uint8=255,
    )

    # ------------------------------------------------------------
    # Estimate FLAME head bounds
    # ------------------------------------------------------------
    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)

    head_width = x_max - x_min
    head_height = y_max - y_min
    head_depth = z_max - z_min

    # ------------------------------------------------------------
    # Combo-specific asset orientation correction
    # ------------------------------------------------------------
    _apply_combo_specific_hair_orientation(
        hair_mesh=hair_mesh,
        appearance_render_plan=appearance_render_plan,
    )

    # ------------------------------------------------------------
    # Source asset bounds
    # ------------------------------------------------------------
    hx_min, hy_min, hz_min = hair_mesh.vertices.min(axis=0)
    hx_max, hy_max, hz_max = hair_mesh.vertices.max(axis=0)

    hair_width = hx_max - hx_min
    hair_height = hy_max - hy_min
    hair_depth = hz_max - hz_min

    if hair_width <= 1e-8 or hair_height <= 1e-8 or hair_depth <= 1e-8:
        print("[appearance_scene] Hair asset has invalid bounds.")
        return None

    # ------------------------------------------------------------
    # Scalp/root anchor instead of bbox center
    # ------------------------------------------------------------
    source_anchor_x = 0.5 * (hx_min + hx_max)
    source_anchor_y = float(np.quantile(hair_mesh.vertices[:, 1], 0.76))
    source_anchor_z = 0.5 * (hz_min + hz_max)

    hair_mesh.vertices -= np.array(
        [source_anchor_x, source_anchor_y, source_anchor_z],
        dtype=np.float64,
    )[None, :]

    # ------------------------------------------------------------
    # Uniform scale (width-based)
    # ------------------------------------------------------------
    uniform_scale = 1.08 * head_width / hair_width
    hair_mesh.vertices *= float(uniform_scale)

    fit_adjustments = _resolve_hair_fit_adjustments(appearance_render_plan)
    hair_mesh.vertices *= np.array(
        [
            fit_adjustments["width_scale"],
            fit_adjustments["height_scale"],
            fit_adjustments["depth_scale"],
        ],
        dtype=np.float64,
    )[None, :]

    # ------------------------------------------------------------
    # Place on FLAME head
    # ------------------------------------------------------------
    target_center_x = 0.5 * (x_min + x_max)
    target_anchor_y = (
        y_max
        - 0.02 * head_height
        + fit_adjustments["y_offset_ratio"] * head_height
    )
    target_center_z = (
        0.5 * (z_min + z_max)
        - 0.02 * head_depth
        + fit_adjustments["z_offset_ratio"] * head_depth
    )

    hair_mesh.vertices += np.array(
        [target_center_x, target_anchor_y, target_center_z],
        dtype=np.float64,
    )[None, :]

    return hair_mesh


def _build_clustered_freckle_strength(
    vertices: np.ndarray,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    z_norm: np.ndarray,
    freckle_level: str,
) -> np.ndarray:
    """
    Build a natural freckle intensity map.

    Main goals:
    - nose bridge / upper nose / upper cheeks / forehead get the highest density
    - the rest of the face still gets some lighter spillover
    - eyes and mouth stay comparatively cleaner
    - "some" stays sparse while "extensive" broadens coverage
    """
    if freckle_level == "none":
        return np.zeros_like(x_norm, dtype=np.float32)

    front = _smoothstep(0.26, 0.95, z_norm)
    full_face_gate = _smoothstep(0.06, 0.92, y_norm)
    broad_width_gate = np.exp(-((x_norm / 0.90) ** 4))

    left_eye_mask = np.exp(
        -(((x_norm + 0.23) / 0.12) ** 2 + ((y_norm - 0.60) / 0.07) ** 2)
    )
    right_eye_mask = np.exp(
        -(((x_norm - 0.23) / 0.12) ** 2 + ((y_norm - 0.60) / 0.07) ** 2)
    )
    mouth_mask = np.exp(
        -(((x_norm) / 0.28) ** 2 + ((y_norm - 0.28) / 0.09) ** 2)
    )
    eye_mouth_exclusion = np.clip(
        1.0 - 0.78 * (left_eye_mask + right_eye_mask) - 0.92 * mouth_mask,
        0.0,
        1.0,
    )

    nose_bridge_mask = np.exp(
        -(((x_norm) / 0.09) ** 2 + ((y_norm - 0.63) / 0.07) ** 2)
    )
    upper_nose_mask = np.exp(
        -(((x_norm) / 0.12) ** 2 + ((y_norm - 0.55) / 0.10) ** 2)
    )
    left_upper_cheek_mask = np.exp(
        -(((x_norm + 0.20) / 0.16) ** 2 + ((y_norm - 0.57) / 0.12) ** 2)
    )
    right_upper_cheek_mask = np.exp(
        -(((x_norm - 0.20) / 0.16) ** 2 + ((y_norm - 0.57) / 0.12) ** 2)
    )
    forehead_mask = np.exp(
        -(((x_norm) / 0.42) ** 2 + ((y_norm - 0.83) / 0.11) ** 2)
    )
    left_temple_mask = np.exp(
        -(((x_norm + 0.42) / 0.13) ** 2 + ((y_norm - 0.68) / 0.15) ** 2)
    )
    right_temple_mask = np.exp(
        -(((x_norm - 0.42) / 0.13) ** 2 + ((y_norm - 0.68) / 0.15) ** 2)
    )
    left_lower_cheek_mask = np.exp(
        -(((x_norm + 0.30) / 0.18) ** 2 + ((y_norm - 0.42) / 0.16) ** 2)
    )
    right_lower_cheek_mask = np.exp(
        -(((x_norm - 0.30) / 0.18) ** 2 + ((y_norm - 0.42) / 0.16) ** 2)
    )
    chin_mask = np.exp(
        -(((x_norm) / 0.26) ** 2 + ((y_norm - 0.14) / 0.08) ** 2)
    )

    upper_cheek_mask = left_upper_cheek_mask + right_upper_cheek_mask
    temple_mask = left_temple_mask + right_temple_mask
    lower_cheek_mask = left_lower_cheek_mask + right_lower_cheek_mask

    if freckle_level == "some":
        diffuse_weight = 0.20
        forehead_weight = 0.52
        temple_weight = 0.20
        lower_face_weight = 0.10
        chin_weight = 0.08
    else:  # extensive
        diffuse_weight = 0.36
        forehead_weight = 0.78
        temple_weight = 0.34
        lower_face_weight = 0.20
        chin_weight = 0.16

    core_region = (
        1.00 * nose_bridge_mask
        + 0.88 * upper_nose_mask
        + 0.92 * upper_cheek_mask
        + forehead_weight * forehead_mask
        + temple_weight * temple_mask
    )
    diffuse_region = (
        0.26 * full_face_gate * broad_width_gate
        + 0.18 * upper_cheek_mask
        + 0.16 * forehead_mask
        + 0.15 * temple_mask
        + lower_face_weight * lower_cheek_mask
        + chin_weight * chin_mask
    )

    macro_noise = _vertex_hash_noise(
        vertices,
        scale=34.0,
        offset=(2.8, 4.3, 9.1),
    )
    macro_modulation = 0.82 + 0.18 * _smoothstep(0.18, 0.90, macro_noise)

    # This returns a broader density field. Small individual freckles are
    # sampled later from this field using higher-frequency noise.
    strength = (
        core_region
        + diffuse_weight * diffuse_region
    ) * macro_modulation

    return np.clip(
        strength * front * full_face_gate * eye_mouth_exclusion,
        0.0,
        1.35,
    )


def build_skin_tone_vertex_colors(
    vertices: np.ndarray,
    appearance_render_plan: Optional[Dict[str, Any]],
) -> np.ndarray:
    """
    Build adaptive per-vertex RGBA colors for the face mesh.

    Includes:
    - better base skin handling for very_light / light / dark
    - adaptive region-aware skin variation
    - refined freckles with stronger natural clustering and lighter spillover
    """
    base_rgba = resolve_face_rgba(appearance_render_plan)
    base_rgb = base_rgba[:3].astype(np.float32)

    base_luma = float(
        np.dot(
            base_rgb,
            np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        )
    )

    # 0 = darker tones, 1 = lighter tones
    tone_t = float(np.clip((base_luma - 0.18) / 0.62, 0.0, 1.0))

    x_norm, y_norm, z_norm = _normalize_face_vertex_coordinates(vertices)
    front = _smoothstep(0.28, 0.88, z_norm)

    cheek_mask = np.exp(
        -(
            ((np.abs(x_norm) - 0.42) / 0.18) ** 2
            + ((y_norm - 0.58) / 0.16) ** 2
        )
    ) * front

    nose_mask = np.exp(
        -(
            (x_norm / 0.14) ** 2
            + ((y_norm - 0.56) / 0.16) ** 2
        )
    ) * front

    forehead_mask = np.exp(
        -(
            (x_norm / 0.50) ** 2
            + ((y_norm - 0.82) / 0.15) ** 2
        )
    ) * _smoothstep(0.20, 0.95, z_norm)

    jaw_mask = np.exp(
        -(((y_norm - 0.20) / 0.11) ** 2)
    ) * (0.35 + 0.65 * np.clip(np.abs(x_norm), 0.0, 1.0)) * (0.15 + 0.85 * front)

    chin_center_mask = np.exp(
        -(
            (x_norm / 0.22) ** 2
            + ((y_norm - 0.18) / 0.08) ** 2
        )
    ) * front

    vertex_rgb = np.tile(base_rgb[None, :], (vertices.shape[0], 1))

    # Use brightness-only modulation so the chosen skin tone keeps its hue.
    cheek_lift = 0.008 + 0.010 * tone_t
    nose_lift = 0.005 + 0.007 * tone_t
    forehead_lift = 0.004 + 0.006 * tone_t
    jaw_shadow = 0.006 + 0.007 * tone_t
    chin_lift = 0.002 + 0.003 * tone_t

    brightness_scale = np.ones((vertices.shape[0],), dtype=np.float32)
    brightness_scale += cheek_lift * cheek_mask
    brightness_scale += nose_lift * nose_mask
    brightness_scale += forehead_lift * forehead_mask
    brightness_scale -= jaw_shadow * jaw_mask
    brightness_scale += chin_lift * chin_center_mask
    brightness_scale = np.clip(brightness_scale, 0.82, 1.18)

    vertex_rgb = vertex_rgb * brightness_scale[:, None]
    accent_region = np.clip(
        1.10 * nose_mask
        + 0.90 * cheek_mask
        + 0.20 * forehead_mask,
        0.0,
        1.0,
    )
    soft_region = np.clip(
        0.85 * nose_mask
        + 0.80 * cheek_mask
        + 0.45 * forehead_mask
        + 0.25 * front,
        0.0,
        1.0,
    )

    # ------------------------------------------------------------
    # Freckles
    # ------------------------------------------------------------
    freckle_level = _resolve_freckle_level(appearance_render_plan)

    if freckle_level != "none":
        freckle_strength = _build_clustered_freckle_strength(
            vertices=vertices,
            x_norm=x_norm,
            y_norm=y_norm,
            z_norm=z_norm,
            freckle_level=freckle_level,
        )

        freckle_color = resolve_freckle_rgb()
        noise_micro_a = _vertex_hash_noise(
            vertices,
            scale=460.0,
            offset=(0.0, 0.0, 0.0),
        )
        noise_micro_b = _vertex_hash_noise(
            vertices,
            scale=735.0,
            offset=(4.2, 9.1, 1.7),
        )
        noise_micro_c = _vertex_hash_noise(
            vertices,
            scale=995.0,
            offset=(6.4, 2.7, 8.5),
        )
        noise_small_a = _vertex_hash_noise(
            vertices,
            scale=250.0,
            offset=(8.8, 1.9, 3.7),
        )
        noise_small_b = _vertex_hash_noise(
            vertices,
            scale=365.0,
            offset=(1.4, 7.7, 5.8),
        )
        noise_dust = _vertex_hash_noise(
            vertices,
            scale=185.0,
            offset=(8.3, 1.6, 6.5),
        )
        noise_halo = _vertex_hash_noise(
            vertices,
            scale=96.0,
            offset=(2.1, 5.4, 7.9),
        )

        if freckle_level == "some":
            halo_alpha = (0.010 + 0.006 * tone_t) * _smoothstep(
                0.30,
                0.72,
                freckle_strength,
            )
            halo_alpha *= _smoothstep(0.93, 1.0, noise_halo)

            dust_alpha = (0.024 + 0.012 * tone_t) * _smoothstep(
                0.22,
                0.72,
                freckle_strength,
            )
            dust_alpha *= _smoothstep(0.95, 1.0, noise_dust)

            soft_mask = (
                (freckle_strength > 0.28)
                & (
                    (
                        (noise_small_a > 0.985)
                        & (noise_small_b > 0.78)
                    )
                    | (
                        (noise_small_a > 0.972)
                        & (noise_micro_c > 0.92)
                    )
                )
            )
            micro_mask = (
                (freckle_strength > 0.22)
                & (
                    (
                        (noise_micro_a > 0.968)
                        & (noise_micro_b > 0.73)
                    )
                    | (
                        (noise_micro_b > 0.985)
                        & (noise_micro_c > 0.52)
                    )
                )
            )
            accent_mask = (
                (freckle_strength > 0.34)
                & (noise_micro_a > 0.992)
                & (noise_micro_c > 0.80)
            )
            soft_alpha_value = 0.09
            micro_alpha_value = 0.48

        else:  # extensive
            halo_alpha = (0.012 + 0.007 * tone_t) * _smoothstep(
                0.22,
                0.68,
                freckle_strength,
            )
            halo_alpha *= _smoothstep(0.90, 1.0, noise_halo)

            dust_alpha = (0.034 + 0.018 * tone_t) * _smoothstep(
                0.16,
                0.66,
                freckle_strength,
            )
            dust_alpha *= _smoothstep(0.91, 1.0, noise_dust)

            soft_mask = (
                (freckle_strength > 0.18)
                & (
                    (
                        (noise_small_a > 0.978)
                        & (noise_small_b > 0.70)
                    )
                    | (
                        (noise_small_b > 0.988)
                        & (noise_micro_a > 0.64)
                    )
                )
            )
            micro_mask = (
                (
                    (freckle_strength > 0.12)
                    & (noise_micro_a > 0.949)
                    & (noise_micro_b > 0.54)
                )
                | (
                    (freckle_strength > 0.16)
                    & (noise_micro_b > 0.968)
                    & (noise_micro_c > 0.45)
                )
            )
            accent_mask = (
                (freckle_strength > 0.24)
                & (noise_micro_a > 0.988)
                & (noise_micro_c > 0.70)
            )
            soft_alpha_value = 0.10
            micro_alpha_value = 0.54

        soft_mask = soft_mask & (soft_region > 0.24)
        accent_mask = accent_mask & (accent_region > 0.24)

        accent_mask = accent_mask.astype(bool)
        micro_mask = micro_mask.astype(bool) & ~accent_mask
        soft_mask = soft_mask.astype(bool) & ~(micro_mask | accent_mask)

        halo_alpha = np.clip(halo_alpha, 0.0, 0.035)
        dust_alpha = np.clip(dust_alpha, 0.0, 0.075)

        vertex_rgb = _blend_rgb_toward_color(
            vertex_rgb=vertex_rgb,
            target_rgb=freckle_color,
            alpha=halo_alpha,
        )
        vertex_rgb = _blend_rgb_toward_color(
            vertex_rgb=vertex_rgb,
            target_rgb=freckle_color,
            alpha=dust_alpha,
        )

        soft_alpha = np.zeros((vertices.shape[0],), dtype=np.float32)
        soft_alpha[soft_mask] = soft_alpha_value
        vertex_rgb = _blend_rgb_toward_color(
            vertex_rgb=vertex_rgb,
            target_rgb=freckle_color,
            alpha=soft_alpha,
        )

        micro_alpha = np.zeros((vertices.shape[0],), dtype=np.float32)
        micro_alpha[micro_mask] = (
            micro_alpha_value
            * (0.55 + 0.45 * accent_region[micro_mask])
        )
        vertex_rgb = _blend_rgb_toward_color(
            vertex_rgb=vertex_rgb,
            target_rgb=freckle_color,
            alpha=micro_alpha,
        )

        vertex_rgb[accent_mask] = freckle_color

    vertex_rgb = np.clip(vertex_rgb, 0.0, 1.0)

    alpha = np.full((vertices.shape[0], 1), base_rgba[3], dtype=np.float32)
    vertex_rgba = np.concatenate([vertex_rgb, alpha], axis=1)

    return rgba_float_to_uint8(vertex_rgba)


def build_face_mesh_with_skin_tone(
    vertices: np.ndarray,
    faces: np.ndarray,
    appearance_render_plan: Optional[Dict[str, Any]] = None,
) -> trimesh.Trimesh:
    """
    Build the FLAME face mesh with adaptive, region-aware skin-tone vertex coloring.
    """
    vertex_colors = build_skin_tone_vertex_colors(
        vertices=vertices,
        appearance_render_plan=appearance_render_plan,
    )

    return trimesh.Trimesh(
        vertices=vertices.copy(),
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )


def _safe_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.copy()
    return vec / norm


def _axis_angle_rotation_matrix(
    axis: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    axis = _safe_normalize(axis.astype(np.float64))
    x, y, z = axis
    c = float(np.cos(angle_radians))
    s = float(np.sin(angle_radians))
    C = 1.0 - c

    return np.array(
        [
            [x * x * C + c,     x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c,     y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def _rotation_from_z_to_vector(target_direction: np.ndarray) -> np.ndarray:
    """
    Returns a 3x3 rotation matrix that maps +Z onto target_direction.
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    target = _safe_normalize(target_direction.astype(np.float64))

    dot = float(np.clip(np.dot(z_axis, target), -1.0, 1.0))

    if dot > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)

    if dot < -1.0 + 1e-8:
        return _axis_angle_rotation_matrix(
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.pi,
        )

    axis = np.cross(z_axis, target)
    angle = float(np.arccos(dot))
    return _axis_angle_rotation_matrix(axis, angle)


def build_colored_eyeball_mesh(
    center: np.ndarray,
    radius: float,
    iris_rgb: Sequence[int],
    gaze_direction: Sequence[float] = (0.0, 0.0, 1.0),
    sclera_rgb: Sequence[int] = (245, 245, 245),
    pupil_rgb: Sequence[int] = (20, 20, 20),
    iris_angle_degrees: float = 22.0,
    pupil_angle_degrees: float = 9.5,
    subdivisions: int = 3,
) -> trimesh.Trimesh:
    """
    Build one eyeball mesh as an icosphere with vertex colors for:
    - sclera
    - iris
    - pupil

    The iris is centered around gaze_direction.
    """
    if radius <= 0.0:
        raise ValueError("Eyeball radius must be positive.")

    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=float(radius))
    vertices = sphere.vertices.copy()

    rotation = _rotation_from_z_to_vector(np.array(gaze_direction, dtype=np.float64))
    vertices = (rotation @ vertices.T).T
    vertices = vertices + np.asarray(center, dtype=np.float64)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=sphere.faces.copy(),
        process=False,
    )

    local_dirs = vertices - np.asarray(center, dtype=np.float64)
    local_dirs = local_dirs / np.clip(
        np.linalg.norm(local_dirs, axis=1, keepdims=True),
        1e-8,
        None,
    )

    gaze_dir = _safe_normalize(np.asarray(gaze_direction, dtype=np.float64))
    cos_to_gaze = np.clip(local_dirs @ gaze_dir, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_to_gaze))

    iris_mask = angles <= float(iris_angle_degrees)
    pupil_mask = angles <= float(pupil_angle_degrees)

    vertex_colors = np.zeros((vertices.shape[0], 4), dtype=np.uint8)
    vertex_colors[:, :3] = np.array(sclera_rgb, dtype=np.uint8)
    vertex_colors[:, 3] = 255

    vertex_colors[iris_mask, :3] = np.array(iris_rgb, dtype=np.uint8)
    vertex_colors[pupil_mask, :3] = np.array(pupil_rgb, dtype=np.uint8)

    mesh.visual.vertex_colors = vertex_colors
    return mesh


def estimate_eye_centers_and_radii_from_landmarks(
    landmarks: np.ndarray,
    center_depth_offset_scale: float = 0.65,
    radius_from_width_scale: float = 0.32,
) -> Dict[str, np.ndarray | float]:
    """
    Estimate eyeball centers and radii from 68-point landmarks.

    Expected landmark indices:
    - left eye: 36:42
    - right eye: 42:48
    - left width landmarks: 36, 39
    - right width landmarks: 42, 45

    This function assumes the landmarks are already in the render/world frame
    where the camera faces roughly toward +Z, so eyeball centers are moved
    slightly backward along -Z.
    """
    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError("landmarks must have shape (N, 3)")
    if landmarks.shape[0] < 48:
        raise ValueError("Expected at least 48 landmarks for eye estimation.")

    left_eye_ring = landmarks[36:42]
    right_eye_ring = landmarks[42:48]

    left_width = float(np.linalg.norm(landmarks[36] - landmarks[39]))
    right_width = float(np.linalg.norm(landmarks[42] - landmarks[45]))

    left_radius = max(1e-5, radius_from_width_scale * left_width)
    right_radius = max(1e-5, radius_from_width_scale * right_width)

    left_center = np.mean(left_eye_ring, axis=0)
    right_center = np.mean(right_eye_ring, axis=0)

    left_center = left_center + np.array(
        [0.0, 0.0, -center_depth_offset_scale * left_radius],
        dtype=np.float64,
    )
    right_center = right_center + np.array(
        [0.0, 0.0, -center_depth_offset_scale * right_radius],
        dtype=np.float64,
    )

    return {
        "left_center": left_center,
        "right_center": right_center,
        "left_radius": left_radius,
        "right_radius": right_radius,
    }


def build_eye_meshes_from_landmarks(
    landmarks: np.ndarray,
    appearance_render_plan: Optional[Dict[str, Any]],
    gaze_direction: Sequence[float] = (0.0, 0.0, 1.0),
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Build left and right eyeball meshes from landmarks and appearance metadata.

    Pass landmarks AFTER your demo/world transform so the default gaze_direction
    of +Z points toward the camera in the current render setup.
    """
    params = estimate_eye_centers_and_radii_from_landmarks(landmarks)

    iris_rgb = [90, 90, 90]
    if appearance_render_plan is not None and "eye_color_rgb" in appearance_render_plan:
        iris_rgb = list(appearance_render_plan["eye_color_rgb"])

    left_mesh = build_colored_eyeball_mesh(
        center=np.asarray(params["left_center"], dtype=np.float64),
        radius=float(params["left_radius"]),
        iris_rgb=iris_rgb,
        gaze_direction=gaze_direction,
    )

    right_mesh = build_colored_eyeball_mesh(
        center=np.asarray(params["right_center"], dtype=np.float64),
        radius=float(params["right_radius"]),
        iris_rgb=iris_rgb,
        gaze_direction=gaze_direction,
    )

    return left_mesh, right_mesh


def render_meshes_offscreen(
    meshes: List[trimesh.Trimesh],
    image_size: Tuple[int, int] = (720, 720),
) -> np.ndarray:
    """
    Render multiple trimesh meshes in one scene.
    This is the multi-mesh replacement for the single-mesh renderer.
    """
    if len(meshes) == 0:
        raise ValueError("meshes must contain at least one mesh.")

    scene = pyrender.Scene(
        bg_color=[0.96, 0.96, 0.96, 1.0],
        ambient_light=[0.22, 0.22, 0.22],
    )

    all_vertices = np.concatenate([m.vertices for m in meshes], axis=0)
    extents = np.ptp(all_vertices, axis=0)
    scale = float(max(extents.max(), 1e-3))

    for tri_mesh in meshes:
        scene.add(pyrender.Mesh.from_trimesh(tri_mesh, smooth=True))

    camera = pyrender.PerspectiveCamera(
        yfov=np.pi / 5.2,
        znear=0.01,
        zfar=10.0 * scale,
    )

    camera_pose = np.eye(4)
    camera_pose[1, 3] = 0.02 * scale
    camera_pose[2, 3] = 2.15 * scale
    scene.add(camera, pose=camera_pose)

    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.2)
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.6)
    rim_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.4)

    key_pose = (
        transformations.rotation_matrix(np.radians(-25.0), [1, 0, 0])
        @ transformations.rotation_matrix(np.radians(35.0), [0, 1, 0])
    )
    key_pose[2, 3] = 2.2 * scale

    fill_pose = (
        transformations.rotation_matrix(np.radians(-10.0), [1, 0, 0])
        @ transformations.rotation_matrix(np.radians(-40.0), [0, 1, 0])
    )
    fill_pose[2, 3] = 2.4 * scale

    rim_pose = (
        transformations.rotation_matrix(np.radians(20.0), [1, 0, 0])
        @ transformations.rotation_matrix(np.radians(180.0), [0, 1, 0])
    )
    rim_pose[2, 3] = 2.8 * scale

    scene.add(key_light, pose=key_pose)
    scene.add(fill_light, pose=fill_pose)
    scene.add(rim_light, pose=rim_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=image_size[0],
        viewport_height=image_size[1],
        point_size=1.0,
    )
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    return color
