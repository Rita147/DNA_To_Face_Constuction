from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyrender
import trimesh
import trimesh.transformations as transformations

from pathlib import Path

HAIR_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "hair" / "wolf_haircut.glb"

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


def _rgb_to_float(rgb: Sequence[float]) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


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


def _select_region_vertices(
    vertices: np.ndarray,
    mask: np.ndarray,
    fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape[0] == verts.shape[0] and np.any(mask_bool):
        return verts[mask_bool]
    if fallback is not None and len(fallback) > 0:
        return np.asarray(fallback, dtype=np.float64)
    return verts


def _compute_head_hair_fit_guides(vertices: np.ndarray) -> Dict[str, float]:
    verts = np.asarray(vertices, dtype=np.float64)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    head_height = float(maxs[1] - mins[1])
    head_depth = float(maxs[2] - mins[2])

    x_norm, y_norm, z_norm = _normalize_face_vertex_coordinates(verts)

    upper_head_mask = y_norm > 0.58
    crown_mask = y_norm > 0.86
    temple_mask = (
        (y_norm > 0.50)
        & (y_norm < 0.82)
        & (np.abs(x_norm) > 0.36)
        & (np.abs(x_norm) < 0.90)
    )
    forehead_mask = (
        (y_norm > 0.62)
        & (y_norm < 0.90)
        & (np.abs(x_norm) < 0.54)
        & (z_norm > 0.46)
    )

    upper_head_pts = _select_region_vertices(verts, upper_head_mask)
    crown_pts = _select_region_vertices(verts, crown_mask, fallback=upper_head_pts)
    temple_pts = _select_region_vertices(verts, temple_mask, fallback=upper_head_pts)
    forehead_pts = _select_region_vertices(verts, forehead_mask, fallback=upper_head_pts)

    upper_width = float(
        np.quantile(upper_head_pts[:, 0], 0.90)
        - np.quantile(upper_head_pts[:, 0], 0.10)
    )
    temple_width = float(
        np.quantile(temple_pts[:, 0], 0.90)
        - np.quantile(temple_pts[:, 0], 0.10)
    )

    scalp_width = max(1e-6, max(upper_width * 1.24, temple_width * 1.26))
    front_z = float(np.quantile(forehead_pts[:, 2], 0.60))
    crown_z = float(np.median(crown_pts[:, 2]))

    return {
        "center_x": float(np.median(crown_pts[:, 0])),
        "anchor_y": float(maxs[1] - 0.078 * head_height),
        "anchor_z": float(0.60 * front_z + 0.40 * crown_z - 0.004 * head_depth),
        "scalp_width": scalp_width,
    }


def _compute_asset_hair_fit_guides(vertices: np.ndarray) -> Dict[str, float]:
    verts = np.asarray(vertices, dtype=np.float64)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)

    x_norm, y_norm, z_norm = _normalize_face_vertex_coordinates(verts)

    upper_cap_mask = y_norm > 0.70
    crown_mask = y_norm > 0.88
    frontal_root_mask = (
        (y_norm > 0.58)
        & (y_norm < 0.84)
        & (np.abs(x_norm) < 0.60)
        & (z_norm > 0.60)
    )

    upper_cap_pts = _select_region_vertices(verts, upper_cap_mask)
    crown_pts = _select_region_vertices(verts, crown_mask, fallback=upper_cap_pts)
    frontal_root_pts = _select_region_vertices(
        verts,
        frontal_root_mask,
        fallback=upper_cap_pts,
    )

    upper_z = float(np.median(upper_cap_pts[:, 2]))
    front_z = float(np.quantile(frontal_root_pts[:, 2], 0.45))

    return {
        "anchor_x": float(np.median(crown_pts[:, 0])),
        "anchor_y": float(np.quantile(upper_cap_pts[:, 1], 0.70)),
        "anchor_z": float(0.80 * upper_z + 0.20 * front_z),
        "width": float(maxs[0] - mins[0]),
        "height": float(maxs[1] - mins[1]),
        "depth": float(maxs[2] - mins[2]),
    }


def _apply_hair_root_fit(vertices: np.ndarray) -> np.ndarray:
    fitted = np.asarray(vertices, dtype=np.float64).copy()
    x_norm, y_norm, _ = _normalize_face_vertex_coordinates(fitted)

    root_contact = _smoothstep(0.54, 0.94, y_norm)
    side_contact = root_contact * _smoothstep(0.34, 0.90, np.abs(x_norm))
    crown_contact = root_contact * (
        1.0 - 0.45 * _smoothstep(0.30, 0.92, np.abs(x_norm))
    )

    fitted[:, 0] *= 1.0 - 0.055 * side_contact
    fitted[:, 2] *= 1.0 - 0.085 * crown_contact
    fitted[:, 1] -= 0.012 * float(np.ptp(fitted[:, 1])) * crown_contact

    return fitted


def _build_hair_vertex_colors(
    vertices: np.ndarray,
    hair_rgb_uint8: Sequence[float],
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    base_rgb = _rgb_to_float(hair_rgb_uint8)
    x_norm, y_norm, z_norm = _normalize_face_vertex_coordinates(verts)

    strand_noise = _vertex_hash_noise(
        verts,
        scale=210.0,
        offset=(1.7, 4.9, 8.3),
    )
    clump_noise = _vertex_hash_noise(
        verts,
        scale=72.0,
        offset=(6.2, 2.7, 3.9),
    )

    brightness = float(
        np.dot(base_rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
    )
    highlight_strength = 0.04 + 0.10 * brightness
    shadow_strength = 0.09 - 0.03 * brightness

    root_shadow = _smoothstep(0.38, 0.96, y_norm)
    tip_lift = _smoothstep(0.10, 0.92, 1.0 - y_norm)
    crown_highlight = np.exp(
        -(
            ((x_norm / 0.72) ** 2)
            + (((y_norm - 0.84) / 0.22) ** 2)
            + (((z_norm - 0.70) / 0.32) ** 2)
        )
    )
    side_shadow = _smoothstep(0.42, 1.00, np.abs(x_norm))

    value_scale = (
        0.92
        + highlight_strength * crown_highlight
        + 0.05 * tip_lift
        + 0.08 * (clump_noise - 0.5)
        + 0.04 * (strand_noise - 0.5)
        - shadow_strength * root_shadow
        - 0.04 * side_shadow
    )
    value_scale = np.clip(value_scale, 0.68, 1.12)

    vertex_rgb = base_rgb[None, :] * value_scale[:, None]

    warm_target = np.clip(
        base_rgb * np.array([1.05, 1.00, 0.93], dtype=np.float32),
        0.0,
        1.0,
    )
    warm_mix = np.clip(0.03 + 0.05 * crown_highlight + 0.02 * tip_lift, 0.0, 0.10)
    vertex_rgb = (
        (1.0 - warm_mix[:, None]) * vertex_rgb
        + warm_mix[:, None] * warm_target[None, :]
    )

    vertex_rgba = np.concatenate(
        [
            np.clip(vertex_rgb, 0.0, 1.0),
            np.ones((verts.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    return rgba_float_to_uint8(vertex_rgba)


def build_hair_mesh(
    vertices: np.ndarray,
    appearance_render_plan: Optional[Dict[str, Any]],
) -> Optional[trimesh.Trimesh]:
    """
    Load a single hair asset, fit it to the upper skull, and tint it
    using the dataset hair color.
    """

    if appearance_render_plan is None:
        return None

    if not HAIR_ASSET_PATH.exists():
        print(f"[appearance_scene] Hair asset not found: {HAIR_ASSET_PATH}")
        return None

    hair_mesh = load_asset_as_single_trimesh(HAIR_ASSET_PATH)
    if hair_mesh is None:
        return None

    hair_rgb = appearance_render_plan.get("hair_color_rgb", [92, 62, 42])
    head_guides = _compute_head_hair_fit_guides(vertices)
    asset_guides = _compute_asset_hair_fit_guides(hair_mesh.vertices)

    if (
        asset_guides["width"] <= 1e-8
        or asset_guides["height"] <= 1e-8
        or asset_guides["depth"] <= 1e-8
    ):
        print("[appearance_scene] Hair asset has invalid bounds.")
        return None

    hair_mesh.vertices -= np.array(
        [
            asset_guides["anchor_x"],
            asset_guides["anchor_y"],
            asset_guides["anchor_z"],
        ],
        dtype=np.float64,
    )[None, :]

    hair_thickness_label = str(
        appearance_render_plan.get("hair_thickness_label", "medium")
    ).strip().lower()
    hair_texture_label = str(
        appearance_render_plan.get("hair_texture_label", "unknown")
    ).strip().lower()

    thickness_scale = {
        "fine": 0.97,
        "medium": 1.00,
        "thick": 1.04,
    }.get(hair_thickness_label, 1.00)

    texture_scale = {
        "straight": 0.99,
        "wavy": 1.00,
        "curly": 1.02,
    }.get(hair_texture_label, 1.00)

    head_width = float(vertices[:, 0].max() - vertices[:, 0].min())
    target_hair_width = max(head_guides["scalp_width"], 0.98 * head_width)
    uniform_scale = (
        target_hair_width / asset_guides["width"]
    ) * thickness_scale * texture_scale
    hair_mesh.vertices *= float(uniform_scale)
    hair_mesh.vertices = _apply_hair_root_fit(hair_mesh.vertices)
    hair_mesh.visual.vertex_colors = _build_hair_vertex_colors(
        hair_mesh.vertices,
        hair_rgb,
    )

    hair_mesh.vertices += np.array(
        [
            head_guides["center_x"],
            head_guides["anchor_y"],
            head_guides["anchor_z"],
        ],
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


def _interpolate_curve_y(
    x_ctrl: np.ndarray,
    y_ctrl: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    order = np.argsort(x_ctrl)
    x_sorted = np.asarray(x_ctrl[order], dtype=np.float64)
    y_sorted = np.asarray(y_ctrl[order], dtype=np.float64)

    unique_x: List[float] = []
    unique_y: List[float] = []
    for x_val, y_val in zip(x_sorted.tolist(), y_sorted.tolist()):
        if unique_x and abs(x_val - unique_x[-1]) < 1e-8:
            unique_y[-1] = 0.5 * (unique_y[-1] + y_val)
        else:
            unique_x.append(float(x_val))
            unique_y.append(float(y_val))

    if len(unique_x) == 1:
        return np.full_like(x_query, unique_y[0], dtype=np.float64)

    return np.interp(
        np.asarray(x_query, dtype=np.float64),
        np.asarray(unique_x, dtype=np.float64),
        np.asarray(unique_y, dtype=np.float64),
    )


def build_colored_eyeball_mesh(
    eye_landmarks: np.ndarray,
    iris_rgb: Sequence[int],
    gaze_direction: Sequence[float] = (0.0, 0.0, 1.0),
    sclera_rgb: Sequence[int] = (238, 239, 235),
    pupil_rgb: Sequence[int] = (18, 14, 12),
    grid_cols: int = 28,
    grid_rows: int = 11,
) -> trimesh.Trimesh:
    """
    Build one visible eye-surface mesh shaped by the eyelid landmarks.

    This intentionally renders the exposed eye surface rather than a full
    spherical eyeball so the visible silhouette reads like an eye opening,
    not a round sphere sitting in the socket.
    """
    eye_landmarks = np.asarray(eye_landmarks, dtype=np.float64)
    if eye_landmarks.shape != (6, 3):
        raise ValueError("eye_landmarks must have shape (6, 3).")

    gaze_dir = _safe_normalize(np.asarray(gaze_direction, dtype=np.float64))

    corner_start = eye_landmarks[0]
    corner_end = eye_landmarks[3]
    upper_mid = np.mean(eye_landmarks[1:3], axis=0)
    lower_mid = np.mean(eye_landmarks[4:6], axis=0)

    width_vec = corner_end - corner_start
    width = float(np.linalg.norm(width_vec))
    if width <= 1e-8:
        raise ValueError("Eye landmarks produced a degenerate eye width.")

    eye_right = width_vec - gaze_dir * float(np.dot(width_vec, gaze_dir))
    if np.linalg.norm(eye_right) < 1e-8:
        eye_right = width_vec
    eye_right = _safe_normalize(eye_right)

    vertical_hint = upper_mid - lower_mid
    eye_up = (
        vertical_hint
        - gaze_dir * float(np.dot(vertical_hint, gaze_dir))
        - eye_right * float(np.dot(vertical_hint, eye_right))
    )
    if np.linalg.norm(eye_up) < 1e-8:
        eye_up = np.cross(gaze_dir, eye_right)
    eye_up = _safe_normalize(eye_up)
    if float(np.dot(eye_up, vertical_hint)) < 0.0:
        eye_up = -eye_up

    eye_forward = _safe_normalize(np.cross(eye_right, eye_up))
    if float(np.dot(eye_forward, gaze_dir)) < 0.0:
        eye_forward = -eye_forward
        eye_up = _safe_normalize(np.cross(eye_forward, eye_right))

    origin = 0.25 * (corner_start + corner_end + upper_mid + lower_mid)

    local_x = (eye_landmarks - origin[None, :]) @ eye_right
    local_y = (eye_landmarks - origin[None, :]) @ eye_up
    local_z = (eye_landmarks - origin[None, :]) @ eye_forward

    upper_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    lower_indices = np.array([0, 5, 4, 3], dtype=np.int64)

    x_min = float(min(local_x[0], local_x[3]))
    x_max = float(max(local_x[0], local_x[3]))
    x_grid = np.linspace(x_min, x_max, int(grid_cols), dtype=np.float64)

    upper_y = _interpolate_curve_y(local_x[upper_indices], local_y[upper_indices], x_grid)
    lower_y = _interpolate_curve_y(local_x[lower_indices], local_y[lower_indices], x_grid)
    upper_z = _interpolate_curve_y(local_x[upper_indices], local_z[upper_indices], x_grid)
    lower_z = _interpolate_curve_y(local_x[lower_indices], local_z[lower_indices], x_grid)

    mid_opening_height = float(
        np.linalg.norm(upper_mid - lower_mid)
    )
    mid_aperture = max(0.5 * mid_opening_height, 1e-5)
    half_width = max(0.5 * width, 1e-5)
    sclera_rgb_float = _rgb_to_float(sclera_rgb)
    iris_rgb_float = _rgb_to_float(iris_rgb)
    pupil_rgb_float = _rgb_to_float(pupil_rgb)

    vertices_local: List[np.ndarray] = []
    local_samples: List[Tuple[float, float, float, float]] = []
    faces: List[List[int]] = []

    base_depth = 0.11 * width
    cornea_depth = 0.022 * width
    iris_center_y = 0.10 * mid_aperture
    iris_radius = max(
        min(0.24 * width, 1.12 * mid_aperture),
        0.16 * width,
    )
    pupil_radius = 0.34 * iris_radius

    for col_idx, x_val in enumerate(x_grid):
        y_upper = float(upper_y[col_idx])
        y_lower = float(lower_y[col_idx])
        z_upper = float(upper_z[col_idx])
        z_lower = float(lower_z[col_idx])
        if y_upper < y_lower:
            y_upper, y_lower = y_lower, y_upper
            z_upper, z_lower = z_lower, z_upper

        aperture_half = max(0.5 * (y_upper - y_lower), 1e-5)
        center_y = 0.5 * (y_upper + y_lower)

        for row_idx, t in enumerate(np.linspace(0.0, 1.0, int(grid_rows), dtype=np.float64)):
            y_val = y_lower + t * (y_upper - y_lower)

            x_norm = x_val / half_width
            y_norm = (y_val - center_y) / aperture_half
            surface_term = max(0.0, 1.0 - x_norm ** 2 - 0.72 * y_norm ** 2)

            iris_dx = x_val
            iris_dy = y_val - iris_center_y
            iris_r_norm = np.sqrt(iris_dx ** 2 + iris_dy ** 2) / max(iris_radius, 1e-6)
            cornea_term = max(0.0, 1.0 - iris_r_norm ** 2)

            rim_z = z_lower + t * (z_upper - z_lower)
            z_val = (
                rim_z
                + base_depth * np.sqrt(surface_term)
                + cornea_depth * (cornea_term ** 2)
            )

            vertices_local.append(
                x_val * eye_right + y_val * eye_up + z_val * eye_forward
            )
            local_samples.append((x_norm, y_norm, iris_dx, iris_dy))

            if col_idx < len(x_grid) - 1 and row_idx < int(grid_rows) - 1:
                idx = col_idx * int(grid_rows) + row_idx
                faces.append([idx, idx + int(grid_rows), idx + 1])
                faces.append(
                    [
                        idx + int(grid_rows),
                        idx + int(grid_rows) + 1,
                        idx + 1,
                    ]
                )

    vertices = origin[None, :] + np.asarray(vertices_local, dtype=np.float64)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )

    local_sample_arr = np.asarray(local_samples, dtype=np.float64)
    x_norm = local_sample_arr[:, 0]
    y_norm = local_sample_arr[:, 1]
    iris_dx = local_sample_arr[:, 2]
    iris_dy = local_sample_arr[:, 3]

    iris_r_norm = np.sqrt(iris_dx ** 2 + iris_dy ** 2) / max(iris_radius, 1e-6)
    iris_mask = iris_r_norm <= 1.0
    pupil_mask = iris_r_norm <= (pupil_radius / max(iris_radius, 1e-6))
    iris_only_mask = iris_mask & ~pupil_mask

    vertex_rgb = np.tile(sclera_rgb_float[None, :], (vertices.shape[0], 1))

    upper_lid_shadow = _smoothstep(0.00, 0.95, y_norm)
    eye_corner_shadow = _smoothstep(0.42, 1.00, np.abs(x_norm))
    lower_sclera_lift = _smoothstep(0.12, 1.00, -y_norm)
    sclera_shade = (
        0.95
        - 0.22 * upper_lid_shadow
        - 0.05 * eye_corner_shadow
        + 0.04 * lower_sclera_lift
    )
    vertex_rgb *= np.clip(sclera_shade[:, None], 0.70, 1.02)

    iris_azimuth = np.arctan2(iris_dy, iris_dx + 1e-8)
    iris_spokes = 0.91 + 0.09 * np.cos(iris_azimuth * 11.0)
    iris_core_lift = 0.82 + 0.20 * (1.0 - np.clip(iris_r_norm, 0.0, 1.0))
    iris_top_shadow = 1.0 - 0.18 * upper_lid_shadow
    limbal_ring = _smoothstep(0.70, 1.00, np.clip(iris_r_norm, 0.0, 1.0))

    iris_colors = iris_rgb_float[None, :] * (
        iris_spokes * iris_core_lift * iris_top_shadow
    )[:, None]
    iris_colors = iris_colors * (1.0 - 0.30 * limbal_ring[:, None])
    iris_colors = np.clip(iris_colors, 0.0, 1.0)
    vertex_rgb[iris_only_mask] = iris_colors[iris_only_mask]

    pupil_colors = pupil_rgb_float[None, :] * (
        0.92 - 0.10 * upper_lid_shadow
    )[:, None]
    pupil_colors = np.clip(pupil_colors, 0.0, 1.0)
    vertex_rgb[pupil_mask] = pupil_colors[pupil_mask]

    highlight_x = -0.18 * iris_radius
    highlight_y = 0.24 * iris_radius
    highlight_r = np.sqrt(
        ((iris_dx - highlight_x) / max(0.34 * iris_radius, 1e-6)) ** 2
        + ((iris_dy - highlight_y) / max(0.24 * iris_radius, 1e-6)) ** 2
    )
    highlight_strength = 1.0 - _smoothstep(0.0, 1.0, highlight_r)
    highlight_strength *= (
        0.70 * iris_mask.astype(np.float32) + 0.16 * (~iris_mask).astype(np.float32)
    )
    highlight_mix = 0.40 * highlight_strength
    vertex_rgb = (
        (1.0 - highlight_mix[:, None]) * vertex_rgb
        + highlight_mix[:, None] * np.ones((1, 3), dtype=np.float32)
    )

    vertex_rgba = np.concatenate(
        [
            np.clip(vertex_rgb, 0.0, 1.0),
            np.ones((vertices.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    mesh.visual.vertex_colors = rgba_float_to_uint8(vertex_rgba)
    return mesh


def estimate_eye_centers_and_radii_from_landmarks(
    landmarks: np.ndarray,
    center_depth_offset_scale: float = 0.58,
    radius_from_width_scale: float = 0.36,
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
    iris_rgb = [90, 90, 90]
    if appearance_render_plan is not None and "eye_color_rgb" in appearance_render_plan:
        iris_rgb = list(appearance_render_plan["eye_color_rgb"])

    base_gaze = _safe_normalize(np.asarray(gaze_direction, dtype=np.float64))
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    gaze_right = np.cross(world_up, base_gaze)
    if np.linalg.norm(gaze_right) < 1e-8:
        gaze_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    gaze_right = _safe_normalize(gaze_right)
    gaze_up = _safe_normalize(np.cross(base_gaze, gaze_right))
    base_gaze = _safe_normalize(base_gaze - 0.03 * gaze_up)

    vergence = 0.035
    left_gaze = _safe_normalize(base_gaze + vergence * gaze_right)
    right_gaze = _safe_normalize(base_gaze - vergence * gaze_right)

    left_mesh = build_colored_eyeball_mesh(
        eye_landmarks=np.asarray(landmarks[36:42], dtype=np.float64),
        iris_rgb=iris_rgb,
        gaze_direction=left_gaze,
    )

    right_mesh = build_colored_eyeball_mesh(
        eye_landmarks=np.asarray(landmarks[42:48], dtype=np.float64),
        iris_rgb=iris_rgb,
        gaze_direction=right_gaze,
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
