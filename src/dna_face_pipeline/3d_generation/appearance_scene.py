from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyrender
import trimesh
import trimesh.transformations as transformations

from pathlib import Path

HAIR_ASSET_PATH = Path(__file__).resolve().parents[4] / "assets" / "hair" / "wolf_haircut.glb"

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

def build_hair_mesh(
    vertices: np.ndarray,
    appearance_render_plan: Optional[Dict[str, Any]],
) -> Optional[trimesh.Trimesh]:
    """
    Load a hair asset, scale/translate it to the FLAME head,
    and tint it using hair_color_rgb from appearance_render_plan.

    First-pass demonstration using a single hair mesh asset.
    """

    if appearance_render_plan is None:
        return None

    if not HAIR_ASSET_PATH.exists():
        print(f"[appearance_scene] Hair asset not found: {HAIR_ASSET_PATH}")
        return None

    hair_mesh = load_asset_as_single_trimesh(HAIR_ASSET_PATH)
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
    # Optional asset orientation correction
    # ------------------------------------------------------------
    # rot = trimesh.transformations.rotation_matrix(np.radians(180.0), [0, 1, 0])
    # hair_mesh.apply_transform(rot)

    # rot = trimesh.transformations.rotation_matrix(np.radians(90.0), [1, 0, 0])
    # hair_mesh.apply_transform(rot)

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
    uniform_scale = 0.96 * head_width / hair_width
    hair_mesh.vertices *= float(uniform_scale)

    # ------------------------------------------------------------
    # Place on FLAME head
    # ------------------------------------------------------------
    target_center_x = 0.5 * (x_min + x_max)
    target_anchor_y = y_max - 0.02 * head_height
    target_center_z = 0.5 * (z_min + z_max) - 0.02 * head_depth

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
    Build a tighter, more natural freckle intensity map.

    Main goals:
    - keep freckles on nose bridge / upper nose / upper cheeks
    - avoid lower-face and mouth-adjacent placement
    - make spots smaller and less blotchy
    """
    if freckle_level == "none":
        return np.zeros_like(x_norm, dtype=np.float32)

    front = _smoothstep(0.35, 0.92, z_norm)

    # Strongly suppress freckles below the upper-mid face
    upper_face_gate = _smoothstep(0.46, 0.60, y_norm)

    # Keep freckles near the central face width-wise
    central_width_gate = np.exp(-((x_norm / 0.42) ** 2))

    if freckle_level == "some":
        cluster_specs = [
            (0.00, 0.61, 0.055, 0.050, 0.95),  # bridge
            (0.00, 0.55, 0.065, 0.050, 0.70),  # upper nose
            (-0.17, 0.58, 0.080, 0.055, 0.72), # left upper cheek
            (0.17, 0.58, 0.080, 0.055, 0.72),  # right upper cheek
            (-0.25, 0.55, 0.070, 0.050, 0.38), # left outer upper cheek
            (0.25, 0.55, 0.070, 0.050, 0.38),  # right outer upper cheek
        ]
        noise_threshold = 0.982
    else:  # extensive
        cluster_specs = [
            (0.00, 0.62, 0.065, 0.055, 1.05),  # bridge
            (0.00, 0.56, 0.078, 0.058, 0.98),  # upper nose
            (-0.15, 0.60, 0.090, 0.060, 0.90), # left inner upper cheek
            (0.15, 0.60, 0.090, 0.060, 0.90),  # right inner upper cheek
            (-0.25, 0.57, 0.095, 0.060, 0.80), # left upper cheek
            (0.25, 0.57, 0.095, 0.060, 0.80),  # right upper cheek
            (-0.32, 0.54, 0.080, 0.055, 0.55), # left outer upper cheek
            (0.32, 0.54, 0.080, 0.055, 0.55),  # right outer upper cheek
        ]
        noise_threshold = 0.958

    cluster_field = np.zeros_like(x_norm, dtype=np.float32)
    for cx, cy, sx, sy, weight in cluster_specs:
        cluster_field += weight * np.exp(
            -(((x_norm - cx) / sx) ** 2 + ((y_norm - cy) / sy) ** 2)
        )

    cluster_field = np.clip(cluster_field, 0.0, 1.22)

    noise_primary = _vertex_hash_noise(
        vertices,
        scale=185.0,
        offset=(0.0, 0.0, 0.0),
    )
    noise_secondary = _vertex_hash_noise(
        vertices,
        scale=96.0,
        offset=(6.1, 8.7, 3.4),
    )

    discrete_spots = _smoothstep(noise_threshold, 1.0, noise_primary)
    spot_modulation = 0.85 + 0.15 * _smoothstep(0.40, 1.0, noise_secondary)

    return (
        cluster_field
        * discrete_spots
        * spot_modulation
        * front
        * upper_face_gate
        * central_width_gate
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
    - refined freckles with smaller clustered spots
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

        freckle_color = EXACT_FRECKLE_RGB.copy()

        if freckle_level == "some":
            freckle_threshold = 0.28
            freckle_alpha_base = 0.07 + 0.04 * tone_t

            freckle_alpha = freckle_alpha_base * _smoothstep(
                freckle_threshold - 0.035,
                freckle_threshold + 0.025,
                freckle_strength,
            )

            freckle_alpha = np.clip(freckle_alpha, 0.0, 0.24)

        else:  # extensive
            freckle_threshold = 0.14
            freckle_alpha_base = 0.14 + 0.07 * tone_t

            freckle_alpha = freckle_alpha_base * _smoothstep(
                freckle_threshold - 0.030,
                freckle_threshold + 0.040,
                freckle_strength,
            )

            freckle_alpha = np.clip(freckle_alpha, 0.0, 0.30)

        vertex_rgb = (
            (1.0 - freckle_alpha[:, None]) * vertex_rgb
            + freckle_alpha[:, None] * freckle_color[None, :]
        )

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
