from flame_pytorch.compat import apply_compat_patches
apply_compat_patches()

import imageio.v2 as imageio

from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
import torch
import pyrender
import trimesh
import trimesh.transformations as transformations

from flame_pytorch import FLAME, get_config
from .measurement_definitions import get_measurement_by_name, MeasurementDef
from .measurement_utils import (
    compute_measurements,
    build_measurement_metadata,
)


from .appearance_mappings import (
    extract_appearance_traits_from_row,
    build_appearance_render_plan,
)


from .appearance_scene import (
    build_face_mesh_with_skin_tone,
    build_eye_meshes_from_landmarks,
    build_hair_mesh,
    render_meshes_offscreen,
)

def assemble_shape_from_selected_pcs(
    theta: torch.Tensor,
    selected_pcs: List[int],
    total_shape_dims: int,
    device: torch.device,
) -> torch.Tensor:
    shape = torch.zeros(1, total_shape_dims, device=device)
    for j, pc_idx in enumerate(selected_pcs):
        shape[:, pc_idx] = theta[:, j]
    return shape


def build_measurement_weight_tensor(
    measurement_defs: List[MeasurementDef],
    weight_overrides: Dict[str, float],
    default_weight: float,
    device: torch.device,
) -> torch.Tensor:
    weights = []
    for m in measurement_defs:
        weights.append(float(weight_overrides.get(str(m["name"]), default_weight)))
    return torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(0)


def build_target_measurements_from_relative_controls(
    neutral_measurements: torch.Tensor,
    measurement_names: List[str],
    relative_controls: Dict[str, float],
) -> torch.Tensor:
    target = neutral_measurements.clone()

    for idx, name in enumerate(measurement_names):
        offset = float(relative_controls.get(name, 0.0))
        target[:, idx] = neutral_measurements[:, idx] * (1.0 + offset)

    return target


def format_run_name(
    experiment_name: str,
    selected_pcs: List[int],
    lambda_reg: float,
    lambda_preserve: float,
) -> str:
    pc_count = len(selected_pcs)
    max_pc = max(selected_pcs) + 1 if selected_pcs else 0
    reg_part = f"reg{lambda_reg:g}"
    preserve_part = f"pres{lambda_preserve:g}"
    return f"run_{experiment_name}_pc{pc_count}_max{max_pc}_{reg_part}_{preserve_part}"


def visualize_result(
    vertices: np.ndarray,
    faces: np.ndarray,
    rotation_degrees: float = 70.0,
) -> None:
    vertex_colors = np.ones((vertices.shape[0], 4)) * [0.5, 0.5, 0.5, 0.85]
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)

    rot_matrix = transformations.rotation_matrix(
        np.radians(rotation_degrees), [1, 0, 0]
    )
    tri_mesh.apply_transform(rot_matrix)

    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(tri_mesh))

    print("Displaying generated result...")
    pyrender.Viewer(scene, use_raymond_lighting=True)


def scale_relative_controls(
    relative_controls: Dict[str, float],
    scale: float,
    max_abs_offset: float = 0.35,
) -> Dict[str, float]:
    scaled: Dict[str, float] = {}
    for name, value in relative_controls.items():
        new_value = float(value) * scale
        new_value = max(-max_abs_offset, min(max_abs_offset, new_value))
        scaled[name] = new_value
    return scaled


def scale_weight_overrides(
    weight_overrides: Dict[str, float],
    scale: float,
    min_weight: float = 0.1,
) -> Dict[str, float]:
    return {
        name: max(min_weight, float(weight) * scale)
        for name, weight in weight_overrides.items()
    }


def apply_demo_mode_to_experiment(
    experiment_config: Dict[str, object],
    control_scale: float = 1.6,
    measurement_weight_scale: float = 1.10,
    preserve_weight_scale: float = 0.80,
    max_abs_offset: float = 0.35,
) -> Dict[str, object]:
    return {
        "relative_controls": scale_relative_controls(
            experiment_config["relative_controls"],  # type: ignore
            scale=control_scale,
            max_abs_offset=max_abs_offset,
        ),
        "measurement_weight_overrides": scale_weight_overrides(
            experiment_config["measurement_weight_overrides"],  # type: ignore
            scale=measurement_weight_scale,
        ),
        "preserve_measurements": list(experiment_config["preserve_measurements"]),  # type: ignore
        "preserve_weight_overrides": scale_weight_overrides(
            experiment_config["preserve_weight_overrides"],  # type: ignore
            scale=preserve_weight_scale,
        ),
    }


def ease_in_out(t: float) -> float:
    return float(t * t * (3.0 - 2.0 * t))


def build_demo_transform(
    vertices: np.ndarray,
    pitch_degrees: float = -10.0,
    yaw_degrees: float = 0.0,
) -> np.ndarray:
    center = np.mean(vertices, axis=0)

    translate = transformations.translation_matrix(-center)
    rot_y = transformations.rotation_matrix(np.radians(yaw_degrees), [0, 1, 0])
    rot_x = transformations.rotation_matrix(np.radians(pitch_degrees), [1, 0, 0])

    return rot_x @ rot_y @ translate


def apply_transform_to_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    hom = np.concatenate([points, ones], axis=1)
    transformed = (transform @ hom.T).T
    return transformed[:, :3]


def build_demo_scene_meshes(
    vertices: np.ndarray,
    landmarks: np.ndarray,
    faces: np.ndarray,
    appearance_render_plan: Optional[Dict[str, Any]],
    pitch_degrees: float = -10.0,
    yaw_degrees: float = 0.0,
    extra_rotation_transform: Optional[np.ndarray] = None,
) -> List[trimesh.Trimesh]:
    """
    Build the full 3D scene meshes for one frame:
    - skintone-colored FLAME face mesh
    - left/right 3D eyeball meshes

    The demo transform places the head in the presentation frame.
    extra_rotation_transform is used for spin-GIF frames.
    """
    demo_transform = build_demo_transform(
        vertices=vertices,
        pitch_degrees=pitch_degrees,
        yaw_degrees=yaw_degrees,
    )

    transformed_landmarks = apply_transform_to_points(landmarks, demo_transform)

    gaze_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if extra_rotation_transform is not None:
        transformed_landmarks = apply_transform_to_points(
            transformed_landmarks,
            extra_rotation_transform,
        )
        gaze_direction = extra_rotation_transform[:3, :3] @ gaze_direction

    face_mesh = build_face_mesh_with_skin_tone(
        vertices=vertices,
        faces=faces,
        appearance_render_plan=appearance_render_plan,
    )
    face_mesh.apply_transform(demo_transform)

    if extra_rotation_transform is not None:
        face_mesh.apply_transform(extra_rotation_transform)

    left_eye_mesh, right_eye_mesh = build_eye_meshes_from_landmarks(
        landmarks=transformed_landmarks,
        appearance_render_plan=appearance_render_plan,
        gaze_direction=gaze_direction,
    )

    hair_mesh = build_hair_mesh(
        vertices=vertices,
        appearance_render_plan=appearance_render_plan,
    )

    if hair_mesh is not None:
        hair_mesh.apply_transform(demo_transform)
        if extra_rotation_transform is not None:
            hair_mesh.apply_transform(extra_rotation_transform)

    meshes = [face_mesh, left_eye_mesh, right_eye_mesh]
    if hair_mesh is not None:
        meshes.append(hair_mesh)
    return meshes


def save_morph_gif(
    flame: FLAME,
    final_theta: torch.Tensor,
    selected_pcs: List[int],
    total_shape_dims: int,
    zeros_exp: torch.Tensor,
    zeros_pose: torch.Tensor,
    device: torch.device,
    faces: np.ndarray,
    output_path: Path,
    num_frames: int = 30,
    fps: int = 12,
    image_size: Tuple[int, int] = (640, 640),
    appearance_render_plan: Optional[Dict[str, Any]] = None,
) -> None:
        frames: List[np.ndarray] = []

        with torch.no_grad():
            for alpha in np.linspace(0.0, 1.0, num_frames):
                eased = ease_in_out(float(alpha))
                current_theta = final_theta * eased

                current_shape = assemble_shape_from_selected_pcs(
                    theta=current_theta,
                    selected_pcs=selected_pcs,
                    total_shape_dims=total_shape_dims,
                    device=device,
                )

                current_vertices, current_landmarks = flame(
                    current_shape,
                    zeros_exp,
                    zeros_pose,
                )
                vertices_cpu = current_vertices[0].detach().cpu().numpy().squeeze()
                landmarks_cpu = current_landmarks[0].detach().cpu().numpy().squeeze()

                scene_meshes = build_demo_scene_meshes(
                    vertices=vertices_cpu,
                    landmarks=landmarks_cpu,
                    faces=faces,
                    appearance_render_plan=appearance_render_plan,
                    pitch_degrees=-10.0,
                    yaw_degrees=0.0,
                    extra_rotation_transform=None,
                )

                frame = render_meshes_offscreen(
                    meshes=scene_meshes,
                    image_size=image_size,
                )
                frames.append(frame)

        frames = [frames[0]] * 4 + frames + [frames[-1]] * 6
        imageio.mimsave(output_path, frames, fps=fps, loop=0)


def save_spin_gif(
    vertices: np.ndarray,
    faces: np.ndarray,
    landmarks: np.ndarray,
    output_path: Path,
    num_frames: int = 36,
    fps: int = 14,
    image_size: Tuple[int, int] = (640, 640),
    appearance_render_plan: Optional[Dict[str, Any]] = None,
) -> None:
        frames: List[np.ndarray] = []

        for angle_deg in np.linspace(0.0, 360.0, num_frames, endpoint=False):
            rot_y = transformations.rotation_matrix(
                np.radians(float(angle_deg)),
                [0, 1, 0],
            )

            scene_meshes = build_demo_scene_meshes(
                vertices=vertices,
                landmarks=landmarks,
                faces=faces,
                appearance_render_plan=appearance_render_plan,
                pitch_degrees=-10.0,
                yaw_degrees=0.0,
                extra_rotation_transform=rot_y,
            )

            frame = render_meshes_offscreen(
                meshes=scene_meshes,
                image_size=image_size,
            )
            frames.append(frame)

        imageio.mimsave(output_path, frames, fps=fps, loop=0)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(torch.sum(weights), min=1e-8)
    return torch.sum(values * weights) / denom


def build_desired_direction_tensor(
    measurement_names: List[str],
    relative_controls: Dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    directions: List[float] = []
    for name in measurement_names:
        offset = float(relative_controls.get(name, 0.0))
        if offset > 0.0:
            directions.append(1.0)
        elif offset < 0.0:
            directions.append(-1.0)
        else:
            directions.append(0.0)

    return torch.tensor(directions, dtype=torch.float32, device=device).unsqueeze(0)


def merge_nested_settings(
    base_settings: Dict[str, object],
    override_settings: Dict[str, object],
) -> Dict[str, object]:
    merged = {
        "pc_selection": dict(base_settings["pc_selection"]),   # type: ignore
        "optimization": dict(base_settings["optimization"]),   # type: ignore
        "enable_visualization": bool(base_settings["enable_visualization"]),
    }

    if "pc_selection" in override_settings:
        merged["pc_selection"].update(override_settings["pc_selection"])  # type: ignore

    if "optimization" in override_settings:
        merged["optimization"].update(override_settings["optimization"])  # type: ignore

    if "enable_visualization" in override_settings:
        merged["enable_visualization"] = bool(override_settings["enable_visualization"])

    return merged


def get_control_defaults(experiment_name: str) -> Dict[str, object]:
    """
    Region-aware defaults for:
    - automatic PC selection
    - optimizer settings
    """
    base_defaults: Dict[str, object] = {
        "pc_selection": {
            "mode": "auto_sensitivity",
            "candidate_pool_size": 100,
            "top_k": 32,
            "probe_value": 1.0,
            "preserve_tradeoff": 0.40,
            "normalization_eps": 1e-6,
        },
        "optimization": {
            "lambda_reg": 5e-5,
            "lambda_preserve": 0.4,
            "learning_rate": 0.05,
            "steps": 700,
        },
        "enable_visualization": True,
    }

    overrides_by_experiment: Dict[str, Dict[str, object]] = {
        "nose": {
            "pc_selection": {
                "top_k": 24,
                "candidate_pool_size": 80,
                "preserve_tradeoff": 0.50,
            }
        },
        "nose_width_only": {
            "pc_selection": {
                "top_k": 20,
                "candidate_pool_size": 80,
                "preserve_tradeoff": 0.55,
            }
        },
        "nose_height_only": {
            "pc_selection": {
                "top_k": 20,
                "candidate_pool_size": 80,
                "preserve_tradeoff": 0.55,
            }
        },
        "mouth": {
            "pc_selection": {
                "top_k": 28,
                "candidate_pool_size": 90,
                "preserve_tradeoff": 0.45,
            }
        },
        "mouth_width_only": {
            "pc_selection": {
                "top_k": 20,
                "candidate_pool_size": 90,
                "preserve_tradeoff": 0.60,
            }
        },
        "mouth_opening_only": {
            "pc_selection": {
                "top_k": 20,
                "candidate_pool_size": 90,
                "preserve_tradeoff": 0.55,
            }
        },
        "eyes_spacing": {
            "pc_selection": {
                "top_k": 18,
                "candidate_pcs": [pc for pc in range(80) if pc not in {0, 1}],
                "preserve_tradeoff": 0.75,
            }
        },
        "eyes_shape": {
            "pc_selection": {
                "top_k": 36,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.28,
            }
        },
        "chin": {
            "pc_selection": {
                "top_k": 24,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.55,
            }
        },
        "jaw": {
            "pc_selection": {
                "top_k": 24,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.55,
            }
        },
        "lower_face": {
            "pc_selection": {
                "top_k": 36,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.40,
            }
        },
        "brows": {
            "pc_selection": {
                "top_k": 28,
                "candidate_pool_size": 90,
                "preserve_tradeoff": 0.42,
            }
        },
        "global_face": {
            "pc_selection": {
                "top_k": 44,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.35,
            }
        },
        "nose_mouth_combo": {
            "pc_selection": {
                "top_k": 40,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.42,
            }
        },
        "eyes_brows_combo": {
            "pc_selection": {
                "top_k": 48,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.38,
            }
        },
        "eyes_brows_staged_combo": {
            "pc_selection": {
                "top_k": 48,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.38,
            }
        },
        "eyes_spacing_brows_combo": {
            "pc_selection": {
                "top_k": 40,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.45,
            }
        },
        "eyes_shape_brows_combo": {
            "pc_selection": {
                "top_k": 44,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.32,
            }
        },
        "eyes_shape_diagnostic": {
            "pc_selection": {
                "top_k": 40,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.22,
            }
        },
        "eyes_shape_diagnostic_soft": {
            "pc_selection": {
                "top_k": 40,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.22,
            }
        },
        "mouth_lowerface_combo": {
            "pc_selection": {
                "top_k": 44,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.40,
            }
        },
    }

    return merge_nested_settings(
        base_defaults,
        overrides_by_experiment.get(experiment_name, {}),
    )


def resolve_candidate_pcs(
    pc_selection_config: Dict[str, object],
    total_shape_dims: int,
) -> List[int]:
    if "candidate_pcs" in pc_selection_config:
        raw_candidate_pcs = pc_selection_config["candidate_pcs"]
        if not isinstance(raw_candidate_pcs, list):
            raise ValueError("pc_selection['candidate_pcs'] must be a list of ints.")
        candidate_pcs = [int(v) for v in raw_candidate_pcs]
    else:
        raw_candidate_pool_size = pc_selection_config.get(
            "candidate_pool_size",
            total_shape_dims,
        )
        if isinstance(raw_candidate_pool_size, list):
            raise ValueError(
                "pc_selection['candidate_pool_size'] must be an integer, not a list. "
                "If you want an explicit PC list, use 'candidate_pcs' instead."
            )

        candidate_pool_size = int(raw_candidate_pool_size)
        candidate_pool_size = min(candidate_pool_size, total_shape_dims)
        candidate_pcs = list(range(candidate_pool_size))

    deduped: List[int] = []
    seen: Set[int] = set()

    for pc_idx in candidate_pcs:
        if pc_idx < 0 or pc_idx >= total_shape_dims:
            raise ValueError(
                f"Invalid PC index {pc_idx}. Valid range: [0, {total_shape_dims - 1}]"
            )
        if pc_idx not in seen:
            seen.add(pc_idx)
            deduped.append(pc_idx)

    if not deduped:
        raise ValueError("Candidate PC set is empty.")

    return deduped


def select_shape_pcs_for_experiment(
    flame: FLAME,
    total_shape_dims: int,
    zeros_exp: torch.Tensor,
    zeros_pose: torch.Tensor,
    selected_measurements: List[MeasurementDef],
    preserve_measurements: List[MeasurementDef],
    neutral_measurements: torch.Tensor,
    neutral_preserve_measurements: torch.Tensor,
    measurement_weights: torch.Tensor,
    preserve_weights: torch.Tensor,
    measurement_names: List[str],
    relative_controls: Dict[str, float],
    pc_selection_config: Dict[str, object],
    device: torch.device,
) -> Tuple[List[int], List[Dict[str, object]]]:
    mode = str(pc_selection_config.get("mode", "auto_sensitivity"))

    if mode == "manual":
        if "selected_pcs" in pc_selection_config:
            raw_selected_pcs = pc_selection_config["selected_pcs"]
            if not isinstance(raw_selected_pcs, list):
                raise ValueError("pc_selection['selected_pcs'] must be a list of ints.")
            selected_pcs = [int(v) for v in raw_selected_pcs]
        else:
            manual_pc_count = int(pc_selection_config.get("manual_pc_count", 60))
            selected_pcs = list(range(min(manual_pc_count, total_shape_dims)))

        diagnostics = [
            {
                "pc_idx": int(pc_idx),
                "selection_mode": "manual",
            }
            for pc_idx in selected_pcs
        ]
        return selected_pcs, diagnostics

    if mode != "auto_sensitivity":
        raise ValueError(f"Unsupported pc_selection mode: {mode}")

    candidate_pcs = resolve_candidate_pcs(pc_selection_config, total_shape_dims)
    top_k = int(pc_selection_config.get("top_k", 32))
    top_k = min(top_k, len(candidate_pcs))

    probe_value = float(pc_selection_config.get("probe_value", 1.0))
    preserve_tradeoff = float(pc_selection_config.get("preserve_tradeoff", 0.40))
    normalization_eps = float(pc_selection_config.get("normalization_eps", 1e-6))

    if probe_value <= 0.0:
        raise ValueError("pc_selection['probe_value'] must be > 0.")

    desired_directions = build_desired_direction_tensor(
        measurement_names=measurement_names,
        relative_controls=relative_controls,
        device=device,
    )

    target_scale = torch.clamp(torch.abs(neutral_measurements), min=normalization_eps)
    preserve_scale = torch.clamp(
        torch.abs(neutral_preserve_measurements),
        min=normalization_eps,
    )

    diagnostics: List[Dict[str, object]] = []

    with torch.no_grad():
        for pc_idx in candidate_pcs:
            plus_shape = torch.zeros(1, total_shape_dims, device=device)
            minus_shape = torch.zeros(1, total_shape_dims, device=device)

            plus_shape[:, pc_idx] = probe_value
            minus_shape[:, pc_idx] = -probe_value

            _, plus_landmarks = flame(plus_shape, zeros_exp, zeros_pose)
            _, minus_landmarks = flame(minus_shape, zeros_exp, zeros_pose)

            plus_target = compute_measurements(plus_landmarks, selected_measurements)
            minus_target = compute_measurements(minus_landmarks, selected_measurements)

            target_derivative = (plus_target - minus_target) / (2.0 * probe_value)
            normalized_target_derivative = target_derivative / target_scale

            aligned_if_positive_theta = torch.relu(
                normalized_target_derivative * desired_directions
            )
            aligned_if_negative_theta = torch.relu(
                (-normalized_target_derivative) * desired_directions
            )

            target_alignment_score_pos = float(
                weighted_mean(aligned_if_positive_theta, measurement_weights).item()
            )
            target_alignment_score_neg = float(
                weighted_mean(aligned_if_negative_theta, measurement_weights).item()
            )

            if target_alignment_score_pos >= target_alignment_score_neg:
                best_target_alignment_score = target_alignment_score_pos
                preferred_theta_sign = +1
            else:
                best_target_alignment_score = target_alignment_score_neg
                preferred_theta_sign = -1

            target_magnitude_score = float(
                weighted_mean(
                    torch.abs(normalized_target_derivative),
                    measurement_weights,
                ).item()
            )

            if len(preserve_measurements) > 0:
                plus_preserve = compute_measurements(plus_landmarks, preserve_measurements)
                minus_preserve = compute_measurements(minus_landmarks, preserve_measurements)

                preserve_derivative = (plus_preserve - minus_preserve) / (2.0 * probe_value)
                normalized_preserve_derivative = torch.abs(preserve_derivative) / preserve_scale

                preserve_drift_score = float(
                    weighted_mean(normalized_preserve_derivative, preserve_weights).item()
                )
            else:
                preserve_drift_score = 0.0

            net_score = best_target_alignment_score - (
                preserve_tradeoff * preserve_drift_score
            )

            diagnostics.append(
                {
                    "pc_idx": int(pc_idx),
                    "target_alignment_score": best_target_alignment_score,
                    "target_alignment_score_pos": target_alignment_score_pos,
                    "target_alignment_score_neg": target_alignment_score_neg,
                    "target_magnitude_score": target_magnitude_score,
                    "preserve_drift_score": preserve_drift_score,
                    "net_score": net_score,
                    "preferred_theta_sign": preferred_theta_sign,
                    "probe_value": probe_value,
                    "preserve_tradeoff": preserve_tradeoff,
                }
            )

    ranked_diagnostics = sorted(
        diagnostics,
        key=lambda item: (
            float(item["net_score"]),
            float(item["target_alignment_score"]),
            float(item["target_magnitude_score"]),
        ),
        reverse=True,
    )

    positive_ranked = [item for item in ranked_diagnostics if float(item["net_score"]) > 0.0]
    selected_pool = positive_ranked if len(positive_ranked) >= top_k else ranked_diagnostics

    selected_ranked = selected_pool[:top_k]
    selected_pcs = [int(item["pc_idx"]) for item in selected_ranked]

    if not selected_pcs:
        raise RuntimeError("PC selection produced an empty selected_pcs list.")

    return selected_pcs, ranked_diagnostics


def print_pc_selection_summary(
    selected_pcs: List[int],
    pc_selection_diagnostics: List[Dict[str, object]],
    max_lines: int = 12,
) -> None:
    print("\nSelected PCs for optimization:")
    print(f"  {selected_pcs}")

    print("\nTop PC sensitivity ranking:")
    for item in pc_selection_diagnostics[:max_lines]:
        if "net_score" not in item:
            print(f"  - pc={item['pc_idx']}")
            continue

        print(
            f"  - pc={int(item['pc_idx']):3d} "
            f"net={float(item['net_score']):.6f} "
            f"target_align={float(item['target_alignment_score']):.6f} "
            f"target_mag={float(item['target_magnitude_score']):.6f} "
            f"preserve={float(item['preserve_drift_score']):.6f} "
            f"preferred_sign={int(item['preferred_theta_sign']):+d}"
        )


def get_base_region_configs() -> Dict[str, Dict[str, object]]:
    """
    Base single-region configs.
    These are used both for single-region and multi-region combined experiments.
    """
    return {
        "nose": {
            "relative_controls": {
                "nose_width": -0.15,
                "nose_height": -0.15,
                "nose_bridge_width_proxy": -0.10,
            },
            "measurement_weight_overrides": {
                "nose_width": 3.0,
                "nose_height": 2.5,
                "nose_bridge_width_proxy": 2.0,
            },
            "preserve_measurements": [
                "face_width",
                "jaw_width",
                "outer_eye_width",
                "inner_eye_distance",
                "mouth_width",
                "lower_face_height",
            ],
            "preserve_weight_overrides": {
                "face_width": 1.2,
                "jaw_width": 1.2,
                "outer_eye_width": 1.0,
                "inner_eye_distance": 1.0,
                "mouth_width": 0.8,
                "lower_face_height": 1.0,
            },
        },
        "nostrils": {
            "relative_controls": {
                "nostril_width_proxy": -0.15,
            },
            "measurement_weight_overrides": {
                "nostril_width_proxy": 3.4,
            },
            "preserve_measurements": [
                "nose_height",
                "nose_bridge_width_proxy",
                "face_width",
                "mouth_width",
                "outer_eye_width",
            ],
            "preserve_weight_overrides": {
                "nose_height": 1.0,
                "nose_bridge_width_proxy": 1.2,
                "face_width": 0.8,
                "mouth_width": 0.7,
                "outer_eye_width": 0.7,
            },
        },
        "nose_width_only": {
            "relative_controls": {
                "nose_width": -0.15,
            },
            "measurement_weight_overrides": {
                "nose_width": 3.2,
            },
            "preserve_measurements": [
                "nose_height",
                "nose_bridge_width_proxy",
                "face_width",
                "jaw_width",
                "outer_eye_width",
                "inner_eye_distance",
                "mouth_width",
                "lower_face_height",
            ],
            "preserve_weight_overrides": {
                "nose_height": 1.2,
                "nose_bridge_width_proxy": 1.4,
                "face_width": 1.3,
                "jaw_width": 1.2,
                "outer_eye_width": 1.0,
                "inner_eye_distance": 1.0,
                "mouth_width": 0.8,
                "lower_face_height": 1.0,
            },
        },
        "nose_height_only": {
            "relative_controls": {
                "nose_height": -0.15,
            },
            "measurement_weight_overrides": {
                "nose_height": 3.0,
            },
            "preserve_measurements": [
                "nose_width",
                "nose_bridge_width_proxy",
                "face_width",
                "jaw_width",
                "outer_eye_width",
                "inner_eye_distance",
                "mouth_width",
                "lower_face_height",
            ],
            "preserve_weight_overrides": {
                "nose_width": 1.4,
                "nose_bridge_width_proxy": 1.4,
                "face_width": 1.3,
                "jaw_width": 1.2,
                "outer_eye_width": 1.0,
                "inner_eye_distance": 1.0,
                "mouth_width": 0.8,
                "lower_face_height": 1.0,
            },
        },
        "mouth": {
            "relative_controls": {
                "mouth_width": +0.08,
                "inner_mouth_width": +0.08,
                "mouth_opening_height": +0.05,
                "mouth_to_chin_height": -0.06,
                "philtrum_height_proxy": -0.04,
            },
            "measurement_weight_overrides": {
                "mouth_width": 3.0,
                "inner_mouth_width": 2.5,
                "mouth_opening_height": 2.0,
                "mouth_to_chin_height": 2.0,
                "philtrum_height_proxy": 1.5,
            },
            "preserve_measurements": [
                "nose_width",
                "nose_height",
                "outer_eye_width",
                "inner_eye_distance",
                "face_width",
                "jaw_width",
            ],
            "preserve_weight_overrides": {
                "nose_width": 1.0,
                "nose_height": 1.0,
                "outer_eye_width": 1.0,
                "inner_eye_distance": 1.0,
                "face_width": 1.0,
                "jaw_width": 1.0,
            },
        },
        "mouth_width_only": {
            "relative_controls": {
                "mouth_width": +0.08,
                "inner_mouth_width": +0.08,
            },
            "measurement_weight_overrides": {
                "mouth_width": 3.0,
                "inner_mouth_width": 2.4,
            },
            "preserve_measurements": [
                "mouth_opening_height",
                "mouth_to_chin_height",
                "philtrum_height_proxy",
                "nose_width",
                "nose_height",
                "outer_eye_width",
                "inner_eye_distance",
                "face_width",
                "jaw_width",
            ],
            "preserve_weight_overrides": {
                "mouth_opening_height": 1.6,
                "mouth_to_chin_height": 1.2,
                "philtrum_height_proxy": 1.2,
                "nose_width": 1.0,
                "nose_height": 1.0,
                "outer_eye_width": 1.0,
                "inner_eye_distance": 1.0,
                "face_width": 1.4,
                "jaw_width": 1.4,
            },
        },
        "mouth_opening_only": {
            "relative_controls": {
                "mouth_opening_height": +0.05,
            },
            "measurement_weight_overrides": {
                "mouth_opening_height": 3.0,
            },
            "preserve_measurements": [
                "mouth_width",
                "inner_mouth_width",
                "mouth_to_chin_height",
                "philtrum_height_proxy",
                "nose_width",
                "nose_height",
                "outer_eye_width",
                "inner_eye_distance",
                "face_width",
                "jaw_width",
            ],
            "preserve_weight_overrides": {
                "mouth_width": 1.6,
                "inner_mouth_width": 1.4,
                "mouth_to_chin_height": 1.2,
                "philtrum_height_proxy": 1.2,
                "nose_width": 1.0,
                "nose_height": 1.0,
                "outer_eye_width": 1.0,
                "inner_eye_distance": 1.0,
                "face_width": 1.2,
                "jaw_width": 1.2,
            },
        },
        "lip_thickness": {
            "relative_controls": {
                "upper_lip_thickness_proxy": +0.12,
                "lower_lip_thickness_proxy": +0.12,
            },
            "measurement_weight_overrides": {
                "upper_lip_thickness_proxy": 3.0,
                "lower_lip_thickness_proxy": 3.0,
            },
            "preserve_measurements": [
                "mouth_width",
                "inner_mouth_width",
                "mouth_opening_height",
                "philtrum_height_proxy",
                "mouth_to_chin_height",
                "nose_to_mouth_center_height",
            ],
            "preserve_weight_overrides": {
                "mouth_width": 1.4,
                "inner_mouth_width": 1.3,
                "mouth_opening_height": 1.8,
                "philtrum_height_proxy": 1.3,
                "mouth_to_chin_height": 1.0,
                "nose_to_mouth_center_height": 1.0,
            },
        },
        "eyes_spacing": {
            "relative_controls": {
                "outer_eye_width": -0.12,
                "inner_eye_distance": -0.15,
            },
            "measurement_weight_overrides": {
                "outer_eye_width": 3.0,
                "inner_eye_distance": 3.2,
            },
            "preserve_measurements": [
                "left_eye_width",
                "right_eye_width",
                "left_eye_opening_height",
                "right_eye_opening_height",
                "nose_width",
                "mouth_width",
                "face_width",
                "jaw_width",
            ],
            "preserve_weight_overrides": {
                "left_eye_width": 1.4,
                "right_eye_width": 1.4,
                "left_eye_opening_height": 1.2,
                "right_eye_opening_height": 1.2,
                "nose_width": 1.4,
                "mouth_width": 1.1,
                "face_width": 2.0,
                "jaw_width": 1.4,
            },
        },
        "eye_size": {
            "relative_controls": {
                "left_eye_width": -0.12,
                "right_eye_width": -0.12,
            },
            "measurement_weight_overrides": {
                "left_eye_width": 3.4,
                "right_eye_width": 3.4,
            },
            "preserve_measurements": [
                "left_eye_opening_height",
                "right_eye_opening_height",
                "outer_eye_width",
                "inner_eye_distance",
                "brow_width_left",
                "brow_width_right",
                "left_brow_to_eye_distance",
                "right_brow_to_eye_distance",
                "nose_width",
                "face_width",
            ],
            "preserve_weight_overrides": {
                "left_eye_opening_height": 1.4,
                "right_eye_opening_height": 1.4,
                "outer_eye_width": 1.5,
                "inner_eye_distance": 1.5,
                "brow_width_left": 1.0,
                "brow_width_right": 1.0,
                "left_brow_to_eye_distance": 1.0,
                "right_brow_to_eye_distance": 1.0,
                "nose_width": 0.8,
                "face_width": 0.8,
            },
        },
        "eyes_shape": {
            "relative_controls": {
                "left_eye_width": -0.08,
                "right_eye_width": -0.08,
                "left_eye_opening_height": -0.10,
                "right_eye_opening_height": -0.10,
            },
            "measurement_weight_overrides": {
                "left_eye_width": 2.6,
                "right_eye_width": 2.6,
                "left_eye_opening_height": 2.0,
                "right_eye_opening_height": 2.0,
            },
            "preserve_measurements": [
                "outer_eye_width",
                "inner_eye_distance",
                "nose_width",
                "mouth_width",
                "face_width",
            ],
            "preserve_weight_overrides": {
                "outer_eye_width": 1.5,
                "inner_eye_distance": 1.5,
                "nose_width": 1.0,
                "mouth_width": 0.8,
                "face_width": 1.0,
            },
        },
        "chin": {
            "relative_controls": {
                "chin_width_proxy": -0.12,
                "mouth_to_chin_height": -0.10,
                "lower_face_height": -0.06,
            },
            "measurement_weight_overrides": {
                "chin_width_proxy": 2.6,
                "mouth_to_chin_height": 2.2,
                "lower_face_height": 1.8,
            },
            "preserve_measurements": [
                "nose_width",
                "nose_height",
                "outer_eye_width",
                "inner_eye_distance",
                "mouth_width",
                "jaw_width",
                "face_width",
            ],
            "preserve_weight_overrides": {
                "nose_width": 1.0,
                "nose_height": 1.0,
                "outer_eye_width": 1.0,
                "inner_eye_distance": 1.0,
                "mouth_width": 1.1,
                "jaw_width": 1.6,
                "face_width": 1.6,
            },
        },
        "jaw": {
            "relative_controls": {
                "jaw_width": -0.12,
                "face_width": -0.08,
                "lower_face_height": -0.06,
            },
            "measurement_weight_overrides": {
                "jaw_width": 2.6,
                "face_width": 2.2,
                "lower_face_height": 1.8,
            },
            "preserve_measurements": [
                "chin_width_proxy",
                "mouth_to_chin_height",
                "mouth_width",
                "inner_mouth_width",
                "nose_width",
                "nose_height",
                "outer_eye_width",
                "inner_eye_distance",
            ],
            "preserve_weight_overrides": {
                "chin_width_proxy": 1.4,
                "mouth_to_chin_height": 1.2,
                "mouth_width": 1.1,
                "inner_mouth_width": 1.0,
                "nose_width": 1.0,
                "nose_height": 1.0,
                "outer_eye_width": 1.0,
                "inner_eye_distance": 1.0,
            },
        },
        "lower_face": {
            "relative_controls": {
                "face_width": -0.10,
                "chin_center_to_nose_bridge": -0.15,
                "lower_face_height": -0.12,
                "jaw_width": -0.10,
                "chin_width_proxy": -0.10,
                "mouth_to_chin_height": -0.08,
            },
            "measurement_weight_overrides": {
                "face_width": 2.0,
                "chin_center_to_nose_bridge": 2.0,
                "lower_face_height": 2.5,
                "jaw_width": 1.8,
                "chin_width_proxy": 1.8,
                "mouth_to_chin_height": 1.8,
            },
            "preserve_measurements": [
                "nose_width",
                "nose_height",
                "outer_eye_width",
                "inner_eye_distance",
                "mouth_width",
                "inner_mouth_width",
            ],
            "preserve_weight_overrides": {
                "nose_width": 1.0,
                "nose_height": 1.0,
                "outer_eye_width": 1.0,
                "inner_eye_distance": 1.0,
                "mouth_width": 1.0,
                "inner_mouth_width": 1.0,
            },
        },
        "brows": {
            "relative_controls": {
                "brow_width_left": +0.10,
                "brow_width_right": +0.10,
                "left_brow_to_eye_distance": +0.08,
                "right_brow_to_eye_distance": +0.08,
            },
            "measurement_weight_overrides": {
                "brow_width_left": 2.0,
                "brow_width_right": 2.0,
                "left_brow_to_eye_distance": 2.0,
                "right_brow_to_eye_distance": 2.0,
            },
            "preserve_measurements": [
                "outer_eye_width",
                "inner_eye_distance",
                "nose_width",
                "face_width",
            ],
            "preserve_weight_overrides": {
                "outer_eye_width": 1.0,
                "inner_eye_distance": 1.0,
                "nose_width": 1.0,
                "face_width": 1.0,
            },
        },
        "global_face": {
            "relative_controls": {
                "face_width": -0.08,
                "midface_width": -0.08,
                "chin_center_to_nose_bridge": -0.10,
                "lower_face_height": -0.10,
                "jaw_width": -0.08,
            },
            "measurement_weight_overrides": {
                "face_width": 2.0,
                "midface_width": 2.0,
                "chin_center_to_nose_bridge": 2.0,
                "lower_face_height": 2.0,
                "jaw_width": 1.8,
            },
            "preserve_measurements": [
                "nose_width",
                "nose_height",
                "outer_eye_width",
                "inner_eye_distance",
                "mouth_width",
            ],
            "preserve_weight_overrides": {
                "nose_width": 1.0,
                "nose_height": 1.0,
                "outer_eye_width": 1.0,
                "inner_eye_distance": 1.0,
                "mouth_width": 1.0,
            },
        },
        "cheekbones": {
            "relative_controls": {
                "left_cheekbone_height_proxy": -0.12,
                "right_cheekbone_height_proxy": -0.12,
            },
            "measurement_weight_overrides": {
                "left_cheekbone_height_proxy": 3.0,
                "right_cheekbone_height_proxy": 3.0,
            },
            "preserve_measurements": [
                "face_width",
                "midface_width",
                "outer_eye_width",
                "inner_eye_distance",
                "nose_width",
                "jaw_width",
            ],
            "preserve_weight_overrides": {
                "face_width": 1.2,
                "midface_width": 1.6,
                "outer_eye_width": 1.0,
                "inner_eye_distance": 1.0,
                "nose_width": 0.9,
                "jaw_width": 1.0,
            },
        },
    }


def combine_region_configs(region_names: List[str]) -> Dict[str, object]:
    """
    Merge multiple base region configs into one combined experiment config.
    """
    base_configs = get_base_region_configs()

    relative_controls: Dict[str, float] = {}
    measurement_weight_overrides: Dict[str, float] = {}
    preserve_measurements: List[str] = []
    preserve_weight_overrides: Dict[str, float] = {}

    for region_name in region_names:
        if region_name not in base_configs:
            raise ValueError(
                f"Unknown region '{region_name}'. Available: {list(base_configs.keys())}"
            )

        cfg = base_configs[region_name]

        for k, v in cfg["relative_controls"].items():  # type: ignore
            relative_controls[k] = float(v)

        for k, v in cfg["measurement_weight_overrides"].items():  # type: ignore
            if k in measurement_weight_overrides:
                measurement_weight_overrides[k] = max(
                    measurement_weight_overrides[k], float(v)
                )
            else:
                measurement_weight_overrides[k] = float(v)

        for name in cfg["preserve_measurements"]:  # type: ignore
            if name not in preserve_measurements:
                preserve_measurements.append(name)

        for k, v in cfg["preserve_weight_overrides"].items():  # type: ignore
            if k in preserve_weight_overrides:
                preserve_weight_overrides[k] = max(
                    preserve_weight_overrides[k], float(v)
                )
            else:
                preserve_weight_overrides[k] = float(v)

    target_names: Set[str] = set(relative_controls.keys())
    preserve_measurements = [m for m in preserve_measurements if m not in target_names]
    preserve_weight_overrides = {
        k: v for k, v in preserve_weight_overrides.items() if k not in target_names
    }

    return {
        "relative_controls": relative_controls,
        "measurement_weight_overrides": measurement_weight_overrides,
        "preserve_measurements": preserve_measurements,
        "preserve_weight_overrides": preserve_weight_overrides,
    }


def apply_manual_overrides(
    config: Dict[str, object],
    measurement_weight_updates: Dict[str, float] | None = None,
    preserve_weight_updates: Dict[str, float] | None = None,
) -> Dict[str, object]:
    updated = {
        "relative_controls": dict(config["relative_controls"]),  # type: ignore
        "measurement_weight_overrides": dict(config["measurement_weight_overrides"]),  # type: ignore
        "preserve_measurements": list(config["preserve_measurements"]),  # type: ignore
        "preserve_weight_overrides": dict(config["preserve_weight_overrides"]),  # type: ignore
    }

    if measurement_weight_updates:
        updated["measurement_weight_overrides"].update(measurement_weight_updates)

    if preserve_weight_updates:
        updated["preserve_weight_overrides"].update(preserve_weight_updates)

    return updated


def apply_relative_control_overrides(
    config: Dict[str, object],
    relative_control_updates: Dict[str, float] | None = None,
    measurement_weight_updates: Dict[str, float] | None = None,
    preserve_weight_updates: Dict[str, float] | None = None,
) -> Dict[str, object]:
    updated = {
        "relative_controls": dict(config["relative_controls"]),  # type: ignore
        "measurement_weight_overrides": dict(config["measurement_weight_overrides"]),  # type: ignore
        "preserve_measurements": list(config["preserve_measurements"]),  # type: ignore
        "preserve_weight_overrides": dict(config["preserve_weight_overrides"]),  # type: ignore
    }

    if relative_control_updates:
        updated["relative_controls"].update(relative_control_updates)

    if measurement_weight_updates:
        updated["measurement_weight_overrides"].update(measurement_weight_updates)

    if preserve_weight_updates:
        updated["preserve_weight_overrides"].update(preserve_weight_updates)

    return updated

def build_scaled_region_config(
    region_name: str,
    scale: float,
) -> Dict[str, object]:
    """
    Build a scaled copy of one base region config.
    scale = +1.0 keeps the original direction
    scale = -1.0 flips the direction
    """
    base_configs = get_base_region_configs()

    if region_name not in base_configs:
        raise ValueError(f"Unknown base region: {region_name}")

    cfg = base_configs[region_name]

    return {
        "relative_controls": {
            name: float(value) * float(scale)
            for name, value in cfg["relative_controls"].items()  # type: ignore
        },
        "measurement_weight_overrides": dict(cfg["measurement_weight_overrides"]),  # type: ignore
        "preserve_measurements": list(cfg["preserve_measurements"]),  # type: ignore
        "preserve_weight_overrides": dict(cfg["preserve_weight_overrides"]),  # type: ignore
    }


def merge_custom_region_configs(
    configs: List[Dict[str, object]],
    max_abs_offset: float = 0.20,
) -> Dict[str, object]:
    """
    Merge multiple already-scaled configs.
    Unlike combine_region_configs(...), this SUMS overlapping relative controls
    so multiple phenotype traits can jointly affect the same measurement.
    """
    if not configs:
        raise ValueError("No configs provided to merge_custom_region_configs.")

    relative_controls: Dict[str, float] = {}
    measurement_weight_overrides: Dict[str, float] = {}
    preserve_measurements: List[str] = []
    preserve_weight_overrides: Dict[str, float] = {}

    for cfg in configs:
        for name, value in cfg["relative_controls"].items():  # type: ignore
            relative_controls[name] = relative_controls.get(name, 0.0) + float(value)

        for name, value in cfg["measurement_weight_overrides"].items():  # type: ignore
            if name in measurement_weight_overrides:
                measurement_weight_overrides[name] = max(
                    measurement_weight_overrides[name], float(value)
                )
            else:
                measurement_weight_overrides[name] = float(value)

        for name in cfg["preserve_measurements"]:  # type: ignore
            if name not in preserve_measurements:
                preserve_measurements.append(name)

        for name, value in cfg["preserve_weight_overrides"].items():  # type: ignore
            if name in preserve_weight_overrides:
                preserve_weight_overrides[name] = max(
                    preserve_weight_overrides[name], float(value)
                )
            else:
                preserve_weight_overrides[name] = float(value)

    # clip merged target offsets so one row does not become too extreme
    for name in list(relative_controls.keys()):
        relative_controls[name] = float(
            np.clip(relative_controls[name], -max_abs_offset, max_abs_offset)
        )

    target_names = set(relative_controls.keys())
    preserve_measurements = [m for m in preserve_measurements if m not in target_names]
    preserve_weight_overrides = {
        k: v for k, v in preserve_weight_overrides.items() if k not in target_names
    }

    return {
        "relative_controls": relative_controls,
        "measurement_weight_overrides": measurement_weight_overrides,
        "preserve_measurements": preserve_measurements,
        "preserve_weight_overrides": preserve_weight_overrides,
    }


def get_dataset_row_control_defaults() -> Dict[str, object]:
    """
    More suitable defaults for a multi-trait phenotype fit.
    """
    return merge_nested_settings(
        get_control_defaults("global_face"),
        {
            "pc_selection": {
                "top_k": 48,
                "candidate_pool_size": 100,
                "preserve_tradeoff": 0.50,
            },
            "optimization": {
                "learning_rate": 0.04,
                "steps": 850,
                "lambda_preserve": 0.45,
            },
            "enable_visualization": True,
        },
    )


def build_dataset_row_experiment_config(
    dataset_csv_path: Path,
    sample_id: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Build one custom FLAME experiment config from a dataset row.

    - reads a dataset row
    - extracts appearance traits
    - maps supported geometry traits to control votes
    - merges the active region configs into one experiment config
    """
    if not dataset_csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_csv_path}")

    df = pd.read_csv(dataset_csv_path)

    matches = df[df["sample_id"] == sample_id]
    if matches.empty:
        raise ValueError(f"sample_id '{sample_id}' not found in dataset.")

    row = matches.iloc[0].to_dict()

    appearance_traits = extract_appearance_traits_from_row(row)
    appearance_render_plan = build_appearance_render_plan(appearance_traits)

    geometry_traits = {
        "face_width": row["face_width"],
        "face_height": row["face_height"],
        "jaw_shape": row["jaw_shape"],
        "chin_prominence": row["chin_prominence"],
        "nose_size" : row["nose_size"],
        "nose_width": row["nose_width"],
        "nose_bridge_width": row["nose_bridge_width"],
        "nose_bridge_height": row["nose_bridge_height"],
        "eye_distance": row["eye_distance"],
        "eyebrow_thickness": row["eyebrow_thickness"],
        "eyebrow_arch": row["eyebrow_arch"],
        "mouth_width": row["mouth_width"],
        "philtrum_depth": row["philtrum_depth"],
        "nostril_width": row["nostril_width"],
        "lip_thickness": row["lip_thickness"],
        "cheekbone_height": row["cheekbone_height"],
        "eye_size": row["eye_size"],
        "eye_shape": row["eye_shape"],
    }

    control_votes: Dict[str, List[float]] = {}

    def add_vote(control_name: str, vote: float) -> None:
        control_votes.setdefault(control_name, []).append(float(vote))

    # ------------------------------------------------------------
    # Trait -> control mapping (rebalanced first-pass geometry only)
    # ------------------------------------------------------------

    # global face
    # Reduced from the previous stronger values so it does not overpower chin/mouth.
    if geometry_traits["face_width"] == "wide":
        add_vote("global_face", -0.55)
    elif geometry_traits["face_width"] == "narrow":
        add_vote("global_face", +0.55)

    if geometry_traits["face_height"] == "long":
        add_vote("global_face", -0.55)
    elif geometry_traits["face_height"] == "short":
        add_vote("global_face", +0.55)

    # jaw
    # Reduced because square-jaw expansion was competing too much with receding chin.
    if geometry_traits["jaw_shape"] == "square":
        add_vote("jaw", -0.45)
    elif geometry_traits["jaw_shape"] == "narrow":
        add_vote("jaw", +0.70)

    # chin
    # Keep this strong so "receding" remains visible against face/jaw broadening.
    if geometry_traits["chin_prominence"] == "receding":
        add_vote("chin", +1.15)
    elif geometry_traits["chin_prominence"] == "prominent":
        add_vote("chin", -1.15)
    

    # cheekbone height
    if geometry_traits["cheekbone_height"] == "high":
        add_vote("cheekbones", +0.55)
    elif geometry_traits["cheekbone_height"] == "low":
        add_vote("cheekbones", -0.55)

    # nose size
    if geometry_traits["nose_size"] == "small":
        add_vote("nose_width_only", +0.35)
        add_vote("nose_height_only", +0.25)
    elif geometry_traits["nose_size"] == "large":
        add_vote("nose_width_only", -0.35)
        add_vote("nose_height_only", -0.25)

    # nose width
    if geometry_traits["nose_width"] == "narrow":
        add_vote("nose_width_only", +0.55)
    elif geometry_traits["nose_width"] == "wide":
        add_vote("nose_width_only", -0.55)

    # nostril width
    if geometry_traits["nostril_width"] == "narrow":
        add_vote("nostrils", +0.55)
    elif geometry_traits["nostril_width"] == "wide":
        add_vote("nostrils", -0.55)

    # nose bridge height -> approximate with nose_height_only
    if geometry_traits["nose_bridge_height"] == "high":
        add_vote("nose_height_only", -0.55)
    elif geometry_traits["nose_bridge_height"] == "flat":
        add_vote("nose_height_only", +0.55)

    # nose bridge width -> approximate with nose_bridge_width_proxy
    if geometry_traits["nose_bridge_width"] == "wide":
        add_vote("nose", -0.55)
    elif geometry_traits["nose_bridge_width"] == "narrow":
        add_vote("nose", +0.55)

    # eye size
    if geometry_traits["eye_size"] == "small":
        add_vote("eye_size", +0.70)
    elif geometry_traits["eye_size"] == "large":
        add_vote("eye_size", -0.70)

    # eye shape
    if geometry_traits["eye_shape"] == "hooded":
        add_vote("eyes_shape", +0.80)
    elif geometry_traits["eye_shape"] == "almond":
        add_vote("eyes_shape", +0.25)
    elif geometry_traits["eye_shape"] == "round":
        add_vote("eyes_shape", -0.70)

    # eye spacing
    if geometry_traits["eye_distance"] == "close":
        add_vote("eyes_spacing", +0.55)
    elif geometry_traits["eye_distance"] == "wide":
        add_vote("eyes_spacing", -0.55)

    # brows
    if geometry_traits["eyebrow_thickness"] == "full":
        add_vote("brows", +0.45)
    elif geometry_traits["eyebrow_thickness"] == "thin":
        add_vote("brows", -0.65)
    elif geometry_traits["eyebrow_thickness"] == "very_full":
        add_vote("brows", +0.80)    

    if geometry_traits["eyebrow_arch"] == "high":
        add_vote("brows", +0.65)
    elif geometry_traits["eyebrow_arch"] == "slight":
        add_vote("brows", -0.35)

    # mouth width
    if geometry_traits["mouth_width"] == "wide":
        add_vote("mouth_width_only", +0.40)
    elif geometry_traits["mouth_width"] == "narrow":
        add_vote("mouth_width_only", -0.40)

    # lip thickness
    if geometry_traits["lip_thickness"] == "thin":
        add_vote("lip_thickness", -0.65)
    elif geometry_traits["lip_thickness"] == "full":
        add_vote("lip_thickness", +0.45)
    elif geometry_traits["lip_thickness"] == "very_full":
        add_vote("lip_thickness", +0.80)

    # philtrum depth
    if geometry_traits["philtrum_depth"] == "shallow":
        add_vote("mouth", +0.40)
    elif geometry_traits["philtrum_depth"] == "deep":
        add_vote("mouth", -0.40)

    # ------------------------------------------------------------
    # Build final control scores
    # ------------------------------------------------------------
    control_scores: Dict[str, float] = {}
    for control_name, votes in control_votes.items():
        if len(votes) == 0:
            continue
        control_scores[control_name] = float(np.clip(np.mean(votes), -1.25, 1.25))

    if not control_scores:
        raise RuntimeError(
            f"No supported geometry traits were extracted from dataset row '{sample_id}'."
        )

    scaled_configs: List[Dict[str, object]] = []
    for control_name, score in control_scores.items():
        if abs(score) < 1e-8:
            continue
        scaled_configs.append(build_scaled_region_config(control_name, score))

    experiment_config = merge_custom_region_configs(
        scaled_configs,
        max_abs_offset=0.20,
    )

    ignored_geometry_traits = {
    }

    metadata = {
        "sample_id": sample_id,
        "geometry_traits": geometry_traits,
        "control_scores": control_scores,
        "ignored_geometry_traits": ignored_geometry_traits,
        "appearance_traits": appearance_traits,
        "appearance_render_plan": appearance_render_plan,
    }

    return experiment_config, metadata    


def get_experiment_config(experiment_name: str) -> Dict[str, object]:
    """
    Supports both single-region and combined experiments.
    """
    base_configs = get_base_region_configs()

    if experiment_name in base_configs:
        return base_configs[experiment_name]

    if experiment_name == "nose_mouth_combo":
        return combine_region_configs(["nose", "mouth"])

    if experiment_name == "eyes_brows_combo":
        return combine_region_configs(["eyes_spacing", "eyes_shape", "brows"])

    if experiment_name == "eyes_brows_staged_combo":
        base_combo = combine_region_configs(["eyes_spacing", "eyes_shape", "brows"])
        return apply_manual_overrides(
            base_combo,
            measurement_weight_updates={
                "outer_eye_width": 2.0,
                "inner_eye_distance": 2.2,
                "left_eye_width": 3.4,
                "right_eye_width": 3.4,
                "left_eye_opening_height": 3.2,
                "right_eye_opening_height": 3.2,
                "brow_width_left": 3.0,
                "brow_width_right": 3.0,
                "left_brow_to_eye_distance": 3.0,
                "right_brow_to_eye_distance": 3.0,
            },
            preserve_weight_updates={
                "nose_width": 0.8,
                "mouth_width": 0.7,
                "face_width": 0.8,
            },
        )

    if experiment_name == "eyes_spacing_brows_combo":
        return apply_manual_overrides(
            combine_region_configs(["eyes_spacing", "brows"]),
            measurement_weight_updates={
                "outer_eye_width": 3.2,
                "inner_eye_distance": 3.4,
                "brow_width_left": 2.6,
                "brow_width_right": 2.6,
                "left_brow_to_eye_distance": 2.6,
                "right_brow_to_eye_distance": 2.6,
            },
            preserve_weight_updates={
                "nose_width": 0.9,
                "mouth_width": 0.7,
                "face_width": 0.9,
                "left_eye_width": 1.5,
                "right_eye_width": 1.5,
                "left_eye_opening_height": 1.4,
                "right_eye_opening_height": 1.4,
            },
        )

    if experiment_name == "eyes_shape_brows_combo":
        return apply_manual_overrides(
            combine_region_configs(["eyes_shape", "brows"]),
            measurement_weight_updates={
                "left_eye_width": 3.4,
                "right_eye_width": 3.4,
                "left_eye_opening_height": 3.0,
                "right_eye_opening_height": 3.0,
                "brow_width_left": 2.6,
                "brow_width_right": 2.6,
                "left_brow_to_eye_distance": 2.6,
                "right_brow_to_eye_distance": 2.6,
            },
            preserve_weight_updates={
                "outer_eye_width": 1.6,
                "inner_eye_distance": 1.6,
                "nose_width": 0.9,
                "mouth_width": 0.7,
                "face_width": 0.9,
            },
        )

    if experiment_name == "eyes_shape_diagnostic":
        return apply_relative_control_overrides(
            base_configs["eyes_shape"],
            relative_control_updates={
                "left_eye_width": -0.08,
                "right_eye_width": -0.08,
                "left_eye_opening_height": -0.10,
                "right_eye_opening_height": -0.10,
            },
            measurement_weight_updates={
                "left_eye_width": 3.8,
                "right_eye_width": 3.8,
                "left_eye_opening_height": 3.8,
                "right_eye_opening_height": 3.8,
            },
            preserve_weight_updates={
                "outer_eye_width": 1.8,
                "inner_eye_distance": 1.8,
                "nose_width": 0.8,
                "mouth_width": 0.6,
                "face_width": 0.8,
            },
        )

    if experiment_name == "eyes_shape_diagnostic_soft":
        return apply_relative_control_overrides(
            base_configs["eyes_shape"],
            relative_control_updates={
                "left_eye_width": -0.05,
                "right_eye_width": -0.05,
                "left_eye_opening_height": -0.05,
                "right_eye_opening_height": -0.05,
            },
            measurement_weight_updates={
                "left_eye_width": 3.8,
                "right_eye_width": 3.8,
                "left_eye_opening_height": 3.8,
                "right_eye_opening_height": 3.8,
            },
            preserve_weight_updates={
                "outer_eye_width": 1.8,
                "inner_eye_distance": 1.8,
                "nose_width": 0.8,
                "mouth_width": 0.6,
                "face_width": 0.8,
            },
        )

    if experiment_name == "mouth_lowerface_combo":
        return combine_region_configs(["mouth", "lower_face"])

    available = list(base_configs.keys()) + [
        "nose_mouth_combo",
        "eyes_brows_combo",
        "eyes_brows_staged_combo",
        "eyes_spacing_brows_combo",
        "eyes_shape_brows_combo",
        "eyes_shape_diagnostic",
        "eyes_shape_diagnostic_soft",
        "mouth_lowerface_combo",
    ]
    raise ValueError(
        f"Unknown experiment_name='{experiment_name}'. Available: {available}"
    )


def load_measurements_by_name(names: List[str]) -> List[MeasurementDef]:
    measurements: List[MeasurementDef] = []
    for name in names:
        measurement = get_measurement_by_name(name)
        if measurement is None:
            raise ValueError(f"Unknown measurement requested: {name}")
        measurements.append(measurement)
    return measurements


def main():
    # ------------------------------------------------------------
    # 1) Setup
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flame_config = get_config()
    flame_config.batch_size = 1

    flame = FLAME(flame_config).to(device)

    zeros_exp = torch.zeros(1, flame_config.expression_params, device=device)
    zeros_pose = torch.zeros(1, flame_config.pose_params, device=device)

    print(f"Device: {device}")

    # ------------------------------------------------------------
    # Demo / presentation settings
    # ------------------------------------------------------------
    demo_mode = False
    demo_control_scale = 2.0
    demo_measurement_weight_scale = 1.10
    demo_preserve_weight_scale = 0.75
    demo_preserve_tradeoff_scale = 0.80
    demo_lambda_preserve_scale = 0.70
    demo_max_abs_offset = 0.35

    generate_demo_gifs = True
    morph_gif_frames = 30
    spin_gif_frames = 36
    gif_fps = 12
    gif_image_size = (720, 720)

    # ------------------------------------------------------------
    # 2) Experiment selection
    # ------------------------------------------------------------
    dataset_mode = True
    dataset_csv_path = Path(__file__).resolve().parents[3] / "data" / "synthetic_dataset" / "synthetic_dataset_complete.csv"
    dataset_sample_id = "SYNTH_000000"   # first strong geometry test row

    dataset_metadata: Dict[str, object] | None = None

    if dataset_mode:
        experiment_name = f"dataset_{dataset_sample_id.lower()}"
        experiment_config, dataset_metadata = build_dataset_row_experiment_config(
            dataset_csv_path=dataset_csv_path,
            sample_id=dataset_sample_id,
        )
        control_defaults = get_dataset_row_control_defaults()
    else:
        experiment_name = "nose_height_only"
        experiment_config = get_experiment_config(experiment_name)
        control_defaults = get_control_defaults(experiment_name)

    pc_selection_config: Dict[str, object] = dict(control_defaults["pc_selection"])  # type: ignore
    optimization_config: Dict[str, float] = dict(control_defaults["optimization"])  # type: ignore
    enable_visualization = bool(control_defaults["enable_visualization"])

    if demo_mode:
        experiment_config = apply_demo_mode_to_experiment(
            experiment_config=experiment_config,
            control_scale=demo_control_scale,
            measurement_weight_scale=demo_measurement_weight_scale,
            preserve_weight_scale=demo_preserve_weight_scale,
            max_abs_offset=demo_max_abs_offset,
        )
        pc_selection_config["preserve_tradeoff"] = (
            float(pc_selection_config.get("preserve_tradeoff", 0.40))
            * demo_preserve_tradeoff_scale
        )
        optimization_config["lambda_preserve"] = (
            float(optimization_config["lambda_preserve"])
            * demo_lambda_preserve_scale
        )

    if generate_demo_gifs:
        enable_visualization = False

    relative_controls: Dict[str, float] = experiment_config["relative_controls"]  # type: ignore
    measurement_weight_overrides: Dict[str, float] = experiment_config["measurement_weight_overrides"]  # type: ignore
    preserve_measurement_names: List[str] = experiment_config["preserve_measurements"]  # type: ignore
    preserve_weight_overrides: Dict[str, float] = experiment_config["preserve_weight_overrides"]  # type: ignore

    if dataset_mode and dataset_metadata is not None:
        print(f"\nDataset row selected: {dataset_metadata['sample_id']}")
        print("\nGeometry traits from dataset row:")
        for trait_name, trait_value in dataset_metadata["geometry_traits"].items():  # type: ignore
            print(f"  - {trait_name:24s} {trait_value}")

        print("\nMapped FLAME control scores:")
        for control_name, score in dataset_metadata["control_scores"].items():  # type: ignore
            print(f"  - {control_name:24s} score={float(score):+.3f}")

        print("\nIgnored / deferred geometry traits:")
        for trait_name, trait_value in dataset_metadata["ignored_geometry_traits"].items():  # type: ignore
            print(f"  - {trait_name:24s} {trait_value}")

        print("\nAppearance traits from dataset row:")
        for trait_name, trait_value in dataset_metadata["appearance_traits"].items():  # type: ignore
            print(f"  - {trait_name:24s} {trait_value}")

        print("\nAppearance render plan:")
        for key, value in dataset_metadata["appearance_render_plan"].items():  # type: ignore
            print(f"  - {key:24s} {value}")

    # ------------------------------------------------------------
    # 3) Load measurements
    # ------------------------------------------------------------
    target_measurement_names = list(relative_controls.keys())
    selected_measurements = load_measurements_by_name(target_measurement_names)
    measurement_names = [str(m["name"]) for m in selected_measurements]

    preserve_measurements = load_measurements_by_name(preserve_measurement_names)

    # ------------------------------------------------------------
    # 4) Neutral baseline
    # ------------------------------------------------------------
    neutral_shape = torch.zeros(1, flame_config.shape_params, device=device)

    with torch.no_grad():
        _, neutral_landmarks = flame(neutral_shape, zeros_exp, zeros_pose)

        neutral_measurements = compute_measurements(
            neutral_landmarks,
            selected_measurements,
        ).detach()

        neutral_preserve_measurements = compute_measurements(
            neutral_landmarks,
            preserve_measurements,
        ).detach()

    # ------------------------------------------------------------
    # 5) Build targets from relative controls
    # ------------------------------------------------------------
    target_measurements = build_target_measurements_from_relative_controls(
        neutral_measurements=neutral_measurements,
        measurement_names=measurement_names,
        relative_controls=relative_controls,
    )

    print(f"\nExperiment config: {experiment_name}")

    print("\nRelative controls:")
    for name in measurement_names:
        print(f"  - {name:28s} offset={relative_controls[name]:+.3f}")

    print("\nTarget parameter values:")
    for name, neutral_val, target_val in zip(
        measurement_names,
        neutral_measurements[0].tolist(),
        target_measurements[0].tolist(),
    ):
        print(
            f"  - {name:28s} neutral={neutral_val:.6f} "
            f"target={target_val:.6f} "
            f"delta={target_val - neutral_val:+.6f}"
        )

    print("\nPreserved measurements (kept near neutral):")
    for m, neutral_val in zip(
        preserve_measurements,
        neutral_preserve_measurements[0].tolist(),
    ):
        print(f"  - {m['name']:28s} neutral_target={neutral_val:.6f}")

    # ------------------------------------------------------------
    # 6) Optimization settings
    # ------------------------------------------------------------
    lambda_reg = float(optimization_config["lambda_reg"])
    lambda_preserve = float(optimization_config["lambda_preserve"])
    learning_rate = float(optimization_config["learning_rate"])
    steps = int(optimization_config["steps"])

    default_weight = 1.0

    measurement_weights = build_measurement_weight_tensor(
        measurement_defs=selected_measurements,
        weight_overrides=measurement_weight_overrides,
        default_weight=default_weight,
        device=device,
    )

    preserve_weights = build_measurement_weight_tensor(
        measurement_defs=preserve_measurements,
        weight_overrides=preserve_weight_overrides,
        default_weight=default_weight,
        device=device,
    )

    # ------------------------------------------------------------
    # 7) Region-specific automatic PC selection
    # ------------------------------------------------------------
    selected_pcs, pc_selection_diagnostics = select_shape_pcs_for_experiment(
        flame=flame,
        total_shape_dims=flame_config.shape_params,
        zeros_exp=zeros_exp,
        zeros_pose=zeros_pose,
        selected_measurements=selected_measurements,
        preserve_measurements=preserve_measurements,
        neutral_measurements=neutral_measurements,
        neutral_preserve_measurements=neutral_preserve_measurements,
        measurement_weights=measurement_weights,
        preserve_weights=preserve_weights,
        measurement_names=measurement_names,
        relative_controls=relative_controls,
        pc_selection_config=pc_selection_config,
        device=device,
    )

    print_pc_selection_summary(
        selected_pcs=selected_pcs,
        pc_selection_diagnostics=pc_selection_diagnostics,
    )

    theta = torch.zeros(1, len(selected_pcs), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=learning_rate)

    history = []

    # ------------------------------------------------------------
    # 8) Optimization loop
    # ------------------------------------------------------------
    print(f"\nStarting {experiment_name} optimization...")
    for it in range(steps):
        optimizer.zero_grad()

        current_shape = assemble_shape_from_selected_pcs(
            theta=theta,
            selected_pcs=selected_pcs,
            total_shape_dims=flame_config.shape_params,
            device=device,
        )

        _, current_landmarks = flame(current_shape, zeros_exp, zeros_pose)

        current_measurements = compute_measurements(
            current_landmarks,
            selected_measurements,
        )

        current_preserve_measurements = compute_measurements(
            current_landmarks,
            preserve_measurements,
        )

        diff_target = current_measurements - target_measurements
        weighted_sq_error_target = measurement_weights * (diff_target ** 2)
        loss_data = torch.mean(weighted_sq_error_target)

        diff_preserve = current_preserve_measurements - neutral_preserve_measurements
        weighted_sq_error_preserve = preserve_weights * (diff_preserve ** 2)
        loss_preserve = torch.mean(weighted_sq_error_preserve)

        loss_reg = lambda_reg * torch.mean(theta ** 2)

        loss = loss_data + lambda_preserve * loss_preserve + loss_reg

        loss.backward()
        optimizer.step()

        history.append(
            {
                "iter": it,
                "loss": float(loss.item()),
                "loss_data": float(loss_data.item()),
                "loss_preserve": float(loss_preserve.item()),
                "loss_reg": float(loss_reg.item()),
                "theta": theta.detach()[0].tolist(),
            }
        )

        if it % 50 == 0 or it == steps - 1:
            print(
                f"Iter {it:03d} | "
                f"loss={loss.item():.6f} "
                f"(data={loss_data.item():.6f}, preserve={loss_preserve.item():.6f}, reg={loss_reg.item():.6f}) | "
                f"theta={theta.detach()[0].tolist()}"
            )

    # ------------------------------------------------------------
    # 9) Final evaluation
    # ------------------------------------------------------------
    final_shape = assemble_shape_from_selected_pcs(
        theta=theta.detach(),
        selected_pcs=selected_pcs,
        total_shape_dims=flame_config.shape_params,
        device=device,
    )

    with torch.no_grad():
        final_vertices, final_landmarks = flame(final_shape, zeros_exp, zeros_pose)

        final_measurements = compute_measurements(
            final_landmarks,
            selected_measurements,
        )

        final_preserve_measurements = compute_measurements(
            final_landmarks,
            preserve_measurements,
        )

    neutral_abs_error = torch.abs(neutral_measurements - target_measurements)
    final_abs_error = torch.abs(final_measurements - target_measurements)
    preserve_abs_shift = torch.abs(
        final_preserve_measurements - neutral_preserve_measurements
    )

    print("\n=== RESULTS ===")
    print(f"Experiment: {experiment_name}")
    print(f"Selected PCs: {selected_pcs}")
    print(f"Recovered theta: {theta.detach()[0].tolist()}")

    print("\nTarget-region measurement comparison:")
    for name, neutral_val, target_val, final_val, neutral_err, final_err in zip(
        measurement_names,
        neutral_measurements[0].tolist(),
        target_measurements[0].tolist(),
        final_measurements[0].tolist(),
        neutral_abs_error[0].tolist(),
        final_abs_error[0].tolist(),
    ):
        improvement = neutral_err - final_err
        print(
            f"  - {name:28s} "
            f"neutral={neutral_val:.6f} "
            f"target={target_val:.6f} "
            f"final={final_val:.6f} "
            f"neutral_err={neutral_err:.6f} "
            f"final_err={final_err:.6f} "
            f"improvement={improvement:+.6f}"
        )

    print("\nPreserved-measurement drift from neutral:")
    for m, neutral_val, final_val, shift in zip(
        preserve_measurements,
        neutral_preserve_measurements[0].tolist(),
        final_preserve_measurements[0].tolist(),
        preserve_abs_shift[0].tolist(),
    ):
        print(
            f"  - {m['name']:28s} "
            f"neutral={neutral_val:.6f} "
            f"final={final_val:.6f} "
            f"abs_shift={shift:.6f}"
        )

    print("\nSummary:")
    print(f"  Mean abs error vs target (neutral): {neutral_abs_error.mean().item():.6f}")
    print(f"  Mean abs error vs target (final):   {final_abs_error.mean().item():.6f}")
    print(f"  Mean preserved-measurement shift:   {preserve_abs_shift.mean().item():.6f}")

    # ------------------------------------------------------------
    # 10) Save artifacts
    # ------------------------------------------------------------
    run_name = format_run_name(
        experiment_name=experiment_name,
        selected_pcs=selected_pcs,
        lambda_reg=lambda_reg,
        lambda_preserve=lambda_preserve,
    )
    if demo_mode:
        run_name = f"{run_name}_demo"

    out_dir = Path("experiments/outputs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if dataset_mode and dataset_metadata is not None:
        appearance_preview = {
            "sample_id": dataset_metadata["sample_id"],
            "appearance_traits": dataset_metadata.get("appearance_traits", {}),
            "appearance_render_plan": dataset_metadata.get("appearance_render_plan", {}),
        }
        (out_dir / "appearance_preview.json").write_text(
            json.dumps(appearance_preview, indent=2)
        )    

    target_measurement_metadata = build_measurement_metadata(selected_measurements)
    for item, weight in zip(target_measurement_metadata, measurement_weights[0].tolist()):
        item["weight"] = float(weight)

    preserve_measurement_metadata = build_measurement_metadata(preserve_measurements)
    for item, weight in zip(preserve_measurement_metadata, preserve_weights[0].tolist()):
        item["weight"] = float(weight)

    metrics = {
        "run_name": run_name,
        "experiment_name": experiment_name,
        "device": str(device),
        "selected_pcs": selected_pcs,
        "pc_selection_config": pc_selection_config,
        "pc_selection_diagnostics": pc_selection_diagnostics,
        "learning_rate": learning_rate,
        "lambda_reg": lambda_reg,
        "lambda_preserve": lambda_preserve,
        "steps": steps,
        "target_measurement_metadata": target_measurement_metadata,
        "preserve_measurement_metadata": preserve_measurement_metadata,
        "relative_controls": relative_controls,
        "neutral_measurements": neutral_measurements[0].tolist(),
        "target_measurements": target_measurements[0].tolist(),
        "final_measurements": final_measurements[0].tolist(),
        "neutral_preserve_measurements": neutral_preserve_measurements[0].tolist(),
        "final_preserve_measurements": final_preserve_measurements[0].tolist(),
        "found_theta": theta.detach()[0].tolist(),
        "neutral_abs_error": neutral_abs_error[0].tolist(),
        "final_abs_error": final_abs_error[0].tolist(),
        "preserve_abs_shift": preserve_abs_shift[0].tolist(),
        "mean_abs_error_neutral": float(neutral_abs_error.mean().item()),
        "mean_abs_error_final": float(final_abs_error.mean().item()),
        "mean_preserve_abs_shift": float(preserve_abs_shift.mean().item()),
        "final_loss": float(history[-1]["loss"]),
        "final_loss_data": float(history[-1]["loss_data"]),
        "final_loss_preserve": float(history[-1]["loss_preserve"]),
        "final_loss_reg": float(history[-1]["loss_reg"]),
        "dataset_mode": dataset_mode,
    }

    if dataset_mode and dataset_metadata is not None:
        metrics["dataset_metadata"] = dataset_metadata

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "loss_history.json").write_text(json.dumps(history, indent=2))

    print(f"\nSaved metrics to: {out_dir / 'metrics.json'}")
    print(f"Saved loss history to: {out_dir / 'loss_history.json'}")

    if generate_demo_gifs:
        morph_gif_path = out_dir / f"{experiment_name}_morph.gif"
        spin_gif_path = out_dir / f"{experiment_name}_spin.gif"

        save_morph_gif(
            flame=flame,
            final_theta=theta.detach(),
            selected_pcs=selected_pcs,
            total_shape_dims=flame_config.shape_params,
            zeros_exp=zeros_exp,
            zeros_pose=zeros_pose,
            device=device,
            faces=flame.faces,
            output_path=morph_gif_path,
            num_frames=morph_gif_frames,
            fps=gif_fps,
            image_size=gif_image_size,
            appearance_render_plan=(
                dataset_metadata.get("appearance_render_plan")
                if dataset_mode and dataset_metadata is not None
                else None
            ),
        )

        vertices_cpu = final_vertices[0].detach().cpu().numpy().squeeze()
        final_landmarks_cpu = final_landmarks[0].detach().cpu().numpy().squeeze()

        save_spin_gif(
            vertices=vertices_cpu,
            faces=flame.faces,
            landmarks=final_landmarks_cpu,
            output_path=spin_gif_path,
            num_frames=spin_gif_frames,
            fps=gif_fps,
            image_size=gif_image_size,
            appearance_render_plan=(
                dataset_metadata.get("appearance_render_plan")
                if dataset_mode and dataset_metadata is not None
                else None
            ),
        )

        metrics["demo_mode"] = demo_mode
        metrics["demo_control_scale"] = demo_control_scale
        metrics["demo_measurement_weight_scale"] = demo_measurement_weight_scale
        metrics["demo_preserve_weight_scale"] = demo_preserve_weight_scale
        metrics["demo_preserve_tradeoff_scale"] = demo_preserve_tradeoff_scale
        metrics["demo_lambda_preserve_scale"] = demo_lambda_preserve_scale
        metrics["demo_max_abs_offset"] = demo_max_abs_offset
        metrics["morph_gif_path"] = str(morph_gif_path)
        metrics["spin_gif_path"] = str(spin_gif_path)

        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        print(f"Saved morph GIF to: {morph_gif_path}")
        print(f"Saved spin GIF to:  {spin_gif_path}")

    # ------------------------------------------------------------
    # 11) Optional visualization
    # ------------------------------------------------------------
    if enable_visualization:
        vertices_cpu = final_vertices[0].detach().cpu().numpy().squeeze()
        visualize_result(
            vertices=vertices_cpu,
            faces=flame.faces,
            rotation_degrees=180.0,
        )


if __name__ == "__main__":
    main()
