from typing import Dict, List, Tuple

import torch

from measurement_definitions import MeasurementDef


def validate_pair(pair: Tuple[int, int], num_landmarks: int) -> bool:
    """
    Check whether a landmark pair is valid for the given landmark count.
    """
    a, b = pair
    return 0 <= a < num_landmarks and 0 <= b < num_landmarks


def filter_valid_measurements(
    measurement_defs: List[MeasurementDef],
    num_landmarks: int,
    include_ratio: bool = True,
) -> List[MeasurementDef]:
    """
    Keep only measurements whose required landmark indices exist.

    Supports:
    - pair-based measurements
    - ratio-based measurements (if include_ratio=True)
    """
    valid_measurements: List[MeasurementDef] = []

    for m in measurement_defs:
        measurement_type = m["measurement_type"]

        if "pair" in m:
            if validate_pair(m["pair"], num_landmarks):
                valid_measurements.append(m)
            continue

        if include_ratio and measurement_type == "ratio":
            if "numerator_pair" in m and "denominator_pair" in m:
                if validate_pair(m["numerator_pair"], num_landmarks) and validate_pair(
                    m["denominator_pair"], num_landmarks
                ):
                    valid_measurements.append(m)

    return valid_measurements


def compute_pair_distance(
    landmarks: torch.Tensor,
    pair: Tuple[int, int],
) -> torch.Tensor:
    """
    Compute full 3D Euclidean distance for one landmark pair.

    Args:
        landmarks: Tensor of shape (B, N, 3)
        pair: (a, b)

    Returns:
        Tensor of shape (B,)
    """
    a, b = pair
    pa = landmarks[:, a, :]
    pb = landmarks[:, b, :]
    return torch.linalg.norm(pa - pb, dim=-1)


def compute_horizontal_distance(
    landmarks: torch.Tensor,
    pair: Tuple[int, int],
) -> torch.Tensor:
    """
    Compute absolute x-axis distance for one landmark pair.

    Useful when you want width-like constraints only.
    """
    a, b = pair
    pa = landmarks[:, a, 0]
    pb = landmarks[:, b, 0]
    return torch.abs(pa - pb)


def compute_vertical_distance(
    landmarks: torch.Tensor,
    pair: Tuple[int, int],
) -> torch.Tensor:
    """
    Compute absolute y-axis distance for one landmark pair.

    Useful when you want height-like constraints only.
    """
    a, b = pair
    pa = landmarks[:, a, 1]
    pb = landmarks[:, b, 1]
    return torch.abs(pa - pb)


def compute_measurement(
    landmarks: torch.Tensor,
    measurement_def: MeasurementDef,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute one measurement from landmarks according to measurement_def.

    Supported types:
    - distance
    - horizontal_distance
    - vertical_distance
    - ratio

    Returns:
        Tensor of shape (B,)
    """
    measurement_type = measurement_def["measurement_type"]

    if measurement_type == "distance":
        return compute_pair_distance(landmarks, measurement_def["pair"])

    if measurement_type == "horizontal_distance":
        return compute_horizontal_distance(landmarks, measurement_def["pair"])

    if measurement_type == "vertical_distance":
        return compute_vertical_distance(landmarks, measurement_def["pair"])

    if measurement_type == "ratio":
        numerator = compute_pair_distance(landmarks, measurement_def["numerator_pair"])
        denominator = compute_pair_distance(
            landmarks, measurement_def["denominator_pair"]
        )
        return numerator / (denominator + eps)

    raise ValueError(f"Unsupported measurement_type: {measurement_type}")


def compute_measurements(
    landmarks: torch.Tensor,
    measurement_defs: List[MeasurementDef],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute multiple measurements from landmarks.

    Args:
        landmarks: Tensor of shape (B, N, 3)
        measurement_defs: list of measurement definitions

    Returns:
        Tensor of shape (B, K), where K = number of measurements
    """
    if len(measurement_defs) == 0:
        return torch.empty(
            landmarks.shape[0],
            0,
            dtype=landmarks.dtype,
            device=landmarks.device,
        )

    values = [compute_measurement(landmarks, m, eps=eps) for m in measurement_defs]
    return torch.stack(values, dim=1)


def build_measurement_names(
    measurement_defs: List[MeasurementDef],
) -> List[str]:
    return [str(m["name"]) for m in measurement_defs]


def build_measurement_categories(
    measurement_defs: List[MeasurementDef],
) -> List[str]:
    return [str(m["category"]) for m in measurement_defs]


def build_measurement_metadata(
    measurement_defs: List[MeasurementDef],
) -> List[Dict[str, object]]:
    metadata: List[Dict[str, object]] = []

    for m in measurement_defs:
        item: Dict[str, object] = {
            "name": str(m["name"]),
            "category": str(m["category"]),
            "status": str(m["status"]),
            "measurement_type": str(m["measurement_type"]),
            "description": str(m.get("description", "")),
            "anatomical_note": str(m.get("anatomical_note", "")),
            "caution": str(m.get("caution", "")),
        }

        if "pair" in m:
            item["pair"] = list(m["pair"])

        if "numerator_pair" in m:
            item["numerator_pair"] = list(m["numerator_pair"])

        if "denominator_pair" in m:
            item["denominator_pair"] = list(m["denominator_pair"])

        metadata.append(item)

    return metadata
