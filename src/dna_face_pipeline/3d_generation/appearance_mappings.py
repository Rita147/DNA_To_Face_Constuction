from typing import Any, Dict, Iterable, Optional


# =========================
# Dataset-aligned categories
# =========================

EYE_COLOR_RGB: Dict[str, list[int]] = {
    "green": [92, 122, 84],
    "brown": [92, 61, 38],
    "blue": [88, 125, 168],
    "unknown": [100, 100, 100],
}

HAIR_COLOR_RGB: Dict[str, list[int]] = {
    "red": [150, 75, 40],
    "brown": [92, 62, 42],
    "blonde": [196, 170, 96],
    "unknown": [110, 110, 110],
}

# Match your chosen skin colors
SKIN_TONE_RGB: Dict[str, list[int]] = {
    "very_light": [241,194,125],  # #F2ECBE
    "light": [224,172,105],       # #DFA878
    "dark": [141,85,36],        # #BA704F
    "unknown": [180, 150, 120],
}

HAIRSTYLE_ALIASES: Dict[str, str] = {
    "straight": "straight_medium",
    "wavy": "wavy_medium",
    "curly": "curly_medium",
    "unknown": "unknown",
}


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "" or text in {"nan", "null"}:
        return None
    return text


def _first_present(row: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in row:
            normalized = _normalize_text(row[key])
            if normalized is not None:
                return normalized
    return None


def _canonical_eye_color(label: Optional[str]) -> str:
    if label is None:
        return "unknown"
    mapping = {
        "green": "green",
        "brown": "brown",
        "blue": "blue",
    }
    return mapping.get(label, "unknown")


def _canonical_hair_color(label: Optional[str]) -> str:
    if label is None:
        return "unknown"
    mapping = {
        "red": "red",
        "brown": "brown",
        "blonde": "blonde",
    }
    return mapping.get(label, "unknown")


def _canonical_skin_tone(label: Optional[str]) -> str:
    if label is None:
        return "unknown"
    mapping = {
        "very_light": "very_light",
        "very light": "very_light",
        "light": "light",
        "dark": "dark",
    }
    return mapping.get(label, "unknown")


def _canonical_hair_texture(label: Optional[str]) -> str:
    if label is None:
        return "unknown"
    mapping = {
        "straight": "straight",
        "wavy": "wavy",
        "curly": "curly",
    }
    return mapping.get(label, "unknown")


def _canonical_hair_thickness(label: Optional[str]) -> str:
    if label is None:
        return "unknown"
    mapping = {
        "fine": "fine",
        "medium": "medium",
        "thick": "thick",
    }
    return mapping.get(label, "unknown")


def _canonical_hairline_shape(label: Optional[str]) -> str:
    if label is None:
        return "unknown"
    mapping = {
        "widow_peak": "widow_peak",
        "widow peak": "widow_peak",
        "rounded": "rounded",
        "straight": "straight",
    }
    return mapping.get(label, "unknown")


def _canonical_freckling(label: Optional[str]) -> str:
    if label is None:
        return "unknown"
    mapping = {
        "none": "none",
        "some": "some",
        "extensive": "extensive",
    }
    return mapping.get(label, "unknown")


def _canonical_hairstyle_from_texture(hair_texture: str) -> str:
    return HAIRSTYLE_ALIASES.get(hair_texture, "unknown")


def extract_appearance_traits_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    eye_color = _canonical_eye_color(
        _first_present(row, ["eye_color", "eye_colour"])
    )

    hair_color = _canonical_hair_color(
        _first_present(row, ["hair_color", "hair_colour"])
    )

    skin_tone = _canonical_skin_tone(
        _first_present(row, ["skin_tone", "complexion"])
    )

    hair_texture = _canonical_hair_texture(
        _first_present(row, ["hair_texture", "hair_style", "hairstyle", "hair_type"])
    )

    hair_thickness = _canonical_hair_thickness(
        _first_present(row, ["hair_thickness"])
    )

    hairline_shape = _canonical_hairline_shape(
        _first_present(row, ["hairline_shape"])
    )

    freckling = _canonical_freckling(
        _first_present(row, ["freckling", "freckles"])
    )

    hairstyle = _canonical_hairstyle_from_texture(hair_texture)

    return {
        "eye_color": eye_color,
        "hair_color": hair_color,
        "skin_tone": skin_tone,
        "hair_texture": hair_texture,
        "hair_thickness": hair_thickness,
        "hairline_shape": hairline_shape,
        "hairstyle": hairstyle,
        "freckling": freckling,
    }


def build_appearance_render_plan(appearance_traits: Dict[str, Any]) -> Dict[str, Any]:
    eye_color = str(appearance_traits.get("eye_color", "unknown"))
    hair_color = str(appearance_traits.get("hair_color", "unknown"))
    skin_tone = str(appearance_traits.get("skin_tone", "unknown"))
    hair_texture = str(appearance_traits.get("hair_texture", "unknown"))
    hair_thickness = str(appearance_traits.get("hair_thickness", "unknown"))
    hairline_shape = str(appearance_traits.get("hairline_shape", "unknown"))
    hairstyle = str(appearance_traits.get("hairstyle", "unknown"))
    freckling = str(appearance_traits.get("freckling", "unknown"))

    return {
        "eye_color_label": eye_color,
        "eye_color_rgb": EYE_COLOR_RGB.get(eye_color, EYE_COLOR_RGB["unknown"]),

        "hair_color_label": hair_color,
        "hair_color_rgb": HAIR_COLOR_RGB.get(hair_color, HAIR_COLOR_RGB["unknown"]),

        "skin_tone_label": skin_tone,
        "skin_tone_rgb": SKIN_TONE_RGB.get(skin_tone, SKIN_TONE_RGB["unknown"]),

        "hair_texture_label": hair_texture,
        "hair_thickness_label": hair_thickness,
        "hairline_shape_label": hairline_shape,

        "hairstyle_label": hairstyle,
        "hair_asset_id": hairstyle,

        "freckling_label": freckling,
        "freckles_enabled": freckling in {"some", "extensive"},
    }
