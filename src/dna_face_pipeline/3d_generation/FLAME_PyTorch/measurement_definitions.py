from typing import Dict, List, TypedDict, Literal, Optional


MeasurementType = Literal[
    "distance",
    "vertical_distance",
    "horizontal_distance",
    "ratio",
]
MeasurementStatus = Literal["recommended", "approximate", "unsupported"]


class MeasurementDef(TypedDict, total=False):
    name: str
    category: str
    measurement_type: MeasurementType
    status: MeasurementStatus
    pair: tuple[int, int]
    numerator_pair: tuple[int, int]
    denominator_pair: tuple[int, int]
    description: str
    anatomical_note: str
    caution: str


# ---------------------------------------------------------------------
# Core 68-landmark groups (standard-style indexing assumption)
# ---------------------------------------------------------------------
LANDMARK_GROUPS: Dict[str, List[int]] = {
    "jaw_contour": list(range(0, 17)),
    "left_eyebrow": list(range(17, 22)),
    "right_eyebrow": list(range(22, 27)),
    "nose": list(range(27, 36)),
    "left_eye": list(range(36, 42)),
    "right_eye": list(range(42, 48)),
    "outer_mouth": list(range(48, 60)),
    "inner_mouth": list(range(60, 68)),
}


# ---------------------------------------------------------------------
# Recommended measurements
# These are the strongest first choices for phenotype-fitting work.
# ---------------------------------------------------------------------
RECOMMENDED_MEASUREMENTS: List[MeasurementDef] = [
    {
        "name": "face_width",
        "category": "global_face",
        "measurement_type": "horizontal_distance",
        "status": "recommended",
        "pair": (0, 16),
        "description": "Approximate overall face width using outer jaw contour endpoints.",
        "anatomical_note": "Strong global width control.",
        "caution": "Contour-based, not true bizygomatic width.",
    },
    {
        "name": "chin_center_to_nose_bridge",
        "category": "global_face",
        "measurement_type": "vertical_distance",
        "status": "recommended",
        "pair": (8, 27),
        "description": "Approximate vertical proportion from chin center to upper nose bridge.",
        "anatomical_note": "Useful global lower/mid-face height constraint.",
        "caution": "Not identical to full anatomical face height.",
    },
    {
        "name": "lower_face_height",
        "category": "global_face",
        "measurement_type": "vertical_distance",
        "status": "recommended",
        "pair": (33, 8),
        "description": "Approximate lower-face height from nose base to chin center.",
        "anatomical_note": "Useful for jaw/chin region control.",
        "caution": "Sensitive to chin geometry.",
    },
    {
        "name": "outer_eye_width",
        "category": "eyes",
        "measurement_type": "horizontal_distance",
        "status": "recommended",
        "pair": (36, 45),
        "description": "Approximate width between outer eye corners.",
        "anatomical_note": "Useful global eye-span control.",
        "caution": "Verify landmark interpretation visually.",
    },
    {
        "name": "inner_eye_distance",
        "category": "eyes",
        "measurement_type": "horizontal_distance",
        "status": "recommended",
        "pair": (39, 42),
        "description": "Approximate width between inner eye corners.",
        "anatomical_note": "Useful interocular spacing control.",
        "caution": "Should be visually validated.",
    },
    {
        "name": "left_eye_width",
        "category": "eyes",
        "measurement_type": "horizontal_distance",
        "status": "recommended",
        "pair": (36, 39),
        "description": "Approximate width of the left eye.",
        "anatomical_note": "Good local eye-shape control.",
        "caution": "Eye corner interpretation should be checked in viewer.",
    },
    {
        "name": "right_eye_width",
        "category": "eyes",
        "measurement_type": "horizontal_distance",
        "status": "recommended",
        "pair": (42, 45),
        "description": "Approximate width of the right eye.",
        "anatomical_note": "Good local eye-shape control.",
        "caution": "Eye corner interpretation should be checked in viewer.",
    },
    {
        "name": "nose_width",
        "category": "nose",
        "measurement_type": "horizontal_distance",
        "status": "recommended",
        "pair": (31, 35),
        "description": "Approximate nose width across the alar/base region.",
        "anatomical_note": "Strong nose-shape control.",
        "caution": "Represents base width, not tip width.",
    },
    {
        "name": "nose_height",
        "category": "nose",
        "measurement_type": "vertical_distance",
        "status": "recommended",
        "pair": (27, 33),
        "description": "Approximate nose height from upper bridge to nose base/tip region.",
        "anatomical_note": "Useful vertical nose proportion.",
        "caution": "Depends on exact FLAME landmark geometry.",
    },
    {
        "name": "mouth_width",
        "category": "mouth",
        "measurement_type": "horizontal_distance",
        "status": "recommended",
        "pair": (48, 54),
        "description": "Approximate width between outer mouth corners.",
        "anatomical_note": "Standard mouth-width control.",
        "caution": "Use neutral expression during fitting.",
    },
    {
        "name": "mouth_opening_height",
        "category": "mouth",
        "measurement_type": "vertical_distance",
        "status": "recommended",
        "pair": (62, 66),
        "description": "Approximate inner mouth opening height.",
        "anatomical_note": "Useful for mouth opening / lip separation behavior.",
        "caution": "Should be kept near neutral expression unless intentionally varied.",
    },
    {
        "name": "inner_mouth_width",
        "category": "mouth",
        "measurement_type": "horizontal_distance",
        "status": "recommended",
        "pair": (60, 64),
        "description": "Approximate inner mouth width.",
        "anatomical_note": "Useful inner-lip aperture control.",
        "caution": "More sensitive to mouth shape changes than outer mouth width.",
    },
]


# ---------------------------------------------------------------------
# Approximate / proxy measurements
# These are usable, but should be treated more carefully.
# ---------------------------------------------------------------------
APPROXIMATE_MEASUREMENTS: List[MeasurementDef] = [
    {
        "name": "jaw_width",
        "category": "jaw",
        "measurement_type": "horizontal_distance",
        "status": "approximate",
        "pair": (4, 12),
        "description": "Approximate jaw width using lower contour points.",
        "anatomical_note": "Useful jaw-region proxy.",
        "caution": "Not true skeletal gonial width.",
    },
    {
        "name": "midface_width",
        "category": "global_face",
        "measurement_type": "horizontal_distance",
        "status": "approximate",
        "pair": (3, 13),
        "description": "Approximate mid/lower face width using contour landmarks.",
        "anatomical_note": "Helpful general width proxy.",
        "caution": "Contour-based, not true cheekbone width.",
    },
    {
        "name": "chin_width_proxy",
        "category": "jaw",
        "measurement_type": "horizontal_distance",
        "status": "approximate",
        "pair": (7, 9),
        "description": "Approximate chin width near the chin center.",
        "anatomical_note": "Useful chin-shape proxy.",
        "caution": "Not a standard anthropometric chin breadth.",
    },
    {
        "name": "brow_width_left",
        "category": "eyebrows",
        "measurement_type": "horizontal_distance",
        "status": "approximate",
        "pair": (17, 21),
        "description": "Approximate left eyebrow width.",
        "anatomical_note": "Useful brow-region width proxy.",
        "caution": "Eyebrow geometry may vary with expression/model shape.",
    },
    {
        "name": "brow_width_right",
        "category": "eyebrows",
        "measurement_type": "horizontal_distance",
        "status": "approximate",
        "pair": (22, 26),
        "description": "Approximate right eyebrow width.",
        "anatomical_note": "Useful brow-region width proxy.",
        "caution": "Eyebrow geometry may vary with expression/model shape.",
    },
    {
        "name": "left_brow_to_eye_distance",
        "category": "eyebrows",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (19, 37),
        "description": "Approximate vertical distance from left brow region to left upper eyelid region.",
        "anatomical_note": "Useful upper-eye/brow spacing proxy.",
        "caution": "Not a strict anatomical eyebrow height.",
    },
    {
        "name": "right_brow_to_eye_distance",
        "category": "eyebrows",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (24, 44),
        "description": "Approximate vertical distance from right brow region to right upper eyelid region.",
        "anatomical_note": "Useful upper-eye/brow spacing proxy.",
        "caution": "Not a strict anatomical eyebrow height.",
    },
    {
        "name": "left_eye_opening_height",
        "category": "eyes",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (37, 41),
        "description": "Approximate left eye opening height.",
        "anatomical_note": "Useful local eye aperture proxy.",
        "caution": "May vary with expression and eyelid pose.",
    },
    {
        "name": "right_eye_opening_height",
        "category": "eyes",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (43, 47),
        "description": "Approximate right eye opening height.",
        "anatomical_note": "Useful local eye aperture proxy.",
        "caution": "May vary with expression and eyelid pose.",
    },
    {
        "name": "nose_base_width",
        "category": "nose",
        "measurement_type": "horizontal_distance",
        "status": "approximate",
        "pair": (31, 35),
        "description": "Approximate base width of the nose.",
        "anatomical_note": "Close to nose width, useful as explicitly named phenotype feature.",
        "caution": "Same landmark pair as nose_width; semantically useful but not independent.",
    },
    {
        "name": "nose_bridge_width_proxy",
        "category": "nose",
        "measurement_type": "horizontal_distance",
        "status": "approximate",
        "pair": (32, 34),
        "description": "Approximate width across the mid/bridge-lower nose region.",
        "anatomical_note": "Useful proxy for a narrower central nose width.",
        "caution": "Not true bony bridge width.",
    },
    {
        "name": "nostril_width_proxy",
        "category": "nose",
        "measurement_type": "horizontal_distance",
        "status": "approximate",
        "pair": (31, 35),
        "description": "Approximate nostril/alar span.",
        "anatomical_note": "Useful as a phenotype-facing nostril-width proxy when only 68 landmarks are available.",
        "caution": "Highly correlated with nose_width; not an independent nostril-only measurement.",
    },
    {
        "name": "brow_to_nose_bridge",
        "category": "upper_face",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (19, 27),
        "description": "Approximate upper-face relation from brow region to nose bridge.",
        "anatomical_note": "Weak forehead/upper-face proxy.",
        "caution": "Not true forehead height.",
    },
    {
        "name": "philtrum_height_proxy",
        "category": "mouth",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (33, 51),
        "description": "Approximate vertical distance from nose base to upper lip center.",
        "anatomical_note": "Useful philtrum-region proxy.",
        "caution": "Only an approximation of philtrum height.",
    },
    {
        "name": "nose_to_mouth_center_height",
        "category": "mouth",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (33, 57),
        "description": "Approximate vertical relation from nose base to lower mouth center.",
        "anatomical_note": "Useful lower mid-face proxy.",
        "caution": "Composite proxy, not a standard named anthropometric measure.",
    },
    {
        "name": "mouth_to_chin_height",
        "category": "mouth",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (57, 8),
        "description": "Approximate vertical relation from lower mouth center to chin center.",
        "anatomical_note": "Useful chin-mouth proportion proxy.",
        "caution": "Sensitive to mouth and chin geometry together.",
    },
    {
        "name": "eye_to_mouth_vertical_proxy",
        "category": "global_face",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (39, 57),
        "description": "Approximate vertical relation between eye region and lower mouth area.",
        "anatomical_note": "Can help constrain facial vertical proportions.",
        "caution": "Composite proxy, not a standard anthropometric pair.",
    },
    {
        "name": "upper_lip_thickness_proxy",
        "category": "mouth",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (51, 62),
        "description": "Approximate upper-lip thickness from outer upper-lip center to inner upper-lip center.",
        "anatomical_note": "Most practical pair-based proxy for upper-lip fullness with 68 landmarks.",
        "caution": "Still a contour-based thickness proxy, not true tissue thickness.",
    },
    {
        "name": "lower_lip_thickness_proxy",
        "category": "mouth",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (66, 57),
        "description": "Approximate lower-lip thickness from inner lower-lip center to outer lower-lip center.",
        "anatomical_note": "Most practical pair-based proxy for lower-lip fullness with 68 landmarks.",
        "caution": "Still a contour-based thickness proxy, not true tissue thickness.",
    },
    {
        "name": "left_cheekbone_height_proxy",
        "category": "global_face",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (2, 36),
        "description": "Approximate left cheekbone/upper-cheek height relative to the outer eye corner.",
        "anatomical_note": "Can be used as a weak cheekbone-height proxy.",
        "caution": "Contour-based and not a true zygomatic landmark.",
    },
    {
        "name": "right_cheekbone_height_proxy",
        "category": "global_face",
        "measurement_type": "vertical_distance",
        "status": "approximate",
        "pair": (14, 45),
        "description": "Approximate right cheekbone/upper-cheek height relative to the outer eye corner.",
        "anatomical_note": "Can be used as a weak cheekbone-height proxy.",
        "caution": "Contour-based and not a true zygomatic landmark.",
    },
]


# ---------------------------------------------------------------------
# Ratio-based measurements
# ---------------------------------------------------------------------
RATIO_MEASUREMENTS: List[MeasurementDef] = [
    {
        "name": "eye_to_nose_width_ratio",
        "category": "ratios",
        "measurement_type": "ratio",
        "status": "approximate",
        "numerator_pair": (36, 45),
        "denominator_pair": (31, 35),
        "description": "Ratio of outer-eye width to nose width.",
        "anatomical_note": "Useful when only proportional phenotype constraints are available.",
        "caution": "Requires ratio-loss handling in the fitting script.",
    },
    {
        "name": "mouth_to_nose_width_ratio",
        "category": "ratios",
        "measurement_type": "ratio",
        "status": "approximate",
        "numerator_pair": (48, 54),
        "denominator_pair": (31, 35),
        "description": "Ratio of mouth width to nose width.",
        "anatomical_note": "Helpful for relative lower-face proportion constraints.",
        "caution": "Requires ratio-loss handling in the optimization pipeline.",
    },
    {
        "name": "lowerface_to_facewidth_ratio",
        "category": "ratios",
        "measurement_type": "ratio",
        "status": "approximate",
        "numerator_pair": (33, 8),
        "denominator_pair": (0, 16),
        "description": "Ratio of lower-face height to face width.",
        "anatomical_note": "Useful scale-normalized lower-face proportion.",
        "caution": "Only meaningful if landmark interpretations are validated.",
    },
    {
        "name": "inner_to_outer_eye_width_ratio",
        "category": "ratios",
        "measurement_type": "ratio",
        "status": "approximate",
        "numerator_pair": (39, 42),
        "denominator_pair": (36, 45),
        "description": "Ratio of inner-eye distance to outer-eye width.",
        "anatomical_note": "Useful for eye spacing relative to total orbital span.",
        "caution": "Needs ratio-loss handling.",
    },
    {
        "name": "nose_to_facewidth_ratio",
        "category": "ratios",
        "measurement_type": "ratio",
        "status": "approximate",
        "numerator_pair": (31, 35),
        "denominator_pair": (0, 16),
        "description": "Ratio of nose width to face width.",
        "anatomical_note": "Useful for normalized nose-width control.",
        "caution": "Needs ratio-loss handling.",
    },
    {
        "name": "left_eye_roundness_ratio",
        "category": "eyes",
        "measurement_type": "ratio",
        "status": "approximate",
        "numerator_pair": (37, 41),
        "denominator_pair": (36, 39),
        "description": "Ratio of left-eye opening height to left-eye width.",
        "anatomical_note": "Potentially useful for eye-size/eye-shape tendencies such as round vs narrow eyes.",
        "caution": "Requires ratio-loss handling and may still be weak in FLAME shape-only optimization.",
    },
    {
        "name": "right_eye_roundness_ratio",
        "category": "eyes",
        "measurement_type": "ratio",
        "status": "approximate",
        "numerator_pair": (43, 47),
        "denominator_pair": (42, 45),
        "description": "Ratio of right-eye opening height to right-eye width.",
        "anatomical_note": "Potentially useful for eye-size/eye-shape tendencies such as round vs narrow eyes.",
        "caution": "Requires ratio-loss handling and may still be weak in FLAME shape-only optimization.",
    },
    {
        "name": "upper_to_lower_lip_thickness_ratio",
        "category": "mouth",
        "measurement_type": "ratio",
        "status": "approximate",
        "numerator_pair": (51, 62),
        "denominator_pair": (66, 57),
        "description": "Ratio of upper-lip thickness proxy to lower-lip thickness proxy.",
        "anatomical_note": "Useful if you later want relative lip-balance constraints.",
        "caution": "Requires ratio-loss handling and relies on proxy thickness measurements.",
    },
]


# ---------------------------------------------------------------------
# Unsupported or not-directly-available measurements
# ---------------------------------------------------------------------
UNSUPPORTED_MEASUREMENTS: List[MeasurementDef] = [
    {
        "name": "forehead_height",
        "category": "upper_face",
        "measurement_type": "vertical_distance",
        "status": "unsupported",
        "description": "True forehead height from brow region to hairline/top forehead.",
        "anatomical_note": "Important phenotype feature in real facial description.",
        "caution": (
            "Standard FLAME 68 landmarks do not include a true hairline/top-forehead point. "
            "You need a pseudo-landmark, mesh-derived landmark, or hairline module."
        ),
    },
    {
        "name": "hairline_width",
        "category": "hair",
        "measurement_type": "horizontal_distance",
        "status": "unsupported",
        "description": "Width of frontal hairline.",
        "anatomical_note": "Potentially important if hair is modeled later.",
        "caution": "Not represented by standard FLAME face landmarks.",
    },
    {
        "name": "hairstyle_type",
        "category": "hair",
        "measurement_type": "distance",
        "status": "unsupported",
        "description": "Categorical hairstyle selection such as curly, straight, short, long.",
        "anatomical_note": "Appearance feature rather than core facial geometry.",
        "caution": "Should be implemented as a separate modular appearance layer.",
    },
    {
        "name": "cheekbone_prominence",
        "category": "global_face",
        "measurement_type": "distance",
        "status": "unsupported",
        "description": "Degree of zygomatic/cheekbone prominence in 3D.",
        "anatomical_note": "Important for face shape realism.",
        "caution": "Not directly captured by a simple 68-landmark pair; may require mesh-depth or pseudo-landmarks.",
    },
    {
        "name": "nose_tip_projection",
        "category": "nose",
        "measurement_type": "distance",
        "status": "unsupported",
        "description": "Forward projection of the nose tip in 3D.",
        "anatomical_note": "Important nose-shape descriptor.",
        "caution": "Needs 3D mesh-based measurement, not just a 2D-style landmark pair.",
    },
    {
        "name": "true_nostril_width",
        "category": "nose",
        "measurement_type": "distance",
        "status": "unsupported",
        "description": "True isolated nostril width rather than alar/base proxies.",
        "anatomical_note": "Potentially useful for finer nose phenotype fitting.",
        "caution": "Not robustly encoded by standard 68 landmarks as an independent measurement.",
    },
    {
        "name": "lip_thickness_upper",
        "category": "mouth",
        "measurement_type": "distance",
        "status": "unsupported",
        "description": "True upper-lip thickness.",
        "anatomical_note": "Important mouth phenotype feature.",
        "caution": "Current landmark setup does not directly encode robust tissue thickness; use upper_lip_thickness_proxy instead.",
    },
    {
        "name": "lip_thickness_lower",
        "category": "mouth",
        "measurement_type": "distance",
        "status": "unsupported",
        "description": "True lower-lip thickness.",
        "anatomical_note": "Important mouth phenotype feature.",
        "caution": "Current landmark setup does not directly encode robust tissue thickness; use lower_lip_thickness_proxy instead.",
    },
    {
        "name": "true_eye_shape",
        "category": "eyes",
        "measurement_type": "distance",
        "status": "unsupported",
        "description": "High-level categorical eye shape such as almond, hooded, or round.",
        "anatomical_note": "Important phenotype feature for realism.",
        "caution": "Requires either richer landmark logic, ratio-based handling, or a different representation than the current pair-only fitting path.",
    },
]


# ---------------------------------------------------------------------
# Combined access helpers
# ---------------------------------------------------------------------
ALL_MEASUREMENTS: List[MeasurementDef] = (
    RECOMMENDED_MEASUREMENTS
    + APPROXIMATE_MEASUREMENTS
    + RATIO_MEASUREMENTS
    + UNSUPPORTED_MEASUREMENTS
)


def get_measurement_by_name(name: str) -> Optional[MeasurementDef]:
    for measurement in ALL_MEASUREMENTS:
        if measurement["name"] == name:
            return measurement
    return None


def get_measurements_by_status(status: MeasurementStatus) -> List[MeasurementDef]:
    return [m for m in ALL_MEASUREMENTS if m["status"] == status]


def get_measurements_by_category(category: str) -> List[MeasurementDef]:
    return [m for m in ALL_MEASUREMENTS if m["category"] == category]


def get_pair_measurements_only(
    include_approximate: bool = True,
) -> List[MeasurementDef]:
    """
    Return only measurements directly defined by a single landmark pair.
    Useful for the current pair-based fitting pipeline.
    """
    allowed_statuses = {"recommended"}
    if include_approximate:
        allowed_statuses.add("approximate")

    return [
        m
        for m in ALL_MEASUREMENTS
        if "pair" in m and m["status"] in allowed_statuses
    ]


def get_recommended_pair_measurements() -> List[MeasurementDef]:
    """
    Return the best first set of direct pair-based measurements
    for the current FLAME fitting experiments.
    """
    return [m for m in RECOMMENDED_MEASUREMENTS if "pair" in m]
