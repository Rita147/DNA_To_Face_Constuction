from .utils import most_common

# Eye color mappings
EYE_MAP = {
    "blue": 0,
    "intermediate": 1,
    "brown": 2,

    # extra categories observed in 41-SNP model
    "blue_shift": 0,
    "strong_blue_shift": 0,
    "light": 0,
    "medium": 1,
    "dark": 2,
}

# Hair color mappings
HAIR_MAP = {
    "non_red": 0,
    "red_carrier": 1,
    "strong_red": 2,

    "dark": 0,
    "medium": 1,
    "light": 2,
    "brown": 0,
    "light_brown": 1,
    "dark_blonde": 1,
    "blonde": 2,
}

# Skin tone mappings
SKIN_MAP = {
    "light": 0,
    "medium": 1,
    "dark": 2,

    "lighter": 0,
    "darker": 2,
}

def encode_trait_vector(trait_dict):
    """
    Takes dictionary:
      { trait_name: [trait_values...] }
    Returns numeric trait vector dictionary.

    Output example:
      { "eye_color": 0, "hair_color": 1, "skin_tone": 2 }
    """
    eye_vals  = trait_dict.get("eye_color",  ["unknown"])
    hair_vals = trait_dict.get("hair_color", ["unknown"])
    skin_vals = trait_dict.get("skin_tone",  ["unknown"])

    return {
        "eye_color": EYE_MAP.get(most_common(eye_vals), -1),
        "hair_color": HAIR_MAP.get(most_common(hair_vals), -1),
        "skin_tone": SKIN_MAP.get(most_common(skin_vals), -1),
    }
    
