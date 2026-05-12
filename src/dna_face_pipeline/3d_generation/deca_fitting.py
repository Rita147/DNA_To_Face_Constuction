"""
DECA image-fitting integration for FLAME parameter estimation.

Wraps the DECA model (https://github.com/yfeng95/DECA) to fit FLAME
shape / expression / pose parameters directly from a 2-D portrait image.

Setup
-----
1. Clone the DECA repository:
       git clone https://github.com/yfeng95/DECA \
           src/dna_face_pipeline/3d_generation/DECA
2. Download DECA model weights as described in the DECA README and place
   them in DECA/data/.
3. Install DECA's Python dependencies (see DECA/requirements.txt).

Alternatively, set the environment variable ``DECA_DIR`` to the root of
the cloned repository if you keep it elsewhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_THREE_D_DIR = Path(__file__).resolve().parent
_DEFAULT_DECA_DIR = _THREE_D_DIR / "DECA"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_deca_dir() -> Optional[Path]:
    """Return the DECA repo root, checking the env var first, then defaults."""
    env_val = os.environ.get("DECA_DIR", "").strip()
    if env_val:
        p = Path(env_val)
        if p.is_dir() and (p / "decalib").is_dir():
            return p

    candidates = [
        _DEFAULT_DECA_DIR,
        _THREE_D_DIR.parent / "DECA",
        _THREE_D_DIR.parents[2] / "DECA",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "decalib").is_dir():
            return candidate

    return None


def _preprocess_image(image_path: Path, image_size: int = 224):
    """
    Load a portrait image and return a (1, 3, H, W) float32 tensor in [-1, 1].
    DECA expects the face to roughly fill the frame; a centre-crop is applied
    if the image is not already square.
    """
    import torch
    from PIL import Image as PILImage

    img = PILImage.open(str(image_path)).convert("RGB")

    # Centre-square crop so the face stays centred
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    img = img.resize((image_size, image_size), PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0          # [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor * 2.0 - 1.0                               # [-1, 1]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fit_deca(
    image_path: Path,
    device_str: str = "cpu",
    deca_dir: Optional[Path] = None,
    use_tex: bool = True,
    rasterizer_type: str = "standard",
) -> Dict[str, Any]:
    """
    Fit FLAME parameters to a portrait image using DECA.

    Parameters
    ----------
    image_path:
        Path to the 2-D portrait (any PIL-readable format).
    device_str:
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
    deca_dir:
        Root of the cloned DECA repository.  Auto-detected if *None*.
    use_tex:
        Whether to decode DECA's neural albedo texture map.
    rasterizer_type:
        ``"standard"`` (pure Python, no extra deps) or ``"pytorch3d"``.

    Returns
    -------
    dict
        ``vertices``  – (N, 3) float64 ndarray, FLAME-space 3-D vertices
        ``faces``     – (F, 3) int32  ndarray, FLAME face connectivity
        ``landmarks`` – (68, 3) float64 ndarray, 3-D landmarks
        ``albedo``    – (3, H, W) float32 ndarray in [0, 1], or *None*
        ``codedict``  – raw DECA parameter dict (numpy arrays)
    """
    import torch

    root = deca_dir or _find_deca_dir()
    if root is None:
        raise RuntimeError(
            "DECA repository not found.\n"
            "Clone https://github.com/yfeng95/DECA to:\n"
            f"  {_DEFAULT_DECA_DIR}\n"
            "or set the DECA_DIR environment variable to the repo root,\n"
            "then download DECA model weights as described in the DECA README."
        )

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from decalib.deca import DECA                    # noqa: PLC0415
    from decalib.utils.config import cfg as deca_cfg  # noqa: PLC0415

    deca_cfg.model.use_tex = use_tex
    deca_cfg.rasterizer_type = rasterizer_type

    device = torch.device(device_str)
    deca_model = DECA(config=deca_cfg, device=device)

    image_tensor = _preprocess_image(image_path).to(device)

    with torch.no_grad():
        codedict = deca_model.encode(image_tensor)
        opdict, _ = deca_model.decode(codedict)

    vertices = opdict["verts"][0].detach().cpu().numpy().astype(np.float64)
    landmarks = opdict["landmarks3d"][0].detach().cpu().numpy().astype(np.float64)

    # FLAME face topology (same across all DECA outputs)
    faces_tensor = deca_model.flame.faces_tensor
    faces = faces_tensor.detach().cpu().numpy().astype(np.int32)

    raw_albedo = opdict.get("albedo")
    albedo_np: Optional[np.ndarray] = None
    if raw_albedo is not None:
        albedo_np = raw_albedo[0].detach().cpu().numpy().astype(np.float32)
        # Ensure channel-first layout (3, H, W)
        if albedo_np.ndim == 3 and albedo_np.shape[2] == 3:
            albedo_np = albedo_np.transpose(2, 0, 1)
        albedo_np = np.clip(albedo_np, 0.0, 1.0)

    raw_codedict: Dict[str, Any] = {
        k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
        for k, v in codedict.items()
    }

    return {
        "vertices": vertices,
        "faces": faces,
        "landmarks": landmarks,
        "albedo": albedo_np,
        "codedict": raw_codedict,
    }
