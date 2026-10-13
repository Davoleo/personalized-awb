from typing import Literal

import numpy as np

from src import Metadata
from src.ciexyz import convert_to_ciexyz
from src.config import RenderConfig
from src.wbalance import IlluminantEstimate, gains

Space = Literal["camera", "xyz_d50", "xyz_libraw", "srgb_linear"]

def render_manual(x: np.ndarray, est: IlluminantEstimate, meta: Metadata, cfg: RenderConfig) -> np.ndarray:
    """
    x: float32 RGB [0,1], black-subtracted, not white balanced
    returns a float image: camera RGB, if cfg.space == "raw", CIE XYZ otherwise
    """

    to_xyz = cfg.space != "raw"

    # global estimate: balance with single illuminant then optionally convert.
    if not est.is_local:
        illuminant = est.colors[0]
        balanced = x * gains(illuminant)
        return convert_to_ciexyz(balanced, meta, illuminant) if to_xyz else balanced

    # local estimate - one term per light weighted by that light's alpha map
    #     blend: each term is the image balanced with that light alone
    #     mix:   every term is the image balanced with the per-pixel mixed illuminant: only the forward matrix differs per term.
    shared = x * gains(est.mixed()) if cfg.local_mode == "mix" else None

    out = np.zeros_like(x)
    for k, lum in enumerate(est.colors):
        balanced = shared if shared is not None else x * gains(lum)
        if to_xyz:
            balanced = convert_to_ciexyz(balanced, meta, lum)
        if est.alpha is None:
            raise ValueError("Local wb rendering without alpha map!")
        # ... => leading axes | k:k+1 => divide the matrix by the last channel, but keep the last dimension (H,W,3) -> (H,W,1) instead of (H,W)
        out += est.alpha[..., k:k + 1] * balanced  # (H,W,1) * (H,W,3) -> per-pixel multiplication of RGB by alpha
    return out  # out will contain sum over k of (alpha_k * balanced_k)