"""
Standalone rawpy postprocessing to output a preview of one or more places

uv run python -m src.rawpy_postprocess --png-root data/sony --raw-root data/sony_raw --places 333 1 800 --algorithm illuminant1
"""

import cv2
import numpy as np
import rawpy

import src.preprocess as preprocess
from src.wbalance import IlluminantEstimate

MAX_UINT16 = 65535
ALGS = ["illuminant1", "illuminant2", "illuminant3", "grey_world", "white_patch", "illu_map_mean"]

def _apply_wb_mosaic(raw: rawpy.RawPy, illuminant: np.ndarray, black: int):
    """
    In-place white balance on the Bayer Mosaic [LSMI: apply_wb_raw]
    illuminant: RGB illuminant colour, either (3,) or illuminant map (h,w,3), these are the dataset Light* values and not gains.
    """
    # these slice and sizes describe the visible area of the whole sensor raw
    si = raw.sizes
    sli = (slice(si.top_margin, si.top_margin + si.height),
          slice(si.left_margin, si.left_margin + si.width))
    # view into the writable raw buffer
    vis = raw.raw_image[sli]
    # Bayer pattern colors mosaic of indices, in case of Bayer pattern: (0=R, 1=G1, 2=B, 3=G2)
    colors = raw.raw_colors[sli]
    H, W = vis.shape

    illu = np.asarray(illuminant, dtype=np.float32)
    # global illuminant
    if illu.ndim == 1:
        r_gain = np.full((H, W), illu[1] / illu[0], np.float32)
        b_gain = np.full((H, W), illu[1] / illu[2], np.float32)
    else:
        # map size should be compatible with the half-size grid for each 2x2 Bayer cell
        h2, w2 = H//2, W//2
        map = cv2.resize(illu, (w2, h2), interpolation=cv2.INTER_LINEAR)
        # red and blue gain multiplier maps
        r = np.repeat(np.repeat(map[..., 1] / map[..., 0], 2, axis=0), 2, axis=1)
        b = np.repeat(np.repeat(map[..., 1] / map[..., 2], 2, axis=0), 2, axis=1)
        pad = ((0, H-2*h2), (0, W-2*w2))
        r_gain = np.pad(r, pad, mode="edge")
        b_gain = np.pad(b, pad, mode="edge")

    # remove black level and clip negatives to 0
    mosaic = np.clip(vis.astype(np.float32) - black, 0, None)
    mult = np.ones_like(mosaic)
    mult[colors==0] = r_gain[colors==0]
    mult[colors==2] = b_gain[colors==2]
    vis[:] = np.clip(mosaic * mult + black, 0, 65535).astype(vis.dtype)

def render_rawpy(raw_path, est: IlluminantEstimate) -> np.ndarray:
    """
    White balance on the mosaic (global illuminant or per-pixel map),
    -> libraw demosaic and conversion to sRGB.
    Returns a uint8 BGR image.
    """
    with rawpy.imread(str(raw_path)) as raw:
        black = int(min(raw.black_level_per_channel))
        _apply_wb_mosaic(raw, est.mixed(), black)
        rgb = raw.postprocess(
            user_black=black,
            user_wb=[1, 1, 1, 1],            # white balance is already applied to the mosaic
            no_auto_bright=True,
            half_size=True,
            output_color=rawpy.ColorSpace.sRGB,
            #gamma=(2.4, 12.92),              # dcraw's setting for the sRGB curve (rawpy's default is BT.709)
            output_bps=8,
        )

    rgb = preprocess.resize(rgb, 720)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def rawpy_refs(raw_path: str) -> dict[str, np.ndarray]:
    with rawpy.imread(raw_path) as raw:
        return {
            "awb": raw.postprocess(use_auto_wb=True, no_auto_bright=True, half_size=True),
            "daylight": raw.postprocess(user_wb=raw.daylight_whitebalance, no_auto_bright=True, half_size=True),
            "camera": raw.postprocess(user_wb=raw.camera_whitebalance, no_auto_bright=True, half_size=True),
        }