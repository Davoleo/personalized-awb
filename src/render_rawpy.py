"""
rawpy postprocessing module to output an image in sRGB space.
"""

import cv2
import numpy as np
import rawpy

import src.preprocess as preprocess
from src.wbalance import IlluminantEstimate

def _wb_channel(vis: np.ndarray, mask: np.ndarray, black: int, gain):
    """Apply (v - black) * gain + black with clipping to the masked pixels in place."""
    v = vis[mask].astype(np.float32)
    v -= black
    np.maximum(v, 0, out=v)
    v *= gain
    v += black
    np.clip(v, 0, 65535, out=v)
    vis[mask] = v.astype(vis.dtype)


def _apply_wb_mosaic(raw: rawpy.RawPy, illuminant: np.ndarray, black: int, mask_r: np.ndarray, mask_b: np.ndarray, sli):
    """
    In-place white balance on the Bayer Mosaic [LSMI: apply_wb_raw]
    illuminant: RGB illuminant colour, either (3) or illuminant map (h,w,3), these are the dataset Light* values and not gains.
    mask_r/mask_b: precomputed boolean masks of red/blue pixels in the visible area;
        must be captured before any postprocess() call, since libraw clobbers
        its internal color pattern afterwards (raw_colors then reads back garbage).
    sli: slice of the visible area; likewise captured up front, because
        postprocess(half_size=True) rewrites imgdata.sizes in place.
    """
    # view into the writable raw buffer
    vis = raw.raw_image[sli]
    H, W = vis.shape

    illu = np.asarray(illuminant, dtype=np.float32)
    # global illuminant: scalar gains, only red and blue pixels are touched
    if illu.ndim == 1:
        _wb_channel(vis, mask_r, black, float(illu[1] / illu[0]))
        _wb_channel(vis, mask_b, black, float(illu[1] / illu[2]))
        return

    # local illuminant map: per-pixel gain maps on the half-size Bayer grid
    h2, w2 = H // 2, W // 2
    imap = cv2.resize(illu, (w2, h2), interpolation=cv2.INTER_LINEAR)
    r_gain = np.pad(np.repeat(np.repeat(imap[..., 1] / imap[..., 0], 2, axis=0), 2, axis=1),
                    ((0, H - 2 * h2), (0, W - 2 * w2)), mode="edge")
    b_gain = np.pad(np.repeat(np.repeat(imap[..., 1] / imap[..., 2], 2, axis=0), 2, axis=1),
                    ((0, H - 2 * h2), (0, W - 2 * w2)), mode="edge")
    _wb_channel(vis, mask_r, black, r_gain[mask_r])
    _wb_channel(vis, mask_b, black, b_gain[mask_b])


class RawpyRenderer:
    def __init__(self, raw_path):
        self.raw = rawpy.imread(str(raw_path))
        self.black = int(min(self.raw.black_level_per_channel))
        # capture geometry and Bayer mask BEFORE any postprocess: libraw
        # clobbers its internal color pattern during dcraw_process (raw_colors
        # then reads back (6), so WB masks silently match nothing) and
        # rewrites imgdata.sizes in place when half_size is set.
        si = self.raw.sizes
        self.sli = (slice(si.top_margin, si.top_margin + si.height),
                    slice(si.left_margin, si.left_margin + si.width))
        colors = self.raw.raw_colors[self.sli]
        self.mask_r = colors == 0
        self.mask_b = colors == 2
        self.orig = self.raw.raw_image.copy()  # copy of the original mosaic

    def _postprocess(self) -> np.ndarray:
        return self.raw.postprocess(
            user_black=self.black,
            user_wb=[1, 1, 1, 1],  # white balance is already applied to the mosaic
            no_auto_bright=True,
            half_size=True,
            output_color=rawpy.ColorSpace.sRGB,
            #gamma=(2.4, 12.92),              # dcraw's setting for the sRGB curve (rawpy's default is BT.709)
            output_bps=8,
        )

    def render(self, illuminant) -> np.ndarray:
        self.raw.raw_image[:] = self.orig  # restore original mosaic
        _apply_wb_mosaic(self.raw, illuminant, self.black, self.mask_r, self.mask_b, self.sli)
        rgb = self._postprocess()
        rgb = preprocess.resize(rgb, 720)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self):
        self.raw.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def render_rawpy(raw_path, est: IlluminantEstimate) -> np.ndarray:
    """
    White balance on the mosaic (global illuminant or per-pixel map),
    -> libraw demosaic and conversion to sRGB.
    Returns a uint8 BGR image.
    """
    with RawpyRenderer(str(raw_path)) as r:
        return r.render(est.mixed())


def rawpy_refs(raw_path: str) -> dict[str, np.ndarray]:
    with rawpy.imread(raw_path) as raw:
        return {
            "awb": raw.postprocess(use_auto_wb=True, no_auto_bright=True, half_size=True),
            "daylight": raw.postprocess(user_wb=raw.daylight_whitebalance, no_auto_bright=True, half_size=True),
            "camera": raw.postprocess(user_wb=raw.camera_whitebalance, no_auto_bright=True, half_size=True),
        }
