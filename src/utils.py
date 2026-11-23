import os

import cv2 as cv
import numpy as np
import rawpy


def read_image(path: str | os.PathLike) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"The image at {str(path)} does not exist")
    return img

def load2rgb_uin16(path: str | os.PathLike) -> np.ndarray:
    """PNG (BGR, uint16) -> float32 RGB in [0,1]"""
    img = read_image(path)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32) / 65535

def fit_cam_to_linear_srgb(raw_path, ):
    """
    Fits the 3x3 matrix M that rawpy applies after white balance, so that
    linear_srgb ~= M @ wb_camera_rgb. wb is the 4-element user_wb (R, G, B, G2).
    """

    test_wbs = [
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 1.0, 1.6, 1.0],
        [1.5, 1.0, 2.5, 1.0],
    ]

    params = {
        "user_wb": test_wbs[0],
        "use_camera_wb": False,
        "use_auto_wb": False,
        "no_auto_bright": True,
        "gamma": (1, 1),
        "output_bps": 16,
        "half_size": True
    }

    with rawpy.imread(raw_path) as raw:
        cam = raw.postprocess(output_color=rawpy.ColorSpace.raw, **params)  # ty: ignore[invalid-argument-type]
        srgb = raw.postprocess(output_color=rawpy.ColorSpace.sRGB, **params) # ty: ignore[invalid-argument-type]

    cam = cam.reshape(-1,3).astype(np.float64) / 65535
    srgb = srgb.reshape(-1,3).astype(np.float64) / 65535

    # boolean mask to clip values
    ok = (cam.max(axis=1) < 0.98) & (srgb.max(axis=1) < 0.98) & (srgb.min(axis=1) > 0.002)

    # solve cam @ M.T = srgb in the least squares sense
    m_transposed, residuals, _, _ = np.linalg.lstsq(cam[ok], srgb[ok], rcond=None)
    M = m_transposed.T

    error = np.abs(cam[ok] @ M.T - srgb[ok]).mean()
    #print("M:\n", M)
    #print("row sums: ", M.sum(axis=1), "| mean abs error: ", error)
    return M


if __name__ == "__main__":
    fit_cam_to_linear_srgb("data/sony_superraw/Place18/Place18_12.arw")