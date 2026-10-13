import colour

from src.config import RenderConfig
import cv2 as cv
import numpy as np

def finalize(img: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    """float render from render_manual -> BGR final image"""
    if cfg.space == 'srgb':
        rgb = colour.XYZ_to_sRGB(img)
        out = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    else:
        out = (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16)
    return cv.cvtColor(out, cv.COLOR_RGB2BGR)

def display_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    peak = np.iinfo(img.dtype).max
    return (np.power(img.astype(np.float32) / peak, gamma) * peak).astype(img.dtype)

def show_samples(images: list[np.ndarray], title="images", cols=4, gamma=None):
    padded = images + [np.zeros_like(images[0])] * (-len(images) % cols)
    rows = [np.concatenate(padded[i:i + cols], axis=1) for i in range(0, len(padded), cols)]
    full = np.concatenate(rows, axis=0)
    cv.imshow(title, display_gamma(full, gamma) if gamma else full)
    cv.waitKey(0)

