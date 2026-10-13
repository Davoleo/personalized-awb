import os

import numpy as np
import cv2 as cv

def read_image(path: str | os.PathLike) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"The image at {str(path)} does not exist")
    return img

def load_linear_rgb(path: str | os.PathLike) -> np.ndarray:
    """PNG (BGR, uint16) -> float32 RGB in [0,1]"""
    img = read_image(path)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32) / 65535