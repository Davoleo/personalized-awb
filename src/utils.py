import os

import numpy as np
import cv2 as cv

def read_image(path: str | os.PathLike) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"The image at {str(path)} does not exist")

    return img