from pylab import ndarray
from enum import Enum
import json
from pathlib import Path

import torch
from torch import nn
import numpy as np
import cv2 as cv

from src import get_project_dir

MAX_UINT16 = 65535

class WBAlgorithm(Enum):
    WHITE_PATCH = 1
    GREY_WORLD = 2
    JSON_DATA = 3

def white_balance(algorithm: WBAlgorithm, img: ndarray, filename: str) -> cv.typing.MatLike:
    wbImage: ndarray
    coeffs: ndarray

    match algorithm:
        case WBAlgorithm.WHITE_PATCH:
            # max: reducing the first 2 dimensions (keep channels, as per openCV shape)
            imageMax = np.amax(img, (0,1))
            print("image maxes: ", imageMax)
            # L2 Norm to normalize illuminant vector
            imageMax /= np.linalg.norm(imageMax)
            print("norm image maxes: ", imageMax)
            # White Patch coeffs
            coeffs = 1.0 / imageMax
        case WBAlgorithm.GREY_WORLD:
            # mean: reducing the first 2 dimensions (keep channels, as per openCV shape)
            imageMean = np.mean(img, axis=(0,1))
            print("image means: ", imageMean)
            imageMean /= np.linalg.norm(imageMean)
            print("image means norm: ", imageMean)
            # Grey World coeffs
            coeffs = 0.5 / imageMean
        case WBAlgorithm.JSON_DATA:
            # take illuminant from json data of the specific image and use it to whitebalance the image
            metadata_path = get_project_dir() / "data" / 'Gehler-Shi' / Path(filename.strip(".png") + "_metadata.json")
            with open(metadata_path, 'r') as file:
                data = json.load(file)

            illu = data['illuminant_color_raw']
            # Convert to BGR
            illu = np.asarray([illu[2], illu[1], illu[0]])
            # L2 Norm to normalize illuminant vector
            illu /= np.linalg.norm(illu)
            # json coeffs
            coeffs = (np.ones((1,3)) / illu)

    # Patch application
    wbImage = img * coeffs
    return wbImage


def gamma_correction(image, gamma: float):
    lookup_table = np.empty((1, MAX_UINT16+1), np.uint16)
    for i in range(MAX_UINT16+1):
        lookup_table[0,i] = np.clip(pow(i / (MAX_UINT16), gamma) * (MAX_UINT16), 0, MAX_UINT16)

    return cv.LUT(image, lookup_table)


## OLD white balancing via torch tensor modules
class WhiteBalance(nn.Module):
    """
    White Balance transform.
    Supports different 'WBAlgorithm's
    """
    def __init__(self, algorithm: WBAlgorithm) -> None:
        super().__init__()
        self.algorithm = algorithm

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        match self.algorithm:
            case WBAlgorithm.WHITE_PATCH:
                # max: reducing the last 2 dimensions (keep channels)
                imageMaxRGB = img.amax((1,2))
                # White Patch
                coeffs = imageMaxRGB / 1
                # TODO : Normalize coeffs
                print('WP coeffs: ', coeffs)
                # Patch application // need to fill last 2 dimensions w/None
                wbImage = img * coeffs[:, None, None]
                return wbImage
            case WBAlgorithm.GREY_WORLD:
                # mean: reducing the last 2 dimensions (keep channels)
                imageMeanRGB = img.mean(dim=(1,2))
                # gray world
                coeffs = 0.5 / imageMeanRGB
                # TODO : Normalize coeffs
                print('GW coeffs: ', coeffs)
                # apply
                wbImage = img * coeffs[:, None, None]
                wbImage = wbImage.clamp(0, 1)
                return wbImage
            case WBAlgorithm.JSON_DATA:

                return torch.Tensor()