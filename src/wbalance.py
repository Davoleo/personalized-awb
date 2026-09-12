from pylab import ndarray
from enum import Enum

import torch
from torch import nn
import numpy as np
import cv2 as cv

from src import Metadata

MAX_UINT16 = 65535

class WBAlgorithm(Enum):
    ILLUMINANT1 = 1
    ILLUMINANT2 = 2
    ILLUMINANT3 = 3
    WHITE_PATCH = 4
    GREY_WORLD = 5

def white_balance(algorithm: WBAlgorithm, img: ndarray, meta: Metadata) -> cv.typing.MatLike:
    coeffs: ndarray

    match algorithm:
        case WBAlgorithm.WHITE_PATCH:
            # max: reducing the first 2 dimensions (keep channels, as per openCV shape)
            image_max = np.amax(img, (0,1))
            #print("image maxes: ", image_max)
            # L2 Norm to normalize illuminant vector
            image_max /= np.linalg.norm(image_max)
            #print("norm image maxes: ", image_max)
            # White Patch coeffs
            coeffs = 1.0 / image_max
        case WBAlgorithm.GREY_WORLD:
            # mean: reducing the first 2 dimensions (keep channels, as per openCV shape)
            image_mean = np.mean(img, axis=(0,1))
            #print("image means: ", image_mean)
            image_mean /= np.linalg.norm(image_mean)
            #print("image means norm: ", image_mean)
            # Grey World coeffs
            coeffs = 0.5 / image_mean
        case WBAlgorithm.ILLUMINANT1 | WBAlgorithm.ILLUMINANT2 | WBAlgorithm.ILLUMINANT3:
            alg = algorithm.value-1
            if alg >= meta.n_illums:
                raise "error: can't w_balance on illuminant that doesn't exist on the image!"

            ill = meta.illuminants[alg]
            ill /= np.linalg.norm(ill)
            coeffs = np.ones(3) / ill

    # Patch application
    wb_image = img * coeffs
    return wb_image


def gamma_correction(image, gamma: float):
    lookup_table = np.empty((1, MAX_UINT16+1), np.uint16)
    for i in range(MAX_UINT16+1):
        lookup_table[0,i] = np.clip(pow(i / MAX_UINT16, gamma) * MAX_UINT16, 0, MAX_UINT16)

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