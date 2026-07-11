from enum import Enum

import torch
from torch import nn
import numpy as np;
import cv2 as cv;

MAX_UINT16 = 65535

class WBAlgorithm(Enum):
    WHITE_PATCH = 1
    GREY_WORLD = 2

def white_balance(algorithm: WBAlgorithm, img) -> cv.typing.MatLike:
    match algorithm:
        case WBAlgorithm.WHITE_PATCH:
            img_f = img.astype(np.float32) / MAX_UINT16
            # max: reducing the first 2 dimensions (keep channels, as per openCV shape)
            imageMax = np.amax(img_f, (0,1))
            print("image maxes: ", imageMax)
            # White Patch
            coeffs = 1.0 / imageMax
            # normalize coeffs : subtract minimum value to be 0 and rescale interval to be between 0 and 1
            print("pre-norm coeffs:", coeffs)
            # Geometric mean to preserve luminance
            coeffs /= np.prod(coeffs)**(1/3)
            print('WP coeffs: ', coeffs)
            # Patch application // need to fill last 2 dimensions w/None
            print(coeffs[None, None, :])
            wbImage = img_f * coeffs
            # cv.imshow("debug", wbImage)
            # cv.waitKey(0)
            # exit(0)
            return np.clip(wbImage * MAX_UINT16, 0, MAX_UINT16).astype(np.uint16)
        case WBAlgorithm.GREY_WORLD:
            raise NotImplementedError()


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
            # TODO : Add white balance from json data illuminant.