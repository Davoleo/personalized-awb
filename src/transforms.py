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
    
    # Convert to float32 typing
    img_f = img.astype(np.float32)
    img_f /= MAX_UINT16
    wbImage: ndarray
    match algorithm:
        case WBAlgorithm.WHITE_PATCH:
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
            wbImage = img_f * coeffs
        case WBAlgorithm.GREY_WORLD:
            # mean: reducing the first 2 dimensions (keep channels, as per openCV shape)
            imageMean = np.mean(img_f, axis=(0,1))
            print("image means: ", imageMean)
            # Grey World
            coeffs = 0.5 / imageMean
            print('pre-norm coeffs: ', coeffs)
            coeffs /= np.prod(coeffs)**(1/3)
            print('GW coeffs: ', coeffs)
            # Apply patch
            wbImage = img_f * coeffs
        case WBAlgorithm.JSON_DATA:
            print(filename)
            # take illuminant from json data of the specific image and use it to whitebalance the image
            metadata_path = get_project_dir() / "data" / 'Gehler-Shi' / Path(filename.strip(".png") + "_metadata.json")
            with open(metadata_path, 'r') as file:
                data = json.load(file)
            
            illu = np.array(data['illuminant_color_raw'])

            #([1,1,1] - [0.2442, 1, 0.52135872])+1 (to be multiplied as coeffs with the pixel intensities)
            #([1,1,1] / [0.2442, 1, 0.52135872])

            coeffs = (np.ones((1,3)) - illu)+1
            #coeffs = (np.ones((1,3)) / illu)

            # Geometric mean to preserve luminance
            coeffs /= np.prod(coeffs)**(1/3)
            
            print("JSON coeffs ", coeffs)
            wbImage = img_f * coeffs


    return np.clip(wbImage * MAX_UINT16, 0, MAX_UINT16).astype(np.uint16)


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