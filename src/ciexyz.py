import os
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

from numpy.typing import ArrayLike
import numpy as np
import cv2 as cv
import colour

from src import get_project_dir

STANDARD_A_CCT = 2856
D65_CCT = 6500

@dataclass
class Metadata:
    """Gehler-shi metadata"""
    illuminant: np.typing.ArrayLike
    color_matrix_1: np.typing.ArrayLike
    color_matrix_2: np.typing.ArrayLike
    forward_matrix_1: np.typing.ArrayLike
    forward_matrix_2: np.typing.ArrayLike


def convert_to_ciexyz(image, filename: str):
    """Converts image colors to be device-independent"""
    metadata_path = get_project_dir() / "data" / 'Gehler-Shi' / Path(filename.strip(".png") + "_metadata.json")
    meta = extract_metadata(metadata_path)

    warnings.filterwarnings("error")
    cct = approximate_cct(meta)
    warnings.filterwarnings("default")
    
    forward_matrix = interpolate_ccm(cct, m1=meta.forward_matrix_1, m2=meta.forward_matrix_2)
    row, col, _ = image.shape

    # swap color channel flatten all dimensions except for colors
    image = image.transpose(2,0,1).reshape(3, row*col)
    image = forward_matrix @ image
    # recompose image
    image = image.reshape(3, row, col).transpose(1,2,0)

    return image


def approximate_cct(meta: Metadata):
    xy: ArrayLike = [0.3127, 0.3290]
    cct_white = 6508
    i = 0

    while i < 1000:
        # convert to UV first to use Robertson's algorithm which is more robust than the default xy_to_CCT of the colour-science lib
        try:
            uv = colour.models.xy_to_UCS_uv(xy)
            cct, _ = colour.temperature.uv_to_CCT_Robertson1968(uv)
        except RuntimeWarning:
            print("WARN: CCT approximation failed - returning base cct value")
            return cct_white

        print(cct)
        color_matrix = interpolate_ccm(cct, meta.color_matrix_1, meta.color_matrix_2)
        color_matrix_inv = np.linalg.inv(color_matrix)
        xyz = color_matrix_inv @ np.transpose(meta.illuminant)
        X, Y, Z = np.asarray(xyz).flatten()
        print("X Y Z: ", X, Y, Z)
        xy_new = [X / (X+Y+Z), Y / (X+Y+Z)]
        if np.allclose(xy, xy_new, atol=1e-6):
            return cct
        xy = xy_new
        i += 1

    print("WARN: CCT approximation did not converge -> returning base cct value")
    return cct_white

def extract_metadata(metapath: Path) -> Metadata:
    with open(metapath, 'r') as file:
        data = json.load(file)
    
    illu = np.array(data['illuminant_color_raw'])
    
    cm1 = np.array(data['cm1'])
    cm2 = np.array(data['cm2'])
    fm1 = np.array(data['fm1'])
    fm2 = np.array(data['fm2'])

    return Metadata(illu, cm1, cm2, fm1, fm2)

def interpolate_ccm(cct, m1: ArrayLike, m2: ArrayLike) -> np.ndarray:
    """cct is the interpolator temperature value"""
    num = (1 / cct) - (1 / D65_CCT)
    den = (1 / STANDARD_A_CCT) - (1 / D65_CCT)
    g = num / den   # Ratio of interpolation
    
    # Interpolation of matrices
    CM = g*m1 + (1-g)*m2
    return CM

if __name__ == '__main__':
    image_path = get_project_dir() / 'data' / 'Gehler-Shi' / 'IMG_0596_sensorname_Canon5D.png'
    image = cv.imread(image_path, flags=cv.IMREAD_UNCHANGED)
    converted = convert_to_ciexyz(image, os.path.basename(image_path))