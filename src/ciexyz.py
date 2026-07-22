import os
from dataclasses import dataclass
import json
from pathlib import Path

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

    _, cct = approximate_cct(meta)
    
    if (cct is None):
        raise LookupError()

    
    forward_matrix = interpolate_ccm(cct, m1=meta.forward_matrix_1, m2=meta.forward_matrix_2)

    img_xyz = np.empty(image.shape)
    row, col, _ = image.shape
    for y in range(row):
        for x in range(col):
            img_xyz[y][x] = forward_matrix @ image[y][x]

    return img_xyz


def approximate_cct(meta: Metadata):
    xy: ArrayLike = [0.3127, 0.3290]
    i = 0
    while i < 100:
        cct = colour.temperature.xy_to_CCT(xy)
        print(cct)
        color_matrix = interpolate_ccm(cct, meta.color_matrix_1, meta.color_matrix_2)
        color_matrix_inv = np.linalg.inv(color_matrix)
        xyz = color_matrix_inv @ np.transpose(meta.illuminant)
        X, Y, Z = np.asarray(xyz).flatten()
        print("X Y Z: ", X, Y, Z)
        xy_new = [X / (X+Y+Z), Y / (X+Y+Z)]
        if np.allclose(xy, xy_new, atol=1e-6):
            return xyz, cct
        xy = xy_new
        i += 1

    print(f"!!! didn't find cct in {i} iterations, returning None !!!")
    return (None,None)

def extract_metadata(metapath: Path) -> Metadata:
    with open(metapath, 'r') as file:
        data = json.load(file)
    
    illu = np.array(data['illuminant_color_raw'])
    # BGR format
    illu[[0,2]] = illu[[2,0]]
    

    # BGR: Swap color plane rows (first and last rows)
    cm1 = np.matrix(data['cm1'])
    cm1[[0,2], :] = cm1[[2,0], :]
    cm2 = np.matrix(data['cm2'])
    cm2[[0,2], :] = cm2[[2,0], :]

    # BGR: Swap color plane columns (first and last columns)
    fm1 = np.matrix(data['fm1'])
    fm1[:, [0,2]] = fm1[:, [2,0]] 
    fm2 = np.matrix(data['fm2'])
    fm2[:, [0,2]] = fm2[:, [2,0]] 


    return Metadata(illu, cm1, cm2, fm1, fm2)

def interpolate_ccm(cct, m1: ArrayLike, m2: ArrayLike):
    """cct is the interpolator temperature value"""
    num = (1 / cct) - (1 / D65_CCT)
    den = (1 / STANDARD_A_CCT) - (1 / D65_CCT)
    g = num / den   # Ratio of interpolation
    
    # Interpolation of matrices
    CM = g*m1 + (1-g)*m2
    return CM

if __name__ == '__main__':
    image_path = get_project_dir() / 'data' / 'Gehler-Shi' / '8D5U5534_sensorname_Canon1D.png'
    image = cv.imread(image_path, flags=cv.IMREAD_UNCHANGED)
    converted = convert_to_ciexyz(image, os.path.basename(image_path))