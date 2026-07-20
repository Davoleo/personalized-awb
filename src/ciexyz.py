import os
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import cv2 as cv

from src import get_project_dir

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

    

    print("illu from metadata object: ", meta.illuminant)    
    pass

def extract_metadata(metapath: Path) -> Metadata:
    with open(metapath, 'r') as file:
        data = json.load(file)
    
    illu = np.array(data['illuminant_color_raw'])
    # BGR format
    illu[[0,2]] = illu[[2,0]]
    

    # BGR: Swap color plane rows (first and last columns)
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

if __name__ == '__main__':
    image_path = get_project_dir() / 'data' / 'Gehler-Shi' / '8D5U5527_sensorname_Canon1D.png'
    image = cv.imread(image_path, flags=cv.IMREAD_UNCHANGED)
    converted = convert_to_ciexyz(image, os.path.basename(image_path))