from numpy import ndarray
import typing

import argparse
import os
import random

import torch
import cv2 as cv
import numpy as np
import skimage as ski

from src import get_project_dir
from src.transforms import gamma_correction, white_balance, WBAlgorithm
from src.ciexyz import convert_to_ciexyz

WB_ALGORITHMS = {
    'white_patch': WBAlgorithm.WHITE_PATCH,
    'gray_world': WBAlgorithm.GREY_WORLD,
    'json_data': WBAlgorithm.JSON_DATA
}

MAX_UINT16 = 65535
MAX_UINT8 = 255

parser = argparse.ArgumentParser()
parser.add_argument('--input', default="Gehler-Shi", help="The dataset path (in the data folder) to use as input data to enhance")
parser.add_argument('--output', default=None, help="The output subfolder (in data/)")
parser.add_argument('--wbalgorithm', required=True, choices=list(WB_ALGORITHMS), help="The White Balancing Algorithm to be used to enhance the dataset")
parser.add_argument('--ciexyz', action='store_true', default=False, help="Whether the software should perform CIE XYZ conversion or not (after WB)")
parser.add_argument('--srgb', action='store_true', default=False, help="Whether the conversion to sRGB 8bit channels should be performed at the end")
args = parser.parse_args()

def get_device() -> str:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'  # type: ignore
    print(f"Accelerator: {device}")
    return device

def show_samples(images: list[str] | list[np.ndarray], title: str = "images", cols: int = 4, gamma = None):
    first = images[0]
    if isinstance(first, str):
        images = typing.cast(list[str], images)
        loaded = [cv.imread(p, cv.IMREAD_UNCHANGED) for p in images]
    else:
        loaded = images

    loaded = typing.cast(list[ndarray], loaded)

    n = len(loaded)
    rows = (n + cols - 1) // cols
    row_panels = [np.concatenate(loaded[r * cols : (r + 1) * cols], axis=1) for r in range(rows)]
    full = np.concatenate(row_panels, axis=0)
    if gamma is not None:
        full = gamma_correction(full, gamma)
    cv.imshow(title, full)
    cv.waitKey(0)


def pipeline(datapath, save_loc, algorithm: WBAlgorithm):
    """
    Enhance image files in @datapath with @transforms and write them to the @save_loc folders
    """
    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(datapath)
        for f in files if f.lower().endswith(('.png'))
    ]

    random.shuffle(paths)

    if len(paths) == 0:
        print(f"Data folder is empty or datapath: {datapath} is wrong")
        exit()

    sample_toshow = random.sample(paths, 16)
    to_show = []

    for path in paths:
        #? cv.IMREAD_COLOR_RGB doesn't seem to work to convert directly on read https://docs.opencv.org/4.13.0/d8/d6a/group__imgcodecs__flags.html#gga61d9b0126a3e57d9277ac48327799c80a18afb429fb71972a327314b2f0d8d56a
        image = cv.imread(path, flags=cv.IMREAD_UNCHANGED)
        if image is None: 
            continue

        # convert to RGB format
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        filename = os.path.basename(path)

        # Convert to float32 typing
        image = image.astype(np.float32)
        image /= MAX_UINT16

        # run white balancing
        image = white_balance(algorithm, image, filename)

        # run conversion to camera-independent color space
        if args.ciexyz:
            image = convert_to_ciexyz(image, filename)

        if args.srgb:
            image = ski.color.xyz2rgb(image)
        
        # convert back to UINT16 and clip any value that goes over max
        if args.srgb:
            final = np.clip(image * MAX_UINT8, 0, MAX_UINT8).astype(np.uint8)
        else:
            final = np.clip(image * MAX_UINT16, 0, MAX_UINT16).astype(np.uint16)

        final = cv.cvtColor(final, cv.COLOR_RGB2BGR)
        if (path in sample_toshow):
            to_show.append(final)
        
        newpath = os.path.join(save_loc, os.path.basename(path))
        print(newpath)

        cv.imwrite(newpath, final)

    show_samples(to_show, title="white balanced samples", )


def main():   
    get_device()
    dir = get_project_dir()
    # by default numpy prints with 8 precision
    #np.set_printoptions(precision=17)
    algorithm = None
    match (args.wbalgorithm):
        case 'white_patch': algorithm = WBAlgorithm.WHITE_PATCH
        case 'gray_world': algorithm = WBAlgorithm.GREY_WORLD
        case 'json_data': algorithm = WBAlgorithm.JSON_DATA
        case _: raise ValueError("--algorithm should be set to one of the following values (white_patch|gray_world|json_data)")
    
    pipeline(datapath=dir.joinpath('data', args.input), save_loc=dir.joinpath('data', args.output if args.output is not None else args.wbalgorithm), algorithm=algorithm)


def hello_cv():
    print("OpenCV: ", cv.__version__)
    img = np.zeros((120, 400, 3), dtype=np.uint8)
    cv.putText(img, "OpenCV OK", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    cv.imshow("hello", img)
    cv.waitKey(0)

if __name__ == "__main__":
    main()