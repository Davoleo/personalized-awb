import json
from pathlib import Path

from numpy import ndarray
import typing

import argparse
import os
import random

import torch
import cv2 as cv
import numpy as np
import skimage as ski

from src import get_project_dir, Metadata
from src.wbalance import gamma_correction, white_balance, WBAlgorithm
from src.ciexyz import convert_to_ciexyz

WB_ALGORITHMS = {
    'white_patch': WBAlgorithm.WHITE_PATCH,
    'grey_world': WBAlgorithm.GREY_WORLD,
    'illuminant1': WBAlgorithm.ILLUMINANT1,
    'illuminant2': WBAlgorithm.ILLUMINANT2,
    'illuminant3': WBAlgorithm.ILLUMINANT3,
}

MAX_UINT16 = 65535
MAX_UINT8 = 255

parser = argparse.ArgumentParser()
parser.add_argument('--input', default="Gehler-Shi", help="The dataset path (in the data folder) to use as input data to enhance")
parser.add_argument('--output', default=None, help="The output subfolder (in data/)")
parser.add_argument('--wbalgorithm', required=True, choices=list(WB_ALGORITHMS), help="The White Balancing Algorithm to be used to enhance the dataset")
parser.add_argument('--ciexyz', action='store_true', default=False, help="Whether the software should perform CIE XYZ conversion or not (after WB)")
parser.add_argument('--srgb', action='store_true', default=False, help="Whether the conversion to sRGB 8bit channels should be performed at the end")
parser.add_argument('--dry-run', action='store_true', default=False, help="Whether the output should be saved to --output or just perform the operation and show the samples")
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

def extract_metadata(global_meta_obj, local_meta) -> Metadata:
    with open(local_meta, 'r') as l_file:
        local_data = json.load(l_file)

    n_illum = global_meta_obj["NumOfLights"]
    illums = []
    for i in range(1, n_illum+1):
        illums.append(global_meta_obj[f"Light{i}"])

    cm1 = np.array(local_data['ColorMatrix1'])
    cm2 = np.array(local_data['ColorMatrix2'])
    fm1 = np.array(local_data['ForwardMatrix1'])
    fm2 = np.array(local_data['ForwardMatrix2'])

    return Metadata(n_illum, illums, cm1, cm2, fm1, fm2)


def pipeline(datapath: Path, save_loc: str, wb_algorithm: WBAlgorithm):
    """
    Enhance image files in @datapath with @transforms and write them to the @save_loc folders
    """
    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(datapath)
        for f in files if f.lower().endswith('.png')
    ]

    random.shuffle(paths)

    if len(paths) == 0:
        print(f"Data folder is empty or datapath: {datapath} is wrong")
        exit()

    sample_toshow = random.sample(paths, 16)
    to_show = []

    # skip everything we don't plan to show in dry runs
    if args.dry_run:
        paths = sample_toshow

    # build  data/{brand}/{brand}_meta.json.
    brand = datapath.parts[-1]
    global_metapath = datapath / f"{brand}_meta.json"

    with open(global_metapath, 'r') as f:
        global_meta = json.load(f)

    for path in paths:
        #? cv.IMREAD_COLOR_RGB doesn't seem to work to convert directly on read https://docs.opencv.org/4.13.0/d8/d6a/group__imgcodecs__flags.html#gga61d9b0126a3e57d9277ac48327799c80a18afb429fb71972a327314b2f0d8d56a
        image = cv.imread(path, flags=cv.IMREAD_UNCHANGED)
        if image is None: 
            continue

        print(f"working on: {path}")

        filename = os.path.basename(path)

        local_metapath = path.replace('.png', '_meta.json')
        meta = extract_metadata(global_meta[filename.split('_')[0]], local_metapath)

        if wb_algorithm == WBAlgorithm.ILLUMINANT3 and meta.n_illums < 3:
            print(f"skipping because n_illums: {meta.n_illums} < 3")
            continue

        final = process_image(image, meta, wb_algorithm)

        if path in sample_toshow:
            to_show.append(final)

        newpath = path.replace(brand, save_loc)
        print(newpath)
        os.makedirs(newpath.replace(newpath.split(os.path.sep)[-1], ''), exist_ok=True)
        cv.imwrite(newpath, final)

    gamma = None if args.srgb else 0.4
    show_samples(to_show, title="white balanced samples", gamma=gamma)

def process_image(image: np.ndarray, meta: Metadata, wb_algorithm: WBAlgorithm):
    # convert to RGB format
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Convert to float32 typing
    image = image.astype(np.float32)
    image /= MAX_UINT16

    print(f"running white balance solution: {wb_algorithm.name}")
    image = white_balance(wb_algorithm, image, meta)

    # run conversion to camera-independent color space
    if args.ciexyz:
        print("converting to CIE XYZ color space...")
        image = convert_to_ciexyz(image, meta)

    if args.srgb:
        print("converting to sRGB...")
        image = ski.color.xyz2rgb(image)

    # convert back to UINT16 and clip any value that goes over max
    if args.srgb:
        final = np.clip(image * MAX_UINT8, 0, MAX_UINT8).astype(np.uint8)
    else:
        final = np.clip(image * MAX_UINT16, 0, MAX_UINT16).astype(np.uint16)

    final = cv.cvtColor(final, cv.COLOR_RGB2BGR)
    return final

def main():   
    get_device()
    proj_dir = get_project_dir()
    # by default numpy prints with 8 precision
    #np.set_printoptions(precision=17)
    algorithm = WB_ALGORITHMS[args.wbalgorithm]

    if algorithm is None:
        raise ValueError(f"--algorithm should be set to one of the following values: {str.join('|', WB_ALGORITHMS.keys())}")
    
    pipeline(datapath=proj_dir.joinpath('data', args.input), save_loc=args.output if args.output is not None else args.wbalgorithm, wb_algorithm=algorithm)


def hello_cv():
    print("OpenCV: ", cv.__version__)
    img = np.zeros((120, 400, 3), dtype=np.uint8)
    cv.putText(img, "OpenCV OK", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    cv.imshow("hello", img)
    cv.waitKey(0)

if __name__ == "__main__":
    main()