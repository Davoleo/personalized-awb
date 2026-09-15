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
from src.wbalance import gamma_correction, compute_wb_variants, WBAlgorithm
from src.ciexyz import convert_to_ciexyz
import src.utils as utils

WB_ALGORITHMS = {
    'white_patch': WBAlgorithm.WHITE_PATCH,
    'grey_world': WBAlgorithm.GREY_WORLD,
    'illuminant1': WBAlgorithm.ILLUMINANT1,
    'illuminant2': WBAlgorithm.ILLUMINANT2,
    'illuminant3': WBAlgorithm.ILLUMINANT3,
    'illu_map_mean': WBAlgorithm.ILLU_MAP_MEAN,
    'illu_map_wmean': WBAlgorithm.ILLU_MAP_W_MEAN,
    'local_illu_blend': WBAlgorithm.LOCAL_ILLU_BLEND
}

MAX_UINT16 = 65535
MAX_UINT8 = 255

SONY_N_PLACES = 1317

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

    # build  data/{brand}/{brand}_meta.json.
    brand = datapath.parts[-1]
    global_metapath = datapath / f"{brand}_meta.json"

    with open(global_metapath, 'r') as f:
        global_meta = json.load(f)

    sample_toshow = random.randint(0, SONY_N_PLACES)
    to_show = []

    # TODO Adapt
    # skip everything we don't plan to show in dry runs
    if args.dry_run:
        paths = sample_toshow

    for path_i in range(SONY_N_PLACES):
        n_illu = global_meta[f"Place{path_i}"]['NumOfLights']

        full_img_ending = "12" if n_illu == 2 else "123"
        path = datapath / f"Place{path_i}" / f"Place{path_i}_{full_img_ending}.png"

        image = utils.read_image(path)

        print(f"working on: {path}")

        local_metapath = str(path).replace('.png', '_meta.json')
        meta = extract_metadata(global_meta[f"Place{path_i}"], local_metapath)

        if wb_algorithm == WBAlgorithm.ILLUMINANT3 and meta.n_illums < 3:
            print(f"skipping because n_illums: {meta.n_illums} < 3")
            continue

        final = process_image(image, meta, wb_algorithm, dir_path=datapath, place_i=path_i)

        if path_i == sample_toshow:
            to_show.append(final)

        newpath = Path(f"data/{save_loc}/{os.path.basename(path)}")
        print(newpath)
        cv.imwrite(newpath, final)

    gamma = None if args.srgb else 0.4
    show_samples(to_show, title="white balanced samples", gamma=gamma)

def process_image(image: np.ndarray, meta: Metadata, wb_algorithm: WBAlgorithm, dir_path: Path, place_i: int):
    # convert to RGB format
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Convert to float32 typing
    image = image.astype(np.float32)
    image /= MAX_UINT16

    print(f"running white balance solution: {wb_algorithm.name}")
    variants = compute_wb_variants(wb_algorithm, image, meta, dir_path=dir_path, place_i=place_i)

    # conversion to CIE XYZ should be done if enabled or forced if the algorithm is based on local wb blending
    to_xyz = args.ciexyz or wb_algorithm == WBAlgorithm.LOCAL_ILLU_BLEND

    accum = None
    for variant in variants:
        stage = variant.image
        if to_xyz:
            # if illuminant to compute cct from was not specified fallback to the first one
            ref_idx = variant.cct_illuminant_index if variant.cct_illuminant_index is not None else 0
            print("converting to CIE XYZ color space...")
            # run conversion to camera-independent color space
            stage = convert_to_ciexyz(stage, meta, illuminant_index=ref_idx)
        weight = variant.weight
        if not isinstance(weight, np.ndarray):
            # fallback to full map to 1 [noop]
            weight = np.full(stage.shape[:2], weight, dtype=np.float32)
        term = weight[:,:,None]*stage
        accum = term if accum is None else accum + term

    image = accum

    # convert back to UINT16 and clip any value that goes over max
    if args.srgb:
        print("converting to sRGB...")
        image = ski.color.xyz2rgb(image)
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