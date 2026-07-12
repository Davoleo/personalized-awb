from numpy import ndarray
import typing

import os
import random

import torch
import cv2 as cv
import numpy as np

from src import get_project_dir
from .transforms import gamma_correction, white_balance, WBAlgorithm

def get_device() -> str:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'  # type: ignore
    print(f"Accelerator: {device}")
    return device

def show_samples(images: list[str] | list[np.ndarray], title: str = "images", cols: int = 4):
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
    full = gamma_correction(full, 0.4)
    cv.imshow(title, full)
    cv.waitKey(0)


def enhance(datapath, save_loc):
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
        image = cv.imread(path, flags=cv.IMREAD_UNCHANGED)
        if image is None: 
            continue
        wbalanced = white_balance(WBAlgorithm.JSON_DATA, image, os.path.basename(path))
        if (path in sample_toshow):
            to_show.append(wbalanced)
        newpath = os.path.join(save_loc, os.path.basename(path))
        print(newpath)
        cv.imwrite(newpath, wbalanced)

    show_samples(to_show, title="white balanced samples")


def main():   
    get_device()
    dir = get_project_dir()
    enhance(datapath=dir.joinpath('data', 'Gehler-Shi'), save_loc=dir.joinpath('data', 'json_data'))


def hello_cv():
    print("OpenCV: ", cv.__version__)
    img = np.zeros((120, 400, 3), dtype=np.uint8)
    cv.putText(img, "OpenCV OK", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    cv.imshow("hello", img)
    cv.waitKey(0)

if __name__ == "__main__":
    main()