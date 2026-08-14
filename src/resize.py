import os
from pathlib import Path

import cv2 as cv
import numpy as np
from cv2.typing import Size

DATAPATH = "data/nikon"
SHAPE_25P = (1840,1228)

def main():
    paths = [
        os.path.join(root, f)
        for root, subdir, files in os.walk(DATAPATH)
        for f in files if f.lower().endswith('.nef')
    ]

    for path in paths:
        img = cv.imread(path, cv.IMREAD_UNCHANGED)
        resized = cv.resize(src=img, dsize=SHAPE_25P)
        newpath = Path(path.replace('nikon', 'nikonres'))
        os.makedirs(newpath.parent, exist_ok=True)
        print(newpath)
        cv.imwrite(filename=newpath, img=resized)

if __name__ == "__main__":
    main()