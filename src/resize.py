import os
import argparse
from pathlib import Path

import rawpy
import cv2 as cv

DATAPATH = "data/nikon"
SHAPE_25P = (1840,1228)

DEMOSAIC_ALGO = {
    "AHD": rawpy.DemosaicAlgorithm.AHD,
    "DHT": rawpy.DemosaicAlgorithm.DHT,
    "VNG": rawpy.DemosaicAlgorithm.VNG,
    "linear": rawpy.DemosaicAlgorithm.LINEAR,
}

# NEF : Nikon
# ARW : Sony
# DNG : Galaxy
EXTENSIONS = ['nef', 'arw', 'dng']

def resize(rgb, max_size):
    h, w = rgb.shape[:2]
    scale = max_size / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    # readiometric data -- inter_area -> no ringing/overshoot
    return cv.resize(rgb, (nw, nh), interpolation=cv.INTER_AREA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--demosaic", choices=DEMOSAIC_ALGO.keys(), default='DHT', help="Demosaic algorithm to use [default: DHT]")
    ap.add_argument("--max-size", type=int, default=1024, help="Downsize longer edge to this amount of pixels.")
    ap.add_argument("--format", required=True, choices=EXTENSIONS, help="Format of the input dataset")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(DATAPATH)
        for f in files if f.lower().endswith('.' + args.format)
    ]

    demosaic = DEMOSAIC_ALGO[args.demosaic]

    for path in paths:
        rgb, calibration = decode(path, demosaic)
        rgb = resize(rgb, args.max_size)

        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        cv.imshow("salofa", bgr)
        cv.waitKey(0)
        exit(0)


        img = cv.imread(path, cv.IMREAD_UNCHANGED)
        resized = cv.resize(src=img, dsize=SHAPE_25P)
        newpath = Path(path.replace('nikon', 'nikonres').replace('.nef', '.tiff'))
        os.makedirs(newpath.parent, exist_ok=True)
        print(newpath)
        cv.imwrite(filename=newpath, img=resized)

def decode(path, demosaic_alg):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm = demosaic_alg,
            output_color = rawpy.ColorSpace.raw,
            use_camera_wb = False,
            user_wb = [1.0, 1.0, 1.0, 1.0],
            gamma = (1.0, 1.0),
            no_auto_bright = True,
            output_bps = 16,
            fbdd_noise_reduction = rawpy.FBDDNoiseReductionMode.Off,
            median_filter_passes = 0,
            highlight_mode = rawpy.HighlightMode.Clip
        )
        calibration = {
            "black_level_per_channel": list(raw.black_level_per_channel),
            "white_level": raw.white_level,
            "camera_whitebalance": list(raw.camera_whitebalance),
            "daylight_whitebalance": list(raw.daylight_whitebalance),
            "rgb_xyz_matrix": raw.rgb_xyz_matrix,
            "color_desc": raw.color_desc.decode() if isinstance(raw.color_desc, bytes) else raw.color_desc,
            "raw_pattern": raw.raw_pattern.tolist() if raw.raw_pattern is not None else None,
            "sizes": {
                "raw_width": raw.sizes.raw_width,
                "raw_height": raw.sizes.raw_height,
                "width": raw.sizes.width,
                "height": raw.sizes.height,
            }
        }
        return rgb, calibration

if __name__ == "__main__":
    main()