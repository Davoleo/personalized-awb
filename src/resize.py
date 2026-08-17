import os
import argparse
from pathlib import Path

import rawpy
import cv2 as cv

DATAPATH = "data/nikon"
SHAPE_25P = (1840,1228)

DEMOSAIC_ALGO = {
    # most popular in industry to minimize artifacts -  Quality: 3
    # 1. interpolate green channel both vertically and horizontally
    # 2. Convert the results to CIELAB space
    # 3. build a homogeneity map to select the best direction for each pixel
    "AHD": rawpy.DemosaicAlgorithm.AHD,
    # Quality: 11 - good and fast
    # uses DHT struct to compute luminance distances, calculating horizontal, vertical and diagonal directions
    # to decide the interpolation path
    "DHT": rawpy.DemosaicAlgorithm.DHT,
    "VNG": rawpy.DemosaicAlgorithm.VNG,       # Quality : 1 Better than linear at preserving edges
    "linear": rawpy.DemosaicAlgorithm.LINEAR, # Quality : 0 naive interpolation, can produce artifacts
}

# NEF : Nikon
# ARW : Sony
# DNG : Galaxy
EXTENSIONS = ['nef', 'arw', 'dng']

def resize(rgb, max_size):
    # take size information from the first 2 colunmns
    h, w = rgb.shape[:2]
    # scale is the multiplier needed for the long edge length to be = max_size
    scale = max_size / max(h, w)
    # multiply by scale -> round to integer -> clamp to 1
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
        # decode RAW Image
        rgb, calibration = decode(path, demosaic)
        # convert to BGR for OpenCV processing and writing
        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        # resize to 1024 on the long edge [keeping aspect ratio]
        bgr = resize(bgr, args.max_size)

        newpath = Path(path.replace(args.input_dir, args.output_dir).replace('.'+args.format, '.tiff'))
        os.makedirs(newpath.parent, exist_ok=True)
        print(newpath)
        cv.imwrite(filename=newpath, img=bgr)

def decode(path, demosaic_alg):
    """Decodes the RAW file format in the specified path"""
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm = demosaic_alg,
            output_color = rawpy.ColorSpace.raw,    # keep camera-native color space
            use_camera_wb = False,                  # do not use default as-shot WB values
            use_auto_wb = False,                    # do not automatically calculate WB coeffs
            user_wb = [1.0, 1.0, 1.0, 1.0],         # ensure that rawpy is not using daylight white balance (happens if all wb parameters are False and None)
            gamma = (1.0, 1.0),                     # linear gamma curve
            no_auto_bright = True,
            output_bps = 16,                        # same as original
            fbdd_noise_reduction = rawpy.FBDDNoiseReductionMode.Off
        )
        calibration = {
            "black_level_per_channel": list(raw.black_level_per_channel),
            "white_level": raw.white_level,
            "camera_whitebalance": list(raw.camera_whitebalance),
            "daylight_whitebalance": list(raw.daylight_whitebalance),
            "color_matrix": raw.color_matrix,
            "rgb_xyz_matrix": raw.rgb_xyz_matrix,
            "color_desc": raw.color_desc.decode() if isinstance(raw.color_desc, bytes) else raw.color_desc,
            "raw_pattern": raw.raw_pattern.tolist() if raw.raw_pattern is not None else None,
        }
        return rgb, calibration

if __name__ == "__main__":
    main()