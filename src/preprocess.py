import os
import argparse
import json
from pathlib import Path

import numpy as np
import cv2 as cv

# NEF : Nikon
# ARW : Sony
# DNG : Galaxy
EXTENSIONS = ['nef', 'arw', 'dng']

MAX_UINT16 = 65535

class NumpyEncoder(json.JSONEncoder):
    """
    JsonEncoder override to support numpy arrays in serialization
    """
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

def resize(rgb, max_size):
    """
    Function to resize the image so that the short edge length == max_size
    """
    # take size information from the first 2 colunmns
    h, w = rgb.shape[:2]
    # scale is the multiplier needed for the short edge length to be = max_size
    scale = max_size / min(h, w)
    # multiply by scale -> round to integer -> clamp to 1
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    # readiometric data -- inter_area -> no ringing/overshoot
    return cv.resize(rgb, (nw, nh), interpolation=cv.INTER_AREA)

def demosaic(image: np.ndarray):
    """
    Demosaic the input image with a naive algorithm, dividing total size by 2.
    This method assumes the input is a Bayer pattern image:
    R G R G
    G B G B
    R G R G
    G B G B
    """
    # Numpy slicing syntax: [start:end:step] -> omitting end = till the end of the array
    # Red Channel: even rows / even columns
    r = image[0::2, 0::2]
    # Blue Channel: odd rows / odd columns
    b = image[1::2, 1::2]
    # Green Channel: even rows / odd columns and vice versa
    g1 = image[0::2, 1::2]
    g2 = image[1::2, 0::2]
    # mean between green channels
    g = (g1+g2)/2

    demosaiced = np.empty((image.shape[0]//2, image.shape[1]//2, 3))
    np.stack([r, g, b], axis=2, out=demosaiced)
    return demosaiced

def preprocess_metadata(meta):
    """Preprocess metadata so that it's compatible with numpy arrays [parse strings]"""
    meta1 = meta
    meta1["ColorMatrix1"] = np.array(meta1["ColorMatrix1"].split(' '), dtype=float).reshape((3,3))
    meta1["ColorMatrix2"] = np.array(meta1["ColorMatrix2"].split(' '), dtype=float).reshape((3,3))
    meta1["ForwardMatrix1"] = np.array(meta1["ForwardMatrix1"].split(' '), dtype=float).reshape((3,3))
    meta1["ForwardMatrix2"] = np.array(meta1["ForwardMatrix2"].split(' '), dtype=float).reshape((3,3))
    meta1["CameraCalibration1"] = np.array(meta1["CameraCalibration1"].split(' '), dtype=float).reshape((3,3))
    meta1["CameraCalibration2"] = np.array(meta1["CameraCalibration2"].split(' '), dtype=float).reshape((3,3))

    meta1["AnalogBalance"] = np.array(meta1["AnalogBalance"].split(' '), dtype=float)
    meta1["AsShotNeutral"] = np.array(meta1["AsShotNeutral"].split(' '), dtype=float)

    meta1["CFARepeatPatternDim"] = tuple(int(string) for string in meta1["CFARepeatPatternDim"].split(' '))
    meta1["CFAPattern2"] = np.array(meta1["CFAPattern2"].split(' '), dtype=int).reshape(meta1["CFARepeatPatternDim"])
    meta1["CFAPlaneColor"] = np.array(meta1["CFAPlaneColor"].split(' '), dtype=int)

    meta1["BlackLevelRepeatDim"] = tuple(int(string) for string in meta1["BlackLevelRepeatDim"].split(' '))
    meta1["BlackLevel"] = np.array(meta1["BlackLevel"].split(' '), dtype=int).reshape(meta1["BlackLevelRepeatDim"])
    return meta1

def remove_black_white_level(image, meta, rmwb_per_channel):
    """
    Removes black and white level from the image according to JSON metadata
    also clamps the image between 0 and 1 to avoid negative values and overflow
    """
    white_level = meta["WhiteLevel"]
    black_level = meta["BlackLevel"]

    #print("white_level: ", white_level, " black_level: ", black_level[0])
    if not rmwb_per_channel:
        black_level = black_level[0][0]
    else:
        raise "rmwb_per_channel is not implemented yet"

    image = (image - black_level) / (white_level - black_level)

    # sensor value could go below black_level described in metadata so we just clamp to 0 in that case
    # sensor value could go above white_level described in metadata so we just clamp to 1 in that case
    image = np.clip(image, 0, 1)

    return image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--rmwb-per-channel", action="store_true", help="Remove black and white level from the image per color channel [if metadata's black level has different values for each channel]")
    ap.add_argument("--short-edge", type=int, default=720, help="Downsize longer edge to this amount of pixels.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(args.input_dir)
        for f in files if f.lower().endswith('.tiff')
    ]

    for path in paths:
        print("now preprocessing: " + path)
        # load RAW TIFF
        image = cv.imread(path, flags=cv.IMREAD_UNCHANGED)

        # load METADATA JSON
        meta_path = path.replace('.tiff', '_meta.json')
        with (open(meta_path, 'r')) as file:
            meta = json.load(file)[0]

        print("preprocessing metadata...")
        meta = preprocess_metadata(meta)

        print("demosaicing image...")
        image = demosaic(image)

        print("image max: ", np.max(image), " | image min: ", np.min(image))

        print("removing saturation level...")
        image = remove_black_white_level(image, meta, args.rmwb_per_channel)

        print("image max: ", np.max(image), " | image min: ", np.min(image))

        # resize to 720 on the short edge [keeping aspect ratio]
        print("resize to ", args.short_edge, " ...")
        image = resize(image, args.short_edge)

        # convert to uint16 and clamp 0:1 to avoid overflow
        image = np.clip(image * MAX_UINT16, 0, MAX_UINT16).astype(np.uint16)
        # convert to BGR to be compatible with imwrite
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        newpath = Path(path.replace(args.input_dir, args.output_dir).replace('.tiff', '.png'))
        os.makedirs(newpath.parent, exist_ok=True)
        print("saving to new path: ", newpath)
        cv.imwrite(filename=newpath, img=image)

        print("rewriting metadata...")
        with open(str(newpath).replace('.png', '_meta.json'), 'w') as f:
            json.dump(meta, f, cls=NumpyEncoder)

# def decode(path, demosaic_alg):
#     """Decodes the RAW file format in the specified path"""
#     with rawpy.imread(path) as raw:
#         rgb = raw.postprocess(
#             demosaic_algorithm = demosaic_alg,
#             output_color = rawpy.ColorSpace.raw,    # keep camera-native color space
#             use_camera_wb = False,                  # do not use default as-shot WB values
#             use_auto_wb = False,                    # do not automatically calculate WB coeffs
#             user_wb = [1.0, 1.0, 1.0, 1.0],         # ensure that rawpy is not using daylight white balance (happens if all wb parameters are False and None)
#             gamma = (1.0, 1.0),                     # linear gamma curve
#             no_auto_bright = True,
#             output_bps = 16,                        # same as original
#             fbdd_noise_reduction = rawpy.FBDDNoiseReductionMode.Off
#         )
#         calibration = {
#             "black_level_per_channel": list(raw.black_level_per_channel),
#             "white_level": raw.white_level,
#             "camera_whitebalance": list(raw.camera_whitebalance),
#             "daylight_whitebalance": list(raw.daylight_whitebalance),
#             "color_matrix": raw.color_matrix.tolist(),
#             "rgb_xyz_matrix": raw.rgb_xyz_matrix.tolist(),
#             "color_desc": raw.color_desc.decode() if isinstance(raw.color_desc, bytes) else raw.color_desc,
#             "raw_pattern": raw.raw_pattern.tolist() if raw.raw_pattern is not None else None,
#         }
#         return rgb, calibration

if __name__ == "__main__":
    main()