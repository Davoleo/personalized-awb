import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from src import Metadata

RAW_EXTS = {
    'sony': '.arw',
    'nikon': 'nef',
    'galaxy': 'dng',
}

@dataclass
class Sample:
    name: str
    png_path: Path
    meta: Metadata
    dir_path: Path
    place_i: int
    raw_path: Path | None

def extract_metadata(global_meta_obj, local_meta) -> Metadata:
    with open(local_meta, 'r') as l_file:
        local_data = json.load(l_file)

    # LSMI
    if global_meta_obj is not None:
        n_illum = global_meta_obj["NumOfLights"]
        illums = []
        for i in range(1, n_illum+1):
            illums.append(global_meta_obj[f"Light{i}"])

        cm1 = np.array(local_data['ColorMatrix1'])
        cm2 = np.array(local_data['ColorMatrix2'])
        fm1 = np.array(local_data['ForwardMatrix1'])
        fm2 = np.array(local_data['ForwardMatrix2'])
    else: # gehler-shi
        n_illum = 1
        illums = [local_data['illuminant_color_raw']]
        cm1 = np.array(local_data['cm1'])
        cm2 = np.array(local_data['cm2'])
        fm1 = np.array(local_data['fm1'])
        fm2 = np.array(local_data['fm2'])

    return Metadata(n_illum, illums, cm1, cm2, fm1, fm2)

def iter_lsmi(datapath: Path, raw_root: Path | None, shuffled = False) -> Iterator[Sample]:
    with open(datapath / f"{datapath.name}_meta.json") as f:
        global_meta = json.load(f)

    places = _sorted_or_shuffled((int(k[5:]) for k in global_meta if k.startswith("Place") and k[5:].isdigit()), shuffled)
    for i in places:
        glob = global_meta[f"Place{i}"]
        filename = f"Place{i}_{'12' if glob["NumOfLights"] == 2 else '123'}"
        png = datapath / f"Place{i}" / f"{filename}.png"
        meta = extract_metadata(glob, str(png).replace(".png", "_meta.json"))
        raw_ext = next((value for key, value in RAW_EXTS.items() if key in str(datapath)), None)
        if raw_ext is None:
            raise ValueError("Could not infer brand")
        raw = Path(raw_root) / f"Place{i}" / f"{filename}{raw_ext}" if raw_root else None
        yield Sample(png.name, png, meta, datapath, i, raw)

def iter_gehler_shi(datapath: Path, shuffled = False) -> Iterator[Sample]:
    for i, png in enumerate(_sorted_or_shuffled(datapath.rglob('*.png'), shuffled)):
        meta = extract_metadata(None, str(png).replace(".png", "_metadata.json"))
        yield Sample(png.name, png, meta, datapath, i, None)

def _sorted_or_shuffled(places, shuffle = False) -> list:
    key_fn = None if not shuffle else lambda _: random.random()
    return sorted(places, key=key_fn)