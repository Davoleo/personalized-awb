from argparse import ArgumentError
from typing import Literal
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2 as cv
import numpy as np

import src.utils as utils
from src import Metadata

MAX_UINT16 = 65535
COORDS_RATIO = 0.179

class WBAlgorithm(Enum):
    DISABLED = 0
    ILLUMINANT1 = 1
    ILLUMINANT2 = 2
    ILLUMINANT3 = 3
    WHITE_PATCH = 4
    GREY_WORLD = 5
    ILLU_MAP_MEAN = 6
    ILLU_MAP_W_MEAN = 7
    LOCAL_ILLU_BLEND = 8

@dataclass
class WBVariant:
    """
    image: result of the white_balancing operation for a specific variant
    weight: 1.0 for global algorithms, while per-pixel alpha map for local algos
    cct_illuminant_index: which metadata illuminant to use to approximate the CCT value.
    """
    image: np.ndarray
    weight: np.ndarray | float
    cct_illuminant_index: int | None


def compute_wb_variants(algorithm: WBAlgorithm, img: np.ndarray, meta: Metadata, dir_path, place_i) -> list[WBVariant]:
    match algorithm:
        case WBAlgorithm.DISABLED:
            return [WBVariant(img, 1.0, None)]
        case WBAlgorithm.WHITE_PATCH | WBAlgorithm.GREY_WORLD:
            gain = _global_gain(algorithm, img)
            return [WBVariant(img * gain, 1.0, None)]
        case WBAlgorithm.ILLUMINANT1 | WBAlgorithm.ILLUMINANT2 | WBAlgorithm.ILLUMINANT3:
            idx = algorithm.value - 1
            gain = _illuminant_gain(meta, idx)
            return [WBVariant(img * gain, 1.0, idx)]
        case WBAlgorithm.ILLU_MAP_MEAN | WBAlgorithm.ILLU_MAP_W_MEAN:
            illu = illu_map(img, dir_path, place_i, meta)
            gain = _map_gain(algorithm, illu)
            # ? illuminant used in xyz conversion could be computed based on the weights instead of being hardcoded to 0
            return [WBVariant(img * gain, 1.0, 0)]
        case WBAlgorithm.LOCAL_ILLU_BLEND:
            alphas = compute_alpha_map(img, dir_path, place_i, meta.n_illums)
            variants = []
            for i in range(meta.n_illums):
                gain = _illuminant_gain(meta, i)
                variants.append(WBVariant(img * gain, alphas[i], i))
            return variants
        case _:
            raise NotImplementedError(f"unhandled White Balancing Algorithm: {algorithm}")

def _global_gain(algorithm: WBAlgorithm, img: np.ndarray) -> np.ndarray:
    if algorithm == WBAlgorithm.WHITE_PATCH:
        # max: reducing the first 2 dimensions (keep channels, as per openCV shape)
        image_max = np.amax(img, (0,1))
        #print("image maxes: ", image_max)
        image_max = _illu_norm(image_max, 'euclidean')
        #print("norm image maxes: ", image_max)
        # White Patch gain
        return np.ones(3) / image_max
    elif algorithm == WBAlgorithm.GREY_WORLD:
        # mean: reducing the first 2 dimensions (keep channels, as per openCV shape)
        image_mean = np.mean(img, axis=(0,1))
        #print("image means: ", image_mean)
        image_mean = _illu_norm(image_mean, 'euclidean')
        #print("image means norm: ", image_mean)
        # Grey World gain
        return np.full(3, 0.5) / image_mean
    else:
        raise ValueError(f"{algorithm} is not a global-coefficient algorithm")

def _illuminant_gain(meta: Metadata, idx: int) -> np.ndarray:
    if idx >= meta.n_illums:
        raise ValueError(f"illuminant index {idx} doesn't exist on this image (n_illuminants={meta.n_illums})")
    illu = np.array(meta.illuminants[idx], dtype=np.float64)
    illu = _illu_norm(illu, 'green_based')
    gain = np.ones(3) / illu
    print("gain:", gain, "ratio to G:", gain / gain[1])
    return gain

def _map_gain(algorithm: WBAlgorithm, illuminant_map: np.ndarray) -> np.ndarray:
    if algorithm == WBAlgorithm.ILLU_MAP_MEAN:
        # only keep third dimension [color channels]
        illu = illuminant_map.mean(axis=(0,1))
    elif algorithm == WBAlgorithm.ILLU_MAP_W_MEAN:
        # ? not sure if it's the best weight ...
        weights = illuminant_map[:,:,1]
        illu = np.sum(illuminant_map * weights[:,:,None],axis=(1, 0)) / np.sum(weights)
    else:
        raise ValueError(f"{algorithm} is not a map-coefficient algorithm")

    gain = np.ones(3) / _illu_norm(illu, 'green_based')
    print("gain:", gain, "ratio to G:", gain / gain[1])
    return gain

def _illu_norm(arr: np.ndarray, norm: Literal['euclidean', 'green_based']):
    if len(arr) != 3:
        raise ValueError("Argument passed to _illu_norm is not an illuminant!")
    match norm:
        case 'euclidean':
            return arr / np.linalg.norm(arr)
        case 'green_based':
            return arr / arr[1]

def gamma_correction(image, gamma: float):
    lookup_table = np.empty((1, MAX_UINT16+1), np.uint16)
    for i in range(MAX_UINT16+1):
        lookup_table[0,i] = np.clip(pow(i / MAX_UINT16, gamma) * MAX_UINT16, 0, MAX_UINT16)

    return cv.LUT(image, lookup_table)

def compute_alpha_map(full_img: np.ndarray, dir_path: Path, place_i: int, n_illums: int):
    # convert from float to uint16
    full_img = (full_img*MAX_UINT16).astype(np.uint16)
    # shouldn't be needed since we're only working with the green channel but just to be sure.
    full_img = cv.cvtColor(full_img, cv.COLOR_RGB2BGR)

    img_l1: np.ndarray = utils.read_image(dir_path / f"Place{place_i}" / f"Place{place_i}_1.png")

    if n_illums == 2:
        diff = full_img.astype(np.int32) - img_l1.astype(np.int32)
        img_l2 = np.clip(diff, 0, MAX_UINT16).astype(np.uint16)
        # use green channel as estimate for the scaling term [light intensity] - like paper
        g_l1 = img_l1[:, :, 1]
        g_l2 = img_l2[:, :, 1]
        # alpha = coefficient map
        denom = g_l1 + g_l2
        denom = np.maximum(denom, 1e-8)
        alpha = g_l1 / denom
        alpha = np.clip(alpha, 0, 1)
        return [alpha, 1-alpha]

    elif n_illums == 3:
        img_l12 = utils.read_image(dir_path / f"Place{place_i}" / f"Place{place_i}_12.png")
        diff_l2 = img_l12.astype(np.int32) - img_l1.astype(np.int32)
        img_l2 = np.clip(diff_l2, 0, MAX_UINT16).astype(np.uint16)

        img_l13 = utils.read_image(dir_path / f"Place{place_i}" / f"Place{place_i}_13.png")
        diff_l3 = img_l13.astype(np.int32) - img_l1.astype(np.int32)
        img_l3 = np.clip(diff_l3, 0, MAX_UINT16).astype(np.uint16)

        g_l1 = img_l1[:,:,1]
        g_l2 = img_l2[:,:,1]
        g_l3 = img_l3[:,:,1]

        denom = g_l1 + g_l2 + g_l3

        # avoid /0
        denom = np.maximum(denom, 1e-8)

        alpha1 = g_l1 / denom
        alpha2 = g_l2 / denom
        alpha3 = g_l3 / denom
        return [alpha1, alpha2, alpha3]
    else:
        raise ValueError("metadata n_illums has has an invalid value")


def illu_map(full_img: np.ndarray, dir_path: Path, place_i: int, meta: Metadata):

    l1 = meta.illuminants[0]
    l1 /= np.linalg.norm(l1)
    l2 = meta.illuminants[1]
    l2 /= np.linalg.norm(l2)

    alpha = compute_alpha_map(full_img, dir_path, place_i, meta.n_illums)

    if meta.n_illums == 2:
        # linear combination of coefficient map and the 2 illuminants.
        l12 = alpha[0][:,:,None] * l1 + alpha[1][:,:,None] * l2
        return l12

    elif meta.n_illums == 3:
        # normalize l3
        l3 = meta.illuminants[2]
        l3 /= np.linalg.norm(l3)

        l123 = alpha[0][:,:,None] * l1 + alpha[1][:,:,None] * l2 + alpha[2][:,:,None] * l3
        return l123
    else:
        raise ValueError("metadata n_illums has has an invalid value")