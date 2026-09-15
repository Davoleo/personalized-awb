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


def compute_wb_variants(algorithm: WBAlgorithm, img: np.ndarray, meta: Metadata, dir_path = None, place_i=None) -> list[WBVariant]:
    match algorithm:
        case WBAlgorithm.WHITE_PATCH | WBAlgorithm.GREY_WORLD:
            coeffs = _global_coeffs(algorithm, img)
            return [WBVariant(img * coeffs, 1.0, None)]
        case WBAlgorithm.ILLUMINANT1 | WBAlgorithm.ILLUMINANT2 | WBAlgorithm.ILLUMINANT3:
            idx = algorithm.value - 1
            coeffs = _illuminant_coeffs(meta, idx)
            return [WBVariant(img * coeffs, 1.0, idx)]
        case WBAlgorithm.ILLU_MAP_MEAN | WBAlgorithm.ILLU_MAP_W_MEAN:
            illu = illu_map(img, dir_path, place_i, meta)
            coeffs = _map_coeffs(algorithm, illu)
            # ? illuminant used in xyz conversion could be computed based on the weights instead of being hardcoded to 0
            return [WBVariant(img * coeffs, 1.0, 0)]
        case WBAlgorithm.LOCAL_ILLU_BLEND:
            alphas = compute_alpha_map(img, dir_path, place_i, meta)
            variants = []
            for i in range(meta.n_illums):
                coeffs = _illuminant_coeffs(meta, i)
                variants.append(WBVariant(img * coeffs, alphas[i], i))
            return variants
        case _:
            raise NotImplementedError(f"unhandled White Balancing Algorithm: {algorithm}")

def _global_coeffs(algorithm: WBAlgorithm, img: np.ndarray) -> np.ndarray:
    if algorithm == WBAlgorithm.WHITE_PATCH:
        # max: reducing the first 2 dimensions (keep channels, as per openCV shape)
        image_max = np.amax(img, (0,1))
        #print("image maxes: ", image_max)
        # L2 Norm to normalize illuminant vector
        image_max /= np.linalg.norm(image_max)
        #print("norm image maxes: ", image_max)
        # White Patch coeffs
        return 1.0 / image_max
    elif algorithm == WBAlgorithm.GREY_WORLD:
        # mean: reducing the first 2 dimensions (keep channels, as per openCV shape)
        image_mean = np.mean(img, axis=(0,1))
        #print("image means: ", image_mean)
        image_mean /= np.linalg.norm(image_mean)
        #print("image means norm: ", image_mean)
        # Grey World coeffs
        return 0.5 / image_mean
    else:
        raise ValueError(f"{algorithm} is not a global-coefficient algorithm")

def _illuminant_coeffs(meta: Metadata, idx: int) -> np.ndarray:
    if idx >= meta.n_illums:
        raise ValueError(f"illuminant index {idx} doesn't exist on this image (n_illuminants={meta.n_illums})")
    illu = np.asarray(meta.illuminants[idx], dtype=np.float64)
    illu /= np.linalg.norm(illu)
    return np.ones(3) / illu

def _map_coeffs(algorithm: WBAlgorithm, illuminant_map: np.ndarray) -> np.ndarray:
    if algorithm == WBAlgorithm.ILLU_MAP_MEAN:
        # only keep third dimension [color channels]
        illu = illuminant_map.mean(axis=(0,1))
    elif algorithm == WBAlgorithm.ILLU_MAP_W_MEAN:
        weights = illuminant_map[:,:,1]
        illu = np.sum(illuminant_map * weights[:,:,None],axis=(1, 0)) / np.sum(weights)
    else:
        raise ValueError(f"{algorithm} is not a map-coefficient algorithm")

    illu /= np.linalg.norm(illu)
    return np.ones(3) / illu


def gamma_correction(image, gamma: float):
    lookup_table = np.empty((1, MAX_UINT16+1), np.uint16)
    for i in range(MAX_UINT16+1):
        lookup_table[0,i] = np.clip(pow(i / MAX_UINT16, gamma) * MAX_UINT16, 0, MAX_UINT16)

    return cv.LUT(image, lookup_table)

def compute_alpha_map(full_img: np.ndarray, dir_path: Path, place_i: int, meta: Metadata):
    # convert from float to uint16
    full_img = (full_img*MAX_UINT16).astype(np.uint16)
    img_l1: np.ndarray = utils.read_image(dir_path / f"Place{place_i}" / f"Place{place_i}_1.png")

    if meta.n_illums == 2:
        diff = full_img.astype(np.int32) - img_l1.astype(np.int32)
        img_l2 = np.clip(diff, 0, MAX_UINT16).astype(np.uint16)
        # use green channel as estimate for the scaling term [light intensity]
        g_l1 = img_l1[:, :, 1]
        g_l2 = img_l2[:, :, 1]
        # alpha = coefficient map
        denom = g_l1 + g_l2
        denom = np.maximum(denom, 1e-8)
        alpha = g_l1 / denom
        alpha = np.clip(alpha, 0, 1)
        return [alpha, 1-alpha]

    elif meta.n_illums == 3:
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

    if meta.n_illums == 2:
        alpha = compute_alpha_map(full_img, dir_path, place_i, meta)
        # linear combination of coefficient map and the 2 illuminants.
        l12 = alpha[0][:,:,None] * l1 + alpha[1][:,:,None] * l2
        l12 = l12[..., [2,1,0]]
        return l12

    elif meta.n_illums == 3:
        alpha = compute_alpha_map(full_img, dir_path, place_i, meta)
        # normalize l3
        l3 = meta.illuminants[2]
        l3 /= np.linalg.norm(l3)

        l123 = alpha[0][:,:,None] * l1 + alpha[1][:,:,None] * l2 + alpha[2][:,:,None] * l3
        # convert BGR -> RGB
        l123 = l123[..., [2, 1, 0]]
        return l123
    else:
        raise ValueError("metadata n_illums has has an invalid value")