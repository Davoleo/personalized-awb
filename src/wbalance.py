from enum import Enum
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy import ndarray

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

def white_balance(algorithm: WBAlgorithm, img: ndarray, meta: Metadata, illuminant_map: np.ndarray) -> cv.typing.MatLike:
    coeffs: ndarray

    match algorithm:
        case WBAlgorithm.WHITE_PATCH:
            # max: reducing the first 2 dimensions (keep channels, as per openCV shape)
            image_max = np.amax(img, (0,1))
            #print("image maxes: ", image_max)
            # L2 Norm to normalize illuminant vector
            image_max /= np.linalg.norm(image_max)
            #print("norm image maxes: ", image_max)
            # White Patch coeffs
            coeffs = 1.0 / image_max
        case WBAlgorithm.GREY_WORLD:
            # mean: reducing the first 2 dimensions (keep channels, as per openCV shape)
            image_mean = np.mean(img, axis=(0,1))
            #print("image means: ", image_mean)
            image_mean /= np.linalg.norm(image_mean)
            #print("image means norm: ", image_mean)
            # Grey World coeffs
            coeffs = 0.5 / image_mean
        case WBAlgorithm.ILLUMINANT1 | WBAlgorithm.ILLUMINANT2 | WBAlgorithm.ILLUMINANT3:
            alg = algorithm.value-1
            if alg >= meta.n_illums:
                raise "error: can't w_balance on illuminant that doesn't exist on the image!"

            illu = meta.illuminants[alg]
            illu /= np.linalg.norm(illu)
            coeffs = np.ones(3) / illu
        case WBAlgorithm.ILLU_MAP_MEAN:
            # only keep third dimension [color channels]
            illu = illuminant_map.mean(axis=(0,1))
            illu /= np.linalg.norm(illu)
            coeffs = np.ones(3) / illu
            pass
        case WBAlgorithm.ILLU_MAP_W_MEAN:
            # weighted mean based on the green channel
            weights = illuminant_map[:,:,1]
            illu = np.sum(
                illuminant_map * weights[:,:,None],
                axis=(1, 0)
            ) / np.sum(weights)
            illu /= np.linalg.norm(illu)
            coeffs = np.ones(3) / illu

    # Patch application
    wb_image = img * coeffs
    return wb_image


def gamma_correction(image, gamma: float):
    lookup_table = np.empty((1, MAX_UINT16+1), np.uint16)
    for i in range(MAX_UINT16+1):
        lookup_table[0,i] = np.clip(pow(i / MAX_UINT16, gamma) * MAX_UINT16, 0, MAX_UINT16)

    return cv.LUT(image, lookup_table)


def illu_map(full_img: np.ndarray, dir_path: Path, place_i: int, meta: Metadata):

    #full_img_str = "12" if meta.n_illums == 2 else "123"

    img_l1: np.ndarray = utils.read_image(dir_path / f"Place{place_i}" / f"Place{place_i}_1.png")
    l1 = meta.illuminants[0]
    l1 /= np.linalg.norm(l1)
    l2 = meta.illuminants[1]
    l2 /= np.linalg.norm(l2)

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
        # linear combination of coefficient map and the 2 illuminants.
        l12 = alpha[:,:,None] * l1 + (1-alpha)[:,:,None] * l2
        l12 = l12[..., [2,1,0]]
        return l12

    elif meta.n_illums == 3:
        # normalize l3
        l3 = meta.illuminants[2]
        l3 /= np.linalg.norm(l3)

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

        l123 = alpha1[:,:,None] * l1 + alpha2[:,:,None] * l2 + alpha3[:,:,None] + l3
        # convert BGR -> RGB
        l123 = l123[..., [2, 1, 0]]
        return l123
    else:
        raise ValueError("metadata n_illums has has an invalid value")