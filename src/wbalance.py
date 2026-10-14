from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike

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
    LOCAL_ILLU = 8

@dataclass(frozen=True)
class IlluminantEstimate:
    colors: np.ndarray                  # (K,3) K RGB values -> G always 1
    alpha: np.ndarray | None = None     # (H,W,K) float32 - None => global illuminant estimation
    valid: np.ndarray | None = None     # (H,W) bool dtype : /0 => valid = False

    @property
    def is_local(self) -> bool:
        return self.alpha is not None

    def mixed(self) -> np.ndarray:
        """(3) for global estimates, (H,W,3) map for local wb estimate"""
        if not self.is_local:
            return self.colors[0]
        if self.alpha is None or self.valid is None:
            raise ValueError("alpha and valid maps must not be null for local illuminant estimates")

        h,w = self.alpha.shape[:2]
        imap = np.zeros((h,w,3), dtype=np.float32)
        # alpha coefficient map -> illuminant map
        for k, color in enumerate(self.colors):
            imap += self.alpha[..., k:k + 1] * color  # depth K => 3 | with K being the number of alphas
        return imap

    def __post_init__(self):
        if self.alpha is not None and (self.alpha.ndim != 3 or self.alpha.shape[2] != len(self.colors)):
            raise ValueError(f"alpha must be (H,W, K={len(self.colors)}), got {self.alpha.shape}")

def gains(illuminant: np.ndarray) -> np.ndarray:
    """Per-channel gains to neutralise the illuminant, works both for global and local illuminants"""
    return illuminant[..., 1:2] / illuminant

def _norm(c: ArrayLike) -> np.ndarray:
    """Green-based normalization"""
    c = np.asarray(c, dtype=np.float64)
    return c / c[1]

def estimate_illuminant(algorithm: WBAlgorithm, img: np.ndarray, meta: Metadata, dir_path, place_i) -> IlluminantEstimate:
    """img: float32 RGB image not white-balanced"""

    def normalized(c):
        """returns normalized illuminant estimate array joined on a new axis"""
        return IlluminantEstimate(np.stack([_norm(c)]))

    match algorithm:
        case WBAlgorithm.DISABLED:
            return IlluminantEstimate(np.ones((1,3)))
        case WBAlgorithm.WHITE_PATCH:
            return normalized(img.max(axis=(0,1)))
        case WBAlgorithm.GREY_WORLD:
            return normalized(img.mean(axis=(0,1)))
        case WBAlgorithm.ILLUMINANT1 | WBAlgorithm.ILLUMINANT2 | WBAlgorithm.ILLUMINANT3:
            idx = algorithm.value - 1
            if idx >= meta.n_illums:
                raise ValueError(f"illuminant{idx+1} doesn't exist (n_illums={meta.n_illums})")
            return normalized(meta.illuminants[idx])
        case WBAlgorithm.ILLU_MAP_MEAN | WBAlgorithm.ILLU_MAP_W_MEAN:
            map = _local_estimate(meta, dir_path, place_i).mixed()
            if algorithm == WBAlgorithm.ILLU_MAP_MEAN:
                return normalized(map.mean(axis=(0,1)))
            # weight by pixel brightness
            # map reshapin': (H,W,3) -> (N,3)
            # weight: image green channels flattened to (N) shape [ravel]
            return normalized(np.average(map.reshape(-1, 3), axis=0, weights=img[..., 1].ravel() + 1e-6))
        case WBAlgorithm.LOCAL_ILLU:
            return _local_estimate(meta, dir_path, place_i)
        case _:
            raise NotImplementedError(f"unhandled White Balancing Algorithm: {algorithm}")

def _local_estimate(meta: Metadata, dir_path: Path, place_i: int) -> IlluminantEstimate:
    colors = np.stack([_norm(L) for L in meta.illuminants[:meta.n_illums]])
    alpha, valid = compute_alpha_map(dir_path, place_i, meta.n_illums)
    return IlluminantEstimate(colors, alpha, valid)

def gamma_correction(image, gamma: float):
    lookup_table = np.empty((1, MAX_UINT16+1), np.uint16)
    for i in range(MAX_UINT16+1):
        lookup_table[0,i] = np.clip(pow(i / MAX_UINT16, gamma) * MAX_UINT16, 0, MAX_UINT16)

    return cv.LUT(image, lookup_table)

def compute_alpha_map(dir_path: Path, place_i: int, n_illums: int):
    if n_illums not in (2,3):
        raise ValueError(f"unsupported n_illums: {n_illums}")

    def _read_g(suf: str):
        """Read green channel of image with suf as illu suffix"""
        ipath = dir_path / f"Place{place_i}" / f"Place{place_i}_{suf}.png"
        return utils.read_image(ipath)[..., 1].astype(np.float32) / MAX_UINT16

    g1 = _read_g(suf="1")
    illu_variants = [g1, np.clip(_read_g(suf="12") - g1, 0, None)]
    if n_illums == 3:
        illu_variants.append(np.clip(_read_g(suf="13") - g1, 0, None))

    greens = np.stack(illu_variants, axis=-1)
    denom = greens.sum(axis=-1)
    valid = denom > 0
    alpha = greens / np.maximum(denom, 1e-6)[..., None]
    return alpha.astype(np.float32), valid

def required_illums(alg: WBAlgorithm) -> int:
    match alg:
        case WBAlgorithm.ILLUMINANT1 | WBAlgorithm.ILLUMINANT2 | WBAlgorithm.ILLUMINANT3:
            return alg.value
        case WBAlgorithm.ILLU_MAP_MEAN | WBAlgorithm.ILLU_MAP_W_MEAN | WBAlgorithm.LOCAL_ILLU:
            return 2
        case _:
            return 1