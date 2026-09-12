from dataclasses import dataclass
from pathlib import Path

import numpy as np


def get_project_dir():
    return Path(__file__).resolve().parent.parent

@dataclass
class Metadata:
    """dataset metadata"""
    n_illums: int
    illuminants: list[np.typing.ArrayLike]
    color_matrix_1: np.typing.ArrayLike
    color_matrix_2: np.typing.ArrayLike
    forward_matrix_1: np.typing.ArrayLike
    forward_matrix_2: np.typing.ArrayLike