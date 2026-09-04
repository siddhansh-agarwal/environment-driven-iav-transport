"""Small geometric operations used by the particle dynamics."""

import numpy as np
from numba import njit


@njit(cache=True)
def normalize_vector(v: np.ndarray) -> None:
    """Normalize a three-dimensional vector in place."""
    norm = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if norm > 1.0e-10:
        v[0] /= norm
        v[1] /= norm
        v[2] /= norm
