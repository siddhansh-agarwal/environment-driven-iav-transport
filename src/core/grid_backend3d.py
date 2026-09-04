"""Receptor-coordinate representations used by the manuscript simulations."""

from __future__ import annotations


GRID_BACKEND_GRADIENT_SPARSE_COORDS = "gradient_sparse_coords"
GRID_BACKEND_UNIFORM_SPARSE_COORDS = "uniform_sparse_coords"
SUPPORTED_GRID_BACKENDS = (
    GRID_BACKEND_UNIFORM_SPARSE_COORDS,
    GRID_BACKEND_GRADIENT_SPARSE_COORDS,
)


def normalize_grid_backend(name: str) -> str:
    """Validate a receptor-coordinate representation."""
    value = str(name).strip().lower()
    if value not in SUPPORTED_GRID_BACKENDS:
        raise ValueError(
            f"Unknown receptor-coordinate representation {name!r}. Expected one of: "
            f"{', '.join(SUPPORTED_GRID_BACKENDS)}"
        )
    return value
