"""Common observables calculated directly from stored simulation paths."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class StoredPath:
    """Time, position and orientation arrays read from one trajectory file."""

    times: np.ndarray
    positions: np.ndarray
    orientations: np.ndarray | None
    dimension: int
    completion_reason: str


def load_stored_path(path: Path) -> StoredPath:
    """Load and validate the path representation stored by the simulator."""

    with h5py.File(path, "r") as h5:
        times = np.asarray(h5["trajectory/times"][:], dtype=float)
        states = np.asarray(h5["trajectory/positions"][:], dtype=float)
        dimension_name = str(h5.attrs["dimension"])
        dimension = 2 if dimension_name == "2d" else 3
        positions = states[:, :dimension]
        if dimension == 3 and states.shape[1] >= 6:
            orientations = states[:, 3:6]
        elif dimension == 2 and states.shape[1] >= 3:
            angles = states[:, 2]
            orientations = np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            orientations = None
        completion_reason = str(h5.attrs.get("completion_reason", "unknown"))

    if times.ndim != 1 or times.size < 2 or positions.shape[0] != times.size:
        raise ValueError(f"Invalid trajectory arrays in {path}")
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"Trajectory times are not finite and increasing in {path}")
    if np.any(~np.isfinite(positions)):
        raise ValueError(f"Trajectory positions are not finite in {path}")
    if orientations is not None:
        norms = np.linalg.norm(orientations, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError(f"Trajectory orientations are invalid in {path}")
        orientations = orientations / norms[:, None]
    return StoredPath(
        times=times,
        positions=positions,
        orientations=orientations,
        dimension=dimension,
        completion_reason=completion_reason,
    )


def _interpolate_positions(
    path: StoredPath,
    sample_times: np.ndarray,
    *,
    hold_after_end: bool,
) -> tuple[np.ndarray, np.ndarray]:
    elapsed = path.times - path.times[0]
    query = np.asarray(sample_times, dtype=float)
    valid = query <= elapsed[-1]
    if hold_after_end:
        valid = np.ones(query.shape, dtype=bool)
    clipped = np.minimum(query, elapsed[-1])
    interpolated = np.column_stack(
        [
            np.interp(clipped, elapsed, path.positions[:, axis])
            for axis in range(path.dimension)
        ]
    )
    return interpolated, valid


def ensemble_msd(
    paths: Iterable[StoredPath],
    sample_times: np.ndarray,
    *,
    hold_after_end: bool = False,
) -> dict[str, np.ndarray | int]:
    """Return ensemble displacement variance from each path's initial position.

    With ``hold_after_end=False``, a path contributes only while it is present.
    With ``hold_after_end=True``, its terminal position is retained after a
    physical stopping event, as in the surface-detachment analysis.
    """

    query = np.asarray(sample_times, dtype=float)
    if query.ndim != 1 or query.size < 2 or np.any(np.diff(query) <= 0.0):
        raise ValueError("sample_times must be a strictly increasing 1D array")
    if query[0] < 0.0 or np.any(~np.isfinite(query)):
        raise ValueError("sample_times must be finite and nonnegative")

    squared: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    dimensions: set[int] = set()
    for path in paths:
        dimensions.add(path.dimension)
        positions, valid = _interpolate_positions(
            path, query, hold_after_end=hold_after_end
        )
        displacement = positions - path.positions[0]
        values = np.sum(displacement * displacement, axis=1)
        values[~valid] = np.nan
        squared.append(values)
        masks.append(valid)
    if not squared:
        raise ValueError("At least one trajectory is required")
    if len(dimensions) != 1:
        raise ValueError("All trajectories must have the same dimension")

    matrix = np.vstack(squared)
    return {
        "time": query,
        "msd": np.nanmean(matrix, axis=0),
        "trajectory_count": np.sum(np.vstack(masks), axis=0),
        "trajectory_squared_displacements": matrix,
        "dimension": dimensions.pop(),
    }


def effective_diffusivity_from_msd(
    times: np.ndarray,
    msd: np.ndarray,
    *,
    dimension: int,
    fit_start: float,
    fit_stop: float,
) -> dict[str, float | int]:
    """Fit the stated late-time interval and return slope divided by ``2d``."""

    t = np.asarray(times, dtype=float)
    y = np.asarray(msd, dtype=float)
    selected = (
        np.isfinite(t)
        & np.isfinite(y)
        & (t >= float(fit_start))
        & (t <= float(fit_stop))
    )
    if dimension not in (2, 3) or np.count_nonzero(selected) < 3:
        raise ValueError("The fit needs dimension 2 or 3 and at least three points")
    slope, intercept = np.polyfit(t[selected], y[selected], 1)
    predicted = slope * t[selected] + intercept
    residual = float(np.sum((y[selected] - predicted) ** 2))
    total = float(np.sum((y[selected] - np.mean(y[selected])) ** 2))
    return {
        "effective_diffusivity": float(slope / (2.0 * dimension)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - residual / total) if total > 0.0 else 1.0,
        "fit_points": int(np.count_nonzero(selected)),
    }


def body_axis_alignment(path: StoredPath) -> float:
    """Mean second-Legendre alignment of displacements with the particle axis."""

    if path.orientations is None:
        raise ValueError("Stored orientations are required for body-axis alignment")
    steps = np.diff(path.positions, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    valid = np.isfinite(lengths) & (lengths > 0.0)
    if not np.any(valid):
        return float("nan")
    directions = steps[valid] / lengths[valid, None]
    cosine = np.sum(directions * path.orientations[:-1][valid], axis=1)
    return float(np.mean(0.5 * (3.0 * cosine**2 - 1.0)))


def orientation_correlation(path: StoredPath, *, apolar: bool) -> np.ndarray:
    """Correlation with the initial axis for vectorial or apolar motion."""

    if path.orientations is None:
        raise ValueError("Stored orientations are required for this correlation")
    cosine = path.orientations @ path.orientations[0]
    if apolar:
        return 0.5 * (3.0 * cosine**2 - 1.0)
    return cosine


def one_over_e_crossing(times: np.ndarray, correlation: np.ndarray) -> float:
    """Linearly interpolate the first crossing of ``1/e``."""

    t = np.asarray(times, dtype=float)
    c = np.asarray(correlation, dtype=float)
    if t.ndim != 1 or c.shape != t.shape or t.size < 2:
        raise ValueError("times and correlation must be equal-length 1D arrays")
    below = np.flatnonzero(c <= np.exp(-1.0))
    if not below.size:
        return float("nan")
    index = int(below[0])
    if index == 0:
        return float(t[0])
    c0, c1 = float(c[index - 1]), float(c[index])
    if np.isclose(c0, c1):
        return float(t[index])
    fraction = (np.exp(-1.0) - c0) / (c1 - c0)
    return float(t[index - 1] + fraction * (t[index] - t[index - 1]))


def terminal_range(path: StoredPath) -> float:
    """Distance between the initial and terminal particle positions."""

    displacement = path.positions[-1] - path.positions[0]
    return float(np.linalg.norm(displacement))
