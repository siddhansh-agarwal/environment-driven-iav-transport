"""Trajectory observables used for motion through receptor gradients."""

from __future__ import annotations

import numpy as np


def path_chemotactic_index(
    positions: np.ndarray,
    dense_direction: np.ndarray,
) -> float:
    """Return displacement toward denser receptors divided by path length.

    ``positions`` has shape ``(n, d)`` and ``dense_direction`` is a unit vector
    pointing toward increasing receptor density. The result is zero when the
    sampled path has no finite displacement.
    """

    xyz = np.asarray(positions, dtype=float)
    direction = np.asarray(dense_direction, dtype=float)
    if xyz.ndim != 2 or xyz.shape[0] < 2:
        raise ValueError("positions must contain at least two d-dimensional points")
    if direction.shape != (xyz.shape[1],):
        raise ValueError("dense_direction must match the position dimension")
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("dense_direction must be finite and nonzero")
    steps = np.diff(xyz, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    valid = np.isfinite(lengths) & (lengths > 0.0)
    path_length = float(np.sum(lengths[valid]))
    if path_length == 0.0:
        return 0.0
    net = xyz[-1] - xyz[0]
    return float(np.dot(net, direction / norm) / path_length)


def path_chemotactic_index_until(
    times: np.ndarray,
    positions: np.ndarray,
    dense_direction: np.ndarray,
    query_time: float,
) -> float:
    """Return the path chemotactic index through a fixed observation time.

    The endpoint is linearly interpolated when ``query_time`` lies between two
    stored samples. If a trajectory ends earlier at a physical stopping
    condition, its complete stored path is used.
    """

    t = np.asarray(times, dtype=float)
    xyz = np.asarray(positions, dtype=float)
    if t.ndim != 1 or xyz.ndim != 2 or t.size != xyz.shape[0] or t.size < 2:
        raise ValueError("times and positions must describe at least two samples")
    if np.any(~np.isfinite(t)) or np.any(np.diff(t) < 0.0):
        raise ValueError("times must be finite and nondecreasing")
    if not np.isfinite(query_time) or query_time <= t[0]:
        raise ValueError("query_time must be finite and greater than the initial time")

    stop = int(np.searchsorted(t, float(query_time), side="right"))
    sampled = [np.asarray(xyz[0], dtype=float)]
    if stop > 1:
        sampled.extend(np.asarray(xyz[1:stop], dtype=float))
    if query_time < t[-1] and stop < t.size:
        endpoint = np.array(
            [
                np.interp(float(query_time), t, xyz[:, axis])
                for axis in range(xyz.shape[1])
            ],
            dtype=float,
        )
    else:
        endpoint = np.asarray(xyz[-1], dtype=float)
    if not np.allclose(endpoint, sampled[-1], rtol=0.0, atol=1.0e-12):
        sampled.append(endpoint)
    return path_chemotactic_index(np.asarray(sampled), dense_direction)


def normalized_dense_arrival_rate(
    times: np.ndarray,
    coordinate: np.ndarray,
    dense_boundary: float,
    sparse_boundary: float,
    observation_time: float,
) -> float:
    """Return the normalized dense-boundary arrival score used in simulation.

    A trajectory that reaches the dense boundary first within the observation
    time contributes ``observation_time / arrival_time``. A trajectory that
    reaches the sparse boundary first, or neither boundary, contributes zero.
    """

    t = np.asarray(times, dtype=float)
    z = np.asarray(coordinate, dtype=float)
    if t.ndim != 1 or z.ndim != 1 or t.size != z.size or t.size < 2:
        raise ValueError(
            "times and coordinate must be equal-length one-dimensional arrays"
        )
    if np.any(~np.isfinite(t)) or np.any(np.diff(t) < 0.0):
        raise ValueError("times must be finite and nondecreasing")
    if not np.isfinite(observation_time) or observation_time <= 0.0:
        raise ValueError("observation_time must be finite and positive")

    elapsed = t - t[0]
    eligible = elapsed <= float(observation_time)
    dense_hits = np.flatnonzero(eligible & (z >= float(dense_boundary)))
    sparse_hits = np.flatnonzero(eligible & (z <= float(sparse_boundary)))
    dense_index = int(dense_hits[0]) if dense_hits.size else None
    sparse_index = int(sparse_hits[0]) if sparse_hits.size else None
    if dense_index is None or (sparse_index is not None and sparse_index < dense_index):
        return 0.0
    arrival_time = float(elapsed[dense_index])
    return 0.0 if arrival_time <= 0.0 else float(observation_time) / arrival_time
