"""Compact HDF5 storage for manuscript simulation trajectories."""

from __future__ import annotations

import heapq
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


TRAJECTORY_COMPRESSION = "budgeted_spatial_orientation_path"


def configuration_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable digest of every parameter supplied to a trajectory."""

    def normalize(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    payload = json.dumps(
        normalize(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _format_token(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def parameter_directory(base_dir: Path, config: dict[str, Any]) -> Path:
    """Return the directory used for one molecular parameter set."""
    dimension = str(config.get("DIMENSION", "3d")).strip().lower()
    if dimension == "3d":
        background_motion = (
            f"Dpar_{_format_token(config['THERMAL_PASSIVE_D_PARALLEL'])}_"
            f"Dperp_{_format_token(config['THERMAL_PASSIVE_D_PERP'])}_"
            f"Drot_{_format_token(config['THERMAL_PASSIVE_D_ROT'])}"
        )
    else:
        background_motion = "athermal"
    return (
        Path(base_dir)
        / f"raw_{dimension}"
        / str(config["ligand_pattern"])
        / f"L_{_format_token(config['L'])}"
        / (
            f"alpha_{_format_token(config['ALPHA'])}_"
            f"d_{_format_token(config['RECEPTOR_SPACING'])}"
        )
        / (
            f"gscale_{_format_token(config['GRADIENT_SCALE'])}_"
            f"gtype_{config['GRADIENT_TYPE']}"
        )
        / f"KC_{_format_token(config['K_C'])}"
        / f"nb_{_format_token(config['n_binders'])}"
        / (
            f"KD_{_format_token(config['K_D'])}_"
            f"gtp_{_format_token(config['GAMMA_T_PARALLEL'])}_"
            f"gtper_{_format_token(config['GAMMA_T_PERPENDICULAR'])}_"
            f"gtr_{_format_token(config['GAMMA_R'])}"
        )
        / background_motion
        / f"config_{configuration_fingerprint(config)[:12]}"
    )


def trajectory_path(
    base_dir: Path, config: dict[str, Any], trajectory_index: int
) -> Path:
    return parameter_directory(base_dir, config) / f"traj_{trajectory_index}.h5"


def _segment_max_deviation(
    features: np.ndarray, start: int, end: int
) -> tuple[float, int]:
    if end <= start + 1:
        return 0.0, -1
    a = features[start]
    b = features[end]
    direction = b - a
    length_squared = float(np.dot(direction, direction))
    interior = features[start + 1 : end]
    if length_squared <= 0.0:
        residual = interior - a[None, :]
    else:
        relative = interior - a[None, :]
        fraction = np.clip(np.dot(relative, direction) / length_squared, 0.0, 1.0)
        residual = interior - (a[None, :] + fraction[:, None] * direction[None, :])
    squared_deviation = np.sum(residual * residual, axis=1)
    local_index = int(np.argmax(squared_deviation))
    return float(np.sqrt(squared_deviation[local_index])), start + 1 + local_index


def _budgeted_rdp_indices(features: np.ndarray, max_points: int) -> np.ndarray:
    """Select the largest spatial or orientational bends within a point budget."""
    n_points = len(features)
    if n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)

    keep = np.zeros(n_points, dtype=bool)
    keep[[0, -1]] = True
    queue: list[tuple[float, int, int, int]] = []

    def add_segment(start: int, end: int) -> None:
        deviation, index = _segment_max_deviation(features, start, end)
        if deviation > 1.0e-12 and start < index < end:
            heapq.heappush(queue, (-deviation, start, end, index))

    add_segment(0, n_points - 1)
    n_kept = 2
    while queue and n_kept < max_points:
        _, start, end, index = heapq.heappop(queue)
        if keep[index]:
            continue
        keep[index] = True
        n_kept += 1
        add_segment(start, index)
        add_segment(index, end)
    return np.flatnonzero(keep).astype(np.int64, copy=False)


def _trajectory_features(
    positions: np.ndarray,
    displacement_scale: float,
    angular_scale: float,
) -> np.ndarray:
    xyz = np.asarray(positions[:, :3], dtype=np.float64)
    if positions.shape[1] < 6:
        return xyz
    orientation = np.asarray(positions[:, 3:6], dtype=np.float64)
    weight = displacement_scale / max(angular_scale, 1.0e-8)
    return np.concatenate((xyz, weight * orientation), axis=1)


def compress_trajectory_3d(
    times: np.ndarray,
    positions: np.ndarray,
    max_points: int = 256,
    disp_threshold: float = 10.0,
    angular_threshold: float = 0.25,
    min_uniform_points: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a trajectory while retaining endpoints, large bends and time coverage."""
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    max_points = max(2, int(max_points))
    if len(times) <= max_points:
        return times, positions

    features = _trajectory_features(positions, disp_threshold, angular_threshold)
    indices = _budgeted_rdp_indices(features, max_points)
    if len(indices) < max_points:
        uniform = np.linspace(
            0,
            len(times) - 1,
            min(max_points, max(2, int(min_uniform_points))),
            dtype=np.int64,
        )
        indices = np.unique(np.concatenate((indices, uniform)))
    if len(indices) > max_points:
        indices = indices[_budgeted_rdp_indices(features[indices], max_points)]
    return times[indices], positions[indices]


def merge_trace_sketch_3d(
    stored_times: np.ndarray,
    stored_positions: np.ndarray,
    new_times: np.ndarray,
    new_positions: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Append a trajectory segment and keep a bounded path representation."""
    if len(new_times) == 0:
        return stored_times, stored_positions
    if len(stored_times) == 0:
        times = np.asarray(new_times, dtype=np.float64)
        positions = np.asarray(new_positions, dtype=np.float64)
    else:
        times = np.concatenate((stored_times, np.asarray(new_times, dtype=np.float64)))
        positions = np.concatenate(
            (stored_positions, np.asarray(new_positions, dtype=np.float64)), axis=0
        )
    keep = np.ones(len(times), dtype=bool)
    keep[1:] = times[1:] > times[:-1]
    times = times[keep]
    positions = positions[keep]
    if len(times) > max_points:
        return compress_trajectory_3d(times, positions, max_points=max_points)
    return times, positions


def _positions_for_dimension(positions: np.ndarray, dimension: str) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if dimension != "2d" or positions.shape[1] < 6:
        return positions
    planar = np.empty((len(positions), 3), dtype=np.float64)
    planar[:, :2] = positions[:, :2]
    planar[:, 2] = np.arctan2(positions[:, 4], positions[:, 3])
    return planar


def _coordinate_dtype(*coordinates: np.ndarray) -> np.dtype:
    largest = max((int(np.max(np.abs(x))) if len(x) else 0 for x in coordinates))
    if largest <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    if largest <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def _set_attributes(target: h5py.AttributeManager, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, bool, int, float)):
            target[key] = value


def write_trajectory(
    path: Path,
    *,
    config: dict[str, Any],
    ligand_types: np.ndarray,
    times: np.ndarray,
    positions: np.ndarray,
    state: dict[str, Any],
    completion_reason: str,
    random_seed: int,
    trajectory_max_points: int = 256,
) -> None:
    """Write the path, final cleaved receptors and identifying metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    dimension = str(config.get("DIMENSION", "3d")).lower()
    positions = _positions_for_dimension(positions, dimension)
    times, positions = compress_trajectory_3d(
        times, positions, max_points=trajectory_max_points
    )

    n_cleaved = max(0, int(state.get("n_cleaved", 0)))
    cleaved_ix = np.asarray(state.get("cleaved_ix", []), dtype=np.int64)[:n_cleaved]
    cleaved_iy = np.asarray(state.get("cleaved_iy", []), dtype=np.int64)[:n_cleaved]
    cleaved_iz = np.asarray(state.get("cleaved_iz", []), dtype=np.int64)[:n_cleaved]
    coordinate_dtype = _coordinate_dtype(cleaved_ix, cleaved_iy, cleaved_iz)

    root_attributes: dict[str, Any] = {
        "run_status": "complete",
        "configuration_sha256": configuration_fingerprint(config),
        "completion_reason": completion_reason,
        "dimension": dimension,
        "random_seed": int(random_seed),
        "particle_length": float(config["L"]),
        "interaction_reach": float(config["ALPHA"]),
        "receptor_spacing": float(config["RECEPTOR_SPACING"]),
        "ligand_pattern": str(config["ligand_pattern"]),
        "n_ligands": int(len(ligand_types)),
        "n_binders": int(np.count_nonzero(ligand_types)),
        "n_cleavers": int(len(ligand_types) - np.count_nonzero(ligand_types)),
        "K_D": float(config["K_D"]),
        "K_C": float(config["K_C"]),
        "t_reached": float(state.get("t", times[-1])),
        "target_t_final": float(config["T_FINAL"]),
        "grid_backend": str(state["grid_backend"]),
        "motion_rule": str(state["motion_rule"]),
        "reaction_method": "direct_stochastic_simulation",
        "nearby_cutoff_alpha_mult": float(state["nearby_cutoff_alpha_mult"]),
        "nearby_cutoff_validate_alpha_mult": float(
            state["nearby_cutoff_validate_alpha_mult"]
        ),
        "tail_propensity_eps": float(state["tail_propensity_eps"]),
        "reaction_steps": int(state.get("reaction_steps", 0)),
        "bind_events": int(state.get("bind_events", 0)),
        "unbind_events": int(state.get("unbind_events", 0)),
        "cleavage_events": int(state.get("cleavage_events", 0)),
        "trajectory_compression": TRAJECTORY_COMPRESSION,
        "trajectory_max_points": int(trajectory_max_points),
        "receptor_mobility": str(state.get("receptor_mobility_mode", "fixed")),
    }
    if bool(state.get("thermal_brownian_enabled", False)):
        root_attributes.update(
            {
                "background_motion": "brownian",
                "D_parallel": float(state["thermal_passive_d_parallel"]),
                "D_perpendicular": float(state["thermal_passive_d_perp"]),
                "D_rotational": float(state["thermal_passive_d_rot"]),
                "brownian_step_mode": str(state["thermal_brownian_dt_mode"]),
                "brownian_dt_min": float(state["thermal_brownian_dt_min"]),
                "brownian_dt_max": float(state["thermal_brownian_dt_max"]),
            }
        )
    else:
        root_attributes["background_motion"] = "athermal"
    if str(config["GRADIENT_TYPE"]).lower() != "uniform":
        root_attributes.update(
            {
                "gradient_axis_law": str(state["gradient_axis_law"]),
                "gradient_min_spacing_stop": float(state["gradient_min_spacing_stop"]),
                "gradient_stop_triggered": bool(
                    state.get("gradient_stop_triggered", False)
                ),
            }
        )

    with h5py.File(temporary_path, "w") as output:
        _set_attributes(output.attrs, root_attributes)
        trajectory = output.create_group("trajectory")
        trajectory.attrs["dimension"] = dimension
        trajectory.create_dataset(
            "times", data=np.asarray(times, dtype=np.float32), compression="gzip"
        )
        trajectory.create_dataset(
            "positions",
            data=np.asarray(positions, dtype=np.float32),
            compression="gzip",
        )

        grid = output.create_group("grid_metadata")
        _set_attributes(
            grid.attrs,
            {
                "backend": str(state["grid_backend"]),
                "dimension": dimension,
                "spacing": float(config["RECEPTOR_SPACING"]),
                "gradient_type": str(config["GRADIENT_TYPE"]),
                "gradient_scale": float(config["GRADIENT_SCALE"]),
            },
        )

        substrate = output.create_group("substrate_final")
        substrate.attrs["encoding"] = "sparse_lattice_coordinates"
        substrate.attrs["n_cleaved"] = n_cleaved
        substrate.attrs["coordinate_dtype"] = coordinate_dtype.name
        substrate.create_dataset(
            "cleaved_ix", data=cleaved_ix.astype(coordinate_dtype), compression="gzip"
        )
        substrate.create_dataset(
            "cleaved_iy", data=cleaved_iy.astype(coordinate_dtype), compression="gzip"
        )
        substrate.create_dataset(
            "cleaved_iz", data=cleaved_iz.astype(coordinate_dtype), compression="gzip"
        )
    os.replace(temporary_path, path)


def trajectory_is_complete(
    path: Path,
    target_time: float,
    config: dict[str, Any],
    minimum_stored_points: int,
) -> bool:
    if not Path(path).is_file():
        return False
    try:
        with h5py.File(path, "r") as trajectory:
            reason = str(trajectory.attrs.get("completion_reason", ""))
            reached = float(trajectory.attrs.get("t_reached", 0.0))
            stored_fingerprint = str(
                trajectory.attrs.get("configuration_sha256", "")
            )
            stored_compression = str(
                trajectory.attrs.get("trajectory_compression", "")
            )
            stored_point_budget = int(
                trajectory.attrs.get("trajectory_max_points", 0)
            )
        if stored_fingerprint != configuration_fingerprint(config):
            return False
        if stored_compression != TRAJECTORY_COMPRESSION:
            return False
        if stored_point_budget < int(minimum_stored_points):
            return False
        return reason in {
            "t_final",
            "gradient_threshold",
            "gradient_escape",
            "no_nearby",
        } and (reached >= target_time - 1.0e-9 or reason != "t_final")
    except (OSError, KeyError, TypeError, ValueError):
        return False
