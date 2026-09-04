"""Run the stochastic calculations used in the manuscript."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .core.grid_backend3d import (
    GRID_BACKEND_GRADIENT_SPARSE_COORDS,
    GRID_BACKEND_UNIFORM_SPARSE_COORDS,
)
from .parameters import distribute_ligands
from .simulation3d import (
    initialize_rng,
    initialize_simulation_state_3d,
    run_simulation_chunk_3d,
    validate_dimension_config,
)
from .storage3d import (
    merge_trace_sketch_3d,
    trajectory_is_complete,
    trajectory_path,
    write_trajectory,
)


@dataclass(frozen=True)
class NumericalProfile:
    """Fixed numerical choices for one manuscript geometry."""

    name: str
    dimension: str
    grid_backend: str
    motion_rule: str
    working_cutoff: float
    validation_cutoff: float


PROFILES = {
    "uniform-3d": NumericalProfile(
        name="uniform-3d",
        dimension="3d",
        grid_backend=GRID_BACKEND_UNIFORM_SPARSE_COORDS,
        motion_rule="adaptive_brownian_reaction",
        working_cutoff=3.0,
        validation_cutoff=4.0,
    ),
    "gradient-3d": NumericalProfile(
        name="gradient-3d",
        dimension="3d",
        grid_backend=GRID_BACKEND_GRADIENT_SPARSE_COORDS,
        motion_rule="adaptive_brownian_reaction",
        working_cutoff=2.5,
        validation_cutoff=3.5,
    ),
    "surface-2d": NumericalProfile(
        name="surface-2d",
        dimension="2d",
        grid_backend=GRID_BACKEND_UNIFORM_SPARSE_COORDS,
        motion_rule="athermal_event_driven",
        working_cutoff=3.0,
        validation_cutoff=5.0,
    ),
}


REQUIRED_PARAMETERS = {
    "L",
    "n_binders",
    "ALPHA",
    "RECEPTOR_SPACING",
    "GAMMA_T_PARALLEL",
    "GAMMA_T_PERPENDICULAR",
    "GAMMA_R",
    "GRADIENT_TYPE",
    "GRADIENT_SCALE",
    "ligand_pattern",
    "K_C",
    "K_D",
    "T_FINAL",
}


def load_parameter_file(path: Path) -> list[dict[str, Any]]:
    """Read one configuration or a list of configurations from YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    rows = payload if isinstance(payload, list) else [payload]
    configurations: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each YAML entry must be a parameter mapping")
        configuration = row.get("params", row)
        if not isinstance(configuration, dict):
            raise ValueError("The 'params' entry must be a parameter mapping")
        configurations.append(dict(configuration))
    if not configurations:
        raise ValueError("The parameter file is empty")
    return configurations


def calculation_type(config: dict[str, Any]) -> str:
    """Identify the supported geometry from its dimension and receptor layout."""
    dimension = validate_dimension_config(config)
    gradient_type = str(config.get("GRADIENT_TYPE", "uniform")).lower()
    gradient_scale = float(config.get("GRADIENT_SCALE", 1.0))
    is_uniform = gradient_type == "uniform" or np.isclose(gradient_scale, 1.0)
    if dimension == "2d":
        if not is_uniform:
            raise ValueError(
                "The manuscript uses the 2D model only on uniform surfaces"
            )
        return "surface-2d"
    return "uniform-3d" if is_uniform else "gradient-3d"


def _validate_configuration(config: dict[str, Any]) -> NumericalProfile:
    missing = sorted(REQUIRED_PARAMETERS.difference(config))
    if missing:
        raise ValueError(f"Missing simulation parameters: {', '.join(missing)}")
    profile = PROFILES[calculation_type(config)]
    n_ligands = int(float(config["L"]))
    n_binders = int(config["n_binders"])
    if n_ligands < 1 or not 0 <= n_binders <= n_ligands:
        raise ValueError("n_binders must lie between zero and int(L)")
    if float(config["T_FINAL"]) <= 0.0:
        raise ValueError("T_FINAL must be positive")
    for key in (
        "ALPHA",
        "RECEPTOR_SPACING",
        "GAMMA_T_PARALLEL",
        "GAMMA_T_PERPENDICULAR",
        "GAMMA_R",
    ):
        if float(config[key]) <= 0.0:
            raise ValueError(f"{key} must be positive")
    if (
        profile.name == "gradient-3d"
        and float(config.get("GRADIENT_MIN_SPACING_STOP", 0.0)) <= 0.0
    ):
        raise ValueError("Gradient calculations require GRADIENT_MIN_SPACING_STOP")
    if profile.motion_rule == "adaptive_brownian_reaction":
        for key in (
            "THERMAL_PASSIVE_D_PARALLEL",
            "THERMAL_PASSIVE_D_PERP",
            "THERMAL_PASSIVE_D_ROT",
        ):
            if key not in config or float(config[key]) < 0.0:
                raise ValueError(f"Thermal 3D calculations require nonnegative {key}")
    return profile


def _runtime_configuration(
    config: dict[str, Any], profile: NumericalProfile
) -> dict[str, Any]:
    runtime = dict(config)
    runtime["DIMENSION"] = profile.dimension
    runtime["nearby_cutoff_alpha_mult"] = profile.working_cutoff
    runtime["nearby_cutoff_validate_alpha_mult"] = profile.validation_cutoff
    runtime["tail_propensity_eps"] = 1.0e-3
    runtime["no_nearby_policy"] = "validated_terminal"
    return runtime


def _initial_trace(dimension: str) -> tuple[np.ndarray, np.ndarray]:
    if dimension == "2d":
        return np.array([0.0]), np.array([[0.0, 0.0, 0.0]])
    return np.array([0.0]), np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])


def _prepare_state(
    config: dict[str, Any],
    ligand_types: np.ndarray,
    profile: NumericalProfile,
) -> dict[str, Any]:
    state = initialize_simulation_state_3d(
        config, ligand_types, grid_backend=profile.grid_backend
    )
    state.update(
        {
            "nearby_cutoff_alpha_mult": profile.working_cutoff,
            "nearby_cutoff_validate_alpha_mult": profile.validation_cutoff,
            "tail_propensity_eps": 1.0e-3,
            "no_nearby_policy": "validated_terminal",
        }
    )
    return state


def run_and_save_single_trajectory(
    config: dict[str, Any],
    ligand_types: np.ndarray | None,
    trajectory_index: int,
    output_dir: Path,
    *,
    chunk_steps: int = 50_000,
    trajectory_max_points: int = 256,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one trajectory through the fixed numerical route for its geometry."""
    profile = _validate_configuration(config)
    config = _runtime_configuration(config, profile)
    if ligand_types is None:
        ligand_types = distribute_ligands(
            int(float(config["L"])),
            int(config["n_binders"]),
            str(config["ligand_pattern"]),
        )
    else:
        ligand_types = np.asarray(ligand_types, dtype=np.bool_)

    path = trajectory_path(output_dir, config, trajectory_index)
    if not overwrite and trajectory_is_complete(
        path,
        float(config["T_FINAL"]),
        config,
        trajectory_max_points,
    ):
        return {
            "success": True,
            "trajectory_index": trajectory_index,
            "path": path,
            "skipped": True,
        }

    seed = 42 + int(trajectory_index)
    state = _prepare_state(config, ligand_types, profile)
    initialize_rng(seed)
    times, positions = _initial_trace(profile.dimension)
    completion_reason = "chunk_limit"

    while True:
        chunk = run_simulation_chunk_3d(
            config,
            ligand_types,
            state,
            max_steps=max(1, int(chunk_steps)),
            nearby_cutoff_alpha_mult=profile.working_cutoff,
            nearby_cutoff_validate_alpha_mult=profile.validation_cutoff,
            tail_propensity_eps=1.0e-3,
            no_nearby_policy="validated_terminal",
        )
        times, positions = merge_trace_sketch_3d(
            times,
            positions,
            np.asarray(chunk["times"], dtype=np.float64),
            np.asarray(chunk["positions"], dtype=np.float64),
            max_points=trajectory_max_points,
        )
        if bool(chunk.get("done", False)):
            completion_reason = str(chunk.get("termination_reason", ""))
            break

    accepted_reasons = {"t_final", "gradient_threshold", "gradient_escape", "no_nearby"}
    if completion_reason not in accepted_reasons:
        raise RuntimeError(
            f"Trajectory {trajectory_index} stopped at t={state.get('t', 0.0):g} "
            f"with reason {completion_reason!r}"
        )

    write_trajectory(
        path,
        config=config,
        ligand_types=ligand_types,
        times=times,
        positions=positions,
        state=state,
        completion_reason=completion_reason,
        random_seed=seed,
        trajectory_max_points=trajectory_max_points,
    )
    return {
        "success": True,
        "trajectory_index": trajectory_index,
        "path": path,
        "skipped": False,
        "completion_reason": completion_reason,
        "t_reached": float(state["t"]),
        "n_cleaved": int(state.get("n_cleaved", 0)),
    }


def _run_task(task: tuple[dict[str, Any], int, Path, int, int, bool]) -> dict[str, Any]:
    config, trajectory_index, output_dir, chunk_steps, max_points, overwrite = task
    return run_and_save_single_trajectory(
        config,
        None,
        trajectory_index,
        output_dir,
        chunk_steps=chunk_steps,
        trajectory_max_points=max_points,
        overwrite=overwrite,
    )


def run_ensemble(
    configurations: list[dict[str, Any]],
    *,
    n_trajectories: int,
    output_dir: Path,
    n_jobs: int = 1,
    chunk_steps: int = 50_000,
    trajectory_max_points: int = 256,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Run every configuration with deterministic seeds 42, 43, ... ."""
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be positive")
    for config in configurations:
        _validate_configuration(config)
    tasks = [
        (config, index, Path(output_dir), chunk_steps, trajectory_max_points, overwrite)
        for config in configurations
        for index in range(n_trajectories)
    ]
    if n_jobs <= 1:
        return [_run_task(task) for task in tasks]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_run_task, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(
        results,
        key=lambda result: (str(result["path"]), int(result["trajectory_index"])),
    )
