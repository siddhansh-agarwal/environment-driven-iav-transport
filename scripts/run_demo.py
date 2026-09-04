#!/usr/bin/env python3
"""Run and verify one short example from each manuscript geometry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIGS = {
    "uniform-3d": ROOT / "config" / "demo_uniform_3d.yaml",
    "gradient-3d": ROOT / "config" / "demo_gradient_3d.yaml",
    "surface-2d": ROOT / "config" / "demo_surface_2d.yaml",
}

EXPECTED = {
    "uniform-3d": {
        "dimension": "3d",
        "columns": 6,
        "backend": "uniform_sparse_coords",
        "motion_rule": "adaptive_brownian_reaction",
        "background": "brownian",
        "working_cutoff": 3.0,
        "validation_cutoff": 4.0,
        "pattern": "polarized",
    },
    "gradient-3d": {
        "dimension": "3d",
        "columns": 6,
        "backend": "gradient_sparse_coords",
        "motion_rule": "adaptive_brownian_reaction",
        "background": "brownian",
        "working_cutoff": 2.5,
        "validation_cutoff": 3.5,
        "pattern": "polarized",
    },
    "surface-2d": {
        "dimension": "2d",
        "columns": 3,
        "backend": "uniform_sparse_coords",
        "motion_rule": "athermal_event_driven",
        "background": "athermal",
        "working_cutoff": 3.0,
        "validation_cutoff": 5.0,
        "pattern": "mixed",
    },
}


def _equal(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        return bool(np.isclose(float(actual), expected))
    return actual == expected


def validate_output(
    case: str, output_dir: Path, expected_files: int
) -> dict[str, object]:
    """Check that saved trajectories used the intended physical and numerical route."""
    files = sorted(Path(output_dir).rglob("traj_*.h5"))
    if len(files) != expected_files:
        raise RuntimeError(
            f"Expected {expected_files} trajectories, found {len(files)}"
        )
    expected = EXPECTED[case]
    total_points = 0
    squared_displacements: list[float] = []

    for path in files:
        with h5py.File(path, "r") as trajectory:
            checks = {
                "dimension": expected["dimension"],
                "grid_backend": expected["backend"],
                "motion_rule": expected["motion_rule"],
                "background_motion": expected["background"],
                "reaction_method": "direct_stochastic_simulation",
                "nearby_cutoff_alpha_mult": expected["working_cutoff"],
                "nearby_cutoff_validate_alpha_mult": expected["validation_cutoff"],
                "ligand_pattern": expected["pattern"],
            }
            for name, value in checks.items():
                actual = trajectory.attrs.get(name)
                if not _equal(actual, value):
                    raise RuntimeError(
                        f"{path} has {name}={actual!r}; expected {value!r}"
                    )
            if str(trajectory.attrs["completion_reason"]) != "t_final":
                raise RuntimeError(f"Demo did not reach its target time: {path}")
            if expected["background"] == "brownian":
                for name, value in {
                    "D_parallel": 0.0125,
                    "D_perpendicular": 0.00625,
                    "D_rotational": 0.00025,
                }.items():
                    if not np.isclose(float(trajectory.attrs[name]), value):
                        raise RuntimeError(f"Unexpected {name} in {path}")

            times = np.asarray(trajectory["trajectory/times"][:], dtype=np.float64)
            positions = np.asarray(
                trajectory["trajectory/positions"][:], dtype=np.float64
            )
            if positions.shape != (len(times), int(expected["columns"])):
                raise RuntimeError(f"Unexpected trajectory shape in {path}")
            if len(times) < 2 or not np.all(np.diff(times) > 0.0):
                raise RuntimeError(
                    f"Trajectory times are not strictly increasing in {path}"
                )
            if not np.all(np.isfinite(positions)):
                raise RuntimeError(
                    f"Trajectory contains non-finite positions in {path}"
                )
            displacement = (
                positions[-1, : 2 if case == "surface-2d" else 3]
                - positions[0, : 2 if case == "surface-2d" else 3]
            )
            squared_displacements.append(float(np.dot(displacement, displacement)))
            total_points += len(times)

    return {
        "files": len(files),
        "stored_points": total_points,
        "mean_squared_displacement": float(np.mean(squared_displacements)),
    }


def main() -> None:
    from analysis.trajectory_summary import write_trajectory_summary
    from src.runner3d import load_parameter_file, run_ensemble

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", choices=[*CONFIGS, "all"], nargs="?", default="uniform-3d"
    )
    parser.add_argument("--trajectories", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "demo_output")
    args = parser.parse_args()
    if args.trajectories < 1:
        parser.error("--trajectories must be positive")

    cases = list(CONFIGS) if args.case == "all" else [args.case]
    for case in cases:
        output_dir = args.output / case
        print(f"Running {case}")
        run_ensemble(
            load_parameter_file(CONFIGS[case]),
            n_trajectories=args.trajectories,
            output_dir=output_dir,
            n_jobs=1,
            overwrite=True,
        )
        summary = validate_output(case, output_dir, args.trajectories)
        write_trajectory_summary(output_dir, output_dir / "trajectory_summary.csv")
        print(
            f"Verified {summary['files']} trajectories and "
            f"{summary['stored_points']} stored path points in {output_dir}"
        )


if __name__ == "__main__":
    main()
