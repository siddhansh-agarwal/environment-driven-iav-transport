"""Summarize compact simulation trajectories in a portable CSV table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


SUMMARY_COLUMNS = (
    "trajectory_file",
    "random_seed",
    "dimension",
    "ligand_pattern",
    "K_D",
    "K_C",
    "elapsed_time",
    "stored_points",
    "net_displacement",
    "squared_displacement",
    "stored_path_length",
    "path_CI",
    "cleaved_receptors",
    "completion_reason",
)


def _dense_axis_sign(
    gradient_type: str, gradient_scale: float
) -> tuple[int, float] | None:
    """Return the gradient-axis index and sign pointing toward higher density."""
    if np.isclose(gradient_scale, 1.0):
        return None
    axis = {"x": 0, "y": 1, "z": 2}.get(gradient_type.lower())
    if axis is None:
        return None
    return axis, -1.0 if gradient_scale > 1.0 else 1.0


def summarize_trajectory(path: Path) -> dict[str, object]:
    """Calculate path-level quantities from one compact HDF5 trajectory."""
    with h5py.File(path, "r") as h5:
        times = np.asarray(h5["trajectory/times"][:], dtype=np.float64)
        positions = np.asarray(h5["trajectory/positions"][:], dtype=np.float64)
        dimension = str(h5.attrs["dimension"])
        spatial_dimensions = 2 if dimension == "2d" else 3
        xyz = positions[:, :spatial_dimensions]
        increments = np.diff(xyz, axis=0)
        displacement = xyz[-1] - xyz[0]
        path_length = float(np.linalg.norm(increments, axis=1).sum())

        grid = h5["grid_metadata"].attrs
        dense_axis = _dense_axis_sign(
            str(grid.get("gradient_type", "uniform")),
            float(grid.get("gradient_scale", 1.0)),
        )
        path_ci = np.nan
        if dense_axis is not None and path_length > 0.0:
            axis, sign = dense_axis
            path_ci = float(sign * displacement[axis] / path_length)

        substrate = h5.get("substrate_final")
        n_cleaved = 0
        if substrate is not None and "cleaved_ix" in substrate:
            n_cleaved = int(substrate["cleaved_ix"].shape[0])

        return {
            "trajectory_file": path.as_posix(),
            "random_seed": int(h5.attrs["random_seed"]),
            "dimension": dimension,
            "ligand_pattern": str(h5.attrs["ligand_pattern"]),
            "K_D": float(h5.attrs["K_D"]),
            "K_C": float(h5.attrs["K_C"]),
            "elapsed_time": float(times[-1] - times[0]),
            "stored_points": int(times.size),
            "net_displacement": float(np.linalg.norm(displacement)),
            "squared_displacement": float(np.dot(displacement, displacement)),
            "stored_path_length": path_length,
            "path_CI": path_ci,
            "cleaved_receptors": n_cleaved,
            "completion_reason": str(h5.attrs["completion_reason"]),
        }


def write_trajectory_summary(
    input_dir: Path, output_file: Path
) -> list[dict[str, object]]:
    """Summarize every trajectory below ``input_dir`` and write one CSV file."""
    paths = sorted(input_dir.rglob("traj_*.h5"))
    if not paths:
        raise FileNotFoundError(f"No trajectory files found below {input_dir}")
    rows = [summarize_trajectory(path) for path in paths]
    for path, row in zip(paths, rows):
        row["trajectory_file"] = path.relative_to(input_dir).as_posix()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing trajectory HDF5 files"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()
    output = args.output or args.input_dir / "trajectory_summary.csv"
    rows = write_trajectory_summary(args.input_dir, output)
    print(f"Wrote {len(rows)} trajectory summaries to {output}")


if __name__ == "__main__":
    main()
