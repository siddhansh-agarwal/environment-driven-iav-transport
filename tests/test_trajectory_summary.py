from pathlib import Path

import h5py
import numpy as np

from analysis.trajectory_summary import summarize_trajectory, write_trajectory_summary


def _write_trajectory(path: Path, *, scale: float = 1.001) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["dimension"] = "3d"
        h5.attrs["random_seed"] = 42
        h5.attrs["ligand_pattern"] = "polarized"
        h5.attrs["K_D"] = 100.0
        h5.attrs["K_C"] = 10.0
        h5.attrs["completion_reason"] = "t_final"
        trajectory = h5.create_group("trajectory")
        trajectory.create_dataset("times", data=np.array([0.0, 1.0, 2.0]))
        trajectory.create_dataset(
            "positions",
            data=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, -2.0]]),
        )
        grid = h5.create_group("grid_metadata")
        grid.attrs["gradient_type"] = "z"
        grid.attrs["gradient_scale"] = scale
        substrate = h5.create_group("substrate_final")
        substrate.create_dataset("cleaved_ix", data=np.arange(3))


def test_summarize_trajectory_uses_dense_side_and_stored_path(tmp_path):
    path = tmp_path / "traj_0.h5"
    _write_trajectory(path)
    row = summarize_trajectory(path)
    assert row["elapsed_time"] == 2.0
    assert row["squared_displacement"] == 4.0
    assert row["stored_path_length"] == 2.0
    assert row["path_CI"] == 1.0
    assert row["cleaved_receptors"] == 3


def test_write_trajectory_summary_creates_csv(tmp_path):
    _write_trajectory(tmp_path / "traj_0.h5", scale=1.0)
    output = tmp_path / "summary.csv"
    rows = write_trajectory_summary(tmp_path, output)
    assert len(rows) == 1
    assert np.isnan(rows[0]["path_CI"])
    assert output.read_text(encoding="utf-8").startswith("trajectory_file,random_seed")
