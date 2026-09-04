from pathlib import Path

import h5py
import numpy as np
import pytest

from analysis.simulation_observables import (
    body_axis_alignment,
    effective_diffusivity_from_msd,
    ensemble_msd,
    load_stored_path,
    one_over_e_crossing,
    orientation_correlation,
    terminal_range,
)


def _write_path(path: Path, *, dimension: str = "3d", stop: float = 3.0) -> None:
    times = np.arange(0.0, stop + 1.0)
    with h5py.File(path, "w") as h5:
        h5.attrs["dimension"] = dimension
        h5.attrs["completion_reason"] = "t_final"
        trajectory = h5.create_group("trajectory")
        trajectory.create_dataset("times", data=times)
        if dimension == "3d":
            xyz = np.column_stack((times, np.zeros_like(times), np.zeros_like(times)))
            axes = np.tile([1.0, 0.0, 0.0], (len(times), 1))
            states = np.column_stack((xyz, axes))
        else:
            states = np.column_stack((times, np.zeros_like(times), np.zeros_like(times)))
        trajectory.create_dataset("positions", data=states)


def test_stored_path_observables(tmp_path):
    file = tmp_path / "traj_0.h5"
    _write_path(file)
    path = load_stored_path(file)
    assert terminal_range(path) == pytest.approx(3.0)
    assert body_axis_alignment(path) == pytest.approx(1.0)
    assert np.allclose(orientation_correlation(path, apolar=False), 1.0)
    assert np.allclose(orientation_correlation(path, apolar=True), 1.0)


def test_ensemble_msd_distinguishes_truncated_and_terminal_paths(tmp_path):
    first = tmp_path / "traj_0.h5"
    second = tmp_path / "traj_1.h5"
    _write_path(first, stop=2.0)
    _write_path(second, stop=3.0)
    paths = [load_stored_path(first), load_stored_path(second)]
    times = np.arange(4.0)
    truncated = ensemble_msd(paths, times)
    held = ensemble_msd(paths, times, hold_after_end=True)
    assert np.array_equal(truncated["trajectory_count"], [2, 2, 2, 1])
    assert np.array_equal(held["trajectory_count"], [2, 2, 2, 2])
    assert held["msd"][-1] == pytest.approx((4.0 + 9.0) / 2.0)


def test_diffusivity_fit_and_one_over_e_crossing():
    times = np.linspace(0.0, 10.0, 11)
    result = effective_diffusivity_from_msd(
        times,
        6.0 * 0.25 * times + 2.0,
        dimension=3,
        fit_start=2.0,
        fit_stop=9.0,
    )
    assert result["effective_diffusivity"] == pytest.approx(0.25)
    assert result["r_squared"] == pytest.approx(1.0)
    correlation = np.exp(-times / 4.0)
    assert one_over_e_crossing(times, correlation) == pytest.approx(4.0)
