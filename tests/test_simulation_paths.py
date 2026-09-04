from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.run_demo import CONFIGS, EXPECTED, validate_output
from src.runner3d import PROFILES, calculation_type, load_parameter_file, run_ensemble
from src.storage3d import parameter_directory, trajectory_is_complete


REFERENCE_HASHES = {
    "uniform-3d": {
        "trajectory/times": "fcb5000d097e0fefd16de9fd18d7d9d0a383334521dbbf469c0634f54fdf56e0",
        "trajectory/positions": "fbd994ff2f3cceb70a0d7ac71fc2ced6a67c6da05492135ea52efdbe032038a1",
        "substrate_final/cleaved_ix": "073857cc2ac6f2957879c93f3df7af7c13760e69134281a9c773917c23b1ba7a",
    },
    "gradient-3d": {
        "trajectory/times": "4b5fe8fdfa41cfc56072066d4062b70c0c506b2095b71c31b0aad5bba4a8689c",
        "trajectory/positions": "def74e22b9d49f8b0c717ba07f4aa2d838acb14d58d08c029d452e7eada9fe6f",
        "substrate_final/cleaved_ix": "efc83fe4d91aaecde7ec6c9f9ffbcb3b59893a0719db39ff1ab102fb0557a53c",
    },
    "surface-2d": {
        "trajectory/times": "affc69c87ccdc34f5ca94b52e9421e49dec02d371fcf0767a283192c99a8c331",
        "trajectory/positions": "443aff3c8b1f2881d59950acaa2bafbe9991e6d32615d6b168f96608b35ac0c7",
        "substrate_final/cleaved_ix": "bbfcd8bd5959ba65a561546c6cc70fa1a4d5ce944aa1a78b272607eabe4f75b4",
    },
}


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def test_profiles_fix_the_three_manuscript_routes():
    assert PROFILES["uniform-3d"].grid_backend == "uniform_sparse_coords"
    assert PROFILES["gradient-3d"].grid_backend == "gradient_sparse_coords"
    assert PROFILES["surface-2d"].dimension == "2d"
    assert PROFILES["uniform-3d"].motion_rule == "adaptive_brownian_reaction"
    assert PROFILES["gradient-3d"].motion_rule == "adaptive_brownian_reaction"
    assert PROFILES["surface-2d"].motion_rule == "athermal_event_driven"


def test_calculation_type_rejects_a_2d_gradient():
    assert calculation_type({"DIMENSION": "3d", "GRADIENT_SCALE": 1.0}) == "uniform-3d"
    assert (
        calculation_type(
            {"DIMENSION": "3d", "GRADIENT_TYPE": "z", "GRADIENT_SCALE": 1.001}
        )
        == "gradient-3d"
    )
    with pytest.raises(ValueError, match="2D model only on uniform surfaces"):
        calculation_type(
            {"DIMENSION": "2d", "GRADIENT_TYPE": "x", "GRADIENT_SCALE": 1.01}
        )


def test_background_diffusivities_identify_distinct_output_conditions(tmp_path: Path):
    config = load_parameter_file(CONFIGS["uniform-3d"])[0]
    weaker_background = dict(config)
    weaker_background["THERMAL_PASSIVE_D_PARALLEL"] *= 0.1
    assert parameter_directory(tmp_path, config) != parameter_directory(
        tmp_path, weaker_background
    )
    changed_boundary = dict(config)
    changed_boundary["GRADIENT_MIN_SPACING_STOP"] = 0.11
    assert parameter_directory(tmp_path, config) != parameter_directory(
        tmp_path, changed_boundary
    )


def test_existing_trajectory_is_reused_only_for_the_same_configuration(
    tmp_path: Path,
):
    config = load_parameter_file(CONFIGS["gradient-3d"])[0]
    run_ensemble(
        [config],
        n_trajectories=1,
        output_dir=tmp_path,
        n_jobs=1,
        overwrite=True,
    )
    path = next(tmp_path.rglob("traj_0.h5"))

    from src.runner3d import _runtime_configuration, _validate_configuration

    runtime = _runtime_configuration(config, _validate_configuration(config))
    assert trajectory_is_complete(path, float(config["T_FINAL"]), runtime, 256)
    changed = dict(runtime)
    changed["GRADIENT_MIN_SPACING_STOP"] = 0.11
    assert not trajectory_is_complete(path, float(config["T_FINAL"]), changed, 256)
    assert not trajectory_is_complete(path, float(config["T_FINAL"]), runtime, 512)


@pytest.mark.parametrize("case", ["uniform-3d", "gradient-3d", "surface-2d"])
def test_manuscript_route_matches_frozen_reference(case: str, tmp_path: Path):
    """Guard against changes to the numerical route or seeded trajectory."""
    run_ensemble(
        load_parameter_file(CONFIGS[case]),
        n_trajectories=1,
        output_dir=tmp_path / case,
        n_jobs=1,
        overwrite=True,
    )
    validate_output(case, tmp_path / case, expected_files=1)
    path = next((tmp_path / case).rglob("traj_0.h5"))
    with h5py.File(path, "r") as trajectory:
        for dataset, expected_hash in REFERENCE_HASHES[case].items():
            assert _digest(trajectory[dataset][:]) == expected_hash
        expected = EXPECTED[case]
        assert trajectory.attrs["dimension"] == expected["dimension"]
        assert trajectory.attrs["grid_backend"] == expected["backend"]
