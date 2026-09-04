from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.storage3d import compress_trajectory_3d  # noqa: E402


def _make_positions(xyz: np.ndarray, orient: np.ndarray | None = None) -> np.ndarray:
    if orient is None:
        orient = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (len(xyz), 1))
    return np.concatenate((xyz, orient), axis=1)


def test_compress_trajectory_3d_preserves_large_spatial_bends():
    seg1 = np.column_stack((np.linspace(0.0, 10.0, 60), np.zeros(60), np.zeros(60)))
    seg2 = np.column_stack(
        (np.full(60, 10.0), np.linspace(0.0, 10.0, 60), np.zeros(60))
    )
    seg3 = np.column_stack(
        (np.linspace(10.0, 20.0, 60), np.full(60, 10.0), np.zeros(60))
    )
    xyz = np.vstack((seg1[:-1], seg2[:-1], seg3))
    times = np.linspace(0.0, 1.0, len(xyz), dtype=np.float64)
    positions = _make_positions(xyz)

    t_comp, p_comp = compress_trajectory_3d(
        times, positions, max_points=8, min_uniform_points=2
    )

    assert len(t_comp) <= 8
    np.testing.assert_allclose(p_comp[0, :3], xyz[0])
    np.testing.assert_allclose(p_comp[-1, :3], xyz[-1])

    # Both 90-degree corners should survive compression.
    assert np.any(
        np.linalg.norm(p_comp[:, :3] - np.array([10.0, 0.0, 0.0]), axis=1) < 0.6
    )
    assert np.any(
        np.linalg.norm(p_comp[:, :3] - np.array([10.0, 10.0, 0.0]), axis=1) < 0.6
    )


def test_compress_trajectory_3d_preserves_large_orientation_turns():
    n = 120
    times = np.linspace(0.0, 1.0, n, dtype=np.float64)
    xyz = np.zeros((n, 3), dtype=np.float64)
    theta = np.concatenate(
        (
            np.zeros(40),
            np.linspace(0.0, np.pi / 2.0, 40),
            np.full(40, np.pi / 2.0),
        )
    )
    orient = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(n)))
    positions = _make_positions(xyz, orient)

    t_comp, p_comp = compress_trajectory_3d(
        times,
        positions,
        max_points=10,
        disp_threshold=10.0,
        angular_threshold=0.25,
        min_uniform_points=2,
    )

    assert len(t_comp) <= 10
    np.testing.assert_allclose(p_comp[0, 3:6], orient[0], atol=1e-12)
    np.testing.assert_allclose(p_comp[-1, 3:6], orient[-1], atol=1e-12)

    # The quarter-turn should force at least one interior orientation point.
    assert len(t_comp) >= 3
    assert np.any(np.abs(p_comp[:, 3]) < 0.8)


def test_compress_trajectory_3d_keeps_uniform_support_for_smooth_motion():
    n = 200
    times = np.linspace(0.0, 10.0, n, dtype=np.float64)
    xyz = np.column_stack((np.linspace(0.0, 100.0, n), np.zeros(n), np.zeros(n)))
    positions = _make_positions(xyz)

    t_comp, p_comp = compress_trajectory_3d(
        times, positions, max_points=32, min_uniform_points=16
    )

    assert len(t_comp) == 16
    np.testing.assert_allclose(p_comp[0, :3], xyz[0])
    np.testing.assert_allclose(p_comp[-1, :3], xyz[-1])
