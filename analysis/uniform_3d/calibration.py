"""Calibrate the exposure factor shared by the analytical calculations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .mean_field import Inputs, load_geometries, load_inputs, predict


def load_uniform_calibration_points(source_dir: Path) -> pd.DataFrame:
    """Load the unique nonzero-cleavage states in the two uniform sweeps.

    Two molecular states appear in both displayed sweeps.  They are retained
    once here so that repeated plotting of a state does not give it extra
    weight in the calibration.
    """

    cleavage = pd.read_csv(source_dir / "panel_f_cleavage_simulation.csv")
    cleavage = cleavage.loc[cleavage["K_C"].gt(0.0)].copy()
    cleavage["calibration_sweep"] = "cleavage_strength"

    avidity = pd.read_csv(source_dir / "panel_g_mean_attachments_simulation.csv")
    avidity = avidity.copy()
    avidity["K_C"] = avidity["K_C_with_cleavage"]
    avidity["calibration_sweep"] = "avidity_and_composition"
    columns = [
        "architecture",
        "n_binders",
        "K_D",
        "K_C",
        "diffusivity_shift",
        "calibration_sweep",
    ]
    points = pd.concat([cleavage[columns], avidity[columns]], ignore_index=True)
    state_columns = ["architecture", "n_binders", "K_D", "K_C"]
    spread = points.groupby(state_columns)["diffusivity_shift"].agg(
        ["min", "max"]
    )
    if not np.allclose(spread["min"], spread["max"], atol=1.0e-12, rtol=0.0):
        raise ValueError("Repeated calibration states have inconsistent values")
    return (
        points.sort_values(state_columns + ["calibration_sweep"])
        .drop_duplicates(state_columns, keep="first")
        .reset_index(drop=True)
    )


def predicted_shifts(
    points: pd.DataFrame,
    exposure_factor: float,
    inputs: Inputs | None = None,
) -> np.ndarray:
    """Evaluate the uniform mean field at all calibration points."""

    if not np.isfinite(exposure_factor) or exposure_factor <= 0.0:
        raise ValueError("exposure_factor must be finite and positive")
    base = load_inputs() if inputs is None else inputs
    current = replace(base, cleavage_exposure_factor=float(exposure_factor))
    geometries = load_geometries(current)
    return np.asarray(
        [
            predict(
                str(row.architecture),
                float(row.K_D),
                float(row.K_C),
                int(row.n_binders),
                current,
                geometries,
            )["diffusivity_shift"]
            for row in points.itertuples(index=False)
        ],
        dtype=float,
    )


def exposure_factor_audit(
    points: pd.DataFrame,
    selected_factor: float = 0.002,
) -> dict[str, float | int]:
    """Compare the rounded shared factor with the global least-squares value."""

    observed = points["diffusivity_shift"].to_numpy(float)

    def rmse(log_factor: float) -> float:
        predicted = predicted_shifts(points, float(np.exp(log_factor)))
        return float(np.sqrt(np.mean((predicted - observed) ** 2)))

    optimum = minimize_scalar(
        rmse,
        bounds=(np.log(1.0e-4), np.log(2.0e-2)),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    selected_prediction = predicted_shifts(points, selected_factor)
    selected_rmse = rmse(np.log(selected_factor))
    best_rmse = float(optimum.fun)
    return {
        "point_count": int(len(points)),
        "selected_factor": float(selected_factor),
        "least_squares_factor": float(np.exp(optimum.x)),
        "selected_rmse": selected_rmse,
        "least_squares_rmse": best_rmse,
        "rmse_ratio": selected_rmse / best_rmse,
        "correlation": float(np.corrcoef(observed, selected_prediction)[0, 1]),
    }
