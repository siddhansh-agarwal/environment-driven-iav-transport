"""Trajectory-reversal control for experimental step alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trajectory_reversal_null(
    track_ci: pd.DataFrame,
    *,
    iterations: int = 20_000,
    seed: int = 1_618,
    replicate_column: str = "date",
) -> pd.DataFrame:
    """Independently reverse each trajectory's directional score at random.

    Each iteration follows the experimental hierarchy: trajectories are
    averaged within recordings, then recordings within biological replicates,
    and finally replicates within each condition.
    """

    required = {replicate_column, "condition", "movie", "step_ci"}
    missing = required.difference(track_ci.columns)
    if missing:
        raise ValueError(f"track_ci is missing required columns: {sorted(missing)}")
    n_iterations = int(iterations)
    if n_iterations < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    base = track_ci.reset_index(drop=True).copy()
    values = base["step_ci"].to_numpy(float)
    rows: list[pd.DataFrame] = []
    for iteration in range(n_iterations):
        reversed_scores = base[[replicate_column, "condition", "movie"]].copy()
        reversed_scores["step_ci"] = values * rng.choice((-1.0, 1.0), len(values))
        recording = reversed_scores.groupby(
            [replicate_column, "condition", "movie"], as_index=False
        )["step_ci"].mean()
        replicate = recording.groupby([replicate_column, "condition"], as_index=False)[
            "step_ci"
        ].mean()
        condition = replicate.groupby("condition", as_index=False)["step_ci"].mean()
        condition["iteration"] = iteration
        rows.append(condition)
    return pd.concat(rows, ignore_index=True)
