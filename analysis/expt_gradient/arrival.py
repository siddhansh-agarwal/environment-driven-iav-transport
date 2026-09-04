"""Local first-arrival measurements for experimental virion trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def add_gradient_frame_displacements(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve each displacement parallel and perpendicular to its local gradient.

    Gradient components are evaluated at the start of each displacement by
    :func:`analysis.expt_gradient.gradients.add_event_gradients`.
    """

    required = {
        "dx_um",
        "dy_um",
        "grad_unit_x",
        "grad_unit_y",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events is missing required columns: {sorted(missing)}")
    result = events.copy()
    dx = pd.to_numeric(result["dx_um"], errors="coerce").to_numpy(float)
    dy = pd.to_numeric(result["dy_um"], errors="coerce").to_numpy(float)
    gx = pd.to_numeric(result["grad_unit_x"], errors="coerce").to_numpy(float)
    gy = pd.to_numeric(result["grad_unit_y"], errors="coerce").to_numpy(float)
    parallel = dx * gx + dy * gy
    result["gradient_parallel_displacement_um"] = parallel
    result["gradient_perpendicular_displacement_um"] = dx * (-gy) + dy * gx
    return result


def local_first_arrival_by_track(
    events: pd.DataFrame,
    *,
    receptor_contrast_threshold: float,
    arrival_distance_um: float,
    group_columns: Sequence[str] = ("date", "condition", "movie", "track_uid"),
) -> pd.DataFrame:
    """Measure signed first arrival within contiguous high-contrast episodes.

    Each episode accumulates displacement along the local receptor-gradient
    direction until it first reaches ``+arrival_distance_um`` (denser side) or
    ``-arrival_distance_um`` (sparser side). Dense and sparse arrivals
    contribute ``+1 / t_hit`` and ``-1 / t_hit``; episodes reaching neither
    side contribute zero. Scores are averaged first within each trajectory.
    """

    required = {
        *group_columns,
        "frame_start",
        "frame_end",
        "time_interval_s",
        "particle_scale_receptor_contrast_percent",
        "gradient_parallel_displacement_um",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events is missing required columns: {sorted(missing)}")
    distance = float(arrival_distance_um)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("arrival_distance_um must be finite and positive")

    selected = events.loc[
        pd.to_numeric(
            events["particle_scale_receptor_contrast_percent"], errors="coerce"
        ).ge(float(receptor_contrast_threshold))
    ].copy()
    output_columns = [
        *group_columns,
        "dense_first_arrival_score_s",
        "sparse_first_arrival_score_s",
        "net_first_arrival_score_s",
        "dense_first_arrival_fraction",
        "sparse_first_arrival_fraction",
        "n_episodes",
    ]
    if selected.empty:
        return pd.DataFrame(columns=output_columns)

    sort_columns = [*group_columns, "frame_start"]
    if "time_start_s" in selected.columns:
        sort_columns.append("time_start_s")
    selected = selected.sort_values(sort_columns)
    rows: list[dict[str, float | int | str]] = []
    for key, group in selected.groupby(list(group_columns), sort=False):
        starts = pd.to_numeric(group["frame_start"], errors="coerce").to_numpy(int)
        ends = pd.to_numeric(group["frame_end"], errors="coerce").to_numpy(int)
        breaks = np.flatnonzero(starts[1:] != ends[:-1]) + 1
        episode_starts = np.r_[0, breaks]
        episode_ends = np.r_[breaks, len(group)]
        parallel = pd.to_numeric(
            group["gradient_parallel_displacement_um"], errors="coerce"
        ).to_numpy(float)
        intervals = pd.to_numeric(group["time_interval_s"], errors="coerce").to_numpy(
            float
        )

        dense_scores: list[float] = []
        sparse_scores: list[float] = []
        for start, stop in zip(episode_starts, episode_ends):
            displacement = np.cumsum(parallel[start:stop])
            elapsed = np.cumsum(intervals[start:stop])
            dense_hits = np.flatnonzero(displacement >= distance)
            sparse_hits = np.flatnonzero(displacement <= -distance)
            dense_index = int(dense_hits[0]) if dense_hits.size else None
            sparse_index = int(sparse_hits[0]) if sparse_hits.size else None
            dense_first = dense_index is not None and (
                sparse_index is None or dense_index <= sparse_index
            )
            sparse_first = sparse_index is not None and (
                dense_index is None or sparse_index < dense_index
            )
            if dense_first and elapsed[dense_index] > 0.0:
                dense_scores.append(1.0 / float(elapsed[dense_index]))
                sparse_scores.append(0.0)
            elif sparse_first and elapsed[sparse_index] > 0.0:
                dense_scores.append(0.0)
                sparse_scores.append(1.0 / float(elapsed[sparse_index]))
            else:
                dense_scores.append(0.0)
                sparse_scores.append(0.0)

        key_values = key if isinstance(key, tuple) else (key,)
        row: dict[str, float | int | str] = dict(zip(group_columns, key_values))
        dense = np.asarray(dense_scores, dtype=float)
        sparse = np.asarray(sparse_scores, dtype=float)
        row.update(
            {
                "dense_first_arrival_score_s": float(np.mean(dense)),
                "sparse_first_arrival_score_s": float(np.mean(sparse)),
                "net_first_arrival_score_s": float(np.mean(dense - sparse)),
                "dense_first_arrival_fraction": float(np.mean(dense > 0.0)),
                "sparse_first_arrival_fraction": float(np.mean(sparse > 0.0)),
                "n_episodes": int(dense.size),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


def summarize_first_arrival(
    track_scores: pd.DataFrame,
    *,
    replicate_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average arrival scores from trajectories to recordings and replicates."""

    value_columns = [
        "dense_first_arrival_score_s",
        "sparse_first_arrival_score_s",
        "net_first_arrival_score_s",
    ]
    required = {replicate_column, "condition", "movie", "track_uid", *value_columns}
    missing = required.difference(track_scores.columns)
    if missing:
        raise ValueError(f"track_scores is missing required columns: {sorted(missing)}")
    recording = track_scores.groupby(
        [replicate_column, "condition", "movie"], as_index=False
    )[value_columns].mean()
    replicate = recording.groupby([replicate_column, "condition"], as_index=False)[
        value_columns
    ].mean()
    return recording, replicate
