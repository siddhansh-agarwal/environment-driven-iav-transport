from __future__ import annotations

import numpy as np
import pandas as pd


def _trajectory_groups(trajectories: pd.DataFrame):
    identity = [
        column
        for column in ("date", "condition", "movie", "particle")
        if column in trajectories.columns
    ]
    if "movie" not in identity or "particle" not in identity:
        raise ValueError("trajectories must contain movie and particle columns")
    yield from trajectories.groupby(identity, sort=False)


def _event_identity(identity_columns: list[str], key) -> dict[str, str | int]:
    values = key if isinstance(key, tuple) else (key,)
    row = dict(zip(identity_columns, values))
    movie = str(row["movie"])
    particle = int(row["particle"])
    row["movie"] = movie
    row["particle"] = particle
    row["track_uid"] = f"{movie}::{particle}"
    if "date" in row and "condition" in row:
        row["global_track_uid"] = (
            f"{row['date']}::{row['condition']}::{movie}::{particle}"
        )
    return row


def contiguous_runs(group: pd.DataFrame) -> list[pd.DataFrame]:
    group = group.sort_values("frame").reset_index(drop=True)
    if len(group) == 0:
        return []
    frames = group["frame"].to_numpy(int)
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, len(group)]
    return [
        group.iloc[start:stop].reset_index(drop=True)
        for start, stop in zip(starts, stops)
        if stop - start >= 2
    ]


def exact_frame_events(trajectories: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    if int(lag) != 1:
        raise ValueError("exact-frame events use consecutive frames (lag=1)")
    rows = []
    identity_columns = [
        column
        for column in ("date", "condition", "movie", "particle")
        if column in trajectories.columns
    ]
    for key, group in _trajectory_groups(trajectories):
        identity = _event_identity(identity_columns, key)
        group = group.sort_values("frame").reset_index(drop=True)
        frame = group["frame"].to_numpy(int)
        x = group["x_um"].to_numpy(float)
        y = group["y_um"].to_numpy(float)
        t = group["time_s"].to_numpy(float)
        for i in range(len(group) - 1):
            if frame[i + 1] - frame[i] != 1:
                continue
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            disp = float(np.hypot(dx, dy))
            if disp <= 0:
                continue
            rows.append(
                {
                    **identity,
                    "event_family": "exact_frame",
                    "event_param_um": np.nan,
                    "frame_start": int(frame[i]),
                    "frame_end": int(frame[i + 1]),
                    "time_start_s": float(t[i]),
                    "time_end_s": float(t[i + 1]),
                    "time_interval_s": float(t[i + 1] - t[i]),
                    "p1_x_um": float(x[i]),
                    "p1_y_um": float(y[i]),
                    "p2_x_um": float(x[i + 1]),
                    "p2_y_um": float(y[i + 1]),
                    "dx_um": float(dx),
                    "dy_um": float(dy),
                    "displacement_um": disp,
                    "max_internal_frame_gap": lag,
                }
            )
    return pd.DataFrame(rows)


def first_passage_events(
    trajectories: pd.DataFrame, body_length_um: float
) -> pd.DataFrame:
    rows = []
    identity_columns = [
        column
        for column in ("date", "condition", "movie", "particle")
        if column in trajectories.columns
    ]
    for key, group in _trajectory_groups(trajectories):
        identity = _event_identity(identity_columns, key)
        for run_id, run in enumerate(contiguous_runs(group)):
            frame = run["frame"].to_numpy(int)
            x = run["x_um"].to_numpy(float)
            y = run["y_um"].to_numpy(float)
            t = run["time_s"].to_numpy(float)
            i = 0
            while i < len(run) - 1:
                found = False
                for j in range(i + 1, len(run)):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    disp = float(np.hypot(dx, dy))
                    if disp >= body_length_um:
                        rows.append(
                            {
                                **identity,
                                "event_family": "first_passage",
                                "event_param_um": float(body_length_um),
                                "contiguous_run_id": int(run_id),
                                "frame_start": int(frame[i]),
                                "frame_end": int(frame[j]),
                                "time_start_s": float(t[i]),
                                "time_end_s": float(t[j]),
                                "time_interval_s": float(t[j] - t[i]),
                                "p1_x_um": float(x[i]),
                                "p1_y_um": float(y[i]),
                                "p2_x_um": float(x[j]),
                                "p2_y_um": float(y[j]),
                                "dx_um": float(dx),
                                "dy_um": float(dy),
                                "displacement_um": disp,
                                "max_internal_frame_gap": 1,
                            }
                        )
                        i = j
                        found = True
                        break
                if not found:
                    i += 1
    return pd.DataFrame(rows)


def first_passage_thresholds(config: dict) -> list[float]:
    thresholds = config["physics"].get("first_passage_displacement_thresholds_um")
    if thresholds is None:
        thresholds = [config["physics"]["body_length_um"]]
    return sorted({float(value) for value in thresholds})


def build_event_table(trajectories: pd.DataFrame, config: dict) -> pd.DataFrame:
    exact = exact_frame_events(
        trajectories, lag=int(config["physics"]["exact_frame_lag"])
    )
    first_passages = [
        first_passage_events(trajectories, body_length_um=threshold)
        for threshold in first_passage_thresholds(config)
    ]
    events = pd.concat([exact, *first_passages], ignore_index=True)
    events["event_uid"] = np.arange(len(events), dtype=int)
    return events


def expand_analysis_events(events: pd.DataFrame, config: dict) -> pd.DataFrame:
    frames = []
    for threshold in config["physics"]["exact_displacement_thresholds_um"]:
        sub = events.loc[
            (events["event_family"] == "exact_frame")
            & (events["displacement_um"] >= float(threshold))
        ].copy()
        sub["analysis"] = f"exact_ge_{float(threshold):g}um"
        sub["analysis_family"] = "exact_frame_threshold"
        frames.append(sub)
    for threshold in first_passage_thresholds(config):
        first = events.loc[
            (events["event_family"] == "first_passage")
            & np.isclose(events["event_param_um"], threshold)
        ].copy()
        first["analysis"] = f"first_passage_{threshold:g}um"
        first["analysis_family"] = "first_passage"
        frames.append(first)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
