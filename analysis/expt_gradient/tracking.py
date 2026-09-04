from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import trackpy as tp
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

from .io import (
    MovieMetadata,
    MovieRecord,
    read_channel_frame,
    read_metadata,
    iter_channel_frames,
)


TRACKING_PARAMETER_KEYS = (
    "radius_pixels",
    "quality_threshold",
    "normalize_quality_per_frame",
    "background_radius_pixels",
    "min_spot_snr",
    "spot_snr_center_radius_pixels",
    "spot_snr_ring_inner_radius_pixels",
    "spot_snr_ring_outer_radius_pixels",
    "search_range_pixels",
    "adaptive_stop_pixels",
    "adaptive_step",
    "link_strategy",
    "neighbor_strategy",
    "memory_frames",
    "stub_length_detections",
    "fragment_reconnect_enabled",
    "fragment_reconnect_max_frame_gap",
    "fragment_reconnect_max_step_body_lengths",
    "fragment_reconnect_min_fragment_detections",
    "fragment_reconnect_min_candidate_margin_pixels",
    "fragment_reconnect_max_prediction_error_pixels",
    "fragment_reconnect_min_velocity_cosine",
    "fragment_reconnect_min_quality_quantile",
    "fragment_reconnect_max_exact_fraction_drop",
    "fragment_reconnect_max_near_search_increase",
    "fragment_reconnect_min_rows_gain_fraction",
    "min_distance_pixels",
    "edge_margin_pixels",
    "detection_threshold_probe_frame",
    "detection_threshold_probe_values",
)


def tracking_params_for_movie(config: dict, movie: str) -> dict:
    params = dict(config["tracking"])
    overrides = config.get("tracking_overrides", {}) or {}
    movie_overrides = (overrides.get("movies", {}) or {}).get(str(movie), {}) or {}
    params.update(movie_overrides)
    return params


def applied_tracking_parameter_row(config: dict, movie: str) -> dict:
    params = tracking_params_for_movie(config, movie)
    row = {"movie": str(movie)}
    for key in TRACKING_PARAMETER_KEYS:
        if key in params:
            value = params[key]
            row[key] = (
                "|".join(str(v) for v in value) if isinstance(value, list) else value
            )
    policy = config.get("tracking_policy", {}) or {}
    if policy:
        row["tracking_policy_name"] = policy.get("name", "")
        row["tracking_policy_rationale"] = policy.get("rationale", "")
    return row


def max_pairwise_distance_um(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return np.nan
    dx = xy[:, 0, None] - xy[None, :, 0]
    dy = xy[:, 1, None] - xy[None, :, 1]
    return float(np.sqrt(np.max(dx * dx + dy * dy)))


def max_contiguous_run_span_um(group: pd.DataFrame) -> float:
    group = group.sort_values("frame").reset_index(drop=True)
    if len(group) < 2:
        return np.nan
    frames = group["frame"].to_numpy(int)
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, len(group)]
    spans = []
    for start, stop in zip(starts, stops):
        if stop - start < 2:
            continue
        xy = group.iloc[start:stop][["x_um", "y_um"]].to_numpy(float)
        spans.append(max_pairwise_distance_um(xy))
    return float(np.nanmax(spans)) if spans else np.nan


def motility_metrics_from_trajectories(
    trajectories: pd.DataFrame, config: dict
) -> pd.DataFrame:
    body_length_um = float(config["physics"]["body_length_um"])
    rows = []
    for track_uid, group in trajectories.groupby("track_uid", sort=False):
        group = group.sort_values("frame")
        xy = group[["x_um", "y_um"]].to_numpy(float)
        full_span = max_pairwise_distance_um(xy)
        contiguous_span = max_contiguous_run_span_um(group)
        rows.append(
            {
                "track_uid": track_uid,
                "track_span_um": full_span,
                "contiguous_track_span_um": contiguous_span,
                "track_span_body_lengths": full_span / body_length_um
                if np.isfinite(full_span)
                else np.nan,
                "contiguous_track_span_body_lengths": contiguous_span / body_length_um
                if np.isfinite(contiguous_span)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def augment_track_motility_metrics(
    track_qc: pd.DataFrame, trajectories: pd.DataFrame, config: dict
) -> pd.DataFrame:
    motility = motility_metrics_from_trajectories(trajectories, config)
    drop_cols = [
        col
        for col in motility.columns
        if col != "track_uid" and col in track_qc.columns
    ]
    if drop_cols:
        track_qc = track_qc.drop(columns=drop_cols)
    return track_qc.merge(motility, on="track_uid", how="left")


def hessian_detector(
    frame: np.ndarray,
    radius_pixels: float,
    background_radius_pixels: float,
    normalize: bool = True,
) -> np.ndarray:
    frame_float = frame.astype(np.float32)
    sigma_bg = background_radius_pixels / 3.0
    ksize_bg = int(6 * sigma_bg + 1)
    if ksize_bg % 2 == 0:
        ksize_bg += 1
    background = cv2.GaussianBlur(frame_float, (ksize_bg, ksize_bg), sigma_bg)
    frame_bg = np.maximum(frame_float - background, 0)

    sigma = radius_pixels / np.sqrt(2)
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    smooth = cv2.GaussianBlur(frame_bg, (ksize, ksize), sigma)
    grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    hxx = cv2.Sobel(grad_x, cv2.CV_32F, 1, 0, ksize=3)
    hxy = cv2.Sobel(grad_x, cv2.CV_32F, 0, 1, ksize=3)
    hyy = cv2.Sobel(grad_y, cv2.CV_32F, 0, 1, ksize=3)
    det = np.maximum((hxx * hyy - hxy**2) * sigma**4, 0)
    if normalize and det.max() > 0:
        det = det / det.max()
    return det.astype(np.float32)


def refine_peak_centroid(response: np.ndarray, y: int, x: int) -> tuple[float, float]:
    h, w = response.shape
    y0, y1 = max(0, y - 1), min(h, y + 2)
    x0, x1 = max(0, x - 1), min(w, x + 2)
    patch = response[y0:y1, x0:x1].astype(float)
    total = patch.sum()
    if total <= 0:
        return float(y), float(x)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((yy * patch).sum() / total), float((xx * patch).sum() / total)


def spot_intensity_features(
    frame: np.ndarray, y: float, x: float, params: dict
) -> dict[str, float]:
    center_radius = float(
        params.get(
            "spot_snr_center_radius_pixels",
            max(1.5, 0.55 * float(params["radius_pixels"])),
        )
    )
    ring_inner = float(
        params.get("spot_snr_ring_inner_radius_pixels", float(params["radius_pixels"]))
    )
    ring_outer = float(
        params.get(
            "spot_snr_ring_outer_radius_pixels", 2.4 * float(params["radius_pixels"])
        )
    )
    h, w = frame.shape
    y0 = max(0, int(np.floor(y - ring_outer)))
    y1 = min(h, int(np.ceil(y + ring_outer)) + 1)
    x0 = max(0, int(np.floor(x - ring_outer)))
    x1 = min(w, int(np.ceil(x + ring_outer)) + 1)
    patch = frame[y0:y1, x0:x1].astype(float)
    if patch.size == 0:
        return {
            "spot_snr": np.nan,
            "spot_signal": np.nan,
            "spot_ring_median": np.nan,
            "spot_ring_sigma": np.nan,
        }
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(yy - y, xx - x)
    center = patch[dist <= center_radius]
    ring = patch[(dist >= ring_inner) & (dist <= ring_outer)]
    if center.size == 0 or ring.size < 8:
        return {
            "spot_snr": np.nan,
            "spot_signal": np.nan,
            "spot_ring_median": np.nan,
            "spot_ring_sigma": np.nan,
        }
    ring_median = float(np.median(ring))
    ring_mad = float(np.median(np.abs(ring - ring_median)))
    ring_sigma = 1.4826 * ring_mad if ring_mad > 0 else float(np.std(ring))
    signal = float(np.mean(center) - ring_median)
    snr = signal / (ring_sigma + 1e-6)
    return {
        "spot_snr": float(snr),
        "spot_signal": signal,
        "spot_ring_median": ring_median,
        "spot_ring_sigma": float(ring_sigma),
    }


def detect_frame(frame: np.ndarray, frame_index: int, params: dict) -> pd.DataFrame:
    response = hessian_detector(
        frame,
        radius_pixels=float(params["radius_pixels"]),
        background_radius_pixels=float(params["background_radius_pixels"]),
        normalize=bool(params["normalize_quality_per_frame"]),
    )
    coords = peak_local_max(
        response,
        min_distance=int(params["min_distance_pixels"]),
        threshold_abs=float(params["quality_threshold"]),
        exclude_border=int(params["radius_pixels"]),
    )
    rows = []
    min_spot_snr = params.get("min_spot_snr")
    min_spot_snr = None if min_spot_snr is None else float(min_spot_snr)
    for y, x in coords:
        y_ref, x_ref = refine_peak_centroid(response, int(y), int(x))
        spot = spot_intensity_features(frame, y_ref, x_ref, params)
        if min_spot_snr is not None and np.isfinite(min_spot_snr):
            if not np.isfinite(spot["spot_snr"]) or spot["spot_snr"] < min_spot_snr:
                continue
        rows.append(
            {
                "frame": int(frame_index),
                "y": y_ref,
                "x": x_ref,
                "quality": float(response[int(y), int(x)]),
                **spot,
            }
        )
    return pd.DataFrame(rows)


def detect_movie(
    record: MovieRecord,
    metadata: MovieMetadata,
    config: dict,
    params: dict | None = None,
) -> pd.DataFrame:
    params = (
        tracking_params_for_movie(config, record.movie) if params is None else params
    )
    frames = []
    for frame_index, frame in iter_channel_frames(
        record, config["experiment"]["tracking_channel"]
    ):
        detected = detect_frame(frame, frame_index, params)
        if not detected.empty:
            frames.append(detected)
    if not frames:
        return pd.DataFrame(
            columns=["frame", "y", "x", "quality", "movie", "x_um", "y_um"]
        )
    detections = pd.concat(frames, ignore_index=True)
    detections["movie"] = record.movie
    detections["x_um"] = detections["x"] * metadata.pixel_size_um
    detections["y_um"] = detections["y"] * metadata.pixel_size_um
    return detections.sort_values(["frame", "y", "x"]).reset_index(drop=True)


def _trackpy_link_unfiltered(detections: pd.DataFrame, params: dict) -> pd.DataFrame:
    tp.quiet()
    link_kwargs = {
        "search_range": float(params["search_range_pixels"]),
        "memory": int(params["memory_frames"]),
    }
    adaptive_stop = params.get("adaptive_stop_pixels")
    if adaptive_stop is not None:
        link_kwargs["adaptive_stop"] = float(adaptive_stop)
        link_kwargs["adaptive_step"] = float(params.get("adaptive_step", 0.95))
    link_strategy = params.get("link_strategy")
    if link_strategy:
        link_kwargs["link_strategy"] = str(link_strategy)
    neighbor_strategy = params.get("neighbor_strategy")
    if neighbor_strategy:
        link_kwargs["neighbor_strategy"] = str(neighbor_strategy)
    return tp.link(detections[["x", "y", "frame", "quality"]].copy(), **link_kwargs)


def _velocity_pixels(
    group: pd.DataFrame, from_end: bool, n_steps: int = 3
) -> np.ndarray | None:
    group = group.sort_values("frame")
    if len(group) < 2:
        return None
    window = (
        group.tail(min(len(group), n_steps + 1))
        if from_end
        else group.head(min(len(group), n_steps + 1))
    )
    frames = window["frame"].to_numpy(float)
    xy = window[["x", "y"]].to_numpy(float)
    dt = frames[-1] - frames[0]
    if dt <= 0:
        return None
    return (xy[-1] - xy[0]) / dt


def _link_summary_pixels(
    linked: pd.DataFrame, search_pixels: float
) -> dict[str, float | int]:
    if linked.empty or "particle" not in linked:
        return {
            "rows": 0,
            "tracks": 0,
            "median_track_length": np.nan,
            "tracks_ge_50": 0,
            "tracks_ge_100": 0,
            "mean_exact_pair_fraction": np.nan,
            "near_search_fraction": np.nan,
        }
    lengths = linked.groupby("particle").size()
    exact_fractions = []
    near_search = []
    for _, group in linked.sort_values(["particle", "frame"]).groupby(
        "particle", sort=False
    ):
        if len(group) < 2:
            continue
        xy = group[["x", "y"]].to_numpy(float)
        frames = group["frame"].to_numpy(int)
        gaps = np.diff(frames)
        disp = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        if len(gaps):
            exact_fractions.append(float(np.mean(gaps == 1)))
        exact = disp[gaps == 1]
        if exact.size:
            near_search.extend((exact >= 0.8 * float(search_pixels)).tolist())
    return {
        "rows": int(len(linked)),
        "tracks": int(lengths.size),
        "median_track_length": float(lengths.median()) if lengths.size else np.nan,
        "tracks_ge_50": int((lengths >= 50).sum()),
        "tracks_ge_100": int((lengths >= 100).sum()),
        "mean_exact_pair_fraction": float(np.nanmean(exact_fractions))
        if exact_fractions
        else np.nan,
        "near_search_fraction": float(np.mean(near_search)) if near_search else np.nan,
    }


def _fragment_reconnect_candidates(
    linked: pd.DataFrame,
    metadata: MovieMetadata,
    config: dict,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if linked.empty or "particle" not in linked:
        return linked, pd.DataFrame()
    search_pixels = float(params["search_range_pixels"])
    body_length_um = float(config["physics"]["body_length_um"])
    max_step_pixels = (
        float(params.get("fragment_reconnect_max_step_body_lengths", 3.0))
        * body_length_um
        / float(metadata.pixel_size_um)
    )
    max_frame_gap = int(
        params.get("fragment_reconnect_max_frame_gap", int(params["memory_frames"]) + 1)
    )
    min_fragment_len = int(params.get("fragment_reconnect_min_fragment_detections", 5))
    min_margin = float(
        params.get("fragment_reconnect_min_candidate_margin_pixels", 2.0)
    )
    max_prediction_error = float(
        params.get("fragment_reconnect_max_prediction_error_pixels", search_pixels)
    )
    min_velocity_cosine = float(
        params.get("fragment_reconnect_min_velocity_cosine", -0.25)
    )
    min_quality_quantile = float(
        params.get("fragment_reconnect_min_quality_quantile", 0.25)
    )
    quality_floor = (
        float(linked["quality"].quantile(min_quality_quantile))
        if "quality" in linked and len(linked)
        else -np.inf
    )

    groups = {
        int(particle): group.sort_values("frame").copy()
        for particle, group in linked.groupby("particle", sort=False)
    }
    fragments = []
    for particle, group in groups.items():
        first = group.iloc[0]
        last = group.iloc[-1]
        fragments.append(
            {
                "particle": particle,
                "n": int(len(group)),
                "start_frame": int(first.frame),
                "end_frame": int(last.frame),
                "start_x": float(first.x),
                "start_y": float(first.y),
                "end_x": float(last.x),
                "end_y": float(last.y),
                "start_quality": float(first.quality),
                "end_quality": float(last.quality),
                "start_velocity": _velocity_pixels(group, from_end=False),
                "end_velocity": _velocity_pixels(group, from_end=True),
            }
        )
    fragment_table = pd.DataFrame(fragments)
    starts_by_frame: dict[int, list] = {}
    for row in fragment_table.itertuples(index=False):
        starts_by_frame.setdefault(int(row.start_frame), []).append(row)
    start_trees = {
        frame: (
            cKDTree(
                np.asarray([(row.start_x, row.start_y) for row in rows], dtype=float)
            ),
            rows,
        )
        for frame, rows in starts_by_frame.items()
    }

    candidate_rows = []
    for end in fragment_table.itertuples(index=False):
        if int(end.n) < min_fragment_len or float(end.end_quality) < quality_floor:
            continue
        for start_frame in range(
            int(end.end_frame) + 1, int(end.end_frame) + max_frame_gap + 1
        ):
            cached = start_trees.get(start_frame)
            if cached is None:
                continue
            gap = start_frame - int(end.end_frame)
            distance_limit = max(
                search_pixels,
                min(max_step_pixels * np.sqrt(gap), max_step_pixels * 1.5),
            )
            tree, start_rows = cached
            raw_candidates = []
            for idx in tree.query_ball_point([end.end_x, end.end_y], r=distance_limit):
                start = start_rows[idx]
                if int(start.particle) == int(end.particle):
                    continue
                if (
                    int(start.n) < min_fragment_len
                    or float(start.start_quality) < quality_floor
                ):
                    continue
                distance = float(
                    np.hypot(
                        float(start.start_x) - float(end.end_x),
                        float(start.start_y) - float(end.end_y),
                    )
                )
                prediction_error = np.nan
                prediction_ok = False
                if end.end_velocity is not None:
                    predicted = (
                        np.asarray([end.end_x, end.end_y], dtype=float)
                        + np.asarray(end.end_velocity, dtype=float) * gap
                    )
                    prediction_error = float(
                        np.hypot(
                            float(start.start_x) - predicted[0],
                            float(start.start_y) - predicted[1],
                        )
                    )
                    prediction_ok = prediction_error <= max(
                        max_prediction_error, search_pixels
                    )
                velocity_cosine = np.nan
                velocity_ok = True
                if end.end_velocity is not None and start.start_velocity is not None:
                    v1 = np.asarray(end.end_velocity, dtype=float)
                    v2 = np.asarray(start.start_velocity, dtype=float)
                    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                    if norm > 1e-9:
                        velocity_cosine = float(np.dot(v1, v2) / norm)
                        velocity_ok = velocity_cosine >= min_velocity_cosine
                if not velocity_ok:
                    continue
                if not (
                    distance <= search_pixels
                    or prediction_ok
                    or distance <= max_step_pixels
                ):
                    continue
                if int(end.n) + int(start.n) < int(params["stub_length_detections"]):
                    continue
                cost = prediction_error if np.isfinite(prediction_error) else distance
                raw_candidates.append(
                    {
                        "cost": float(cost),
                        "distance_pixels": distance,
                        "prediction_error_pixels": prediction_error,
                        "velocity_cosine": velocity_cosine,
                        "end_particle": int(end.particle),
                        "start_particle": int(start.particle),
                        "frame_gap": int(gap),
                    }
                )
            raw_candidates = sorted(raw_candidates, key=lambda item: item["cost"])
            if not raw_candidates:
                continue
            best = raw_candidates[0]
            second_cost = (
                raw_candidates[1]["cost"] if len(raw_candidates) > 1 else np.inf
            )
            best["second_best_cost"] = float(second_cost)
            best["candidate_margin_pixels"] = (
                float(second_cost - best["cost"])
                if np.isfinite(second_cost)
                else np.inf
            )
            best["n_candidates_same_end_frame_gap"] = int(len(raw_candidates))
            if best["candidate_margin_pixels"] >= min_margin:
                candidate_rows.append(best)
    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        return linked, candidates

    parent = {particle: particle for particle in groups}

    def find(particle: int) -> int:
        while parent[particle] != particle:
            parent[particle] = parent[parent[particle]]
            particle = parent[particle]
        return particle

    accepted_pairs = set()
    used_end = set()
    used_start = set()
    for row in candidates.sort_values("cost").itertuples(index=False):
        end_particle = int(row.end_particle)
        start_particle = int(row.start_particle)
        if end_particle in used_end or start_particle in used_start:
            continue
        root_end = find(end_particle)
        root_start = find(start_particle)
        if root_end == root_start:
            continue
        parent[root_start] = root_end
        used_end.add(end_particle)
        used_start.add(start_particle)
        accepted_pairs.add((end_particle, start_particle))
    candidates["accepted"] = [
        (int(row.end_particle), int(row.start_particle)) in accepted_pairs
        for row in candidates.itertuples(index=False)
    ]
    if not accepted_pairs:
        return linked, candidates

    roots = {particle: find(particle) for particle in groups}
    compact = {root: idx for idx, root in enumerate(sorted(set(roots.values())))}
    reconnected = linked.copy()
    reconnected["particle"] = [
        compact[roots[int(particle)]] for particle in reconnected["particle"]
    ]
    return reconnected.sort_values(["particle", "frame"]).reset_index(
        drop=True
    ), candidates


def link_movie_with_diagnostics(
    detections: pd.DataFrame,
    metadata: MovieMetadata,
    config: dict,
    params: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str | bool], pd.DataFrame]:
    params = (
        tracking_params_for_movie(config, metadata.movie) if params is None else params
    )
    recovery_enabled = bool(params.get("fragment_reconnect_enabled", False))
    if detections.empty:
        empty = detections.assign(particle=pd.Series(dtype=int))
        return (
            empty,
            {
                "movie": metadata.movie,
                "fragment_reconnect_enabled": recovery_enabled,
                "fragment_reconnect_applied": False,
                "fragment_reconnect_decision": "empty_detections",
                "fragment_reconnect_candidate_edges": 0,
                "fragment_reconnect_accepted_edges": 0,
            },
            pd.DataFrame(),
        )

    linked_unfiltered = _trackpy_link_unfiltered(detections, params)
    stub_length = int(params["stub_length_detections"])
    search_pixels = float(params["search_range_pixels"])
    base_filtered = tp.filter_stubs(
        linked_unfiltered, threshold=stub_length
    ).reset_index(drop=True)
    base_summary = _link_summary_pixels(base_filtered, search_pixels)

    candidate_edges = pd.DataFrame()
    candidate_summary = base_summary
    final = base_filtered
    applied = False
    accepted_edges = 0
    decision = "disabled"
    if recovery_enabled:
        reconnected_unfiltered, candidate_edges = _fragment_reconnect_candidates(
            linked_unfiltered, metadata, config, params
        )
        candidate_filtered = tp.filter_stubs(
            reconnected_unfiltered, threshold=stub_length
        ).reset_index(drop=True)
        candidate_summary = _link_summary_pixels(candidate_filtered, search_pixels)
        accepted_edges = (
            int(candidate_edges["accepted"].sum())
            if not candidate_edges.empty and "accepted" in candidate_edges
            else 0
        )
        row_gain = (candidate_summary["rows"] - base_summary["rows"]) / max(
            float(base_summary["rows"]), 1.0
        )
        long_gain = int(candidate_summary["tracks_ge_50"]) - int(
            base_summary["tracks_ge_50"]
        )
        exact_drop = float(base_summary["mean_exact_pair_fraction"]) - float(
            candidate_summary["mean_exact_pair_fraction"]
        )
        near_increase = float(candidate_summary["near_search_fraction"]) - float(
            base_summary["near_search_fraction"]
        )
        median_drop = float(base_summary["median_track_length"]) - float(
            candidate_summary["median_track_length"]
        )
        exact_drop = 0.0 if not np.isfinite(exact_drop) else exact_drop
        near_increase = 0.0 if not np.isfinite(near_increase) else near_increase
        median_drop = 0.0 if not np.isfinite(median_drop) else median_drop
        if accepted_edges == 0:
            decision = "no_unique_fragment_pairs"
        elif (
            (
                row_gain
                >= float(params.get("fragment_reconnect_min_rows_gain_fraction", 0.002))
                or long_gain > 0
            )
            and exact_drop
            <= float(params.get("fragment_reconnect_max_exact_fraction_drop", 0.01))
            and near_increase
            <= float(params.get("fragment_reconnect_max_near_search_increase", 0.005))
            and median_drop <= 2.0
        ):
            final = candidate_filtered
            applied = True
            decision = "accepted_by_qc_gates"
        else:
            decision = "rejected_by_qc_gates"

    final = final.reset_index(drop=True)
    final["movie"] = metadata.movie
    final["x_um"] = final["x"] * metadata.pixel_size_um
    final["y_um"] = final["y"] * metadata.pixel_size_um
    final["time_s"] = (
        final["frame"].astype(int).map(lambda f: float(metadata.timestamps_s[f]))
    )
    final["track_uid"] = (
        final["movie"].astype(str) + "::" + final["particle"].astype(int).astype(str)
    )
    qc_summary = {
        "movie": metadata.movie,
        "fragment_reconnect_enabled": recovery_enabled,
        "fragment_reconnect_applied": applied,
        "fragment_reconnect_decision": decision,
        "fragment_reconnect_candidate_edges": int(len(candidate_edges))
        if recovery_enabled
        else 0,
        "fragment_reconnect_accepted_edges": accepted_edges,
        "base_rows": int(base_summary["rows"]),
        "candidate_rows": int(candidate_summary["rows"]),
        "final_rows": int(len(final)),
        "base_tracks": int(base_summary["tracks"]),
        "candidate_tracks": int(candidate_summary["tracks"]),
        "final_tracks": int(final["particle"].nunique()) if not final.empty else 0,
        "base_median_track_length": base_summary["median_track_length"],
        "candidate_median_track_length": candidate_summary["median_track_length"],
        "base_mean_exact_pair_fraction": base_summary["mean_exact_pair_fraction"],
        "candidate_mean_exact_pair_fraction": candidate_summary[
            "mean_exact_pair_fraction"
        ],
        "base_near_search_fraction": base_summary["near_search_fraction"],
        "candidate_near_search_fraction": candidate_summary["near_search_fraction"],
        "base_tracks_ge_50": int(base_summary["tracks_ge_50"]),
        "candidate_tracks_ge_50": int(candidate_summary["tracks_ge_50"]),
        "base_tracks_ge_100": int(base_summary["tracks_ge_100"]),
        "candidate_tracks_ge_100": int(candidate_summary["tracks_ge_100"]),
    }
    return (
        final.sort_values(["particle", "frame"]).reset_index(drop=True),
        qc_summary,
        candidate_edges,
    )


def link_movie(
    detections: pd.DataFrame,
    metadata: MovieMetadata,
    config: dict,
    params: dict | None = None,
) -> pd.DataFrame:
    trajectories, _, _ = link_movie_with_diagnostics(
        detections, metadata, config, params
    )
    return trajectories


def track_metrics(
    traj: pd.DataFrame, metadata: MovieMetadata, config: dict
) -> pd.DataFrame:
    rows = []
    margin = float(config["tracking"]["edge_margin_pixels"])
    for particle, group in traj.groupby("particle", sort=False):
        group = group.sort_values("frame")
        xy = group[["x_um", "y_um"]].to_numpy(float)
        frames = group["frame"].to_numpy(int)
        if len(group) > 1:
            disp = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
            dt = np.diff(group["time_s"].to_numpy(float))
            path_length = float(disp.sum())
            net = float(np.hypot(xy[-1, 0] - xy[0, 0], xy[-1, 1] - xy[0, 1]))
            track_span = max_pairwise_distance_um(xy)
            contiguous_span = max_contiguous_run_span_um(group)
            exact_pair_frac = float(np.mean(np.diff(frames) == 1))
            max_gap = int(np.max(np.diff(frames)))
            mean_velocity = float(
                np.nanmean(
                    np.divide(disp, dt, out=np.full_like(disp, np.nan), where=dt > 0)
                )
            )
        else:
            path_length = net = track_span = contiguous_span = exact_pair_frac = (
                mean_velocity
            ) = np.nan
            max_gap = 0
        body_length_um = float(config["physics"]["body_length_um"])
        true_edge = bool(
            (group["x"].min() < margin)
            or (group["y"].min() < margin)
            or (group["x"].max() > metadata.width_px - margin)
            or (group["y"].max() > metadata.height_px - margin)
        )
        rows.append(
            {
                "movie": metadata.movie,
                "particle": int(particle),
                "track_uid": f"{metadata.movie}::{int(particle)}",
                "track_length": int(len(group)),
                "frame_start": int(group["frame"].min()),
                "frame_end": int(group["frame"].max()),
                "path_length_um": path_length,
                "net_displacement_um": net,
                "track_span_um": track_span,
                "contiguous_track_span_um": contiguous_span,
                "track_span_body_lengths": track_span / body_length_um
                if np.isfinite(track_span)
                else np.nan,
                "contiguous_track_span_body_lengths": contiguous_span / body_length_um
                if np.isfinite(contiguous_span)
                else np.nan,
                "confinement_ratio": net / path_length
                if path_length and path_length > 0
                else np.nan,
                "mean_velocity_um_s": mean_velocity,
                "exact_pair_fraction": exact_pair_frac,
                "max_internal_frame_gap": max_gap,
                "hessian_quality_cv": float(
                    group["quality"].std() / group["quality"].mean()
                )
                if group["quality"].mean() > 0
                else np.nan,
                "at_edge_true": true_edge,
            }
        )
    return pd.DataFrame(rows)


def reconnect_candidates(
    traj: pd.DataFrame,
    metadata: MovieMetadata,
    config: dict,
    params: dict | None = None,
) -> dict[str, float | int | str]:
    params = (
        tracking_params_for_movie(config, metadata.movie) if params is None else params
    )
    memory = int(params["memory_frames"])
    radius_um = float(params["search_range_pixels"]) * metadata.pixel_size_um
    starts = []
    ends = []
    for particle, group in traj.groupby("particle", sort=False):
        group = group.sort_values("frame")
        first = group.iloc[0]
        last = group.iloc[-1]
        if int(first.frame) > 0:
            starts.append(
                (int(first.frame), float(first.x_um), float(first.y_um), int(particle))
            )
        if int(last.frame) < metadata.n_frames - 1:
            ends.append(
                (int(last.frame), float(last.x_um), float(last.y_um), int(particle))
            )
    starts_by_frame = {}
    for frame, x_um, y_um, particle in starts:
        starts_by_frame.setdefault(frame, []).append((x_um, y_um, particle))
    trees = {
        frame: (cKDTree(np.asarray([(x, y) for x, y, _ in rows], dtype=float)), rows)
        for frame, rows in starts_by_frame.items()
    }
    candidate_edges = 0
    candidate_ends = set()
    for end_frame, x_um, y_um, particle in ends:
        for start_frame in range(
            end_frame + memory + 1, min(metadata.n_frames, end_frame + memory + 10)
        ):
            cached = trees.get(start_frame)
            if cached is None:
                continue
            tree, rows = cached
            for idx in tree.query_ball_point([x_um, y_um], r=radius_um):
                if rows[idx][2] == particle:
                    continue
                candidate_edges += 1
                candidate_ends.add(particle)
    return {
        "movie": metadata.movie,
        "n_starts_after_frame0": len(starts),
        "n_ends_before_last": len(ends),
        "candidate_reconnect_edges_after_memory": candidate_edges,
        "end_tracks_with_candidate_fraction": len(candidate_ends) / len(ends)
        if ends
        else np.nan,
    }


def detection_threshold_probe(
    record: MovieRecord, config: dict, params: dict | None = None
) -> pd.DataFrame:
    params = (
        tracking_params_for_movie(config, record.movie) if params is None else params
    )
    frame_index = int(params["detection_threshold_probe_frame"])
    frame = read_channel_frame(
        record, config["experiment"]["tracking_channel"], frame=frame_index
    )
    response = hessian_detector(
        frame,
        radius_pixels=float(params["radius_pixels"]),
        background_radius_pixels=float(params["background_radius_pixels"]),
        normalize=bool(params["normalize_quality_per_frame"]),
    )
    rows = []
    for threshold in params["detection_threshold_probe_values"]:
        trial_params = dict(params)
        trial_params["quality_threshold"] = float(threshold)
        coords = peak_local_max(
            response,
            min_distance=int(params["min_distance_pixels"]),
            threshold_abs=float(threshold),
            exclude_border=int(params["radius_pixels"]),
        )
        filtered = detect_frame(frame, frame_index, trial_params)
        rows.append(
            {
                "movie": record.movie,
                "frame": frame_index,
                "threshold": float(threshold),
                "hessian_candidates": int(len(coords)),
                "detections": int(len(filtered)),
                "min_spot_snr": params.get("min_spot_snr", np.nan),
                "median_spot_snr": float(filtered["spot_snr"].median())
                if not filtered.empty and "spot_snr" in filtered
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def process_movie_tracking(
    record: MovieRecord, config: dict, source_dir: Path, qc_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    metadata = read_metadata(record)
    params = tracking_params_for_movie(config, record.movie)
    detections = detect_movie(record, metadata, config, params)
    trajectories, link_qc, fragment_candidates = link_movie_with_diagnostics(
        detections, metadata, config, params
    )
    metrics = track_metrics(trajectories, metadata, config)
    threshold_probe = detection_threshold_probe(record, config, params)
    reconnect = reconnect_candidates(trajectories, metadata, config, params)
    applied = applied_tracking_parameter_row(config, record.movie)

    detections.to_parquet(
        source_dir / f"{record.movie}_detections.parquet", index=False
    )
    trajectories.to_parquet(
        source_dir / f"{record.movie}_trajectories.parquet", index=False
    )
    metrics.to_csv(qc_dir / f"{record.movie}_track_metrics.csv", index=False)
    threshold_probe.to_csv(
        qc_dir / f"{record.movie}_detection_threshold_probe.csv", index=False
    )
    fragment_path = qc_dir / f"{record.movie}_fragment_reconnect_candidates.csv"
    if not fragment_candidates.empty:
        fragment_candidates.to_csv(fragment_path, index=False)
    elif fragment_path.exists():
        fragment_path.unlink()
    return detections, trajectories, metrics, reconnect, applied, link_qc


def _trajectory_step_summary(group: pd.DataFrame) -> dict[str, float | int]:
    exact_displacements = []
    all_displacements = []
    frame_gaps = []
    for _, track in group.sort_values(["particle", "frame"]).groupby(
        "particle", sort=False
    ):
        if len(track) < 2:
            continue
        xy = track[["x_um", "y_um"]].to_numpy(float)
        frames = track["frame"].to_numpy(int)
        disp = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        gaps = np.diff(frames)
        all_displacements.extend(disp.tolist())
        frame_gaps.extend(gaps.tolist())
        exact_displacements.extend(disp[gaps == 1].tolist())
    all_displacements = np.asarray(all_displacements, dtype=float)
    exact_displacements = np.asarray(exact_displacements, dtype=float)
    frame_gaps = np.asarray(frame_gaps, dtype=float)
    if all_displacements.size == 0:
        return {
            "n_link_pairs": 0,
            "exact_frame_pair_fraction_link_weighted": np.nan,
            "exact_step_median_um": np.nan,
            "exact_step_q90_um": np.nan,
            "exact_step_q95_um": np.nan,
            "exact_step_q99_um": np.nan,
            "all_step_median_um": np.nan,
            "all_step_mean_um": np.nan,
        }
    exact_fraction = float(np.mean(frame_gaps == 1)) if frame_gaps.size else np.nan
    return {
        "n_link_pairs": int(all_displacements.size),
        "exact_frame_pair_fraction_link_weighted": exact_fraction,
        "exact_step_median_um": float(np.nanmedian(exact_displacements))
        if exact_displacements.size
        else np.nan,
        "exact_step_q90_um": float(np.nanquantile(exact_displacements, 0.90))
        if exact_displacements.size
        else np.nan,
        "exact_step_q95_um": float(np.nanquantile(exact_displacements, 0.95))
        if exact_displacements.size
        else np.nan,
        "exact_step_q99_um": float(np.nanquantile(exact_displacements, 0.99))
        if exact_displacements.size
        else np.nan,
        "all_step_median_um": float(np.nanmedian(all_displacements)),
        "all_step_mean_um": float(np.nanmean(all_displacements)),
    }


def tracking_movie_qc(
    records: list[MovieRecord],
    trajectories: pd.DataFrame,
    track_qc: pd.DataFrame,
    reconnect: pd.DataFrame,
    config: dict,
    source_dir: Path,
) -> pd.DataFrame:
    rows = []
    reconnect_by_movie = (
        reconnect.set_index("movie")
        if not reconnect.empty and "movie" in reconnect.columns
        else pd.DataFrame()
    )
    track_qc_by_movie = {
        movie: group.copy() for movie, group in track_qc.groupby("movie", sort=False)
    }
    traj_by_movie = {
        movie: group.copy()
        for movie, group in trajectories.groupby("movie", sort=False)
    }
    for record in records:
        metadata = read_metadata(record)
        params = tracking_params_for_movie(config, record.movie)
        detections_path = source_dir / f"{record.movie}_detections.parquet"
        detections = (
            pd.read_parquet(detections_path)
            if detections_path.exists()
            else pd.DataFrame()
        )
        traj = traj_by_movie.get(record.movie, pd.DataFrame())
        metrics = track_qc_by_movie.get(record.movie, pd.DataFrame())
        frame_counts = (
            detections.groupby("frame")
            .size()
            .reindex(range(metadata.n_frames), fill_value=0)
            if not detections.empty
            else pd.Series(np.zeros(metadata.n_frames, dtype=int))
        )
        accepted_frame_counts = (
            traj.groupby("frame").size().reindex(range(metadata.n_frames), fill_value=0)
            if not traj.empty
            else pd.Series(np.zeros(metadata.n_frames, dtype=int))
        )
        step = (
            _trajectory_step_summary(traj)
            if not traj.empty
            else _trajectory_step_summary(
                pd.DataFrame(columns=["particle", "frame", "x_um", "y_um"])
            )
        )
        search_um = float(params["search_range_pixels"]) * metadata.pixel_size_um
        near_search_fraction = np.nan
        if np.isfinite(step["exact_step_q99_um"]) and step["n_link_pairs"]:
            exact_displacements = []
            for _, track in traj.sort_values(["particle", "frame"]).groupby(
                "particle", sort=False
            ):
                if len(track) < 2:
                    continue
                xy = track[["x_um", "y_um"]].to_numpy(float)
                frames = track["frame"].to_numpy(int)
                disp = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
                exact_displacements.extend(disp[np.diff(frames) == 1].tolist())
            if exact_displacements:
                near_search_fraction = float(
                    np.mean(np.asarray(exact_displacements) >= 0.8 * search_um)
                )
        row = {
            "movie": record.movie,
            "n_frames": int(metadata.n_frames),
            "pixel_size_um": float(metadata.pixel_size_um),
            "quality_threshold": float(params["quality_threshold"]),
            "search_range_pixels": float(params["search_range_pixels"]),
            "search_range_um": search_um,
            "adaptive_stop_pixels": params.get("adaptive_stop_pixels", np.nan),
            "memory_frames": int(params["memory_frames"]),
            "stub_length_detections": int(params["stub_length_detections"]),
            "raw_detections": int(len(detections)),
            "raw_detections_per_frame_median": float(frame_counts.median()),
            "raw_detections_per_frame_q90": float(frame_counts.quantile(0.90)),
            "raw_detections_per_frame_max": int(frame_counts.max()),
            "spot_snr_median": float(detections["spot_snr"].median())
            if "spot_snr" in detections and not detections.empty
            else np.nan,
            "spot_snr_q10": float(detections["spot_snr"].quantile(0.10))
            if "spot_snr" in detections and not detections.empty
            else np.nan,
            "spot_snr_q25": float(detections["spot_snr"].quantile(0.25))
            if "spot_snr" in detections and not detections.empty
            else np.nan,
            "configured_min_spot_snr": float(params.get("min_spot_snr", np.nan)),
            "accepted_rows": int(len(traj)),
            "accepted_rows_per_frame_median": float(accepted_frame_counts.median()),
            "accepted_rows_per_raw_detection": float(len(traj) / len(detections))
            if len(detections)
            else np.nan,
            "accepted_tracks": int(traj["track_uid"].nunique())
            if not traj.empty
            else 0,
            "median_track_length": float(metrics["track_length"].median())
            if not metrics.empty
            else np.nan,
            "q90_track_length": float(metrics["track_length"].quantile(0.90))
            if not metrics.empty
            else np.nan,
            "max_track_length": int(metrics["track_length"].max())
            if not metrics.empty
            else 0,
            "track_weighted_exact_pair_fraction": float(
                metrics["exact_pair_fraction"].mean()
            )
            if not metrics.empty
            else np.nan,
            "median_track_velocity_um_s": float(metrics["mean_velocity_um_s"].median())
            if not metrics.empty
            else np.nan,
            "median_contiguous_span_um": float(
                metrics["contiguous_track_span_um"].median()
            )
            if not metrics.empty
            else np.nan,
            "edge_track_fraction": float(metrics["at_edge_true"].mean())
            if not metrics.empty and "at_edge_true" in metrics
            else np.nan,
            "exact_step_near_search_fraction": near_search_fraction,
            **step,
        }
        if not reconnect_by_movie.empty and record.movie in reconnect_by_movie.index:
            rec = reconnect_by_movie.loc[record.movie]
            row["reconnect_candidate_edges_after_memory"] = int(
                rec.get("candidate_reconnect_edges_after_memory", 0)
            )
            row["end_tracks_with_candidate_fraction"] = float(
                rec.get("end_tracks_with_candidate_fraction", np.nan)
            )
        else:
            row["reconnect_candidate_edges_after_memory"] = np.nan
            row["end_tracks_with_candidate_fraction"] = np.nan
        hard_flags = []
        warn_flags = []
        if row["track_weighted_exact_pair_fraction"] < 0.75:
            hard_flags.append("low_exact_frame_continuity")
        elif row["track_weighted_exact_pair_fraction"] < 0.85:
            warn_flags.append("moderate_exact_frame_continuity")
        if row["median_track_length"] < 15:
            hard_flags.append("short_median_tracks")
        elif row["median_track_length"] < 25:
            warn_flags.append("modest_median_tracks")
        if row["exact_step_median_um"] > 0.15:
            hard_flags.append("large_exact_step_median")
        elif row["exact_step_median_um"] > 0.10:
            warn_flags.append("elevated_exact_step_median")
        weak_linking_context = (
            (
                np.isfinite(row["track_weighted_exact_pair_fraction"])
                and row["track_weighted_exact_pair_fraction"] < 0.85
            )
            or (
                np.isfinite(row["median_track_length"])
                and row["median_track_length"] < 25
            )
            or (
                np.isfinite(row["accepted_rows_per_raw_detection"])
                and row["accepted_rows_per_raw_detection"] < 0.25
            )
        )
        detector_candidate_flood = row["raw_detections_per_frame_median"] > 2200
        high_detector_density = row["raw_detections_per_frame_median"] > 1500
        if detector_candidate_flood and weak_linking_context:
            hard_flags.append("detector_candidate_flood_with_weak_linking")
        elif high_detector_density:
            warn_flags.append("high_detector_candidate_density")
        if row["accepted_rows_per_raw_detection"] < 0.25:
            warn_flags.append("low_detection_survival")
        if np.isfinite(row["configured_min_spot_snr"]) and np.isfinite(
            row["spot_snr_q10"]
        ):
            if (
                row["spot_snr_q10"] < row["configured_min_spot_snr"] + 0.2
                and weak_linking_context
            ):
                warn_flags.append("low_spot_snr_margin_with_weak_linking")
        if row["end_tracks_with_candidate_fraction"] > 0.20:
            warn_flags.append("many_possible_splits_after_memory")
        if row["exact_step_near_search_fraction"] > 0.05:
            warn_flags.append("many_exact_steps_near_search_limit")
        row["tracking_qc_status"] = (
            "flagged" if hard_flags else ("warning" if warn_flags else "pass")
        )
        row["tracking_qc_flags"] = ";".join(hard_flags + warn_flags)
        rows.append(row)
    return pd.DataFrame(rows)
