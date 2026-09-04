from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


LOG_D_FLOOR_UM2_S = 1e-8
MIXTURE_RANDOM_SEED = 44


def _longest_contiguous_run(track: pd.DataFrame) -> pd.DataFrame:
    track = track.sort_values("frame").reset_index(drop=True)
    if track.empty:
        return track
    frames = track["frame"].to_numpy(int)
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, len(track)]
    lengths = stops - starts
    best = int(np.argmax(lengths))
    return track.iloc[starts[best] : stops[best]].reset_index(drop=True)


def _tamsd(run: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    run = run.sort_values("frame").reset_index(drop=True)
    x = run["x_um"].to_numpy(float)
    y = run["y_um"].to_numpy(float)
    t = run["time_s"].to_numpy(float)
    rows = []
    for lag in range(1, min(max_lag, len(run) - 1) + 1):
        dt = t[lag:] - t[:-lag]
        valid = dt > 0
        if valid.sum() < 5:
            continue
        sq = (x[lag:] - x[:-lag]) ** 2 + (y[lag:] - y[:-lag]) ** 2
        rows.append(
            {
                "lag_frames": lag,
                "lag_s": float(np.nanmedian(dt[valid])),
                "msd_um2": float(np.nanmean(sq[valid])),
                "n_pairs": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def _weighted_r2(
    x: np.ndarray, y: np.ndarray, fitted: np.ndarray, weights: np.ndarray
) -> float:
    ybar = np.average(y, weights=weights)
    ss_res = float(np.sum(weights * (y - fitted) ** 2))
    ss_tot = float(np.sum(weights * (y - ybar) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def diffusion_from_trajectories(
    trajectories: pd.DataFrame,
    *,
    date: str,
    condition: str,
    selection_label: str = "alltracks",
    min_bbox_span_um: float = 0.0,
    min_contiguous_points: int = 30,
    max_fit_lag: int = 10,
    min_fit_points: int = 4,
    min_fit_r_squared: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    track_rows = []
    msd_rows = []
    for track_uid, track in trajectories.groupby("track_uid", sort=False):
        run = _longest_contiguous_run(track)
        if len(run) < min_contiguous_points:
            continue
        bbox_x_span_um = float(run["x_um"].max() - run["x_um"].min())
        bbox_y_span_um = float(run["y_um"].max() - run["y_um"].min())
        bbox_max_span_um = float(max(bbox_x_span_um, bbox_y_span_um))
        bbox_diag_um = float(np.hypot(bbox_x_span_um, bbox_y_span_um))
        if bbox_max_span_um <= float(min_bbox_span_um):
            continue
        max_lag = min(max_fit_lag, max(1, len(run) // 4))
        msd = _tamsd(run, max_lag=max_lag)
        if len(msd) >= min_fit_points:
            x = msd["lag_s"].to_numpy(float)
            y = msd["msd_um2"].to_numpy(float)
            w = np.sqrt(msd["n_pairs"].to_numpy(float))
            slope, intercept = np.polyfit(x, y, 1, w=w)
            fitted = slope * x + intercept
            d_eff = slope / 4.0
            r2 = _weighted_r2(x, y, fitted, w)
            fit_lag_min_s = float(x.min())
            fit_lag_max_s = float(x.max())
            fit_points = int(len(msd))
        else:
            slope = intercept = d_eff = r2 = np.nan
            fit_lag_min_s = fit_lag_max_s = np.nan
            fit_points = int(len(msd))
        movie = str(run["movie"].iloc[0])
        particle = int(run["particle"].iloc[0])
        global_track_uid = f"{date}::{condition}::{track_uid}"
        if not msd.empty:
            msd = msd.copy()
            msd["date"] = date
            msd["condition"] = condition
            msd["selection_label"] = selection_label
            msd["min_bbox_span_um"] = float(min_bbox_span_um)
            msd["movie"] = movie
            msd["particle"] = particle
            msd["track_uid"] = global_track_uid
            msd_rows.append(msd)
        d_eff_clipped = float(max(d_eff, 0.0)) if np.isfinite(d_eff) else 0.0
        passes_fit_qc = bool(
            np.isfinite(d_eff)
            and np.isfinite(r2)
            and r2 >= float(min_fit_r_squared)
        )
        passes_positive_diffusion_qc = bool(passes_fit_qc and d_eff > 0)
        track_rows.append(
            {
                "date": date,
                "condition": condition,
                "selection_label": selection_label,
                "min_bbox_span_um": float(min_bbox_span_um),
                "movie": movie,
                "particle": particle,
                "track_uid": global_track_uid,
                "contiguous_points": int(len(run)),
                "duration_s": float(run["time_s"].iloc[-1] - run["time_s"].iloc[0]),
                "bbox_x_span_um": bbox_x_span_um,
                "bbox_y_span_um": bbox_y_span_um,
                "bbox_max_span_um": bbox_max_span_um,
                "bbox_diag_um": bbox_diag_um,
                "stuck_by_bbox_0p20": bool(bbox_max_span_um <= 0.20),
                "body_scale_mobile_bbox_0p50": bool(bbox_max_span_um >= 0.50),
                "fit_lag_min_s": fit_lag_min_s,
                "fit_lag_max_s": fit_lag_max_s,
                "fit_points": fit_points,
                "msd_slope_um2_s": float(slope),
                "intercept_um2": float(intercept),
                "d_eff_um2_s": float(d_eff),
                "d_eff_um2_s_clipped": d_eff_clipped,
                "log10_d_eff_clipped_floor": float(
                    np.log10(max(d_eff_clipped, LOG_D_FLOOR_UM2_S))
                ),
                "fit_r2": float(r2),
                "passes_fit_qc": passes_fit_qc,
                "passes_diffusion_qc": passes_positive_diffusion_qc,
                "fast_tail_d_eff_gt_1e_minus_3": bool(d_eff_clipped >= 1e-3),
                "fast_tail_d_eff_gt_1e_minus_2": bool(d_eff_clipped >= 1e-2),
            }
        )
    tracks = pd.DataFrame(track_rows)
    msd_all = pd.concat(msd_rows, ignore_index=True) if msd_rows else pd.DataFrame()
    return tracks, msd_all


def _summarize_diffusion_group(group: pd.DataFrame) -> dict[str, float | int]:
    passed = group.loc[group["passes_diffusion_qc"] & (group["d_eff_um2_s"] > 0)].copy()
    fit_qc = group.loc[group["passes_fit_qc"]].copy()
    out = {
        "n_tracks": int(group["track_uid"].nunique()),
        "n_movies": int(group["movie"].nunique()),
        "n_fit_qc_tracks": int(fit_qc["track_uid"].nunique()),
        "n_positive_diffusion_qc_tracks": int(passed["track_uid"].nunique()),
        "fit_qc_fraction": float(group["passes_fit_qc"].mean())
        if len(group)
        else np.nan,
        "positive_diffusion_qc_fraction": float(group["passes_diffusion_qc"].mean())
        if len(group)
        else np.nan,
        "stuck_fraction_bbox_0p20": float(group["stuck_by_bbox_0p20"].mean())
        if len(group)
        else np.nan,
        "body_scale_mobile_fraction_bbox_0p50": float(
            group["body_scale_mobile_bbox_0p50"].mean()
        )
        if len(group)
        else np.nan,
        "fast_tail_fraction_d_gt_1e_minus_3": float(
            group["fast_tail_d_eff_gt_1e_minus_3"].mean()
        )
        if len(group)
        else np.nan,
        "fast_tail_fraction_d_gt_1e_minus_2": float(
            group["fast_tail_d_eff_gt_1e_minus_2"].mean()
        )
        if len(group)
        else np.nan,
        "median_d_eff_all_clipped_um2_s": float(group["d_eff_um2_s_clipped"].median())
        if len(group)
        else np.nan,
        "mean_d_eff_all_clipped_um2_s": float(group["d_eff_um2_s_clipped"].mean())
        if len(group)
        else np.nan,
        "q90_d_eff_all_clipped_um2_s": float(
            group["d_eff_um2_s_clipped"].quantile(0.90)
        )
        if len(group)
        else np.nan,
        "q99_d_eff_all_clipped_um2_s": float(
            group["d_eff_um2_s_clipped"].quantile(0.99)
        )
        if len(group)
        else np.nan,
        "mean_log10_d_eff_all_clipped_floor": float(
            group["log10_d_eff_clipped_floor"].mean()
        )
        if len(group)
        else np.nan,
    }
    if passed.empty:
        out.update(
            {
                "median_d_eff_um2_s": np.nan,
                "mean_log10_d_eff": np.nan,
                "sd_log10_d_eff": np.nan,
                "q25_d_eff_um2_s": np.nan,
                "q75_d_eff_um2_s": np.nan,
            }
        )
    else:
        log10_d = np.log10(passed["d_eff_um2_s"])
        out.update(
            {
                "median_d_eff_um2_s": float(passed["d_eff_um2_s"].median()),
                "mean_log10_d_eff": float(log10_d.mean()),
                "sd_log10_d_eff": float(log10_d.std(ddof=1)),
                "q25_d_eff_um2_s": float(np.quantile(passed["d_eff_um2_s"], 0.25)),
                "q75_d_eff_um2_s": float(np.quantile(passed["d_eff_um2_s"], 0.75)),
            }
        )
    return out


def summarize_diffusion_tracks(
    diffusion: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if diffusion.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "selection_label" in diffusion.columns:
        group_cols = ["selection_label", "min_bbox_span_um", "date", "condition"]
        condition_cols = ["selection_label", "min_bbox_span_um", "condition"]
    else:
        group_cols = ["date", "condition"]
        condition_cols = ["condition"]
    date_rows = []
    for keys, group in diffusion.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row.update(_summarize_diffusion_group(group))
        date_rows.append(row)
    date_summary = pd.DataFrame(date_rows).sort_values(group_cols)
    condition_summary = (
        date_summary.groupby(condition_cols, as_index=False)
        .agg(
            n_biological_replicates=("date", "nunique"),
            mean_median_d_eff_um2_s=("median_d_eff_um2_s", "mean"),
            sd_median_d_eff_um2_s=("median_d_eff_um2_s", "std"),
            mean_median_d_eff_all_clipped_um2_s=(
                "median_d_eff_all_clipped_um2_s",
                "mean",
            ),
            sd_median_d_eff_all_clipped_um2_s=("median_d_eff_all_clipped_um2_s", "std"),
            mean_stuck_fraction_bbox_0p20=("stuck_fraction_bbox_0p20", "mean"),
            sd_stuck_fraction_bbox_0p20=("stuck_fraction_bbox_0p20", "std"),
            mean_body_scale_mobile_fraction_bbox_0p50=(
                "body_scale_mobile_fraction_bbox_0p50",
                "mean",
            ),
            sd_body_scale_mobile_fraction_bbox_0p50=(
                "body_scale_mobile_fraction_bbox_0p50",
                "std",
            ),
            mean_fast_tail_fraction_d_gt_1e_minus_3=(
                "fast_tail_fraction_d_gt_1e_minus_3",
                "mean",
            ),
            sd_fast_tail_fraction_d_gt_1e_minus_3=(
                "fast_tail_fraction_d_gt_1e_minus_3",
                "std",
            ),
            mean_log10_d_eff_across_biological_replicates=("mean_log10_d_eff", "mean"),
            sd_log10_d_eff_across_biological_replicates=("mean_log10_d_eff", "std"),
        )
        .sort_values(condition_cols)
    )
    return date_summary, condition_summary


def paired_diffusion_contrasts(
    date_summary: pd.DataFrame, selection_label: str = "alltracks"
) -> pd.DataFrame:
    if date_summary.empty or "selection_label" not in date_summary.columns:
        return pd.DataFrame()
    sub = date_summary.loc[date_summary["selection_label"] == selection_label].copy()
    if sub.empty:
        return pd.DataFrame()
    metrics = [
        "median_d_eff_all_clipped_um2_s",
        "mean_d_eff_all_clipped_um2_s",
        "q90_d_eff_all_clipped_um2_s",
        "q99_d_eff_all_clipped_um2_s",
        "stuck_fraction_bbox_0p20",
        "body_scale_mobile_fraction_bbox_0p50",
        "fast_tail_fraction_d_gt_1e_minus_3",
        "median_d_eff_um2_s",
    ]
    wide = sub.pivot(index="date", columns="condition", values=metrics)
    rows = []
    for date in wide.index:
        row = {"date": date, "selection_label": selection_label}
        for metric in metrics:
            no = (
                wide.loc[date, (metric, "noSBA_noNAI")]
                if (metric, "noSBA_noNAI") in wide.columns
                else np.nan
            )
            nai = (
                wide.loc[date, (metric, "noSBA_NAI")]
                if (metric, "noSBA_NAI") in wide.columns
                else np.nan
            )
            row[f"{metric}_noNAI"] = no
            row[f"{metric}_NAI"] = nai
            row[f"delta_{metric}_NAI_minus_noNAI"] = (
                nai - no if np.isfinite(nai) and np.isfinite(no) else np.nan
            )
            if "d_eff" in metric:
                row[f"log2_ratio_{metric}_NAI_over_noNAI"] = (
                    float(
                        np.log2(
                            max(nai, LOG_D_FLOOR_UM2_S) / max(no, LOG_D_FLOOR_UM2_S)
                        )
                    )
                    if np.isfinite(nai) and np.isfinite(no)
                    else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def fit_untreated_diffusivity_mixture(
    diffusion: pd.DataFrame,
    *,
    untreated_condition: str = "noSBA_noNAI",
) -> tuple[GaussianMixture, int]:
    """Fit the two-component mixture used for the control-membrane analysis.

    The fit uses positive apparent diffusivities that passed the MSD-fit
    criterion.  It is fitted once to the untreated tracks and then held fixed
    when the two experimental conditions are compared.
    """

    required = {"condition", "d_eff_um2_s", "passes_diffusion_qc"}
    missing = required.difference(diffusion.columns)
    if missing:
        raise ValueError(f"diffusion is missing required columns: {sorted(missing)}")
    reference = diffusion.loc[
        diffusion["condition"].eq(untreated_condition)
        & diffusion["passes_diffusion_qc"].astype(bool)
        & pd.to_numeric(diffusion["d_eff_um2_s"], errors="coerce").gt(0.0),
        "d_eff_um2_s",
    ].to_numpy(float)
    if reference.size < 10:
        raise ValueError(
            "At least ten accepted untreated tracks are required for the mixture fit"
        )
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=MIXTURE_RANDOM_SEED,
        n_init=100,
        reg_covar=1.0e-5,
    ).fit(np.log10(reference).reshape(-1, 1))
    fast_component = int(np.argmax(model.means_.ravel()))
    return model, fast_component


def assign_fast_component_probability(
    diffusion: pd.DataFrame,
    model: GaussianMixture,
    fast_component: int,
) -> pd.DataFrame:
    """Apply one fixed mixture to all accepted tracks."""

    required = {"d_eff_um2_s", "passes_diffusion_qc"}
    missing = required.difference(diffusion.columns)
    if missing:
        raise ValueError(f"diffusion is missing required columns: {sorted(missing)}")
    result = diffusion.copy()
    accepted = result["passes_diffusion_qc"].astype(bool) & pd.to_numeric(
        result["d_eff_um2_s"], errors="coerce"
    ).gt(0.0)
    result["fast_component_probability"] = np.nan
    if accepted.any():
        log_diffusivity = np.log10(
            result.loc[accepted, "d_eff_um2_s"].to_numpy(float)
        ).reshape(-1, 1)
        result.loc[accepted, "fast_component_probability"] = model.predict_proba(
            log_diffusivity
        )[:, int(fast_component)]
    return result


def summarize_fast_component(
    classified: pd.DataFrame,
    *,
    replicate_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average fast-component probabilities over recordings and replicates."""

    required = {
        replicate_column,
        "condition",
        "movie",
        "fast_component_probability",
    }
    missing = required.difference(classified.columns)
    if missing:
        raise ValueError(f"classified tracks are missing columns: {sorted(missing)}")
    accepted = classified.dropna(subset=["fast_component_probability"])
    recording = accepted.groupby(
        [replicate_column, "condition", "movie"], as_index=False
    )["fast_component_probability"].mean()
    replicate = recording.groupby([replicate_column, "condition"], as_index=False)[
        "fast_component_probability"
    ].mean()
    return recording, replicate


def write_diffusion_outputs(
    trajectories_by_condition: list[tuple[pd.DataFrame, str, str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    track_tables = []
    msd_tables = []
    selections = (
        ("alltracks", 0.0),
        ("bboxspan_0p20", 0.20),
        ("bboxspan_0p50", 0.50),
    )
    for trajectories, date, condition in trajectories_by_condition:
        for selection_label, min_bbox_span_um in selections:
            tracks, msd = diffusion_from_trajectories(
                trajectories,
                date=date,
                condition=condition,
                selection_label=selection_label,
                min_bbox_span_um=min_bbox_span_um,
            )
            if not tracks.empty:
                track_tables.append(tracks)
            if not msd.empty:
                msd_tables.append(msd)
    diffusion = (
        pd.concat(track_tables, ignore_index=True) if track_tables else pd.DataFrame()
    )
    msd_all = pd.concat(msd_tables, ignore_index=True) if msd_tables else pd.DataFrame()
    date_summary, condition_summary = summarize_diffusion_tracks(diffusion)
    paired = paired_diffusion_contrasts(date_summary, selection_label="alltracks")
    diffusion.to_csv(output_dir / "track_diffusion_coefficients.csv", index=False)
    msd_all.to_csv(output_dir / "track_tamsd_curves.csv", index=False)
    date_summary.to_csv(output_dir / "date_level_diffusion_summary.csv", index=False)
    condition_summary.to_csv(
        output_dir / "condition_level_diffusion_summary.csv", index=False
    )
    paired.to_csv(output_dir / "date_paired_diffusion_contrasts.csv", index=False)
