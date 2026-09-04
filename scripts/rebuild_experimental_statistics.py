#!/usr/bin/env python3
"""Recalculate the manuscript's experimental summaries from track-level tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]


def _load_analysis_functions():
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from analysis.expt_gradient import (  # noqa: PLC0415
        assign_fast_component_probability,
        fit_untreated_diffusivity_mixture,
        summarize_fast_component,
        within_replicate_label_permutation,
    )

    return (
        assign_fast_component_probability,
        fit_untreated_diffusivity_mixture,
        summarize_fast_component,
        within_replicate_label_permutation,
    )


def _recording_keys(table: pd.DataFrame) -> set[tuple[str, str, str]]:
    return {
        tuple(map(str, row))
        for row in table[["date", "condition", "movie"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }


def _validate_gradient_inputs(
    tracks: pd.DataFrame,
    recording_table: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    """Check the recordings and analysis settings used for each metric."""

    required_track_columns = {
        "date",
        "condition",
        "movie",
        "global_track_uid",
        "value",
    }
    missing = required_track_columns.difference(tracks.columns)
    if missing:
        raise ValueError(f"{metric} track table is missing {sorted(missing)}")
    if tracks["global_track_uid"].duplicated().any():
        raise ValueError(f"{metric} track identifiers are not unique")
    if not np.isfinite(tracks["value"].to_numpy(float)).all():
        raise ValueError(f"{metric} contains a non-finite track value")
    expected_conditions = {"SBA_noNAI", "SBA_NAI"}
    if set(tracks["condition"].astype(str)) != expected_conditions:
        raise ValueError(f"{metric} contains unexpected treatment labels")

    for column, expected in (
        ("conditioned_top_fraction", 0.10),
        ("conditioned_high_gradient_cutoff_percent", 9.1966917756456),
    ):
        values = recording_table[column].dropna().to_numpy(float)
        if values.size == 0 or not np.allclose(values, expected):
            raise ValueError(f"{metric} does not use the shared {column}")
    if metric == "conditioned_high_gradient_net_first_arrival_score":
        distances = recording_table["conditioned_passage_distance_um"].dropna()
        if distances.empty or not np.allclose(distances.to_numpy(float), 0.75):
            raise ValueError("Arrival analysis does not use the stated 0.75-um boundary")

    included = _included_recordings(recording_table)
    if _recording_keys(included) != _recording_keys(tracks):
        raise ValueError(f"{metric} track and recording tables have different support")
    replicate_conditions = (
        included.groupby("date")["condition"].apply(lambda values: set(values.astype(str)))
    )
    if len(replicate_conditions) != 3 or not all(
        conditions == expected_conditions for conditions in replicate_conditions
    ):
        raise ValueError(f"{metric} must contain both treatments in three replicates")
    return included


def _included_recordings(table: pd.DataFrame) -> pd.DataFrame:
    included = table.copy()
    if "conditioned_included_in_date_average" in included:
        included = included.loc[
            included["conditioned_included_in_date_average"].astype(bool)
        ].copy()
    return included[["date", "condition", "movie"]].drop_duplicates()


def _hierarchical_summary(
    tracks: pd.DataFrame,
    included: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = tracks.merge(included, on=["date", "condition", "movie"], how="inner")
    recordings = selected.groupby(
        ["date", "condition", "movie"], as_index=False
    )["value"].mean()
    replicates = recordings.groupby(["date", "condition"], as_index=False)[
        "value"
    ].mean()
    return recordings, replicates


def _condition_rows(
    metric: str,
    replicates: pd.DataFrame,
    *,
    compare_with_zero: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition, group in replicates.groupby("condition", sort=True):
        values = group.sort_values("date")["value"].to_numpy(float)
        rows.append(
            {
                "metric": metric,
                "condition": condition,
                "n_biological_replicates": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "biological_replicate_values": ";".join(
                    f"{value:.12g}" for value in values
                ),
                "p_one_sample_greater_than_zero": (
                    float(stats.ttest_1samp(values, 0.0, alternative="greater").pvalue)
                    if compare_with_zero
                    else np.nan
                ),
            }
        )
    return rows


def _paired_row(
    metric: str,
    replicates: pd.DataFrame,
    *,
    untreated_condition: str,
    treated_condition: str,
) -> dict[str, object]:
    wide = replicates.pivot(index="date", columns="condition", values="value").dropna()
    differences = wide[treated_condition].to_numpy(float) - wide[
        untreated_condition
    ].to_numpy(float)
    return {
        "metric": metric,
        "condition": "paired_NAI_minus_untreated",
        "n_biological_replicates": len(differences),
        "mean_delta": float(np.mean(differences)),
        "sd_delta": float(np.std(differences, ddof=1)),
        "biological_replicate_deltas": ";".join(
            f"{value:.12g}" for value in differences
        ),
        "p_paired_two_sided": float(
            stats.ttest_1samp(differences, 0.0, alternative="two-sided").pvalue
        ),
    }


def rebuild(input_dir: Path, output_dir: Path) -> None:
    (
        assign_fast_component_probability,
        fit_untreated_diffusivity_mixture,
        summarize_fast_component,
        within_replicate_label_permutation,
    ) = _load_analysis_functions()

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    replicate_sets: dict[str, set[str]] = {}
    analyses = (
        (
            "conditioned_high_gradient_chemotactic_index",
            "ci_track_values.csv",
            "ci_recording_metrics.csv",
            "conditioned_ci",
        ),
        (
            "conditioned_high_gradient_net_first_arrival_score",
            "arrival_track_values.csv",
            "arrival_recording_metrics.csv",
            "conditioned_net_first_passage_score_s",
        ),
    )
    for metric, track_name, recording_name, recording_metric in analyses:
        tracks = pd.read_csv(input_dir / track_name)
        recording_table = pd.read_csv(input_dir / recording_name)
        included = _validate_gradient_inputs(
            tracks,
            recording_table,
            metric=metric,
        )
        recordings, replicates = _hierarchical_summary(tracks, included)
        replicate_sets[metric] = set(replicates["date"].astype(str))
        rows.extend(_condition_rows(metric, replicates))
        paired = _paired_row(
            metric,
            replicates,
            untreated_condition="SBA_noNAI",
            treated_condition="SBA_NAI",
        )
        permutation = within_replicate_label_permutation(
            recordings,
            value_column="value",
            untreated_condition="SBA_noNAI",
            treated_condition="SBA_NAI",
            alternative="less",
            iterations=20_000,
            seed=417 + sum(ord(character) for character in recording_metric),
        )
        paired.update(
            {
                "p_recording_stratified": permutation["p_value"],
                "n_recording_label_randomizations": permutation["n_permutations"],
            }
        )
        rows.append(paired)
        recordings.to_csv(output_dir / f"{metric}_recordings.csv", index=False)
        replicates.to_csv(
            output_dir / f"{metric}_biological_replicates.csv", index=False
        )

    if len({frozenset(values) for values in replicate_sets.values()}) != 1:
        raise ValueError("CI and arrival must use the same biological replicates")
    if len(next(iter(replicate_sets.values()))) != 3:
        raise ValueError("The reported experimental analysis requires three replicates")

    diffusion = pd.read_csv(input_dir / "control_diffusivity_track_values.csv")
    if set(diffusion["condition"]) != {"noSBA_noNAI", "noSBA_NAI"}:
        raise ValueError("Unexpected conditions in the control-diffusivity table")
    if diffusion.duplicated(["date", "condition", "track_uid"]).any():
        raise ValueError("Control-diffusivity track identifiers are not unique")
    replicate_conditions = (
        diffusion.groupby("date")["condition"]
        .apply(lambda values: set(values.astype(str)))
        .to_list()
    )
    if len(replicate_conditions) != 3 or not all(
        conditions == {"noSBA_noNAI", "noSBA_NAI"}
        for conditions in replicate_conditions
    ):
        raise ValueError("Control analysis must contain both treatments in three replicates")
    if set(diffusion["selection_label"]) != {"bboxspan_0p50"} or not np.allclose(
        diffusion["min_bbox_span_um"].to_numpy(float), 0.5
    ):
        raise ValueError("Control diffusivities must use the stated 0.5-um span filter")
    if not diffusion["passes_diffusion_qc"].astype(bool).all():
        raise ValueError("Control-diffusivity input contains a track that failed fit QC")
    if (
        (diffusion["contiguous_points"].to_numpy(int) < 30).any()
        or (diffusion["fit_points"].to_numpy(int) < 4).any()
        or (diffusion["fit_r2"].to_numpy(float) < 0.3).any()
        or (diffusion["d_eff_um2_s"].to_numpy(float) <= 0.0).any()
        or (diffusion["bbox_max_span_um"].to_numpy(float) < 0.5).any()
    ):
        raise ValueError("Control-diffusivity input is inconsistent with its QC flags")
    model, fast_component = fit_untreated_diffusivity_mixture(diffusion)
    classified = assign_fast_component_probability(diffusion, model, fast_component)
    fast_recordings, fast_replicates = summarize_fast_component(classified)
    fast_replicates = fast_replicates.rename(
        columns={"fast_component_probability": "value"}
    )
    rows.extend(
        _condition_rows(
            "control_fast_component_probability",
            fast_replicates,
            compare_with_zero=False,
        )
    )
    fast_pair = _paired_row(
        "control_fast_component_probability",
        fast_replicates,
        untreated_condition="noSBA_noNAI",
        treated_condition="noSBA_NAI",
    )
    fast_permutation = within_replicate_label_permutation(
        fast_recordings,
        value_column="fast_component_probability",
        untreated_condition="noSBA_noNAI",
        treated_condition="noSBA_NAI",
        alternative="less",
    )
    fast_pair.update(
        {
            "p_recording_stratified": fast_permutation["p_value"],
            "n_recording_label_randomizations": fast_permutation["n_permutations"],
        }
    )
    rows.append(fast_pair)
    fast_recordings.to_csv(output_dir / "control_fast_component_recordings.csv", index=False)
    fast_replicates.to_csv(
        output_dir / "control_fast_component_biological_replicates.csv", index=False
    )
    pd.DataFrame(rows).to_csv(output_dir / "recalculated_statistics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "analysis_input" / "experimental",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.input_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
