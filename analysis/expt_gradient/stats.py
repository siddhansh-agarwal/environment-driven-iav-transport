"""Averaging and treatment comparisons for the experimental observables."""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def shared_receptor_contrast_threshold(
    events: pd.DataFrame,
    *,
    retained_fraction: float = 0.10,
) -> float:
    """Return one contrast cutoff pooled across conditions and replicates."""

    if "particle_scale_receptor_contrast_percent" not in events:
        raise ValueError("events must contain particle-scale receptor contrast")
    fraction = float(retained_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("retained_fraction must lie in (0, 1]")
    values = pd.to_numeric(
        events["particle_scale_receptor_contrast_percent"], errors="coerce"
    ).to_numpy(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite receptor contrasts were supplied")
    return float(np.quantile(values, 1.0 - fraction))


def summarize_step_ci(
    events: pd.DataFrame,
    *,
    receptor_contrast_threshold: float,
    replicate_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Average step CI from trajectories to recordings and replicates."""

    track_column = "global_track_uid" if "global_track_uid" in events else "track_uid"
    required = {
        replicate_column,
        "condition",
        "movie",
        track_column,
        "ci",
        "particle_scale_receptor_contrast_percent",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events are missing required columns: {sorted(missing)}")
    selected = events.loc[
        pd.to_numeric(
            events["particle_scale_receptor_contrast_percent"], errors="coerce"
        ).ge(float(receptor_contrast_threshold))
    ].copy()
    track = (
        selected.groupby(
            [replicate_column, "condition", "movie", track_column], as_index=False
        )["ci"]
        .mean()
        .rename(columns={"ci": "step_ci"})
    )
    recording = track.groupby([replicate_column, "condition", "movie"], as_index=False)[
        "step_ci"
    ].mean()
    replicate = recording.groupby([replicate_column, "condition"], as_index=False)[
        "step_ci"
    ].mean()
    return track, recording, replicate


def within_replicate_label_permutation(
    recording_values: pd.DataFrame,
    *,
    value_column: str,
    untreated_condition: str,
    treated_condition: str,
    replicate_column: str = "date",
    alternative: str = "two-sided",
    iterations: int = 20_000,
    seed: int | None = None,
    exact_limit: int = 2_000_000,
) -> dict[str, float | int]:
    """Compare conditions after shuffling recording labels within each replicate.

    Group sizes are held fixed in every biological replicate.  Small null
    spaces are enumerated exactly; larger comparisons use the deterministic
    Monte Carlo procedure used for the receptor-gradient measurements.
    """

    required = {replicate_column, "condition", "movie", value_column}
    missing = required.difference(recording_values.columns)
    if missing:
        raise ValueError(
            f"recording_values are missing required columns: {sorted(missing)}"
        )
    if alternative not in {"less", "greater", "two-sided"}:
        raise ValueError("alternative must be 'less', 'greater', or 'two-sided'")
    if int(iterations) < 1:
        raise ValueError("iterations must be positive")

    observed_differences: list[float] = []
    blocks: list[tuple[np.ndarray, int]] = []
    for _, group in recording_values.groupby(replicate_column, sort=True):
        group = group.loc[
            group["condition"].isin((untreated_condition, treated_condition))
        ].dropna(subset=[value_column])
        values = group[value_column].to_numpy(float)
        labels = group["condition"].to_numpy(str)
        n_treated = int(np.sum(labels == treated_condition))
        n_untreated = int(np.sum(labels == untreated_condition))
        if n_treated == 0 or n_untreated == 0:
            continue
        observed_differences.append(
            float(
                values[labels == treated_condition].mean()
                - values[labels == untreated_condition].mean()
            )
        )
        blocks.append((values, n_treated))
    if not observed_differences:
        raise ValueError("no replicate contained recordings from both conditions")

    total_permutations = 1
    for values, n_treated in blocks:
        total_permutations *= comb(len(values), n_treated)

    exact = total_permutations <= int(exact_limit)
    if exact:
        possible_differences: list[np.ndarray] = []
        for values, n_treated in blocks:
            differences = []
            for treated_indices in combinations(range(len(values)), n_treated):
                treated = np.zeros(len(values), dtype=bool)
                treated[list(treated_indices)] = True
                differences.append(
                    float(values[treated].mean() - values[~treated].mean())
                )
            possible_differences.append(np.asarray(differences, dtype=float))
        null = possible_differences[0]
        for differences in possible_differences[1:]:
            null = (null[:, None] + differences[None, :]).ravel()
        null /= len(possible_differences)
    else:
        resolved_seed = (
            417 + sum(ord(character) for character in value_column)
            if seed is None
            else int(seed)
        )
        rng = np.random.default_rng(resolved_seed)
        null = np.empty(int(iterations), dtype=float)
        filled = 0
        chunk_size = 1_000
        while filled < len(null):
            n_chunk = min(chunk_size, len(null) - filled)
            delta_sum = np.zeros(n_chunk, dtype=float)
            for values, n_treated in blocks:
                draws = rng.random((n_chunk, len(values)))
                treated_indices = np.argpartition(
                    draws, n_treated - 1, axis=1
                )[:, :n_treated]
                treated_sum = values[treated_indices].sum(axis=1)
                n_untreated = len(values) - n_treated
                delta_sum += treated_sum / n_treated - (
                    values.sum() - treated_sum
                ) / n_untreated
            null[filled : filled + n_chunk] = delta_sum / len(blocks)
            filled += n_chunk

    observed = float(np.mean(observed_differences))
    correction = 0 if exact else 1
    denominator = len(null) if exact else len(null) + 1
    if alternative == "less":
        p_value = float((correction + np.sum(null <= observed)) / denominator)
    elif alternative == "greater":
        p_value = float((correction + np.sum(null >= observed)) / denominator)
    else:
        p_value = float(
            (correction + np.sum(np.abs(null) >= abs(observed))) / denominator
        )
    return {
        "observed_treated_minus_untreated": observed,
        "p_value": p_value,
        "p_two_sided": p_value if alternative == "two-sided" else np.nan,
        "n_replicates": len(observed_differences),
        "n_permutations": int(len(null)),
        "exact": bool(exact),
    }
