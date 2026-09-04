#!/usr/bin/env python3
"""Reapply the surface-regime criteria and rebuild every plotted range."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.surface_2d import (  # noqa: E402
    load_geometries,
    load_inputs,
    predict_surface_range,
)


KEYS = ["architecture", "K_C", "K_D", "n_binders"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition-audit",
        type=Path,
        default=(
            ROOT
            / "data/analysis_input/surface_2d/surface_range_condition_audit.csv"
        ),
    )
    parser.add_argument(
        "--reference-table",
        type=Path,
        default=(
            ROOT
            / "data/figure_source/surface_2d/panel_d_recurrence_design_points.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    conditions = pd.read_csv(args.condition_audit.resolve())
    selected = (
        conditions["K_C"].gt(0.0)
        & conditions["detached_fraction"].ge(0.8)
        & conditions["median_path_in_persistence_lengths"].ge(30.0)
        & conditions["recurrence_range_screen"].lt(
            0.9 * conditions["finite_time_free_range"]
        )
    )
    archived_selection = conditions["included_in_repeated_return_comparison"].astype(
        bool
    )
    if not np.array_equal(selected.to_numpy(), archived_selection.to_numpy()):
        raise SystemExit("Stored and recalculated surface-regime selections differ")

    reference = pd.read_csv(args.reference_table.resolve())
    selected_keys = conditions.loc[selected, KEYS].sort_values(KEYS).reset_index(drop=True)
    reference_keys = reference[KEYS].sort_values(KEYS).reset_index(drop=True)
    if not selected_keys.equals(reference_keys):
        raise SystemExit("Selected conditions do not match the plotted range conditions")

    inputs = load_inputs()
    geometries = load_geometries(inputs)
    rebuilt_rows: list[dict[str, float | int | str]] = []
    for row in reference.itertuples(index=False):
        result = predict_surface_range(
            str(row.architecture),
            float(row.K_D),
            float(row.K_C),
            int(row.n_binders),
            float(row.observation_time),
            inputs,
            geometries,
        )
        rebuilt_rows.append(
            {
                "architecture": row.architecture,
                "K_C": row.K_C,
                "K_D": row.K_D,
                "n_binders": row.n_binders,
                "persistence_length": result["ell_one_clock"],
                "effective_trail_width": result["effective_trail_width"],
                "detachment_area": result["detachment_area"],
                "mean_field_range": result["mean_field_range"],
            }
        )
    rebuilt = pd.DataFrame(rebuilt_rows)
    joined = reference.merge(rebuilt, on=KEYS, suffixes=("_archived", "_rebuilt"))
    fields = (
        "persistence_length",
        "effective_trail_width",
        "detachment_area",
        "mean_field_range",
    )
    largest = max(
        float(
            np.max(
                np.abs(
                    joined[f"{field}_archived"].to_numpy(float)
                    - joined[f"{field}_rebuilt"].to_numpy(float)
                )
            )
        )
        for field in fields
    )
    if largest > 1.0e-9:
        raise SystemExit(f"Surface mean-field check failed: difference {largest:.6g}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(args.output_dir / "surface_range_mean_field.csv", index=False)
    print(f"surface selection: {int(selected.sum())} of {len(conditions)} conditions")
    print(f"surface mean field: largest archived difference = {largest:.3g}")


if __name__ == "__main__":
    main()
