#!/usr/bin/env python3
"""Recalculate the explore, escape and exploit maps shown in the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.population import (  # noqa: E402
    evaluate_representative_states,
    rebuild_function_maps,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guidance-tensor",
        type=Path,
        default=ROOT / "data/analysis_input/population/gradient_ci_tensor.npz",
    )
    parser.add_argument(
        "--reference-table",
        type=Path,
        default=ROOT / "data/figure_source/population/panel_ab_function_maps.csv",
    )
    parser.add_argument(
        "--guidance-validation",
        type=Path,
        default=(
            ROOT
            / "data/analysis_input/population/gradient_ci_emulator_validation.csv"
        ),
    )
    parser.add_argument(
        "--representative-states",
        type=Path,
        default=(
            ROOT
            / "data/figure_source/population/panel_c_representative_states.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-tolerance", type=float, default=2.0e-7)
    args = parser.parse_args()

    validation = pd.read_csv(args.guidance_validation.resolve())
    required_bins = {"K_C_le_10", "K_C_10_to_50", "K_C_50_to_250_edge"}
    if set(validation["validation_bin"]) != required_bins:
        raise SystemExit("Guidance validation does not cover the required cleavage bins")
    largest_validation_error = float(validation["absolute_error"].max())
    if largest_validation_error > 1.5e-3:
        raise SystemExit(
            "Guidance interpolation check failed: maximum held-out error "
            f"{largest_validation_error:.6g}"
        )

    rebuilt = rebuild_function_maps(args.guidance_tensor.resolve())
    reference = pd.read_csv(args.reference_table.resolve())
    keys = ["architecture", "chi_C", "phi_b"]
    rebuilt = rebuilt.sort_values(keys).reset_index(drop=True)
    reference = reference.sort_values(keys).reset_index(drop=True)
    if list(rebuilt.columns) != list(reference.columns) or len(rebuilt) != len(reference):
        raise SystemExit("Rebuilt and archived population tables have different shapes")
    numerical_columns = rebuilt.select_dtypes(include=[np.number]).columns
    largest = float(
        np.nanmax(
            np.abs(
                rebuilt[numerical_columns].to_numpy(float)
                - reference[numerical_columns].to_numpy(float)
            )
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(args.output_dir / "panel_ab_function_maps.csv", index=False)

    declared_states = pd.read_csv(args.representative_states.resolve())
    state_scores = evaluate_representative_states(
        args.guidance_tensor.resolve(), declared_states
    )
    declared = declared_states.set_index("function")
    largest_state_difference = 0.0
    for state in state_scores.itertuples(index=False):
        role_column = f"{state.function}_score"
        difference = abs(
            float(getattr(state, role_column))
            - float(declared.loc[state.function, role_column])
        )
        largest_state_difference = max(largest_state_difference, difference)
    state_scores.to_csv(
        args.output_dir / "panel_c_representative_state_full_scores.csv",
        index=False,
    )
    print(f"population function maps: largest archived difference = {largest:.3g}")
    print(
        "gradient-guidance interpolation: maximum held-out absolute error = "
        f"{largest_validation_error:.3g}"
    )
    print(
        "representative-state capacities: largest archived difference = "
        f"{largest_state_difference:.3g}"
    )
    largest = max(largest, largest_state_difference)
    if not np.isfinite(largest) or largest > float(args.check_tolerance):
        raise SystemExit(
            f"Population-map check failed: {largest:.6g} exceeds "
            f"{args.check_tolerance:.6g}"
        )
    print(f"All population-map rows agree within {args.check_tolerance:g}.")


if __name__ == "__main__":
    main()
