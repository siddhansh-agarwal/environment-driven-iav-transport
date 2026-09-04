#!/usr/bin/env python3
"""Rebuild the population allocations and their one-dimensional projection."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.population import (  # noqa: E402
    demand_sweep_allocation,
    projected_population_density,
    reader_cleaver_coordinate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data/figure_source/population",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "recomputed/population",
    )
    parser.add_argument("--check-tolerance", type=float, default=1.0e-10)
    args = parser.parse_args()

    states = pd.read_csv(args.source_dir / "panel_c_representative_states.csv")
    indexed = states.set_index("function")
    capacities = {
        "explore": float(indexed.loc["explore", "explore_score"]),
        "escape": float(indexed.loc["escape", "escape_score"]),
        "exploit": float(indexed.loc["exploit", "exploit_score"]),
    }
    ratios = np.geomspace(0.05, 20.0, 121)
    rows = demand_sweep_allocation(capacities, ratios)
    sweep = pd.DataFrame(rows)
    sweep["simplex_x"] = sweep["escape_fraction"] + 0.5 * sweep["exploit_fraction"]
    sweep["simplex_y"] = (math.sqrt(3.0) / 2.0) * sweep["exploit_fraction"]
    ordered = [
        "explore_to_exploit_demand_ratio",
        "explore_fraction",
        "escape_fraction",
        "exploit_fraction",
        "simplex_x",
        "simplex_y",
        "explore_contribution",
        "escape_contribution",
        "exploit_contribution",
        "explore_demand",
        "escape_demand",
        "exploit_demand",
    ]
    sweep = sweep[ordered]

    shared_center = reader_cleaver_coordinate(
        float(indexed.loc["explore", "phi_b"]),
        float(indexed.loc["explore", "chi_C"]),
    )
    exploit_center = reader_cleaver_coordinate(
        float(indexed.loc["exploit", "phi_b"]),
        float(indexed.loc["exploit", "chi_C"]),
    )
    sigma = abs(exploit_center - shared_center) / (
        2.0 * math.sqrt(2.0 * math.log(2.0))
    )
    coordinate = np.linspace(
        shared_center - 3.5 * sigma,
        exploit_center + 3.5 * sigma,
        900,
    )
    distribution_rows = []
    for ratio in (0.05, 1.0, 20.0):
        row = sweep.iloc[int(np.argmin(abs(sweep.iloc[:, 0].to_numpy(float) - ratio)))]
        fractions = {
            role: float(row[f"{role}_fraction"])
            for role in ("explore", "escape", "exploit")
        }
        density = projected_population_density(
            fractions, shared_center, exploit_center, coordinate
        )
        for index, value in enumerate(coordinate):
            distribution_rows.append(
                {
                    "explore_to_exploit_demand_ratio": ratio,
                    "reader_cleaver_coordinate": value,
                    "explore_density": density["explore_density"][index],
                    "escape_density": density["escape_density"][index],
                    "exploit_density": density["exploit_density"][index],
                    "total_density": density["total_density"][index],
                    **{f"{role}_fraction": fractions[role] for role in fractions},
                    "explore_escape_center": shared_center,
                    "exploit_center": exploit_center,
                    "display_width": density["display_width"],
                }
            )
    distributions = pd.DataFrame(distribution_rows)

    archived_sweep = pd.read_csv(args.source_dir / "panel_c_demand_sweep.csv")
    archived_distributions = pd.read_csv(
        args.source_dir / "panel_c_population_distributions.csv"
    )
    sweep_difference = float(
        np.nanmax(
            np.abs(
                sweep.select_dtypes(include=[np.number]).to_numpy(float)
                - archived_sweep.select_dtypes(include=[np.number]).to_numpy(float)
            )
        )
    )
    distribution_difference = float(
        np.nanmax(
            np.abs(
                distributions.select_dtypes(include=[np.number]).to_numpy(float)
                - archived_distributions.select_dtypes(include=[np.number]).to_numpy(float)
            )
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output_dir / "panel_c_demand_sweep.csv", index=False)
    distributions.to_csv(
        args.output_dir / "panel_c_population_distributions.csv", index=False
    )
    largest = max(sweep_difference, distribution_difference)
    print(f"demand sweep: largest archived difference = {sweep_difference:.3g}")
    print(
        "population projection: largest archived difference = "
        f"{distribution_difference:.3g}"
    )
    if not np.isfinite(largest) or largest > float(args.check_tolerance):
        raise SystemExit(
            f"Population check failed: largest difference {largest:.6g} exceeds "
            f"{args.check_tolerance:.6g}"
        )
    print(f"All population allocation rows agree within {args.check_tolerance:g}.")


if __name__ == "__main__":
    main()
