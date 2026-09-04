#!/usr/bin/env python3
"""Recalculate every mean-field row in the uniform-3D figure tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.uniform_3d import load_geometries, load_inputs, predict  # noqa: E402


DIRECT_FIELDS = {
    "mean_attachments": "mean_attachments",
    "attachment_probability": "attachment_probability",
    "speed": "speed",
    "orientational_persistence_time": "persistence_time",
    "active_diffusivity": "active_diffusivity",
    "mobile_fraction": "mobile_fraction",
    "local_depletion_probability": "local_support_loss_probability",
    "receptor_contrast": "contrast",
    "persistence_length": "persistence_length",
    "diffusivity_shift": "diffusivity_shift",
}


def _prediction(row: pd.Series, inputs, geometries) -> dict[str, float]:
    return predict(
        str(row["architecture"]),
        float(row["K_D"]),
        float(row["K_C"]),
        int(row["n_binders"]),
        inputs,
        geometries,
    )


def _rebuild_direct_table(source: Path, destination: Path, inputs, geometries) -> float:
    table = pd.read_csv(source)
    rebuilt = table.copy()
    differences: list[float] = []
    for index, row in table.iterrows():
        result = _prediction(row, inputs, geometries)
        for column, key in DIRECT_FIELDS.items():
            if column not in table:
                continue
            value = float(result[key])
            differences.append(abs(value - float(row[column])))
            rebuilt.loc[index, column] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(destination, index=False)
    return max(differences, default=0.0)


def _rebuild_single_curve(
    source: Path,
    destination: Path,
    output_column: str,
    result_key: str,
    inputs,
    geometries,
) -> float:
    table = pd.read_csv(source)
    rebuilt = table.copy()
    differences: list[float] = []
    for index, row in table.iterrows():
        result = predict(
            str(row["architecture"]),
            float(row["K_D"]),
            10.0,
            10,
            inputs,
            geometries,
        )
        value = float(result[result_key])
        differences.append(abs(value - float(row[output_column])))
        rebuilt.loc[index, output_column] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(destination, index=False)
    return max(differences, default=0.0)


def _rebuild_contrast_table(source: Path, destination: Path, inputs, geometries) -> float:
    table = pd.read_csv(source)
    rebuilt = table.copy()
    differences: list[float] = []
    for index, row in table.iterrows():
        result = _prediction(row, inputs, geometries)
        values = {
            "receptor_contrast": 1.0
            - float(result["rho_deep"]) / max(float(result["rho_shallow"]), 1.0e-300),
            "locally_depleted_fraction": 1.0 - float(result["mobile_fraction"]),
            "mobile_fraction": float(result["mobile_fraction"]),
            "diffusivity_shift": float(result["diffusivity_shift"]),
        }
        for column, value in values.items():
            differences.append(abs(value - float(row[column])))
            rebuilt.loc[index, column] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(destination, index=False)
    return max(differences, default=0.0)


def _rebuild_attachment_table(source: Path, destination: Path, inputs, geometries) -> float:
    table = pd.read_csv(source)
    rebuilt = table.copy()
    differences: list[float] = []
    for index, row in table.iterrows():
        result = _prediction(row, inputs, geometries)
        particle_friction = (
            inputs.gamma_parallel
            if row["architecture"] == "polarized"
            else inputs.gamma_perp
        )
        values = {
            "mean_attachments": float(result["mean_attachments"]),
            "attachment_probability": float(result["attachment_probability"]),
            "probability_of_at_least_one_attachment": 1.0
            - (1.0 - float(result["attachment_probability"]))
            ** int(row["n_binders"]),
            "relative_mobility": particle_friction
            / (particle_friction + float(result["B_t"])),
            "attachment_lifetime": float(result["tau_off"]),
            "attachment_friction": float(result["B_t"]),
        }
        for column, value in values.items():
            differences.append(abs(value - float(row[column])))
            rebuilt.loc[index, column] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(destination, index=False)
    return max(differences, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data/figure_source/uniform_3d",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "recomputed/uniform_3d",
    )
    parser.add_argument("--check-tolerance", type=float, default=1.0e-9)
    args = parser.parse_args()

    inputs = load_inputs()
    geometries = load_geometries(inputs)
    checks = {
        "panel_d_tau_theory.csv": _rebuild_single_curve(
            args.source_dir / "panel_d_tau_theory.csv",
            args.output_dir / "panel_d_tau_theory.csv",
            "directional_persistence_time",
            "directional_persistence_time",
            inputs,
            geometries,
        ),
        "panel_e_speed_theory.csv": _rebuild_single_curve(
            args.source_dir / "panel_e_speed_theory.csv",
            args.output_dir / "panel_e_speed_theory.csv",
            "speed",
            "speed",
            inputs,
            geometries,
        ),
        "panel_f_cleavage_theory.csv": _rebuild_direct_table(
            args.source_dir / "panel_f_cleavage_theory.csv",
            args.output_dir / "panel_f_cleavage_theory.csv",
            inputs,
            geometries,
        ),
        "panel_g_mean_attachments_theory.csv": _rebuild_direct_table(
            args.source_dir / "panel_g_mean_attachments_theory.csv",
            args.output_dir / "panel_g_mean_attachments_theory.csv",
            inputs,
            geometries,
        ),
        "panel_h_contrast_erasure_competition.csv": _rebuild_contrast_table(
            args.source_dir / "panel_h_contrast_erasure_competition.csv",
            args.output_dir / "panel_h_contrast_erasure_competition.csv",
            inputs,
            geometries,
        ),
        "panel_i_attachment_motion_competition.csv": _rebuild_attachment_table(
            args.source_dir / "panel_i_attachment_motion_competition.csv",
            args.output_dir / "panel_i_attachment_motion_competition.csv",
            inputs,
            geometries,
        ),
    }
    largest = max(checks.values())
    for filename, difference in checks.items():
        print(f"{filename}: largest archived difference = {difference:.3g}")
    if not np.isfinite(largest) or largest > float(args.check_tolerance):
        raise SystemExit(
            f"Uniform mean-field check failed: largest difference {largest:.6g} "
            f"exceeds {args.check_tolerance:.6g}"
        )
    print(f"All uniform mean-field rows agree within {args.check_tolerance:g}.")


if __name__ == "__main__":
    main()
