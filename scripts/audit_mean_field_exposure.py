#!/usr/bin/env python3
"""Audit the shared mean-field cleavage-exposure factor against uniform sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.uniform_3d.calibration import (  # noqa: E402
    exposure_factor_audit,
    load_uniform_calibration_points,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data/figure_source/uniform_3d",
    )
    parser.add_argument("--selected-factor", type=float, default=0.002)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    points = load_uniform_calibration_points(args.source_dir)
    result = exposure_factor_audit(points, args.selected_factor)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if result["rmse_ratio"] > 1.10:
        raise SystemExit("The selected rounded factor is no longer near the global optimum")


if __name__ == "__main__":
    main()
