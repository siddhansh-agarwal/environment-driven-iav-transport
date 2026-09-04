#!/usr/bin/env python3
"""Run a uniform 3D, gradient 3D or uniform 2D manuscript calculation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from analysis.trajectory_summary import write_trajectory_summary
    from src.runner3d import load_parameter_file, run_ensemble

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_file", type=Path)
    parser.add_argument("--n-trajectories", type=int, default=32)
    parser.add_argument("--n-jobs", default="1", help="Parallel workers or 'auto'")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--chunk-steps", type=int, default=50_000)
    parser.add_argument("--trajectory-max-points", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_trajectories < 1:
        parser.error("--n-trajectories must be positive")
    if args.chunk_steps < 1 or args.trajectory_max_points < 2:
        parser.error(
            "chunk steps must be positive and trajectories need at least two points"
        )

    if args.n_jobs == "auto":
        jobs = max(1, min(8, os.cpu_count() or 1))
    else:
        try:
            jobs = int(args.n_jobs)
        except ValueError:
            parser.error("--n-jobs must be 'auto' or a positive integer")
        if jobs < 1:
            parser.error("--n-jobs must be 'auto' or a positive integer")
    configurations = load_parameter_file(args.parameter_file)
    results = run_ensemble(
        configurations,
        n_trajectories=args.n_trajectories,
        output_dir=args.output_dir,
        n_jobs=jobs,
        chunk_steps=args.chunk_steps,
        trajectory_max_points=args.trajectory_max_points,
        overwrite=args.overwrite,
    )
    output = args.output_dir / "trajectory_summary.csv"
    write_trajectory_summary(args.output_dir, output)
    completed = sum(not result["skipped"] for result in results)
    print(f"Completed {completed} trajectories; summary: {output}")


if __name__ == "__main__":
    main()
