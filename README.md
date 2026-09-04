# Environment-driven active transport of influenza A virus

This repository contains the simulation and analysis code accompanying the
manuscript **“Environment-driven active transport of influenza A virus.”** It
includes the stochastic bind–cleave model, mean-field calculations,
experimental trajectory analysis, population-function calculations and the
compact numerical inputs underlying the reported results.

## Contents

- `src/`: stochastic binding, unbinding and cleavage simulations in 2D and 3D
- `analysis/`: trajectory observables, mean-field theory, experimental analysis
  and population-function calculations
- `scripts/`: simulation examples and scripts that recalculate selected reported results
- `config/`: example simulation and experimental-analysis settings
- `data/analysis_input/`: compact inputs used by the analysis scripts
- `data/figure_source/`: numerical values underlying the principal plots
- `tests/`: numerical and regression tests

## Installation

The code uses Python 3.11. Install Miniforge, Miniconda or Anaconda, then run:

```bash
conda env create -f environment.yml
conda activate iav-transport
```

Installation typically takes 3–10 minutes, depending on the package cache and
network connection.

This repository version has been tested with Python 3.11 on macOS 26.6.2 and
Ubuntu 24.04. Exact dependency versions are listed in `environment.yml`. No
non-standard hardware is required; the examples and analysis run on a standard
CPU.

## Simulation examples

Run short examples of the three geometries from the repository root:

```bash
python scripts/run_demo.py uniform-3d
python scripts/run_demo.py gradient-3d
python scripts/run_demo.py surface-2d
```

The examples write trajectories and a summary table under `demo_output/` and
check that each output is valid. Use `python scripts/run_demo.py all` to run all
three, or add `--trajectories 1` for a quicker check. The uniform and gradient
3D examples include translational and rotational Brownian motion. The 2D
surface example uses the athermal between-event dynamics described in the
Supplementary Information. All three use the same stochastic reaction kernel.
After installation, the complete three-case demo normally runs in under one
minute on a standard desktop computer.

Other conditions can be run from a modified example configuration:

```bash
python scripts/run_simulation.py path/to/config.yaml \
  --n-trajectories 32 --n-jobs 1 --output-dir results/example
```

The YAML files specify binder number, ligand organization, molecular rates,
geometry and Brownian diffusivities. Each stored trajectory includes its input
parameters, random seed, particle path, orientation, cleaved receptors,
stopping reason and numerical modes.

## Reported calculations

The following commands recalculate the uniform- and surface-theory curves,
experimental statistics and population allocations from the deposited inputs.
They write their results to the requested output directory and compare them
with the values used in the manuscript figures. The gradient directory contains
the local mean-field relations and the trajectory observables used for the
gradient analysis; its plotted values are provided as figure-source tables.

```bash
PYTHONPATH=. python scripts/audit_mean_field_exposure.py
PYTHONPATH=. python scripts/rebuild_uniform_mean_field.py --output-dir results/uniform
PYTHONPATH=. python scripts/rebuild_surface_ranges.py --output-dir results/surface
PYTHONPATH=. python scripts/rebuild_experimental_statistics.py --output-dir results/experimental
PYTHONPATH=. python scripts/rebuild_population_maps.py --output-dir results/population
PYTHONPATH=. python scripts/rebuild_population_allocation.py --output-dir results/population
```

The experimental code under `analysis/expt_gradient/` implements virion
tracking, local receptor-gradient measurements, displacement alignment,
first-arrival analysis and the statistical tests reported in the manuscript.
The deposited track-level values preserve the hierarchy of trajectories,
recordings and three biological replicates. The shared analysis settings are
recorded in `config/experimental_gradient_analysis.yaml`.

The simulation observables are calculated in
`analysis/simulation_observables.py`. Section-specific theory is grouped under
`analysis/uniform_3d/`, `analysis/surface_2d/`, `analysis/gradient_3d/` and
`analysis/population/`. Model definitions, parameter ranges, run durations and
ensemble sizes are given in the manuscript Methods and Supplementary
Information.

## Tests

Run the complete test suite with:

```bash
PYTHONPATH=. python -m pytest -q tests
```

The tests cover the three simulation geometries, trajectory storage,
observables, mean-field calculations and manuscript-facing analysis scripts.
The same suite runs automatically on GitHub.

## License

The code is released under the [BSD 3-Clause License](LICENSE).
