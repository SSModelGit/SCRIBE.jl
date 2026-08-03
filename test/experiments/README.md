# SCRIBE experiment environment

This directory contains executable research experiments, not package tests.
Its plotting and JLD2 dependencies are isolated in a dedicated environment.

Run experiments from the repository root with one Julia thread and one BLAS
thread unless a larger resource budget has been deliberately allocated:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no \
  --project=test/experiments \
  -e 'include("test/experiments/test_scriptor.jl"); run_scriptor(max_runs=1, make_plots=false)'
```

Plotting is opt-in:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no \
  --project=test/experiments \
  -e 'include("test/experiments/test_scriptor.jl"); run_scriptor(max_runs=1, make_plots=true)'
```

`full_sweep=true` only expands the candidate parameter grid. `max_runs` remains
a hard limit and should be increased gradually while monitoring the host, not
only the Distrobox process namespace.

Run the MHMC experiment with:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no \
  --project=test/experiments \
  -e 'include("test/experiments/test_mhmc.jl")'
```

For the matched three-agent comparison between no communication and distributed
communication, run:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GKSwstype=png \
  julia --startup-file=no --project=test/experiments \
  -e 'include("test/experiments/test_scriptor.jl"); run_communication_comparison()'
```

The comparison resets the same random seed for each case, runs both simulations
before plotting, and writes shared-scale comparison figures to `test/res_plots`.
