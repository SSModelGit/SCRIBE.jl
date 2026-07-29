# IROS workshop publication experiments

This folder contains the four experiments used to update the SCRIBE results
for the IROS workshop submission.

`filtering_experiments.jl` implements:

1. connected-network SCRIBE against a centralized information-filter oracle;
2. connected, split, and reconnected operation under matched model assumptions;
3. the same split-and-reconnect study with a truth outside the SCRIBE basis
   span.

Every filtering backend consumes the same pre-generated sampling locations and
measurements. The comparisons include the centralized oracle, SCRIBE,
independent agents, and an explicitly labeled naive information-sum baseline.
Reported metrics include field error, predictive log likelihood, interval
coverage, normalized estimation error, integrated uncertainty, consensus
error, centralized-posterior error, covariance conservatism, consensus
iterations, message count, and transmitted payload size.

`informative_exploration.jl` implements experiment 4. It uses SCRIBE as the
model-learning backend for VulcanJ and produces the evolving posterior,
prediction and uncertainty metrics, information rewards, and optional
animations for a deliberately misspecified ground truth. Its planning problem
is defined locally in `informative_exploration_problem.jl`; it has no
dependency on the scripts under `examples/`. All SCRIBE posterior figures and
animations use the model states gathered during exploration directly through
the visualization API exported by SCRIBE; the posterior is not reconstructed
in a second reporting pass.

## Environment

Instantiate the workshop plotting environment once from the SCRIBE root:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --startup-file=no --project=iros-workshop-27 \
  -e '
      using Pkg
      Pkg.develop(path=".")
      Pkg.develop(path="/home/shashank/cbase/secondary/jbase/VulcanJ")
      Pkg.instantiate()
  '
```

Change the second development path when VulcanJ has a different local
location.

## Full experiment runs

Run the experiments sequentially. Each command below uses the `full` profile
and writes beneath `iros-workshop-27/res/`.

Experiment 1:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GKSwstype=png \
  julia --startup-file=no --project=iros-workshop-27 \
  iros-workshop-27/filtering_experiments.jl full 1
```

Experiment 2:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GKSwstype=png \
  julia --startup-file=no --project=iros-workshop-27 \
  iros-workshop-27/filtering_experiments.jl full 2
```

Experiment 3:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GKSwstype=png \
  julia --startup-file=no --project=iros-workshop-27 \
  iros-workshop-27/filtering_experiments.jl full 3
```

Experiment 4:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GKSwstype=png \
  julia --startup-file=no --project=iros-workshop-27 \
  iros-workshop-27/informative_exploration.jl full
```

To additionally generate the experiment 4 animations:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GKSwstype=png \
  julia --startup-file=no --project=iros-workshop-27 \
  iros-workshop-27/informative_exploration.jl full animations
```

Do not run the four full profiles concurrently. The filtering studies execute
their Monte Carlo trials sequentially and render only after the corresponding
trials have completed.
