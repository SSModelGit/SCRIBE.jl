# IROS workshop publication experiments

This folder contains the four experiments used to update the SCRIBE results
for the IROS workshop submission.

`filtering_experiments.jl` implements:

1. a continuous-sharing numerical check against a centralized
   information-filter oracle;
2. distance-limited pairwise communication, a total blackout, and
   distance-limited recovery under matched model assumptions;
3. the same communication-loss study with a truth outside the SCRIBE basis
   span.

Every filtering backend consumes the same pre-generated sampling locations and
measurements. The comparisons include the centralized oracle, SCRIBE,
independent agents, a Kalman-filter-only update that does not reconcile
divergent priors, and a Covariance-Intersection-only update that does not
accumulate independent innovations.
Reported metrics include field error, predictive log likelihood, interval
coverage, normalized estimation error, integrated uncertainty, consensus
error, centralized-posterior error, covariance conservatism, consensus
iterations, message count, and transmitted payload size.

`distributed_informative_exploration.jl` implements experiment 4. Four robots
run independent, single-agent instances of the Vulcan information-based
planner from their local environment posteriors. SCRIBE handles model exchange;
there is no joint or collaborative planning. Every distributed comparison uses
distance-limited, pairwise communication, so the four-agent network is never
fully connected. All links are then removed for a fixed blackout interval
before limited-range communication resumes.

Replicated trials compare SCRIBE with ideal instantaneous centralized sharing,
no communication, Kalman-Filter-only fusion, Covariance-Intersection-only
fusion, and direct exchange of up to the two newest unseen observation records
per available link. The experiment reports model error and calibration,
inter-agent disagreement, communication cost, and two trajectory-quality
measures obtained by rebuilding a common pooled-data posterior from the samples
selected under each sharing regime: reconstruction RMSE and integrated field
variance. The latter is the direct information-coverage metric for the
information-based planning study.

Each policy query uses a fixed MCTS rollout count rather than a wall-clock
budget, making paths repeatable for a given seed.

The representative-run figure and animation are generated from the model
states and paths recorded during the same experiment invocation; the
exploration or filtering is never rerun for visualization. The planning
problem is defined in `informative_exploration_problem.jl` and has no
dependency on scripts under `examples/`.

`informative_exploration.jl` retains the original single-robot integration
study as a smaller diagnostic, but it is not the distributed experiment
reported in the abstract.

## Environment

Instantiate the workshop plotting environment once from the SCRIBE root:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --startup-file=no --project=iros-workshop-rose26 \
  -e '
      using Pkg
      Pkg.develop(path=".")
      Pkg.develop(path="/home/shashank/cbase/corespace/jbase/VulcanJ")
      Pkg.instantiate()
  '
```

Change the second development path when VulcanJ has a different local
location.

## Full experiment runs

Run the experiments sequentially. Each command below uses the `full` profile
and writes beneath `iros-workshop-rose26/res/`.

Experiment 1:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GKSwstype=png \
  julia --startup-file=no --project=iros-workshop-rose26 \
  iros-workshop-rose26/filtering_experiments.jl full 1
```

Experiment 2:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GKSwstype=png \
  julia --startup-file=no --project=iros-workshop-rose26 \
  iros-workshop-rose26/filtering_experiments.jl full 2
```

Experiment 3:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GKSwstype=png \
  julia --startup-file=no --project=iros-workshop-rose26 \
  iros-workshop-rose26/filtering_experiments.jl full 3
```

Experiment 4:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GKSwstype=png \
  julia --startup-file=no --project=iros-workshop-rose26 \
  iros-workshop-rose26/distributed_informative_exploration.jl full
```

The full experiment 4 profile generates its representative animation by
default. To skip animation rendering while retaining all numerical results and
static figures:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GKSwstype=png \
  julia --startup-file=no --project=iros-workshop-rose26 \
  iros-workshop-rose26/distributed_informative_exploration.jl full no_animations
```

Do not run the four full profiles concurrently. The filtering studies execute
their Monte Carlo trials sequentially and render only after the corresponding
trials have completed.