# SCRIBE examples

## EOF climate model from ROMS

The examples under `eof-climate-models/` separate offline model construction
from online agent operation. The private `_roms_data.jl` helper keeps its three
jobs explicit: read the MATLAB arrays, prepare wet-cell EOF snapshots, and
serve one prepared snapshot as location-indexed recorded observations.

First, learn the static EOF eigen-model space from the RAMS Head ROMS archive,
declare its coefficient process covariance, and save the complete
`EOFClimateModel` artifact:

```sh
julia --project=. \
    examples/eof-climate-models/construct_roms_eof_model.jl
```

Then load that artifact and assimilate recorded ROMS measurements during
agent operation:

```sh
julia --project=. \
    examples/eof-climate-models/online_roms_eof_model.jl
```

The first example writes `rams_head_u_eof.mat` under
`examples/res/eof-climate-models/offline/`. By default it samples the full archive at
a three-hour cadence, uses the first 80% as a chronological learning window,
and retains the smallest rank reaching 99.5% of anomaly variance. The EOF
basis and mean remain fixed. Online coefficients use an identity random-walk
prior with a declared `Q`; no transition matrix, forcing vector, or temporal
schedule is learned from the archive. The example's `process_variance` is the
coefficient variance admitted per selected update interval and should be
redeclared when the operational cadence or expected rate of change differs.
It also writes three 4×4 diagnostic figures. The first two rows contain four
training snapshots and the last two rows contain four snapshots from the
untouched chronological validation block. Each consecutive pair shows ground
truth beside its best-fit reconstruction, posterior field-covariance diagonal,
or pointwise percent relative error; every reconstruction is annotated with
its full-field RMSE. The posterior
variance uses a full-field observation variance of `1e-4`; because observation
geometry and covariance are fixed, the linear-Gaussian posterior covariance is
the same for all eight snapshots even though their reconstructed means differ.

The second example runs four coefficient-learning demonstrations. Each fixes
the environment at a snapshot from the untouched validation block, initializes
the SCRIBE coefficient mean at the best-fit coordinate of a training snapshot,
and gathers 400 sparse
measurements along a serpentine agent trajectory. The ground truth is supplied
directly through a `DataObserver`; no separate ground-truth `SCRIBEModel` is
constructed. Each scenario writes a side-by-side truth/posterior animation and
a full-field RMSE curve under its own directory in
`examples/res/eof-climate-models/online/`. Each directory also contains a
one-row comparison of the ground truth, deliberately wrong prior
reconstruction, and final posterior reconstruction. The animation starts at
sample zero, so its first posterior panel is the unconditioned prior.
Subsequent frames show the ordinary SCRIBE information-filter updates pulling
the coefficient vector toward the fixed truth coordinate.

The online demonstration therefore combines held-out representation with
sparse-observation state identification. It is not a temporal forecast: the
selected validation snapshot remains fixed while the agent explores it. Its
reference coefficient is the direct projection of that unseen field into the
fixed EOF basis, while full-field RMSE also retains irreducible truncation
error.
Extending it to multiple agents only requires creating one estimator per agent
and connecting those agents through SCRIBE's distributed fusion protocol; the
EOF model and observer code do not change.

This archive does not contain ROMS `pm`, `pn`, or cell-volume metrics, so the
example uses uniform spatial weights. Production loaders should normally pass
area or volume weights when those metrics are available.

## Data-backed SCRIBE observations

SCRIBE can pull observations from an external sensor or recorded data without
constructing a ground-truth `SCRIBEModel`. Wrap a callable data source together
with the observer behavior for the agent's model:

```julia
sensor = (k, X) -> read_sensor_values(k, X)
observer = DataObserver(sensor, EOFObserverBehavior(1e-4))
estimators = initialize_KF(eof_parameters, observer, initial_locations)
```

The callback may return either a vector of values or a `SensorObservation`
with an explicit covariance. `initialize_KF` creates the agent's
`EOFClimateModel` through `initialize_SCRIBEModel_from_parameters`, and
`scribe_observations` converts the sensor result into an `EOFObserverState`.
After the agent's local information-filter update, pull the next observation:

```julia
information = information_filter_update(estimators, k)
progress_agent_env_filter(estimators.system, information, next_locations)
```

For distributed agents, consensus inserts the next information state, so the
corresponding call is:

```julia
progress_agent_env_filter(agent, next_locations)
```

Pre-collected data can instead be pushed explicitly with
`SensorObservation(k, X, z; covariance=R)` at initialization and progression.
The original model-backed overloads remain available for simulations.

## Information-based exploration with VulcanJ

`vulcan_scribe_exploration.jl` uses a SCRIBE Gaussian scalar field as the
environment model supplied to VulcanJ's information-based MCTS planner. The
integration consists of four dispatched methods:

- initialize the planning model;
- evaluate the planner's integrated variance-reduction reward;
- return its posterior predictive measurement distribution;
- condition the model on a simulated or realized measurement.

VulcanJ remains responsible for planning. SCRIBE remains responsible for all
model prediction, conditioning, information calculations, and posterior
visualization. The actual exploration loop stores each conditioned
`SCRIBEModelState` as it is produced. Those model states go directly to the
visualization interface; observations are not replayed afterward.

Prepare the example environment from the SCRIBE root with:

```sh
julia --project=examples -e '
    using Pkg
    Pkg.develop(path=".")
    Pkg.develop(path="/home/shashank/cbase/secondary/jbase/VulcanJ")
    Pkg.instantiate()
'
```

Change the second path when VulcanJ has a different local location.

The complete profile uses an eight-connected 33×33 navigation grid and a
108-sample mission. Its 36 overlapping scalar-field bases have a spatially
correlated coefficient prior. The synthetic truth is a direct analytic
function combining domain-scale variation and two off-grid local features, so
it is not constructed from—or aligned with—the learner's basis dictionary.
The MDP state is simply the physical sampling location, and its transition
clamps an eight-connected motion step to the domain. VulcanJ receives
SCRIBE's scalar posterior predictive `Normal` directly. The planning objective
is mean integrated field-variance reduction over a fixed evaluation grid; it
is defined by the planner integration using the current SCRIBE posterior, not
by SCRIBE itself.

Run the complete experiment with:

```sh
julia --project=examples examples/vulcan_scribe_exploration.jl
```

A short compilation and plotting check is also available:

```sh
julia --project=examples examples/vulcan_scribe_exploration.jl smoke
```

Both profiles write a compact four-panel diagnostic animation, a focused
ground-truth-versus-posterior animation, standalone posterior-mean and
posterior-uncertainty animations, a final surface comparison, information and
error histories, and predicted-versus-ground-truth plots. Results are separated under
`examples/res/vulcan_scribe/<profile>/`, so a smoke render cannot overwrite
the full experiment.

## SCRIBE visualization interface

The package visualization path starts from a regular evaluation grid:

```julia
grid = SCRIBEVisualizationGrid(x, y)
```

First pair the SCRIBE model with its information state and measurement noise:

```julia
initial_model = SCRIBEModelState(smodel, initial_information, R)
```

Given `SCRIBEObservation(X, z, R)` or `LGSFObserverState` entries, only the
observations and initial model are required. Locations are not drawn as a
trajectory unless they are separately supplied through `sampling_locations`:

```julia
visualization = scribe_model_history(
    observations,
    initial_model;
    sampling_locations=nothing,
    grid,
    ground_truth,
)
```

Raw measurement values can use `sampling_locations` for both filtering and
trajectory visualization. Alternatively, supply an observation function and
sampling locations directly:

```julia
visualization = scribe_model_history(
    X -> observe_environment(X),
    sampling_locations,
    initial_model;
    grid,
    ground_truth,
)
```

Set `show_sampling_path=false` to suppress that trajectory. A vector beginning
with the initial `SCRIBEModelState` and followed by updated model states can
also be passed directly to `scribe_model_history`.

Each visualization has its own exported function, such as
`plot_posterior_mean_map`, `plot_posterior_against_ground_truth`,
`animate_posterior_mean_map`, and
`animate_posterior_against_ground_truth`. Batch output is selected explicitly:

```julia
save_static_visualizations(
    visualization;
    output_dir,
    metrics=[:posterior_mean, :posterior_uncertainty],
)

save_animated_visualizations(
    visualization;
    output_dir,
    metrics=[:posterior_mean, :posterior_against_ground_truth],
)
```

## SCRIBE and Gaussian-process comparisons

The comparison facilities run VulcanJ using either SCRIBE or a standard
squared-exponential Gaussian process. The motion model, measurements, fixed
rollout budget, and integrated variance-reduction reward are held fixed so that
the posterior backend is the experimental variable. The two prior
field-uncertainty amplitudes are matched; neither model is fitted to the
ground-truth function. The GP branch uses GaussianProcesses.jl's standard
`GPE`, SE kernel, observation-noise model, posterior prediction, and
conditioning machinery. Both planners operate directly on the physical
sampling location and use scalar predictive measurement distributions. SCRIBE
visualizations consume the posterior states saved during the run; the
observations are not replayed afterward.

The introductory comparison uses a smooth two-lobe field and a single-scale
SCRIBE basis:

```sh
julia --project=examples examples/vulcan_simple_model_comparison.jl
```

The difficult comparison combines domain-wide variation, local anomalies, and
a curved narrow feature. Its SCRIBE model uses local and broad GSFs together
with a scale-aware prior:

```sh
julia --project=examples examples/vulcan_complicated_model_comparison.jl
```

Its SCRIBE backend uses two overlapping GSF scales. A dense 9×9 dictionary of
narrow fields carries local variation, while a sparse 4×4 dictionary of broad
fields carries the domain-scale structure. This allocates spatial coverage
according to scale instead of repeating one center grid for every width. The
two dictionaries use distinct staggered lattices, while the ground-truth wells
are off-grid, rotated, and anisotropic. Consequently, neither well coincides
with a SCRIBE basis and the truth is not exactly represented by the model
dictionary.
Coefficients are spatially correlated within each scale but independent
between scales, avoiding an artificial assumption that broad and local
features are interchangeable. The resulting prior is normalized at the field
level so its mean pointwise variance matches the GP's prior variance. The
complicated field uses a unit prior field deviation for both backends; the smoother
introductory field retains the broader prior used by the original example.

The full comparisons use 12 fixed MCTS rollouts per sample with a depth-four
lookahead. This keeps the backend comparison repeatable while bounding the
matrix-factorization work performed by the multiscale SCRIBE model.

Append `smoke` to either command for its bounded profile. Closed-loop results
are separated by scenario and profile under
`examples/res/vulcan_model_comparison/planner/`, so a smoke run cannot
overwrite a complete experiment.

## Same-data model comparison

`vulcan_model_only_comparison.jl` removes the planning trajectory as a
confounding variable. It constructs one deterministic farthest-point sampling
sequence and one seeded noisy measurement vector, then conditions SCRIBE and
the GP on precisely the same `(location, measurement)` pairs.

```sh
julia --project=examples examples/vulcan_model_only_comparison.jl
julia --project=examples examples/vulcan_model_only_comparison.jl smoke
```

The runner evaluates both the simple and complicated truths. It saves the
shared data in `shared_samples.csv` alongside the comparison figures under
`examples/res/vulcan_model_comparison/model_only/<scenario>/<profile>/`.

## Multirun MCTS ablation

`vulcan_mcts_ablation.jl` evaluates three backends on the complicated field:

- multiscale SCRIBE with dense local and sparse broad GSF dictionaries;
- local-only SCRIBE, which removes the broad GSF dictionary;
- the single-scale squared-exponential GP.

The complete profile aggregates five matched MCTS seeds. Runs execute
sequentially and retain only metric histories, rather than posterior
animations or search trees. The smoke profile uses two seeds.

```sh
julia --project=examples examples/vulcan_mcts_ablation.jl
julia --project=examples examples/vulcan_mcts_ablation.jl smoke
```

The aggregate figure reports the mean trajectory with a one-standard-deviation
ribbon. Per-seed final values are saved to `final_metrics.csv` under
`examples/res/vulcan_model_comparison/ablation/complicated/<profile>/`.

All comparison experiments report:

- RMSE and RMSE normalized by the ground-truth field deviation;
- mean predictive log likelihood using latent posterior variance plus the
  measurement-noise variance;
- empirical coverage of the latent truth by the posterior 95% interval;
- integrated posterior uncertainty, evaluated as mean field variance;
- MAE for model-only runs or planning reward for closed-loop runs.
