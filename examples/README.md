# SCRIBE examples

## EOF climate model from ROMS

`roms_eof_climate_model.jl` demonstrates the user-defined
`eof_model_data_loader` contract on the RAMS Head ROMS velocity archive under
`bigdata/`. The example removes the archive's all-NaN land rows, samples the
hourly history at a configurable interval, learns a fixed EOF basis and linear
coefficient dynamics, and saves the complete learned model to a reusable
MATLAB artifact.

From the repository root, run:

```sh
julia --project=. examples/roms_eof_climate_model.jl
```

The default profile learns 12 EOFs from up to 720 daily `u`-velocity
snapshots. It writes the learned `.mat` artifact, variance spectrum, first EOF,
and coefficient histories under `examples/res/eof_roms/`. The loader is kept
in the example because file variables, land masks, time selection, physical
weights, and multivariate stacking are dataset-specific responsibilities.
This particular archive does not contain ROMS `pm`, `pn`, or cell-volume
metrics, so the example uses uniform spatial weights; production loaders
should pass area or volume weights when those metrics are available.

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
