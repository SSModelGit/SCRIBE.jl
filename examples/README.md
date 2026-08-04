# SCRIBE examples

The examples are grouped by purpose. Each runnable example writes only beneath
the matching directory in `examples/res/`.

## EOF climate models from ROMS

The files under `eof-climate-models/` separate offline EOF construction from
online agent operation. The private `_roms_data.jl` helper has three explicit
jobs: read the MATLAB arrays, prepare wet-cell EOF snapshots, and expose a
prepared snapshot as location-indexed recorded observations.

Learn the fixed EOF eigen-model space from the RAMS Head ROMS archive and save
the complete `EOFClimateModel` artifact:

```sh
julia --project=. examples/eof-climate-models/construct_roms_eof_model.jl
```

The offline run writes the model and three 4×4 diagnostic figures under
`examples/res/eof-climate-models/offline/`. The first two rows of every grid
use training snapshots and the final two rows use held-out chronological
validation snapshots. Reconstruction panels report full-field RMSE.

Load the artifact and assimilate sparse recorded ROMS measurements during
simulated agent operation:

```sh
julia --project=. examples/eof-climate-models/online_roms_eof_model.jl
```

The online run uses four scenarios. Each prior coefficient vector is the EOF
projection of a training snapshot, while the fixed ground truth comes from the
validation block. Every scenario writes its truth/prior/posterior comparison,
RMSE history, and truth-versus-posterior animation beneath
`examples/res/eof-climate-models/online/<scenario>/`.

The learned mean and EOF basis remain static. Online operation estimates only
the EOF coefficient vector using identity random-walk dynamics and the declared
process covariance `Q`. The example is therefore a sparse-observation state
identification experiment, not a learned temporal forecast.

The ROMS archive does not include `pm`, `pn`, or cell-volume metrics, so this
example uses uniform spatial weights. Production loaders should normally
supply cell areas or volumes when available.

The folder also contains two textbook-style references:

- `eof_theory_scribe_roms.pdf` derives EOF construction and explains the ROMS
  implementation;
- `mutual_information_scribe_tutorial.pdf` develops KL divergence, entropy,
  mutual information, SCRIBE information metrics, and planner objectives.

## Data-backed observations

SCRIBE can consume a physical sensor or recorded data without constructing a
ground-truth `SCRIBEModel`. Wrap a callable data source with the appropriate
model-specific behavior:

```julia
sensor = (k, X) -> read_sensor_values(k, X)
observer = DataObserver(sensor, EOFObserverBehavior(1e-4))
estimators = initialize_KF(eof_parameters, observer, initial_locations)
```

The callback may return either raw values or a `SensorObservation` carrying an
explicit covariance. A local update and progression are:

```julia
information = information_filter_update(estimators, k)
progress_agent_env_filter(estimators.system, information, next_locations)
```

After distributed fusion, the fused information state is already inserted, so
the corresponding progression is:

```julia
progress_agent_env_filter(agent.agent, next_locations)
```

Pre-collected measurements can instead be supplied directly as
`SensorObservation(k, X, z; covariance=R)`.

## VulcanJ planner integration

`vulcan-planner-integration/scribe_exploration.jl` is the focused integration
example. It supplies an LGSF posterior to VulcanJ's information-based MCTS
planner through four dispatches:

- initialize the planning model;
- evaluate the integrated field-variance-reduction reward;
- return the posterior predictive observation distribution;
- condition the SCRIBE model on a hypothetical or realized observation.

VulcanJ owns planning and SCRIBE owns prediction, conditioning, uncertainty,
and posterior visualization. The synthetic truth is an analytic field rather
than a second SCRIBE model.

Prepare the example environment from the repository root:

```sh
julia --project=examples -e '
    using Pkg
    Pkg.develop([
        Pkg.PackageSpec(path="."),
        Pkg.PackageSpec(path="/path/to/VulcanJ"),
    ])
    Pkg.instantiate()
'
```

Run the complete or smoke profile:

```sh
julia --project=examples \
    examples/vulcan-planner-integration/scribe_exploration.jl

julia --project=examples \
    examples/vulcan-planner-integration/scribe_exploration.jl smoke
```

Both profiles write beneath
`examples/res/vulcan-planner-integration/<profile>/`, so a smoke run cannot
overwrite the complete experiment. See
`vulcan-planner-integration/README.md` for the model, planner, and output
details.

The former GP comparisons, model-only comparison, and MCTS ablation repeated a
large shared experimental harness and were removed from `examples/`; they were
benchmark experiments rather than distinct SCRIBE API demonstrations.

## Visualization interface

Create a regular evaluation grid and pair a model with its information state:

```julia
grid = SCRIBEVisualizationGrid(x, y)
initial_model = SCRIBEModelState(smodel, initial_information, R)
```

Build a history from observations, from an observation callback, or directly
from saved `SCRIBEModelState` values:

```julia
visualization = scribe_model_history(
    model_states;
    sampling_locations,
    grid,
    ground_truth,
)
```

Save only the diagnostics needed by the example:

```julia
save_static_visualizations(
    visualization;
    output_dir,
    metrics=[:posterior_mean, :posterior_uncertainty],
)

save_animated_visualizations(
    visualization;
    output_dir,
    metrics=[:posterior_against_ground_truth],
)
```

Individual exported plotting functions are also available for posterior means,
uncertainty, ground-truth comparisons, absolute error, metric histories, and
prediction calibration.
