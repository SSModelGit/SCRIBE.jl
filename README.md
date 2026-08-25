# SCRIBE

SCRIBE (Signal-aware Collaboration for Reliable Information-Based
Estimation) is a distributed information-filtering framework for
environmental models that are linear in a shared coefficient vector. The
spatial model may be a dictionary of Linear Gaussian Scalar Fields
(LGSFs) or a fixed Empirical Orthogonal Function (EOF) space learned
from environmental snapshots. The same `SCRIBEModel`,
`SCRIBEModelParameters`, observer, prediction, metric, and
distributed-fusion APIs operate on both backends.

This README is a quickstart. Complete runnable workflows and
generated-output descriptions live in [examples/](examples/),
particularly [eof-climate-models/](examples/eof-climate-models/) and
[vulcan-planner-integration/](examples/vulcan-planner-integration/).

## Installation

Activate the repository and instantiate its dependencies:

```console
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The examples have their own environment and use a local VulcanJ
checkout:

```console
julia --project=examples -e '
    using Pkg
    Pkg.develop([
        Pkg.PackageSpec(path="."),
        Pkg.PackageSpec(path="/path/to/VulcanJ"),
    ])
    Pkg.instantiate()
'
```

## Common API

Every backend supplies a subtype of `SCRIBEModelParameters` and a
subtype of `SCRIBEModel`. Construct parameters first, then use the
common initializer:

```julia
model = initialize_SCRIBEModel_from_parameters(parameters; k=1)
```

During estimation, the model stores the basis, time, and current
coefficient estimate. `KFEnvInfo` stores the Gaussian coefficient
posterior in information form. An observer converts raw measurements
into the model's linear observation dynamics. Consequently, filtering,
prediction, metrics, and distributed fusion do not need separate LGSF
and EOF control flows.

## LGSF quickstart

An LGSF field is a weighted sum of Gaussian spatial basis functions.
`mu` contains one basis center per row; every combination of a center,
`sigma`, and `tau` creates one basis function.

```julia
using LinearAlgebra
using SCRIBE

centers = [
    -1.0 -1.0
     1.0 -1.0
    -1.0  1.0
     1.0  1.0
]
n_phi = size(centers, 1)

lgsf_parameters = LGSFModelParameters(
    μ=centers,
    σ=[1.5],
    τ=[1.0],
    ϕ₀=zeros(n_phi),
    A=Matrix{Float64}(I, n_phi, n_phi),
    Q=1e-3 * Matrix{Float64}(I, n_phi, n_phi),
)
lgsf_model = initialize_SCRIBEModel_from_parameters(lgsf_parameters)

X = [0.0 0.0; 0.5 0.5]
H = prediction_dynamics(lgsf_model, X)
```

The source uses Unicode keyword names. In the Julia REPL and most
editors, enter them with `\mu<Tab>`, `\sigma<Tab>`, `\tau<Tab>`, and
`\phi<Tab>\_0<Tab>`.

`A=I` gives random-walk coefficient dynamics. `Q` is coefficient process
covariance per update. Scalar values supplied to observers and
prediction functions are variances, not standard deviations.

## EOF quickstart

EOF training data must be a finite `n_features × n_snapshots` matrix.
Each column is one complete environmental snapshot; rows must retain the
same feature/grid-cell ordering across all snapshots. `locations` must
have one row per feature in that same order. Optional positive `weights`
define the spatial inner product and should normally contain ROMS cell
areas or volumes.

```julia
using LinearAlgebra
using SCRIBE

decomposition = fit_eof_decomposition(
    snapshots;
    weights=cell_weights,
    variance_fraction=0.995,
)

eof_parameters = EOFClimateModelParameters(
    decomposition;
    locations=locations,
    process_covariance=1e-3,
    interpolation=:inverse_distance,
    interpolation_neighbors=4,
)
eof_model = initialize_SCRIBEModel_from_parameters(eof_parameters)

full_field = reconstruct_eof_field(eof_model)
values_at_X = predict_SCRIBEModel(eof_model, X)

candidate_coefficients = eof_coefficients(eof_model, candidate_snapshot)
candidate_field = reconstruct_eof_field(
    eof_model;
    coefficients=candidate_coefficients,
)
plot_eof_field(eof_model; coefficients=candidate_coefficients)
```

The learned mean and EOF vectors remain fixed online. The filter
estimates only their coefficient vector. `process_covariance` declares
how much unexplained coefficient change is admitted per operational
update; it does not learn or impose a temporal transition law.

## MATLAB data and EOF artifacts

Raw training archives and trained SCRIBE EOF artifacts serve different
purposes.

For a simple raw archive, store the snapshot matrix and its aligned
spatial metadata with MAT.jl:

```julia
using MAT

matwrite(
    "training_snapshots.mat",
    Dict(
        "snapshots" => snapshots,
        "locations" => locations,
        "weights" => cell_weights,
    );
    compress=true,
)
```

Load it and pass the numerical arrays to SCRIBE:

```julia
archive = matread("training_snapshots.mat")
snapshots = Matrix{Float64}(archive["snapshots"])
locations = Matrix{Float64}(archive["locations"])
weights = vec(Float64.(archive["weights"]))

decomposition = fit_eof_decomposition(snapshots; weights)
parameters = EOFClimateModelParameters(
    decomposition;
    locations,
    process_covariance=1e-3,
)
```

Real ROMS files usually require masking land/fill values, selecting
depth and variables, reconciling staggered grids, arranging the feature
rows, and possibly subsampling time. Reusable ROMS loading, wet-cell
preparation, curl construction, EOF fitting, and field rendering live in
`SCRIBE.ROMSTools`. The EOF examples retain only their archive paths, mission
choices, and recorded-snapshot sensor. A different reusable source type can
specialize `eof_model_data_loader(source; ...)` and then use
`initialize_eof_climate_model(source; loader_kwargs, ...)`.

Once the decomposition and operational priors are chosen, save the
complete versioned SCRIBE artifact instead of manually writing
individual fields:

```julia
save_eof_model("climate_eof.mat", parameters)

loaded_parameters = load_eof_model_parameters("climate_eof.mat")
loaded_model = load_eof_climate_model("climate_eof.mat"; k=1)
```

`save_eof_model` stores the `EOFDecomposition`, locations, initial
coefficient mean and covariance, process covariance, interpolation
settings, and metadata. Pass `include_coefficients=false` when the
offline coefficient history is not needed at runtime. Do not use
`matread` plus manual struct construction to load a SCRIBE-generated
model artifact.

## Online observations and filtering

Use `DataObserver` for a physical sensor, middleware callback,
simulator, or recorded dataset. It avoids constructing a separate
ground-truth `SCRIBEModel`:

```julia
sensor = (k, X) -> SensorObservation(
    k,
    X,
    read_sensor_values(k, X);
    covariance=1e-2,
)

behavior = EOFObserverBehavior(1e-2; include_truncation_error=true)
observer = DataObserver(sensor, behavior)
estimators = initialize_KF(eof_parameters, observer, initial_locations)

for k in 1:n_updates
    information = information_filter_update(estimators, k)
    coefficients = recover_estimate_from_info(information)

    if k < n_updates
        progress_agent_env_filter(
            estimators.system,
            information,
            next_locations[k],
        )
    end
end
```

For LGSF, replace the parameters and use
`LGSFObserverBehavior(sensor_variance)`. A sensor callback may return
raw values and use the behavior's default variance, or return
`SensorObservation` with scalar, diagonal, or full measurement
covariance. Pre-collected `SensorObservation` objects can also be passed
directly to `initialize_KF` and `progress_agent_env_filter`.

## Extracting predictions and metrics

Metrics act on the model together with its `KFEnvInfo` posterior:

```julia
model = estimators.system.estimates[end].estimate
information = last(estimators.system.information)

coefficients = posterior_coefficient_moments(information)
field = posterior_model_moments(model, information, evaluation_locations)
field_variance = predict_model_uncertainty(
    model,
    information,
    evaluation_locations;
    metric=:variance,
)

entropy = evaluate_information_metric(
    information;
    metric=:differential_entropy,
)
total_coefficient_variance = evaluate_information_metric(
    information;
    metric=:total_variance,
)

expected_mi = mutual_information(model, information, candidate_X, sensor_R)
realized_gain = D_KL(model, information, candidate_X, observed_z, sensor_R)
field_ivr = integrated_variance_reduction(
    model,
    information,
    candidate_X,
    evaluation_locations,
    sensor_R,
)
```

Available information-state metrics include `:information`,
`:covariance`, `:logdet_information`, `:differential_entropy`,
`:total_variance`, `:mean_variance`, `:minimum_information_eigenvalue`,
and `:condition_number`. Field uncertainty supports `:covariance`,
`:variance`, `:standard_deviation`, `:total_variance`, and
`:mean_variance`.

For an EOF candidate measurement, use its effective measurement
covariance if truncation uncertainty should affect planning:

```julia
R_effective = eof_effective_measurement_covariance(
    eof_model,
    candidate_X,
    sensor_R,
)
expected_mi = mutual_information(
    eof_model,
    information,
    candidate_X,
    R_effective,
)
```

## Distributed online workflow

Every agent must use the same parameter object—or equivalently the same
basis rows in the same order—while retaining its own observer, filter,
and local information state. Define the current undirected communication
graph, initialize each estimator, and wrap it as a `SCRIBEAgent`:

```julia
edges = Dict(
    "agent1" => ["agent2"],
    "agent2" => ["agent1", "agent3"],
    "agent3" => ["agent2"],
)
network = init_network_graph(edges)

for id in keys(edges)
    estimator = initialize_KF(
        shared_parameters,
        observers[id],
        initial_locations[id],
    )
    network.vertices[id] = initialize_agent(id, estimator, network)
end
```

At each model time, agents first compute and exchange SCRIBE consensus
messages. `deliver_consensus_messages` is an in-process transport
helper; real deployments serialize the same `ConsensusMessage` through
their network layer and call `consume_consensus_message!` on receipt.

```julia
complete = Dict(id => false for id in keys(edges))
while !all(values(complete))
    for id in keys(edges)
        complete[id] = distributed_fusion(
            k, id, network, 1e-8, 360,
        )
    end

    for (sender, message) in deliver_consensus_messages(k, network)
        for receiver in network.edges[sender]
            consume_consensus_message!(
                receiver,
                network.vertices[receiver].net_conn,
                message,
                k,
                network,
            )
        end
        network.vertices[sender].net_conn.outbox["msg_out"] = nothing
    end
end

for id in keys(edges)
    vertex = network.vertices[id]
    progress_agent_env_filter(vertex.agent, next_locations[id])
    full_reset_network_connector(vertex.net_conn)
end
```

`distributed_fusion` performs covariance intersection on potentially
correlated priors and average consensus on new measurement innovations.
It inserts the fused `KFEnvInfo` at `k+1`; the final progression call
recovers the coefficient estimate, advances the model, and acquires the
next local observation. Call `update_network_graph_edges` before a
fusion cycle when connectivity changes.

## Detailed examples

  - [EOF climate models](examples/eof-climate-models/): offline ROMS
    training, MATLAB artifact persistence, held-out reconstruction
    diagnostics, and online coefficient estimation from sparse data.
  - [VulcanJ planner integration](examples/vulcan-planner-integration/):
    an LGSF-backed VulcanJ informative-exploration run and visualization
    workflow.
  - [Examples guide](examples/README.md): commands, expected outputs,
    and modeling assumptions for the runnable examples.
