using MAT: matread, matwrite
using Plots
using Random: AbstractRNG, default_rng, randn
using Statistics: mean

import .SCRIBEModels:
    compute_obs_dynamics,
    get_model_time,
    initialize_SCRIBEModel_from_parameters,
    predict_SCRIBEModel,
    scribe_observations,
    update_SCRIBEModel

export EOFDecomposition, EOFClimateModelParameters, EOFClimateModel
export EOFObserverBehavior, EOFObserverState
export eof_model_data_loader, fit_eof_decomposition
export initialize_eof_climate_model, save_eof_model
export load_eof_model_parameters, load_eof_climate_model
export eof_mean, eof_modes, eof_prior_covariance, eof_process_covariance
export eof_residual_variance
export eof_interpolation_matrix, eof_basis_at, eof_mean_at
export eof_residual_variance_at, eof_effective_measurement_covariance
export reconstruct_eof_field, eof_variance_fraction
export eof_coefficients, eof_model_at_coefficients
export eof_mode_values, eof_mode_grid
export plot_eof_field, plot_eof_spectrum, plot_eof_mode, plot_eof_coefficients

const EOF_MODEL_FORMAT_VERSION = 3

# ---------------------------------------------------------------------------
# Part I: offline EOF learning and persistence
# ---------------------------------------------------------------------------

"""
    eof_model_data_loader(source; kwargs...) -> AbstractMatrix

Load an environmental dataset and return the single numerical snapshot matrix
used to learn an EOF model. Users of the EOF backend must add a method
specialized for their own source type.

The returned matrix must have shape `(n_features, n_snapshots)`:

  * each column is the complete environmental state at one time;
  * each row is the same scalar field component/grid location at every time;
  * chronological order should be retained so the stored coefficient history
    can be interpreted and the archive's temporal coverage can be audited;
  * all entries must be finite. Land cells, fill values, and missing values must
    be removed or imputed by the loader;
  * row ordering must agree with the `locations` and optional `weights` passed
    to `EOFClimateModelParameters` or `initialize_eof_climate_model`.

For a multivariate EOF, concatenate variables vertically after applying the
desired physical nondimensionalization. The loader deliberately owns
file-format, masking, staggering, unit-conversion, and subsampling policy;
SCRIBE owns the decomposition after the matrix has been produced.
"""
function eof_model_data_loader(source; kwargs...)
    throw(ArgumentError(
        "No eof_model_data_loader method is defined for $(typeof(source)). " *
        "Implement one that returns an n_features × n_snapshots matrix.",
    ))
end

"""A weighted, truncated EOF decomposition learned from snapshot data."""
struct EOFDecomposition
    mean::Vector{Float64}
    modes::Matrix{Float64}
    eigenvalues::Vector{Float64}
    coefficients::Matrix{Float64}
    weights::Vector{Float64}
    residual_variance::Vector{Float64}
    explained_variance::Float64
    total_variance::Float64
    n_samples::Int

    function EOFDecomposition(
        mean,
        modes,
        eigenvalues,
        coefficients,
        weights,
        residual_variance,
        explained_variance,
        total_variance,
        n_samples,
    )
        μ = Vector{Float64}(mean)
        E = Matrix{Float64}(modes)
        λ = Vector{Float64}(eigenvalues)
        Φ = Matrix{Float64}(coefficients)
        w = Vector{Float64}(weights)
        residual = Vector{Float64}(residual_variance)
        m, r = size(E)

        length(μ) == m ||
            throw(DimensionMismatch("EOF mean and mode row counts differ."))
        length(λ) == r ||
            throw(DimensionMismatch("EOF eigenvalue and mode counts differ."))
        size(Φ, 1) == r ||
            throw(DimensionMismatch("EOF coefficient and mode counts differ."))
        length(w) == m ||
            throw(DimensionMismatch("EOF weight and feature counts differ."))
        length(residual) == m ||
            throw(DimensionMismatch(
                "EOF residual-variance and feature counts differ.",
            ))
        n_samples >= 2 ||
            throw(ArgumentError("An EOF decomposition needs at least 2 samples."))
        size(Φ, 2) in (0, n_samples) ||
            throw(DimensionMismatch(
                "EOF coefficient history must be empty or have n_samples columns.",
            ))
        all(isfinite, μ) && all(isfinite, E) && all(isfinite, λ) &&
            all(isfinite, Φ) && all(isfinite, w) &&
            all(isfinite, residual) ||
            throw(ArgumentError("EOF decomposition contains non-finite values."))
        all(>(0.0), w) ||
            throw(ArgumentError("EOF weights must be strictly positive."))
        all(>=(0.0), λ) ||
            throw(ArgumentError("EOF eigenvalues must be nonnegative."))
        all(>=(0.0), residual) ||
            throw(ArgumentError("EOF residual variances must be nonnegative."))

        new(
            μ,
            E,
            λ,
            Φ,
            w,
            residual,
            Float64(explained_variance),
            Float64(total_variance),
            Int(n_samples),
        )
    end
end

function thin_orthonormal_basis(Y)
    n_columns = min(size(Y)...)
    n_columns == 0 &&
        return Matrix{Float64}(undef, size(Y, 1), 0)
    factor = qr(Y)
    Matrix(factor.Q[:, 1:n_columns])
end

"""Compute a fixed-rank randomized SVD without full singular-vector matrices."""
function randomized_eof_svd(
    Z::AbstractMatrix,
    target_rank::Integer;
    oversample::Integer=10,
    power_iterations::Integer=1,
    rng::AbstractRNG=default_rng(),
)
    m, n = size(Z)
    q = min(m, n)
    1 <= target_rank <= q ||
        throw(ArgumentError("target_rank must lie in 1:$q."))
    oversample >= 0 ||
        throw(ArgumentError("oversample must be nonnegative."))
    power_iterations >= 0 ||
        throw(ArgumentError("power_iterations must be nonnegative."))

    sketch_rank = min(q, target_rank + oversample)
    Ω = randn(rng, n, sketch_rank)
    Q = thin_orthonormal_basis(Z * Ω)
    for _ in 1:power_iterations
        Q = thin_orthonormal_basis(Z * thin_orthonormal_basis(Z' * Q))
    end

    small_factor = svd(Q' * Z; full=false)
    keep = 1:min(target_rank, length(small_factor.S))
    Dict(
        :U => Q * small_factor.U[:, keep],
        :S => small_factor.S[keep],
        :V => small_factor.V[:, keep],
    )
end

function eof_svd(
    Z::AbstractMatrix;
    algorithm::Symbol,
    target_rank::Integer,
    oversample::Integer,
    power_iterations::Integer,
    rng::AbstractRNG,
)
    q = min(size(Z)...)
    selected_algorithm = algorithm == :auto ?
        (target_rank < q && q > 256 ? :randomized : :exact) :
        algorithm

    if selected_algorithm == :exact
        factor = svd(Z; full=false)
        Dict(:U => factor.U, :S => factor.S, :V => factor.V)
    elseif selected_algorithm == :randomized
        randomized_eof_svd(
            Z,
            target_rank;
            oversample,
            power_iterations,
            rng,
        )
    else
        throw(ArgumentError(
            "Unknown EOF decomposition algorithm $algorithm. " *
            "Use :auto, :exact, or :randomized.",
        ))
    end
end

function selected_eof_rank(
    singular_values,
    total_variance;
    rank,
    variance_fraction,
)
    if !isnothing(rank)
        return Int(rank)
    end
    0.0 < variance_fraction <= 1.0 ||
        throw(ArgumentError("variance_fraction must lie in (0, 1]."))
    cumulative = cumsum(abs2.(singular_values)) ./ total_variance
    selected = findfirst(>=(variance_fraction), cumulative)
    isnothing(selected) &&
        throw(ArgumentError(
            "The computed singular spectrum explains only " *
            "$(round(100 * last(cumulative); digits=2))% of total variance. " *
            "Increase max_rank or use algorithm=:exact.",
        ))
    selected
end

"""
    fit_eof_decomposition(data; kwargs...) -> EOFDecomposition

Learn fixed, spatially weighted EOF vectors from a snapshot matrix whose rows
are features and columns are chronological snapshots.

`weights` defines the diagonal spatial inner product `a' * Diagonal(weights) *
b`; ROMS scalar fields should normally use cell area or volume weights. If
`rank` is omitted, the smallest computed rank reaching `variance_fraction` is
retained. For large data, `algorithm=:auto` uses a randomized SVD bounded by
`max_rank` (default 100); specify a larger `max_rank` if necessary.
"""
function fit_eof_decomposition(
    data::AbstractMatrix{<:Real};
    weights=nothing,
    rank::Union{Nothing, Integer}=nothing,
    variance_fraction::Real=0.95,
    max_rank::Union{Nothing, Integer}=nothing,
    algorithm::Symbol=:auto,
    oversample::Integer=10,
    power_iterations::Integer=1,
    rng::AbstractRNG=default_rng(),
)
    m, n = size(data)
    m > 0 || throw(ArgumentError("EOF data matrix has no features."))
    n >= 2 || throw(ArgumentError("EOF data matrix needs at least 2 snapshots."))
    all(isfinite, data) ||
        throw(ArgumentError(
            "EOF data contain NaN or Inf. Clean them in eof_model_data_loader.",
        ))

    q = min(m, n)
    !isnothing(rank) && !(1 <= rank <= q) &&
        throw(ArgumentError("rank must lie in 1:$q."))
    !isnothing(max_rank) && max_rank < 1 &&
        throw(ArgumentError("max_rank must be positive."))
    candidate_rank = if algorithm == :exact
        q
    elseif !isnothing(rank)
        Int(rank)
    elseif isnothing(max_rank)
        min(100, q)
    else
        min(Int(max_rank), q)
    end

    w = isnothing(weights) ? ones(Float64, m) : Vector{Float64}(weights)
    length(w) == m ||
        throw(DimensionMismatch("weights must contain one value per data row."))
    all(isfinite, w) && all(>(0.0), w) ||
        throw(ArgumentError("weights must be finite and strictly positive."))

    μ = vec(mean(data; dims=2))
    sqrt_w = sqrt.(w)
    Z = (sqrt_w .* (Matrix{Float64}(data) .- μ)) ./ sqrt(n - 1)
    total_variance = sum(abs2, Z)
    total_variance > 0.0 ||
        throw(ArgumentError("EOF data have zero anomaly variance."))

    factor = eof_svd(
        Z;
        algorithm,
        target_rank=candidate_rank,
        oversample,
        power_iterations,
        rng,
    )
    retained_rank = selected_eof_rank(
        factor[:S],
        total_variance;
        rank,
        variance_fraction,
    )
    retained_rank <= length(factor[:S]) ||
        throw(ArgumentError(
            "Requested rank $retained_rank exceeds the computed spectrum.",
        ))

    keep = 1:retained_rank
    singular_values = factor[:S][keep]
    eigenvalues = abs2.(singular_values)
    modes = factor[:U][:, keep] ./ sqrt_w
    coefficients =
        sqrt(n - 1) .* (singular_values .* factor[:V][:, keep]')
    point_variance = vec(sum(abs2, Z; dims=2)) ./ w
    represented_variance = abs2.(modes) * eigenvalues
    residual_variance = max.(point_variance - represented_variance, 0.0)
    explained_variance = sum(eigenvalues) / total_variance

    EOFDecomposition(
        μ,
        modes,
        eigenvalues,
        coefficients,
        w,
        residual_variance,
        explained_variance,
        total_variance,
        n,
    )
end

function eof_covariance_matrix(covariance, r::Integer, name::AbstractString)
    matrix = if covariance isa Number
        Float64(covariance) .* Matrix{Float64}(I, r, r)
    elseif covariance isa AbstractVector
        length(covariance) == r ||
            throw(DimensionMismatch("$name must contain $r diagonal entries."))
        Matrix(Diagonal(Float64.(covariance)))
    else
        Matrix{Float64}(covariance)
    end
    size(matrix) == (r, r) ||
        throw(DimensionMismatch("$name must be $r×$r."))
    all(isfinite, matrix) ||
        throw(ArgumentError("$name must contain only finite values."))
    matrix = Matrix(Symmetric((matrix + matrix') / 2))
    tolerance = sqrt(eps(Float64)) * max(opnorm(matrix), 1.0)
    eigmin(Symmetric(matrix)) >= -tolerance ||
        throw(ArgumentError("$name must be positive semidefinite."))
    matrix
end

"""A fixed EOF model space and a noise-permissive coefficient prior.

The retained mean and EOF modes define the model space. Online coefficients
follow the structural random walk `ϕ[k+1] = ϕ[k] + w[k]`, with
`w[k] ~ N(0, Q)`. `P₀` and `Q` may be supplied directly or obtained from
`calibrate_eof_uncertainty`. This remains a random-walk uncertainty model; it
does not fit a directional temporal law from the offline coefficient history.
"""
struct EOFClimateModelParameters <: SCRIBEModelParameters
    nᵩ::Int
    decomposition::EOFDecomposition
    locations::Matrix{Float64}
    ϕ₀::Vector{Float64}
    P₀::Matrix{Float64}
    Q::Matrix{Float64}
    interpolation::Symbol
    interpolation_neighbors::Int
    metadata::Dict{String, Any}

    function EOFClimateModelParameters(
        decomposition::EOFDecomposition,
        locations,
        ϕ₀,
        P₀,
        Q,
        interpolation,
        interpolation_neighbors,
        metadata,
    )
        m, r = size(decomposition.modes)
        locations_matrix = Matrix{Float64}(locations)
        size(locations_matrix, 1) == m ||
            throw(DimensionMismatch(
                "locations must contain one row per EOF feature.",
            ))
        size(locations_matrix, 2) > 0 ||
            throw(ArgumentError("locations must have at least one coordinate."))
        all(isfinite, locations_matrix) ||
            throw(ArgumentError("locations contain non-finite values."))

        initial = Vector{Float64}(ϕ₀)
        initial_covariance = Matrix{Float64}(P₀)
        length(initial) == r ||
            throw(DimensionMismatch("ϕ₀ must contain $r coefficients."))
        all(isfinite, initial) ||
            throw(ArgumentError("ϕ₀ must contain only finite values."))
        size(initial_covariance) == (r, r) ||
            throw(DimensionMismatch("P₀ must be $r×$r."))
        all(isfinite, initial_covariance) ||
            throw(ArgumentError("P₀ must contain only finite values."))
        initial_covariance =
            Matrix(Symmetric((initial_covariance + initial_covariance') / 2))
        cholesky(Symmetric(initial_covariance); check=true)
        process_covariance = eof_covariance_matrix(
            Q,
            r,
            "process_covariance",
        )
        interpolation in (:nearest, :inverse_distance) ||
            throw(ArgumentError(
                "interpolation must be :nearest or :inverse_distance.",
            ))
        interpolation_neighbors >= 1 ||
            throw(ArgumentError("interpolation_neighbors must be positive."))

        new(
            r,
            decomposition,
            locations_matrix,
            initial,
            initial_covariance,
            process_covariance,
            interpolation,
            Int(interpolation_neighbors),
            Dict{String, Any}(string(k) => v for (k, v) in pairs(metadata)),
        )
    end
end

default_eof_locations(n_features) =
    reshape(collect(1.0:n_features), :, 1)

"""
    EOFClimateModelParameters(decomposition; process_covariance, kwargs...)

Construct the shared SCRIBE parameters for a fixed EOF space.
`process_covariance` is the random-walk covariance per model update and is
required explicitly. A scalar creates `qI`, a vector creates a diagonal
covariance, and a matrix supplies the full coefficient covariance. The default
initial coefficient is zero, the coordinate of the learned mean field; the
default initial covariance is the retained archival coefficient covariance.
"""
function EOFClimateModelParameters(
    decomposition::EOFDecomposition;
    process_covariance,
    locations=default_eof_locations(length(decomposition.mean)),
    ϕ₀=zeros(length(decomposition.eigenvalues)),
    prior_covariance=Matrix(Diagonal(max.(
        decomposition.eigenvalues,
        eps(Float64),
    ))),
    interpolation::Symbol=:inverse_distance,
    interpolation_neighbors::Integer=4,
    metadata=Dict{String, Any}(),
)
    EOFClimateModelParameters(
        decomposition,
        locations,
        ϕ₀,
        prior_covariance,
        process_covariance,
        interpolation,
        interpolation_neighbors,
        metadata,
    )
end

"""
    EOFClimateModelParameters(data; process_covariance, kwargs...)

Fit an `EOFDecomposition` from the snapshot columns in `data`, then construct
the corresponding random-walk SCRIBE parameters. No lagged coefficient model
is fitted from the snapshot ordering.
"""
function EOFClimateModelParameters(
    data::AbstractMatrix{<:Real};
    process_covariance,
    locations=default_eof_locations(size(data, 1)),
    weights=nothing,
    rank::Union{Nothing, Integer}=nothing,
    variance_fraction::Real=0.95,
    max_rank::Union{Nothing, Integer}=nothing,
    algorithm::Symbol=:auto,
    oversample::Integer=10,
    power_iterations::Integer=1,
    rng::AbstractRNG=default_rng(),
    ϕ₀=nothing,
    prior_covariance=nothing,
    interpolation::Symbol=:inverse_distance,
    interpolation_neighbors::Integer=4,
    metadata=Dict{String, Any}(),
)
    decomposition = fit_eof_decomposition(
        data;
        weights,
        rank,
        variance_fraction,
        max_rank,
        algorithm,
        oversample,
        power_iterations,
        rng,
    )
    initial = isnothing(ϕ₀) ? zeros(length(decomposition.eigenvalues)) : ϕ₀
    initial_covariance = isnothing(prior_covariance) ?
        Matrix(Diagonal(max.(decomposition.eigenvalues, eps(Float64)))) :
        prior_covariance
    EOFClimateModelParameters(
        decomposition;
        process_covariance,
        locations,
        ϕ₀=initial,
        prior_covariance=initial_covariance,
        interpolation,
        interpolation_neighbors,
        metadata,
    )
end

eof_mean(params::EOFClimateModelParameters) = params.decomposition.mean
eof_modes(params::EOFClimateModelParameters) = params.decomposition.modes
eof_prior_covariance(params::EOFClimateModelParameters) = params.P₀
eof_process_covariance(params::EOFClimateModelParameters) = params.Q
eof_residual_variance(params::EOFClimateModelParameters) =
    params.decomposition.residual_variance

"""A SCRIBE system snapshot represented in a fixed EOF basis."""
struct EOFClimateModel <: SCRIBEModel
    k::Int
    params::EOFClimateModelParameters
    ϕ::Vector{Float64}

    function EOFClimateModel(k::Integer, params, ϕ)
        k >= 1 || throw(ArgumentError("EOF model time must be positive."))
        coefficients = Vector{Float64}(ϕ)
        length(coefficients) == params.nᵩ ||
            throw(DimensionMismatch(
                "EOF state needs $(params.nᵩ) coefficients.",
            ))
        all(isfinite, coefficients) ||
            throw(ArgumentError("EOF coefficients must be finite."))
        new(Int(k), params, coefficients)
    end
end

function initialize_eof_climate_model(
    data::AbstractMatrix{<:Real};
    k::Integer=1,
    kwargs...,
)
    params = EOFClimateModelParameters(data; kwargs...)
    initialize_SCRIBEModel_from_parameters(params; k)
end

"""
    initialize_eof_climate_model(source; loader_kwargs=Dict(), kwargs...)

Load a user-defined environmental source with `eof_model_data_loader`, learn
the fixed EOF space, and initialize its random-walk runtime model.
All keywords other than `loader_kwargs` and `k` are forwarded to
`EOFClimateModelParameters`.
"""
function initialize_eof_climate_model(
    source;
    loader_kwargs::AbstractDict{Symbol}=Dict{Symbol,Any}(),
    k::Integer=1,
    kwargs...,
)
    data = eof_model_data_loader(source; loader_kwargs...)
    data isa AbstractMatrix ||
        throw(ArgumentError(
            "eof_model_data_loader must return an AbstractMatrix; got " *
            "$(typeof(data)).",
        ))
    initialize_eof_climate_model(data; k, kwargs...)
end

function eof_artifact_dictionary(
    params::EOFClimateModelParameters;
    include_coefficients::Bool=true,
)
    decomposition = params.decomposition
    artifact = Dict{String, Any}(
        "scribe_eof_format_version" => EOF_MODEL_FORMAT_VERSION,
        "mean" => decomposition.mean,
        "modes" => decomposition.modes,
        "eigenvalues" => decomposition.eigenvalues,
        "coefficients" => include_coefficients ?
            decomposition.coefficients :
            zeros(Float64, params.nᵩ, 0),
        "weights" => decomposition.weights,
        "residual_variance" => decomposition.residual_variance,
        "explained_variance" => decomposition.explained_variance,
        "total_variance" => decomposition.total_variance,
        "n_samples" => decomposition.n_samples,
        "locations" => params.locations,
        "phi0" => params.ϕ₀,
        "prior_covariance" => params.P₀,
        "process_covariance" => params.Q,
        "interpolation" => String(params.interpolation),
        "interpolation_neighbors" => params.interpolation_neighbors,
        "metadata" => params.metadata,
    )
    artifact
end

"""
    save_eof_model(path, model_or_parameters; include_coefficients=true)

Write a learned EOF model artifact to a MATLAB `.mat` file. The artifact
contains everything required to reconstruct `EOFClimateModelParameters`
without repeating the decomposition.
"""
function save_eof_model(
    path::AbstractString,
    params::EOFClimateModelParameters;
    include_coefficients::Bool=true,
)
    endswith(lowercase(path), ".mat") ||
        throw(ArgumentError("EOF model artifacts must use a .mat extension."))
    matwrite(
        path,
        eof_artifact_dictionary(params; include_coefficients);
        compress=true,
    )
    path
end

save_eof_model(path::AbstractString, model::EOFClimateModel; kwargs...) =
    save_eof_model(path, model.params; kwargs...)

function required_artifact_value(artifact, key)
    haskey(artifact, key) ||
        throw(ArgumentError("EOF model artifact is missing `$key`."))
    artifact[key]
end

function artifact_metadata(value)
    value isa AbstractDict ||
        return Dict{String, Any}("loaded_metadata" => value)
    Dict{String, Any}(string(k) => v for (k, v) in pairs(value))
end

"""Load `EOFClimateModelParameters` from a SCRIBE-generated MATLAB artifact."""
function load_eof_model_parameters(path::AbstractString)
    artifact = matread(path)
    version = Int(required_artifact_value(
        artifact,
        "scribe_eof_format_version",
    ))
    version == EOF_MODEL_FORMAT_VERSION ||
        throw(ArgumentError(
            "Unsupported EOF artifact version $version; expected " *
            "$EOF_MODEL_FORMAT_VERSION.",
        ))

    stored_coefficients = Matrix{Float64}(required_artifact_value(
        artifact,
        "coefficients",
    ))
    n_modes = size(required_artifact_value(artifact, "modes"), 2)
    coefficients = isempty(stored_coefficients) ?
        zeros(Float64, n_modes, 0) :
        stored_coefficients
    decomposition = EOFDecomposition(
        vec(required_artifact_value(artifact, "mean")),
        required_artifact_value(artifact, "modes"),
        vec(required_artifact_value(artifact, "eigenvalues")),
        coefficients,
        vec(required_artifact_value(artifact, "weights")),
        vec(required_artifact_value(artifact, "residual_variance")),
        required_artifact_value(artifact, "explained_variance"),
        required_artifact_value(artifact, "total_variance"),
        Int(required_artifact_value(artifact, "n_samples")),
    )

    EOFClimateModelParameters(
        decomposition;
        process_covariance=required_artifact_value(
            artifact,
            "process_covariance",
        ),
        locations=required_artifact_value(artifact, "locations"),
        ϕ₀=vec(required_artifact_value(artifact, "phi0")),
        prior_covariance=required_artifact_value(
            artifact,
            "prior_covariance",
        ),
        interpolation=Symbol(required_artifact_value(
            artifact,
            "interpolation",
        )),
        interpolation_neighbors=Int(required_artifact_value(
            artifact,
            "interpolation_neighbors",
        )),
        metadata=artifact_metadata(get(
            artifact,
            "metadata",
            Dict{String, Any}(),
        )),
    )
end

function load_eof_climate_model(path::AbstractString; k::Integer=1)
    params = load_eof_model_parameters(path)
    initialize_SCRIBEModel_from_parameters(params; k)
end

# ---------------------------------------------------------------------------
# Part II: online SCRIBE dispatch for a fixed EOF basis
# ---------------------------------------------------------------------------

function initialize_SCRIBEModel_from_parameters(
    params::EOFClimateModelParameters;
    k=1,
)
    EOFClimateModel(k, params, copy(params.ϕ₀))
end

get_model_time(model::EOFClimateModel) = model.k

eof_mean(model::EOFClimateModel) = eof_mean(model.params)
eof_modes(model::EOFClimateModel) = eof_modes(model.params)
eof_prior_covariance(model::EOFClimateModel) =
    eof_prior_covariance(model.params)
eof_process_covariance(model::EOFClimateModel) =
    eof_process_covariance(model.params)
eof_residual_variance(model::EOFClimateModel) =
    eof_residual_variance(model.params)

function sample_eof_process_noise(
    Q::AbstractMatrix{<:Real};
    rng::AbstractRNG=default_rng(),
)
    factor = eigen(Symmetric(Matrix{Float64}(Q)))
    factor.vectors * (sqrt.(max.(factor.values, 0.0)) .* randn(rng, size(Q, 1)))
end

function update_SCRIBEModel(model::EOFClimateModel)
    EOFClimateModel(
        model.k + 1,
        model.params,
        model.ϕ + sample_eof_process_noise(model.params.Q),
    )
end

function update_SCRIBEModel(model::EOFClimateModel, ϕₖ)
    EOFClimateModel(model.k + 1, model.params, ϕₖ)
end

"""Scalar sensor-noise behavior for synthetic or real EOF observations."""
struct EOFObserverBehavior <: SCRIBEObserverBehavior
    variance::Float64
    include_truncation_error::Bool

    function EOFObserverBehavior(
        variance::Real=0.1;
        include_truncation_error::Bool=true,
    )
        variance > 0.0 ||
            throw(ArgumentError("Observation variance must be positive."))
        new(Float64(variance), include_truncation_error)
    end
end

measurement_noise_covariance(
    behavior::EOFObserverBehavior,
    n_samples,
) = behavior.variance * I(n_samples)

"""An EOF observation, including raw values and their reduced dynamics."""
struct EOFObserverState <: SCRIBEObserverState
    k::Int
    nₛ::Int
    X::Matrix{Float64}
    H::Matrix{Float64}
    mean::Vector{Float64}
    v::Dict{Symbol, AbstractArray{Float64}}
    z::Vector{Float64}
end

function eof_query_locations(X)
    if X isa Number
        reshape(Float64[X], 1, 1)
    elseif X isa AbstractVector
        reshape(Vector{Float64}(X), 1, :)
    else
        Matrix{Float64}(X)
    end
end

"""
    eof_interpolation_matrix(params, X)

Construct `L_X`, mapping values at EOF feature locations to query rows in `X`.
`:nearest` uses one feature; `:inverse_distance` uses up to
`interpolation_neighbors`. Coordinates are interpreted in the Euclidean metric
and their columns must therefore use commensurate scales. Latitude/longitude
users should work in a sufficiently local domain or provide projected
coordinates.
"""
function eof_interpolation_matrix(
    params::EOFClimateModelParameters,
    X,
)
    queries = eof_query_locations(X)
    locations = params.locations
    size(queries, 2) == size(locations, 2) ||
        throw(DimensionMismatch(
            "Query locations have $(size(queries, 2)) coordinates but EOF " *
            "locations have $(size(locations, 2)).",
        ))
    n_queries = size(queries, 1)
    n_features = size(locations, 1)
    interpolation = zeros(Float64, n_queries, n_features)
    n_neighbors = min(params.interpolation_neighbors, n_features)

    for query_index in axes(queries, 1)
        query = @view queries[query_index, :]
        distances_squared = vec(sum(
            abs2,
            locations .- query';
            dims=2,
        ))
        nearest = argmin(distances_squared)
        if params.interpolation == :nearest ||
           distances_squared[nearest] <= eps(Float64)
            interpolation[query_index, nearest] = 1.0
        else
            neighbors = partialsortperm(
                distances_squared,
                1:n_neighbors,
            )
            inverse_distances =
                1.0 ./ sqrt.(max.(distances_squared[neighbors], eps(Float64)))
            interpolation[query_index, neighbors] .=
                inverse_distances ./ sum(inverse_distances)
        end
    end
    interpolation
end

function eof_basis_at(model::EOFClimateModel, X)
    eof_interpolation_matrix(model.params, X) *
        eof_modes(model)
end

function eof_mean_at(model::EOFClimateModel, X)
    eof_interpolation_matrix(model.params, X) *
        eof_mean(model)
end

function eof_residual_variance_at(model::EOFClimateModel, X)
    interpolation = eof_interpolation_matrix(model.params, X)
    abs2.(interpolation) * eof_residual_variance(model)
end

function eof_effective_measurement_covariance(
    model::EOFClimateModel,
    X,
    R;
    include_truncation_error::Bool=true,
)
    queries = eof_query_locations(X)
    sensor_covariance = measurement_noise_covariance(R, size(queries, 1))
    include_truncation_error || return Matrix(sensor_covariance)
    Matrix(sensor_covariance) +
        Diagonal(eof_residual_variance_at(model, queries))
end

function compute_obs_dynamics(
    model::EOFClimateModel,
    X::Matrix{Float64};
    kwargs...,
)
    eof_basis_at(model, X), X
end

function scribe_observations(
    X::Matrix{Float64},
    model::EOFClimateModel,
    behavior::EOFObserverBehavior,
)
    n_samples = size(X, 1)
    H = eof_basis_at(model, X)
    mean_values = eof_mean_at(model, X)
    R = eof_effective_measurement_covariance(
        model,
        X,
        behavior.variance;
        include_truncation_error=behavior.include_truncation_error,
    )
    noise = rand(Gaussian(zeros(n_samples), R))
    z = mean_values + H * model.ϕ + noise
    EOFObserverState(
        model.k,
        n_samples,
        X,
        H,
        mean_values,
        Dict{Symbol, AbstractArray{Float64}}(
            :R => R,
            :R_sensor => behavior.variance *
                Matrix{Float64}(I, n_samples, n_samples),
            :k => noise,
        ),
        z,
    )
end

"""
Scribe externally supplied environmental data against a fixed EOF model.
`measurement.z` remains the raw field value; the climatological mean is stored
separately and removed by the EOF information-filter dispatch. When enabled,
the learned truncation variance is added to the supplied sensor covariance.
"""
function scribe_observations(
    measurement::SensorObservation,
    model::EOFClimateModel,
    behavior::EOFObserverBehavior,
)
    measurement.k == model.k ||
        throw(ArgumentError(
            "Observation time $(measurement.k) does not match model time " *
            "$(model.k).",
        ))
    X = measurement.X
    n_samples = length(measurement.z)
    H = eof_basis_at(model, X)
    mean_values = eof_mean_at(model, X)
    sensor_covariance = isnothing(measurement.R) ?
        behavior.variance .* Matrix{Float64}(I, n_samples, n_samples) :
        measurement.R
    R = eof_effective_measurement_covariance(
        model,
        X,
        sensor_covariance;
        include_truncation_error=behavior.include_truncation_error,
    )
    EOFObserverState(
        measurement.k,
        n_samples,
        X,
        H,
        mean_values,
        Dict{Symbol, AbstractArray{Float64}}(
            :R => R,
            :R_sensor => Matrix(sensor_covariance),
            :k => zeros(n_samples),
        ),
        copy(measurement.z),
    )
end

function reconstruct_eof_field(
    params::EOFClimateModelParameters;
    coefficients=params.ϕ₀,
)
    eof_mean(params) .+ eof_modes(params) * coefficients
end

function reconstruct_eof_field(
    model::EOFClimateModel;
    coefficients=model.ϕ,
)
    reconstruct_eof_field(model.params; coefficients)
end

"""Project a complete field snapshot into an existing EOF coordinate system."""
function eof_coefficients(params::EOFClimateModelParameters, snapshot)
    params.decomposition.modes' * (
        params.decomposition.weights .* (snapshot .- params.decomposition.mean)
    )
end

eof_coefficients(model::EOFClimateModel, snapshot) =
    eof_coefficients(model.params, snapshot)

"""Create an EOF model centered at a supplied coefficient vector."""
function eof_model_at_coefficients(params::EOFClimateModelParameters, coefficients)
    centered = EOFClimateModelParameters(
        params.decomposition;
        process_covariance=eof_process_covariance(params),
        locations=params.locations,
        ϕ₀=coefficients,
        prior_covariance=eof_prior_covariance(params),
        interpolation=params.interpolation,
        interpolation_neighbors=params.interpolation_neighbors,
        metadata=copy(params.metadata),
    )
    initialize_SCRIBEModel_from_parameters(centered)
end

eof_model_at_coefficients(model::EOFClimateModel, coefficients) =
    eof_model_at_coefficients(model.params, coefficients)

function predict_SCRIBEModel(model::EOFClimateModel, X)
    queries = eof_query_locations(X)
    prediction = eof_interpolation_matrix(model.params, queries) *
        reconstruct_eof_field(model)
    X isa AbstractMatrix ? prediction : only(prediction)
end

function init_agent_info(
    params::EOFClimateModelParameters;
    prior_covariance::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
    model_time::Integer=1,
)
    P₀ = isnothing(prior_covariance) ?
        params.P₀ :
        Matrix{Float64}(prior_covariance)
    size(P₀) == (params.nᵩ, params.nᵩ) ||
        throw(DimensionMismatch(
            "Initial covariance must be $(params.nᵩ)×$(params.nᵩ).",
        ))
    P₀ = Matrix(Symmetric((P₀ + P₀') / 2))
    factor = cholesky(Symmetric(P₀); check=true)
    Y₀ = Matrix(factor \ Matrix{Float64}(I, params.nᵩ, params.nᵩ))
    KFEnvInfo(
        Y₀ * params.ϕ₀,
        Y₀,
        zeros(params.nᵩ),
        zeros(params.nᵩ, params.nᵩ),
    )
end

function initialize_estimators(
    system::KFEnvScribe,
    params::EOFClimateModelParameters,
)
    A = Matrix{Float64}(I, params.nᵩ, params.nᵩ)
    b = zeros(params.nᵩ)
    get_A(_, _) = A
    get_b(_, _) = b
    get_ϕ(k, system) = system.estimates[k].estimate.ϕ
    get_Q(_, _) = params.Q
    get_H(k, system) = system.estimates[k].observations.H
    get_z(k, system) =
        system.estimates[k].observations.z -
        system.estimates[k].observations.mean
    get_R(k, system) = system.estimates[k].observations.v[:R]
    get_Y(k, system) = system.information[k].Y
    get_y(k, system) = system.information[k].y

    KFEstimators(
        system,
        k -> get_A(k, system),
        k -> get_b(k, system),
        k -> get_ϕ(k, system),
        k -> get_Q(k, system),
        k -> get_H(k, system),
        k -> get_z(k, system),
        k -> get_R(k, system),
        k -> get_Y(k, system),
        k -> get_y(k, system),
    )
end

function posterior_model_moments(
    model::EOFClimateModel,
    info::KFEnvInfo,
    X,
)
    interpolation = eof_interpolation_matrix(model.params, X)
    H = interpolation * eof_modes(model)
    coefficients = posterior_coefficient_moments(info)
    μ = interpolation * eof_mean(model) +
        H * coefficients[:μ]
    resolved_covariance = H * coefficients[:Σ] * H'
    residual_variance =
        abs2.(interpolation) * eof_residual_variance(model)
    Σ = resolved_covariance + Diagonal(residual_variance)
    Dict(:μ => μ, :Σ => prediction_symmetric(Σ))
end

"""
Condition an EOF coefficient information state on raw environmental
measurements. The climatological EOF mean is removed before forming the
information innovation. `R` is used as supplied; pass
`eof_effective_measurement_covariance(model, X, sensor_R)` when truncation
variance should be folded into a real sensor covariance. `EOFObserverState`
already stores this effective covariance when requested by its behavior.
"""
function condition_on_measurement(
    model::EOFClimateModel,
    info::KFEnvInfo,
    X,
    z,
    R,
)
    H = eof_basis_at(model, X)
    anomaly = (z isa Number ? [z] : z) - eof_mean_at(model, X)
    innovation = measurement_information(H, anomaly, R)
    Y⁺ = prediction_symmetric(info.Y + innovation[:δI])
    y⁺ = info.y + innovation[:δi]
    KFEnvInfo(y⁺, Y⁺, innovation[:δi], innovation[:δI])
end

# ---------------------------------------------------------------------------
# Part III: EOF-specific visualization helpers
# ---------------------------------------------------------------------------

function plot_eof_values(
    params::EOFClimateModelParameters,
    values;
    title,
    color=:balance,
    clims=nothing,
)
    if size(params.locations, 2) >= 2
        scatter(
            params.locations[:, 1],
            params.locations[:, 2];
            marker_z=values,
            markerstrokewidth=0,
            color,
            clims,
            aspect_ratio=:equal,
            title,
            xlabel="coordinate 1",
            ylabel="coordinate 2",
            label=nothing,
            colorbar_title="field value",
        )
    else
        plot(
            params.locations[:, 1],
            values;
            color,
            title,
            xlabel="coordinate",
            ylabel="field value",
            label=nothing,
        )
    end
end

"""Plot the field reconstructed from any coefficient vector in an EOF space."""
function plot_eof_field(
    params::EOFClimateModelParameters,
    coefficients=params.ϕ₀;
    title="EOF field",
    color=:balance,
    clims=nothing,
)
    values = reconstruct_eof_field(params; coefficients)
    plot_eof_values(params, values; title, color, clims)
end

function plot_eof_field(
    model::EOFClimateModel;
    coefficients=model.ϕ,
    title="EOF field",
    color=:balance,
    clims=nothing,
)
    plot_eof_field(model.params, coefficients; title, color, clims)
end

"""Return each retained EOF's fraction of total weighted anomaly variance."""
eof_variance_fraction(decomposition::EOFDecomposition) =
    decomposition.eigenvalues ./ decomposition.total_variance

eof_variance_fraction(params::EOFClimateModelParameters) =
    eof_variance_fraction(params.decomposition)

eof_variance_fraction(model::EOFClimateModel) =
    eof_variance_fraction(model.params)

function eof_mode_values(decomposition::EOFDecomposition, mode::Integer)
    1 <= mode <= size(decomposition.modes, 2) ||
        throw(BoundsError(decomposition.modes, (:, mode)))
    decomposition.modes[:, mode]
end

eof_mode_values(params::EOFClimateModelParameters, mode::Integer) =
    eof_mode_values(params.decomposition, mode)

eof_mode_values(model::EOFClimateModel, mode::Integer) =
    eof_mode_values(model.params, mode)

function eof_mode_grid(
    decomposition::EOFDecomposition,
    mode::Integer,
    dimensions::Tuple,
)
    prod(dimensions) == size(decomposition.modes, 1) ||
        throw(DimensionMismatch(
            "Grid dimensions $dimensions do not contain one entry per EOF feature.",
        ))
    reshape(eof_mode_values(decomposition, mode), dimensions)
end

eof_mode_grid(params::EOFClimateModelParameters, mode, dimensions) =
    eof_mode_grid(params.decomposition, mode, dimensions)

eof_mode_grid(model::EOFClimateModel, mode, dimensions) =
    eof_mode_grid(model.params, mode, dimensions)

"""Plot individual and cumulative retained variance fractions."""
function plot_eof_spectrum(decomposition::EOFDecomposition)
    fractions = eof_variance_fraction(decomposition)
    modes = collect(eachindex(fractions))
    plot(
        modes,
        fractions;
        marker=:circle,
        xlabel="EOF mode",
        ylabel="fraction of total variance",
        label="individual",
    )
    plot!(
        modes,
        cumsum(fractions);
        marker=:diamond,
        label="cumulative",
    )
end

plot_eof_spectrum(params::EOFClimateModelParameters) =
    plot_eof_spectrum(params.decomposition)

plot_eof_spectrum(model::EOFClimateModel) =
    plot_eof_spectrum(model.params)

"""
    plot_eof_mode(model_or_parameters, mode; grid_shape=nothing)

Plot an EOF as a heatmap when `grid_shape` is supplied, or as a colored
scatterplot at the stored two-dimensional feature locations otherwise.
"""
function plot_eof_mode(
    params::EOFClimateModelParameters,
    mode::Integer;
    grid_shape=nothing,
)
    values = eof_mode_values(params, mode)
    variance = 100 * eof_variance_fraction(params)[mode]
    title = "EOF $mode ($(round(variance; digits=2))% variance)"
    if !isnothing(grid_shape)
        heatmap(
            eof_mode_grid(params, mode, Tuple(grid_shape));
            title,
            xlabel="grid column",
            ylabel="grid row",
            color=:balance,
        )
    elseif size(params.locations, 2) >= 2
        scatter(
            params.locations[:, 1],
            params.locations[:, 2];
            marker_z=values,
            markerstrokewidth=0,
            color=:balance,
            title,
            xlabel="coordinate 1",
            ylabel="coordinate 2",
            label=nothing,
            colorbar_title="EOF amplitude",
        )
    else
        plot(
            params.locations[:, 1],
            values;
            title,
            xlabel="coordinate",
            ylabel="EOF amplitude",
            label="mode $mode",
        )
    end
end

plot_eof_mode(model::EOFClimateModel, mode::Integer; kwargs...) =
    plot_eof_mode(model.params, mode; kwargs...)

"""Plot the learned chronological coefficient histories."""
function plot_eof_coefficients(
    decomposition::EOFDecomposition;
    modes=1:min(5, size(decomposition.coefficients, 1)),
)
    isempty(decomposition.coefficients) &&
        throw(ArgumentError(
            "This EOF artifact does not contain coefficient history.",
        ))
    selected = collect(modes)
    plot(
        decomposition.coefficients[selected, :]';
        xlabel="snapshot",
        ylabel="EOF coefficient",
        label=permutedims(["ϕ$i" for i in selected]),
    )
end

plot_eof_coefficients(params::EOFClimateModelParameters; kwargs...) =
    plot_eof_coefficients(params.decomposition; kwargs...)

plot_eof_coefficients(model::EOFClimateModel; kwargs...) =
    plot_eof_coefficients(model.params; kwargs...)
