export EOFUncertaintyCalibration, calibrate_eof_uncertainty
export with_eof_uncertainty

"""Offline uncertainty learned for a fixed SCRIBE EOF coordinate system.

`P₀` describes uncertainty in an initial coefficient vector. `Q` describes
coefficient change over one archive timestep. Both are learned from ensembles
of chronological coefficient histories and regularized in EOF coordinates.
"""
@with_kw_noshow struct EOFUncertaintyCalibration
    P₀::Matrix{Float64}
    Q::Matrix{Float64}
    initial_samples::Int
    transition_samples::Int
    initial_shrinkage::Float64
    process_shrinkage::Float64
end

function eof_sample_covariance(samples::AbstractMatrix, shrinkage, variance_floor)
    dimension, count = size(samples)
    count >= 2 || throw(ArgumentError("uncertainty calibration needs at least two samples"))
    anomalies = Matrix{Float64}(samples) .- mean(samples; dims=2)
    covariance = anomalies * anomalies' / (count - 1)
    diagonal = Diagonal(diag(covariance))
    regularized = (1 - shrinkage) .* covariance + shrinkage .* diagonal
    scale = max(sum(diag(regularized)) / max(dimension, 1), 1.0)
    Matrix(Symmetric(regularized + variance_floor * scale * I))
end

function eof_transition_samples(histories)
    transitions = Matrix{Float64}[]
    for history in histories
        size(history, 2) >= 2 || continue
        push!(transitions, diff(Matrix{Float64}(history); dims=2))
    end
    isempty(transitions) && throw(ArgumentError(
        "process calibration needs at least one chronological coefficient transition",
    ))
    hcat(transitions...)
end

"""
    calibrate_eof_uncertainty(decomposition; histories, kwargs...)

Estimate the initial and random-walk covariance of an EOF model offline.
Each history is an `n_modes × n_times` matrix in the decomposition's fixed EOF
basis. History boundaries are kept separate so unrelated missions do not create
false transitions. When held-out SCRIBE experiments are available,
`initial_errors` should contain coefficient estimation errors and
`process_innovations` should contain one-step coefficient innovations. Otherwise
the archival coefficient ensemble and its within-history differences provide a
conservative default.
"""
function calibrate_eof_uncertainty(
    decomposition::EOFDecomposition;
    histories=[decomposition.coefficients],
    initial_errors=nothing,
    process_innovations=nothing,
    initial_shrinkage::Real=0.1,
    process_shrinkage::Real=0.2,
    variance_floor::Real=sqrt(eps(Float64)),
)
    dimension = length(decomposition.eigenvalues)
    coefficient_histories = Matrix{Float64}.(histories)
    all(size(history, 1) == dimension for history in coefficient_histories) ||
        throw(DimensionMismatch("coefficient histories must use the decomposition's EOF basis"))
    initial_ensemble = isnothing(initial_errors) ?
        hcat(coefficient_histories...) : Matrix{Float64}(initial_errors)
    transitions = isnothing(process_innovations) ?
        eof_transition_samples(coefficient_histories) :
        Matrix{Float64}(process_innovations)
    size(initial_ensemble, 1) == dimension ||
        throw(DimensionMismatch("initial errors must use the decomposition's EOF basis"))
    size(transitions, 1) == dimension ||
        throw(DimensionMismatch("process innovations must use the decomposition's EOF basis"))
    EOFUncertaintyCalibration(
        P₀=eof_sample_covariance(
            initial_ensemble, Float64(initial_shrinkage), Float64(variance_floor),
        ),
        Q=eof_sample_covariance(
            transitions, Float64(process_shrinkage), Float64(variance_floor),
        ),
        initial_samples=size(initial_ensemble, 2),
        transition_samples=size(transitions, 2),
        initial_shrinkage=Float64(initial_shrinkage),
        process_shrinkage=Float64(process_shrinkage),
    )
end

"""Return the same EOF model space with calibrated `P₀` and `Q`."""
function with_eof_uncertainty(
    params::EOFClimateModelParameters,
    calibration::EOFUncertaintyCalibration;
    initial_coefficients=params.ϕ₀,
)
    EOFClimateModelParameters(
        params.decomposition;
        process_covariance=calibration.Q,
        locations=params.locations,
        ϕ₀=initial_coefficients,
        prior_covariance=calibration.P₀,
        interpolation=params.interpolation,
        interpolation_neighbors=params.interpolation_neighbors,
        metadata=merge(params.metadata, Dict(
            "uncertainty_initial_samples" => calibration.initial_samples,
            "uncertainty_transition_samples" => calibration.transition_samples,
        )),
    )
end
