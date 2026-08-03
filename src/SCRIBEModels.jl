module SCRIBEModels

export SCRIBEModel, SCRIBEModelParameters, initialize_SCRIBEModel_from_parameters, update_SCRIBEModel, predict_SCRIBEModel
export SCRIBEObserverBehavior, SCRIBEObserverState, compute_obs_dynamics, scribe_observations
export SensorObservation, DataObserver, observe_data

using GaussianDistributions: Gaussian
using LinearAlgebra: Diagonal, I, Symmetric, cholesky, norm, ⋅, rank

"""Abstract type defined for specialization during model instantiation.

Currently defined types:
    - `LGSFModelParameters`
    - `EOFClimateModelParameters`
"""
abstract type SCRIBEModelParameters end

"""Generic model initialization function.

Takes parameter structure of a type inheriting from `SCRIBEModelParameters`.\\
Produces a model of a type inheriting from `SCRIBEModel`.
"""
function initialize_SCRIBEModel_from_parameters(params::SCRIBEModelParameters; k)
    # This function has no implementation and is intended to be specialized
    error("`initialize_model_from_parameters` is not implemented for the abstract type SCRIBEModelParameters. Please provide a specific implementation.")
end

"""Abstract type that collects model types. Useful for specialization.

Currently defined model types:
    - `LGSFModel`
    - `EOFClimateModel`
"""
abstract type SCRIBEModel end

"""Generic model update function.
"""
function update_SCRIBEModel(smodel::SCRIBEModel)
    # This function has no implementation and is intended to be specialized
    error("`update_SCRIBEModel` is not implemented for the abstract type SCRIBEModel. Please provide a specific implementation.")
end

"""Generic model prediction function.
"""
function predict_SCRIBEModel(smodel::SCRIBEModel)
    # This function has no implementation and is intended to be specialized
    error("`predict_SCRIBEModel` is not implemented for the abstract type SCRIBEModel. Please provide a specific implementation.")
end

"""Generic helper function for acquiring current model time.
"""
function get_model_time(smodel::SCRIBEModel)
    error("`get_model_time` is not implemented for the abstract type SCRIBEModel. Please provide a specific implementation.")
end

"""Abstract type that defines observer behavior.

This can include:
* Sensors parameters, such as the observation noise covariance
* Details that impact observations, such as a fixed set or function of sensing locations

Currently defined observer behavior types:
    - `LGSFObserverBehavior`
    - `EOFObserverBehavior`
    - `DataObserver`, which wraps either behavior around external data
"""
abstract type SCRIBEObserverBehavior end

"""Abstract type that collects the current state of the observer at timestep k.

This is notably the state *after* observations have been collected.

Currently defined observer state types:
    - `LGSFObserverState`
    - `EOFObserverState`
"""
abstract type SCRIBEObserverState end

"""
    SensorObservation(k, X, z; covariance=nothing)

A measurement obtained from a physical sensor, recorded dataset, simulator,
or other source outside the SCRIBE model. `X` contains one observation
location per row and `z` contains the corresponding raw measurements.

`covariance` is the sensor covariance associated with `z`. It may be a scalar,
a vector of marginal variances, or a full covariance matrix. When omitted, the
wrapped model-specific observer behavior supplies its default covariance.
"""
struct SensorObservation
    k::Int
    X::Matrix{Float64}
    z::Vector{Float64}
    R::Union{Nothing, Matrix{Float64}}

    function SensorObservation(
        k::Integer,
        X,
        z;
        covariance=nothing,
    )
        k >= 1 || throw(ArgumentError("Observation time must be positive."))
        locations = X isa AbstractVector ?
            reshape(Vector{Float64}(X), 1, :) :
            Matrix{Float64}(X)
        values = z isa Number ? Float64[z] : Vector{Float64}(z)
        size(locations, 1) == length(values) ||
            throw(DimensionMismatch(
                "SensorObservation needs one value per location row.",
            ))
        all(isfinite, locations) ||
            throw(ArgumentError("Observation locations must be finite."))
        all(isfinite, values) ||
            throw(ArgumentError("Observation values must be finite."))

        R = isnothing(covariance) ?
            nothing :
            sensor_observation_covariance(covariance, length(values))
        new(Int(k), locations, values, R)
    end
end

function sensor_observation_covariance(covariance, n::Integer)
    R = if covariance isa Number
        Float64(covariance) .* Matrix{Float64}(I, n, n)
    elseif covariance isa AbstractVector
        length(covariance) == n ||
            throw(DimensionMismatch(
                "Sensor covariance vector must have $n entries.",
            ))
        Matrix(Diagonal(Float64.(covariance)))
    else
        Matrix{Float64}(covariance)
    end
    size(R) == (n, n) ||
        throw(DimensionMismatch("Sensor covariance must be $n×$n."))
    all(isfinite, R) ||
        throw(ArgumentError("Sensor covariance must be finite."))
    R = Matrix(Symmetric((R + R') / 2))
    cholesky(Symmetric(R); check=true)
    R
end

"""
    DataObserver(sensor, behavior)

Wrap an external sensor or data source for use by SCRIBE. `sensor` must be
callable as `sensor(k, X)` and return either raw values `z` or a
`SensorObservation`. `behavior` remains responsible for model-specific sensor
semantics, such as EOF truncation-error covariance.

Unlike model-backed simulation observers, a `DataObserver` never evaluates a
ground-truth `SCRIBEModel`.
"""
struct DataObserver{S, B<:SCRIBEObserverBehavior} <: SCRIBEObserverBehavior
    sensor::S
    behavior::B
end

function observe_data(observer::DataObserver, k::Integer, X)
    locations = X isa AbstractVector ?
        reshape(Vector{Float64}(X), 1, :) :
        Matrix{Float64}(X)
    result = observer.sensor(k, locations)
    measurement = result isa SensorObservation ?
        result :
        SensorObservation(k, locations, result)
    measurement.k == k ||
        throw(ArgumentError(
            "Sensor returned time $(measurement.k) for requested time $k.",
        ))
    measurement.X == locations ||
        throw(ArgumentError(
            "Sensor returned locations different from those requested.",
        ))
    measurement
end

"""Generic observation dynamics calculation function.
"""
function compute_obs_dynamics(smodel::SCRIBEModel, X::Matrix{Float64})
    # This function has no implementation and is intended to be specialized
    error("`compute_obs_dynamics` is not implemented for the abstract type SCRIBEModel. Please provide a specific implementation.")
end

"""Generic observation function.
"""
function scribe_observations(X::Matrix{Float64}, smodel::SCRIBEModel, o_b::SCRIBEObserverBehavior)
    # This function has no implementation and is intended to be specialized
    error("`scribe_observations` is not implemented for the abstract types SCRIBEModel and SCRIBEObserverBehavior. Please provide a specific implementation.")
end

function scribe_observations(
    measurement::SensorObservation,
    smodel::SCRIBEModel,
    behavior::SCRIBEObserverBehavior,
)
    error(
        "Direct data observation is not implemented for $(typeof(smodel)) " *
        "and $(typeof(behavior)). Define a model-specific " *
        "scribe_observations(::SensorObservation, ::$(typeof(smodel)), " *
        "::$(typeof(behavior))) method.",
    )
end

"""Acquire external data and scribe it against the agent's current model."""
function scribe_observations(
    X::Matrix{Float64},
    smodel::SCRIBEModel,
    observer::DataObserver,
)
    measurement = observe_data(observer, get_model_time(smodel), X)
    scribe_observations(measurement, smodel, observer.behavior)
end

include("lineargaussianscalarfields.jl")

end
