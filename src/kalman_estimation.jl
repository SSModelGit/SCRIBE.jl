export KFEnvInfo, KFEnvScribe, initialize_scribe, next_agent_state, next_agent_time, next_agent_info_state
export KFEstimators, initialize_estimators, compute_info_priors, compute_innov_from_obs
export information_filter_update, recover_estimate_from_info
export progress_agent_env_filter, initialize_KF

"""Current discrete-time estimate of the linear system being observed.

Will store the discrete time of estimation for redundancy checking.
"""
struct KFEnvEstimate
    k::Integer
    estimate::SCRIBEModel
    observations::SCRIBEObserverState
    function KFEnvEstimate(k::Integer, estimate::SCRIBEModel, observations::SCRIBEObserverState)
        new(k, estimate, observations)
    end
end

"""Initialize an estimate and acquire its observation from a simulated world."""
function init_agent_estimate(world::SCRIBEModel, k::Integer,
                             params::SCRIBEModelParameters, bhv::SCRIBEObserverBehavior,
                             X::VecOrMat{Float64})
    KFEnvEstimate(k, initialize_SCRIBEModel_from_parameters(params, k=k),
                     scribe_observations(X,world,bhv))
end

"""Initialize an estimate by pulling an observation from external data."""
function init_agent_estimate(
    params::SCRIBEModelParameters,
    observer::DataObserver,
    X::Matrix{Float64},
    model_time::Integer=1,
)
    model = initialize_SCRIBEModel_from_parameters(params; k=model_time)
    KFEnvEstimate(
        model_time,
        model,
        scribe_observations(X, model, observer),
    )
end

"""Initialize an estimate from a measurement already acquired by a sensor."""
function init_agent_estimate(
    params::SCRIBEModelParameters,
    behavior::SCRIBEObserverBehavior,
    measurement::SensorObservation,
)
    model = initialize_SCRIBEModel_from_parameters(params; k=measurement.k)
    model_behavior = behavior isa DataObserver ?
        behavior.behavior :
        behavior
    KFEnvEstimate(
        measurement.k,
        model,
        scribe_observations(measurement, model, model_behavior),
    )
end

"""Current discrete-time information.

This is separate from the actual model estimate - this is what the Kalman Filter interacts with.
The model estimate is a representation of the system, which is recovered from this.

Combines y(t=k), Y(t=k), δi(t=k-1), and δI(t=k-1).
These should truthfully be discussed separately.
But to keep the implementation smoother, we aim to integrate them together.

Innovation is only understood after z(k) is gained.
However, y/Y(t=k) is necessary to update ϕⱼ(t=k), which then influences our choice of X(t=k) and thus z(k).
Hence, we cannot create a structure that defines δi/δI(t=k) at the same time as y/Y(t=k).

To allow for proper streamlining, we then say that δi/δI(t=k) is stored with y/Y(t=k+1).
This is fine, because we never actually return to δi/δI(t=k) after y/Y(t=k+1) is learned.
Essentially, the moment we calculate δi/δI(t=k), we immediately also calculate y/Y(t=k+1).
From then on, δi/δI(t=k) becomes useless, so this treatment of δi/δI(t=k) with y/Y(t=k+1) is okay.
"""
struct KFEnvInfo
    y::Vector{Float64}
    Y::Matrix{Float64}
    i::Vector{Float64}
    I::Matrix{Float64}

    KFEnvInfo(y::Vector{Float64}, Y::Matrix{Float64},
                 i::Vector{Float64}, I::Matrix{Float64}) = new(y,Y,i,I)
end


"""Construct a finite, positive-definite initial information state.

`prior_covariance` defaults to a weak prior of `1e3I`. Observations are deliberately
excluded here: the observation at `k=1` is incorporated exactly once by the first
filter update.
"""
function init_agent_info(params::SCRIBEModelParameters;
                         prior_covariance::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
                         model_time::Integer=1)
    nᵩ = params.nᵩ
    P₀ = isnothing(prior_covariance) ?
         1e3 * Matrix{Float64}(I, nᵩ, nᵩ) :
         Matrix{Float64}(prior_covariance)
    @assert size(P₀) == (nᵩ, nᵩ) "Initial covariance must be $(nᵩ)×$(nᵩ)."
    P₀ = Matrix(Symmetric((P₀ + P₀') / 2))
    P₀_factor = cholesky(Symmetric(P₀); check=true)
    Y₀ = Matrix(P₀_factor \ Matrix{Float64}(I, nᵩ, nᵩ))
    y₀ = Y₀ * params.ϕ₀
    KFEnvInfo(y₀, Y₀, zeros(nᵩ), zeros(nᵩ, nᵩ))
end

"""Specializing the `copy` function for KFEnvInfo.
"""
Base.copy(i::KFEnvInfo) = KFEnvInfo(copy(i.y), copy(i.Y), copy(i.i), copy(i.I))

"""This is the collection of the system over time.

This is a linearized representation.
The mutating elements are the vectors, which are appended to.

The timing of this is slightly unintuitive.
* Estimates carry two properties - the estimate of the system, and the observation.
    * System estimates are our estimates at time (k) **before** taking observations.
    * Observations are taken at time (k), but are not fused into the estimates until (k+1).
* Information carry two properties - the information values and the innovations.
    * Both are calculated **after** taking the observation at time (k).
    * The innovations values represent the innovation gained after observation at time (k).
    * The information values represent the updated information about the state after observation at time (k).
    * Accordingly, the system estimate at time (k+1) will be recovered from the information at time (k).
* Agent Update Process:
* Agents are initialized with an observation at t=1
        * The observation may come from a SCRIBE ground-truth model, a `DataObserver`, or a `SensorObservation`.
        * Initial values are estimate at t=1, observation at t=1, and initial information at t=1.
    * Each update step at t=k requires:
        * Compute innovation gained from observation, generating δi & δI for t=k
            * Recall that the information state at t=k also contains the innovation for t=k-1
        * Compute information gained from observation, generating information state for t=k+1
        * Refine estimate of the system, determining estimate values for t=k+1
        * Construct "estimated system state", generating both estimate for t=k+1 and observation for t=k+1
        * Progress internal clock from t=k to t=k+1
"""
mutable struct KFEnvScribe <: EnvScribe
    k::Integer
    cwrld_sync::Integer
    params::SCRIBEModelParameters
    bhv::SCRIBEObserverBehavior
    estimates::Vector{KFEnvEstimate}
    information::Vector{KFEnvInfo}

    KFEnvScribe(k::Integer, cwrld_sync::Integer, params::SCRIBEModelParameters, bhv::SCRIBEObserverBehavior,
                estimates::Vector{KFEnvEstimate},
                information::Vector{KFEnvInfo}) = new(k, cwrld_sync, params, bhv, estimates, information)
end

function initialize_scribe(params::SCRIBEModelParameters, bhv::SCRIBEObserverBehavior,
                           cwrld::SCRIBEModel, X₀::Matrix{Float64};
                           prior_covariance::Union{Nothing, AbstractMatrix{<:Real}}=nothing)
    let k=1, cwrld_sync=get_model_time(cwrld)-1, params=params, bhv=bhv
        estimates = [init_agent_estimate(cwrld, 1, params, bhv, X₀)]
        information = [init_agent_info(params; prior_covariance)]
        KFEnvScribe(k, cwrld_sync, params, bhv, estimates, information)
    end
end

"""Initialize a filter whose observations are pulled from external data."""
function initialize_scribe(
    params::SCRIBEModelParameters,
    observer::DataObserver,
    X₀::Matrix{Float64};
    prior_covariance::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
    model_time::Integer=1,
)
    estimate = init_agent_estimate(params, observer, X₀, model_time)
    information = [init_agent_info(params; prior_covariance, model_time)]
    KFEnvScribe(1, 0, params, observer, [estimate], information)
end

"""Initialize a filter from an explicitly supplied sensor measurement."""
function initialize_scribe(
    params::SCRIBEModelParameters,
    behavior::SCRIBEObserverBehavior,
    measurement::SensorObservation;
    prior_covariance::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
)
    estimate = init_agent_estimate(params, behavior, measurement)
    information = [init_agent_info(
        params;
        prior_covariance,
        model_time=measurement.k,
    )]
    KFEnvScribe(1, 0, params, behavior, [estimate], information)
end

"""Add a new estimate and synthesize its observation from a SCRIBE world."""
function next_agent_state(agent::KFEnvScribe, ϕₖ::Vector{Float64}, cwrld::SCRIBEModel, X::Matrix{Float64})
    let k=agent.k,
        new_estimate=update_SCRIBEModel(agent.estimates[k].estimate, ϕₖ),
        new_obs=scribe_observations(X, cwrld, agent.bhv)
        push!(agent.estimates, KFEnvEstimate(k+1, new_estimate, new_obs))
    end
end

"""Add a new estimate and pull its observation from a `DataObserver`."""
function next_agent_state(
    agent::KFEnvScribe,
    ϕₖ::Vector{Float64},
    X::Matrix{Float64},
)
    agent.bhv isa DataObserver ||
        throw(ArgumentError(
            "Location-only progression requires a DataObserver; supply a " *
            "world or SensorObservation for this filter.",
        ))
    model = update_SCRIBEModel(agent.estimates[agent.k].estimate, ϕₖ)
    observation = scribe_observations(X, model, agent.bhv)
    push!(agent.estimates, KFEnvEstimate(agent.k + 1, model, observation))
end

"""Add a new estimate using a sensor measurement supplied by the caller."""
function next_agent_state(
    agent::KFEnvScribe,
    ϕₖ::Vector{Float64},
    measurement::SensorObservation,
)
    expected_time = get_model_time(agent.estimates[agent.k].estimate) + 1
    measurement.k == expected_time ||
        throw(ArgumentError(
            "Expected a time-$expected_time observation; received time " *
            "$(measurement.k).",
        ))
    model = update_SCRIBEModel(agent.estimates[agent.k].estimate, ϕₖ)
    behavior = agent.bhv isa DataObserver ?
        agent.bhv.behavior :
        agent.bhv
    observation = scribe_observations(measurement, model, behavior)
    push!(agent.estimates, KFEnvEstimate(expected_time, model, observation))
end

"""Insert newest information state gained from info fusion into the agent representation.
"""
next_agent_info_state(agent::KFEnvScribe, info::KFEnvInfo) = push!(agent.information, info)

"""Update agent-internal clock.
"""
next_agent_time(agent::KFEnvScribe) = agent.k+=1

struct KFEstimators <: EnvEstimators
    system::KFEnvScribe
    A::Function
    b::Function
    ϕ::Function
    Q::Function
    H::Function
    z::Function
    R::Function
    Y::Function
    y::Function

    KFEstimators(system::KFEnvScribe, A::Function, b::Function,
                     ϕ::Function, Q::Function,
                     H::Function, z::Function, R::Function,
                     Y::Function, y::Function) =
        new(system, A, b, ϕ, Q, H, z, R, Y, y)
end

"""Instantiates estimator functions for the LGSF model.

Note that the observation matrix H must be seperately computed each timestep.
The `H` used here is the current agent approximation of the true observation matrix.

The initialization dispatches on the parameter type. The EOF backend provides
its corresponding specialization in `eofclimatemodels.jl`.
"""
function initialize_estimators(system::KFEnvScribe, params::LGSFModelParameters)
    get_A(k, system) = system.estimates[k].estimate.params.A
    get_b(_, _) = zeros(params.nᵩ)
    get_ϕ(k, system) = system.estimates[k].estimate.ϕ
    get_Q(k, system) = params.w[:Q]
    get_H(k, system) = compute_obs_dynamics(system.estimates[k].estimate, system.estimates[k].observations.X)[1]
    get_z(k, system) = system.estimates[k].observations.z
    get_R(k, system) = system.estimates[k].observations.v[:R]
    get_Y(k, system) = system.information[k].Y
    get_y(k, system) = system.information[k].y

    KFEstimators(system, kA->get_A(kA, system), kb->get_b(kb, system),
                 kϕ->get_ϕ(kϕ, system), kQ->get_Q(kQ, system),
                 kH->get_H(kH, system), kz->get_z(kz, system), kR->get_R(kR, system),
                 kY->get_Y(kY, system), ky->get_y(ky, system))
end

"""Local computation of the prior update **of the next step** Y⁻(k+1).

Takes two inputs:
* Estimator functions (`Ef::KFEstimators`)
* The **current** timestep `k`. It will use this along `Ef` to lookup corresponding system information.
"""
function compute_info_priors(Ef::KFEstimators, k::Integer)
    @unpack A, b, Q, Y, y = Ef
    let A=A(k), b=b(k), Y=Matrix(Symmetric((Y(k) + Y(k)') / 2)),
        Q=Q(k), y=y(k)
        Y_factor = cholesky(Symmetric(Y); check=true)
        P = Matrix(Y_factor \ Matrix{Float64}(I, size(Y)...))
        ϕ = Y_factor \ y
        P_predicted = A * P * A' + Q
        P⁻ = Matrix(Symmetric((P_predicted + P_predicted') / 2))
        P⁻_factor = cholesky(Symmetric(P⁻); check=true)
        Y⁻ = Matrix(P⁻_factor \ Matrix{Float64}(I, size(P⁻)...))
        y⁻ = Y⁻ * (A * ϕ + b)
        return Y⁻, y⁻
    end
end

"""Computes the current innovation gained by observation at time k.

Takes two inputs:
* Estimator functions (`Ef::KFEstimators`)
* The **current** timestep `k`. It will use this along `Ef` to lookup corresponding system information.
"""
function compute_innov_from_obs(Ef::KFEstimators, k::Integer)
    @unpack H, z, R, = Ef
    R_factor = cholesky(Symmetric(Matrix(R(k))); check=true)
    δI_raw = H(k)' * (R_factor \ H(k))
    δI = Matrix(Symmetric((δI_raw + δI_raw') / 2))
    δi = H(k)' * (R_factor \ z(k))
    return δI, δi
end

"""
Advance one estimator's information filter from time `k` to `k+1` using its
local observation. Multi-agent information must instead be reconciled through
SCRIBE's distributed fusion protocol.
"""
function information_filter_update(estimators::EnvEstimators, k::Integer)
    Y⁻, y⁻ = compute_info_priors(estimators, k)
    δI, δi = compute_innov_from_obs(estimators, k)
    Y_next = Y⁻ + δI
    Y = Matrix(Symmetric((Y_next + Y_next') / 2))
    KFEnvInfo(y⁻ + δi, Y, δi, δI)
end

"""Recover the state estimate vector from given info state.
"""
function recover_estimate_from_info(info::KFEnvInfo)
    Y = Matrix(Symmetric((info.Y + info.Y') / 2))
    cholesky(Symmetric(Y); check=true) \ info.y
end

recover_estimate_from_info(agent::KFEnvScribe, k::Integer) = recover_estimate_from_info(agent.information[k])

"""Advance a simulated filter with fused information and a SCRIBE world."""
function progress_agent_env_filter(agent::KFEnvScribe, info::KFEnvInfo, world::SCRIBEModel, X::Matrix{Float64})
    ϕₖ=recover_estimate_from_info(info) # acquire ϕⱼ(t=k+1)
    next_agent_info_state(agent, info) # set info(t=k+1)
    next_agent_state(agent, ϕₖ, world, X) # set ϕⱼ(t=k+1); acquire and set z(t=k+1)
    next_agent_time(agent) # k ⟵ k+1
end

"""Advance a data-backed filter and pull its next observation at `X`."""
function progress_agent_env_filter(
    agent::KFEnvScribe,
    info::KFEnvInfo,
    X::Matrix{Float64},
)
    agent.bhv isa DataObserver ||
        throw(ArgumentError(
            "Location-only progression requires a DataObserver.",
        ))
    ϕₖ = recover_estimate_from_info(info)
    next_agent_info_state(agent, info)
    next_agent_state(agent, ϕₖ, X)
    next_agent_time(agent)
end

"""Advance a filter using a pre-collected next sensor measurement."""
function progress_agent_env_filter(
    agent::KFEnvScribe,
    info::KFEnvInfo,
    measurement::SensorObservation,
)
    expected_time = get_model_time(agent.estimates[agent.k].estimate) + 1
    measurement.k == expected_time ||
        throw(ArgumentError(
            "Expected a time-$expected_time observation; received time " *
            "$(measurement.k).",
        ))
    ϕₖ = recover_estimate_from_info(info)
    next_agent_info_state(agent, info)
    next_agent_state(agent, ϕₖ, measurement)
    next_agent_time(agent)
end

"""Initialize the Kalman Filter system.
"""
function initialize_KF(params::SCRIBEModelParameters, observer::SCRIBEObserverBehavior,
                       init_loc::Matrix{Float64}, ground_state::SCRIBEModel;
                       prior_covariance::Union{Nothing, AbstractMatrix{<:Real}}=nothing)
    let scribe = initialize_scribe(params, observer, ground_state, init_loc;
                                   prior_covariance)
        return initialize_estimators(scribe, params)
    end
end

"""Initialize a Kalman information filter backed by external data."""
function initialize_KF(
    params::SCRIBEModelParameters,
    observer::DataObserver,
    init_loc::Matrix{Float64};
    prior_covariance::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
    model_time::Integer=1,
)
    scribe = initialize_scribe(
        params,
        observer,
        init_loc;
        prior_covariance,
        model_time,
    )
    initialize_estimators(scribe, params)
end

"""Initialize a Kalman information filter from a sensor measurement."""
function initialize_KF(
    params::SCRIBEModelParameters,
    behavior::SCRIBEObserverBehavior,
    measurement::SensorObservation;
    prior_covariance::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
)
    scribe = initialize_scribe(
        params,
        behavior,
        measurement;
        prior_covariance,
    )
    initialize_estimators(scribe, params)
end
