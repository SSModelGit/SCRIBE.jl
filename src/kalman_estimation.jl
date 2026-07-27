using Combinatorics: combinations
using Statistics: mean

export KFEnvInfo, KFEnvScribe, initialize_scribe, next_agent_state, next_agent_time, next_agent_info_state
export KFEstimators, initialize_estimators, compute_info_priors, compute_innov_from_obs
export centralized_fusion, recover_estimate_from_info, progress_agent_env_filter, initialize_KF

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

function init_agent_estimate(world::SCRIBEModel, k::Integer,
                             params::SCRIBEModelParameters, bhv::SCRIBEObserverBehavior,
                             X::VecOrMat{Float64})
    KFEnvEstimate(k, initialize_SCRIBEModel_from_parameters(params, k=k),
                     scribe_observations(X,world,bhv))
end

# function new_agent_estimate(k::Integer,
#                             old_estimate::SCRIBEModel, ϕₖ::Vector{Float64},
#                             world::SCRIBEModel, bhv::SCRIBEObserverBehavior, X::Matrix{Float64})
#     KFEnvEstimate(k, update_SCRIBEModel(old_estimate, ϕₖ), scribe_observations(X, world, bhv))
# end

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
                         prior_covariance::Union{Nothing, AbstractMatrix{<:Real}}=nothing)
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
    * Agents are initialized at a location at t=1
        * Initial values are estimate at t=1, observation at t=1 and initial location, initial information at t=1
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

"""Adds new internal system estimate for t=k+1.
"""
function next_agent_state(agent::KFEnvScribe, ϕₖ::Vector{Float64}, cwrld::SCRIBEModel, X::Matrix{Float64})
    let k=agent.k,
        new_estimate=update_SCRIBEModel(agent.estimates[k].estimate, ϕₖ),
        new_obs=scribe_observations(X, cwrld, agent.bhv)
        push!(agent.estimates, KFEnvEstimate(k+1, new_estimate, new_obs))
    end
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
    ϕ::Function
    Q::Function
    H::Function
    z::Function
    R::Function
    Y::Function
    y::Function

    KFEstimators(system::KFEnvScribe, A::Function, ϕ::Function, Q::Function,
                     H::Function, z::Function, R::Function,
                     Y::Function, y::Function) = new(system, A, ϕ, Q, H, z, R, Y, y)
end

"""Instantiates some simple estimator functions for the LGSF model.

Note that the observation matrix H must be seperately computed each timestep.
The `H` used here is the current agent approximation of the true observation matrix.

The initialization will dispatch on the params type.
Currently, only the LGSFModelParameters are implemented.
"""
function initialize_estimators(system::KFEnvScribe, params::LGSFModelParameters)
    get_A(k, system) = system.estimates[k].estimate.params.A
    get_ϕ(k, system) = system.estimates[k].estimate.ϕ
    get_Q(k, system) = params.w[:Q]
    get_H(k, system) = compute_obs_dynamics(system.estimates[k].estimate, system.estimates[k].observations.X)[1]
    get_z(k, system) = system.estimates[k].observations.z
    get_R(k, system) = system.estimates[k].observations.v[:R]
    get_Y(k, system) = system.information[k].Y
    get_y(k, system) = system.information[k].y

    KFEstimators(system, kA->get_A(kA, system), kϕ->get_ϕ(kϕ, system), kQ->get_Q(kQ, system),
                 kH->get_H(kH, system), kz->get_z(kz, system), kR->get_R(kR, system),
                 kY->get_Y(kY, system), ky->get_y(ky, system))
end

"""Local computation of the prior update **of the next step** Y⁻(k+1).

Takes two inputs:
* Estimator functions (`Ef::KFEstimators`)
* The **current** timestep `k`. It will use this along `Ef` to lookup corresponding system information.
"""
function compute_info_priors(Ef::KFEstimators, k::Integer)
    @unpack A, Q, Y, y = Ef
    let A=A(k), Y=Matrix(Symmetric((Y(k) + Y(k)') / 2)), Q=Q(k), y=y(k)
        Y_factor = cholesky(Symmetric(Y); check=true)
        P = Matrix(Y_factor \ Matrix{Float64}(I, size(Y)...))
        ϕ = Y_factor \ y
        P_predicted = A * P * A' + Q
        P⁻ = Matrix(Symmetric((P_predicted + P_predicted') / 2))
        P⁻_factor = cholesky(Symmetric(P⁻); check=true)
        Y⁻ = Matrix(P⁻_factor \ Matrix{Float64}(I, size(P⁻)...))
        y⁻ = Y⁻ * A * ϕ
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

"""Computes the information filter update for y(t=k+1) and Y(t=k+1) from z(k) and y/Y(t=k).
"""
function centralized_fusion(agent_estimators::Vector{KFEstimators}, k::Integer)
    # Compute priors from current system state at t=k and previous information state at t=k-1
    priors = map(ef->compute_info_priors(ef, k), agent_estimators)
    nₐ=size(priors,1)

    # Ensure all priors are the same
    if nₐ>1
        for prior_pair in combinations(priors, 2)
            let prior_a=prior_pair[1], prior_b=prior_pair[2]
                @assert isapprox(prior_a[1], prior_b[1]) "Information matrix priors for Y⁻(k+1) are diverged!"
                @assert isapprox(prior_a[2], prior_b[2]) "Information value priors for y⁻(k+1) are diverged!"
            end
        end
    end

    # Compute innovations for t=k+1 from current observation at t=k
    innovs = map(ef->compute_innov_from_obs(ef, k), agent_estimators)
    δĪ=mean(map(x->x[1], innovs))
    δī=mean(map(x->x[2], innovs))

    return [KFEnvInfo(priors[a][2]+nₐ*δī, priors[a][1]+nₐ*δĪ, δī, δĪ) for a in 1:nₐ]
end

"""Recover the state estimate vector from given info state.
"""
function recover_estimate_from_info(info::KFEnvInfo)
    Y = Matrix(Symmetric((info.Y + info.Y') / 2))
    cholesky(Symmetric(Y); check=true) \ info.y
end

recover_estimate_from_info(agent::KFEnvScribe, k::Integer) = recover_estimate_from_info(agent.information[k])

"""Consolidated update process given new fused information estimates y/Y(t=k+1)

Also requires the new state of the world and the new sampling locations.
"""
function progress_agent_env_filter(agent::KFEnvScribe, info::KFEnvInfo, world::SCRIBEModel, X::Matrix{Float64})
    ϕₖ=recover_estimate_from_info(info) # acquire ϕⱼ(t=k+1)
    next_agent_info_state(agent, info) # set info(t=k+1)
    next_agent_state(agent, ϕₖ, world, X) # set ϕⱼ(t=k+1); acquire and set z(t=k+1)
    next_agent_time(agent) # k ⟵ k+1
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
