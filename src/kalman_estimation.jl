using Parameters: @unpack
using Combinatorics: combinations

export AgentEnvModel, initialize_agent, next_agent_state, next_agent_time, next_agent_info_state
export SystemEstimators, simple_LGSF_Estimators, compute_info_priors, compute_innov_from_obs
export centralized_fusion, recover_estimate_from_info, progress_agent_env_filter

"""Current discrete-time estimate of the linear system being observed.

Will store the discrete time of estimation for redundancy checking.
"""
struct AgentEnvEstimate
    k::Integer
    estimate::SCRIBEModel
    observations::SCRIBEObserverState
    function AgentEnvEstimate(k::Integer, estimate::SCRIBEModel, observations::SCRIBEObserverState)
        new(k, estimate, observations)
    end
end

function init_agent_estimate(world::SCRIBEModel, k::Integer,
                             params::SCRIBEModelParameters, bhv::SCRIBEObserverBehavior,
                             X::VecOrMat{Float64})
    AgentEnvEstimate(k, initialize_SCRIBEModel_from_parameters(params, k=k),
                     scribe_observations(X,world,bhv))
end

# function new_agent_estimate(k::Integer,
#                             old_estimate::SCRIBEModel, ϕₖ::Vector{Float64},
#                             world::SCRIBEModel, bhv::SCRIBEObserverBehavior, X::Matrix{Float64})
#     AgentEnvEstimate(k, update_SCRIBEModel(old_estimate, ϕₖ), scribe_observations(X, world, bhv))
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
struct AgentEnvInfo
    y::Vector{Float64}
    Y::Matrix{Float64}
    i::Vector{Float64}
    I::Matrix{Float64}

    AgentEnvInfo(y::Vector{Float64}, Y::Matrix{Float64},
                 i::Vector{Float64}, I::Matrix{Float64}) = new(y,Y,i,I)
end


"""Initial information associated per agent.

There is no information about the agent state, so all set to zero.

This initial innovation state is not truly an "innovation state", but a meaningless prior.
So, this is arbitrarily set to zero. This is described further in the docstring for AgentEnvInfo.
"""
function init_agent_info(nᵩ::Integer)
    AgentEnvInfo(zeros(nᵩ), zeros(nᵩ,nᵩ), zeros(nᵩ), zeros(nᵩ,nᵩ))
end

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
mutable struct AgentEnvModel
    k::Integer
    cwrld_sync::Integer
    params::SCRIBEModelParameters
    bhv::SCRIBEObserverBehavior
    estimates::Vector{AgentEnvEstimate}
    information::Vector{AgentEnvInfo}

    AgentEnvModel(k::Integer, cwrld_sync::Integer, params::SCRIBEModelParameters, bhv::SCRIBEObserverBehavior,
                  estimates::Vector{AgentEnvEstimate},
                  information::Vector{AgentEnvInfo}) = new(k, cwrld_sync, params, bhv, estimates, information)
end

function initialize_agent(params::SCRIBEModelParameters, bhv::SCRIBEObserverBehavior, cwrld::SCRIBEModel, X₀::Matrix{Float64})
    AgentEnvModel(1, cwrld.k-1, params, bhv, [init_agent_estimate(cwrld, 1, params, bhv, X₀)], [init_agent_info(params.nᵩ)])
end

"""Adds new internal system estimate for t=k+1.
"""
function next_agent_state(agent::AgentEnvModel, ϕₖ::Vector{Float64}, cwrld::SCRIBEModel, X::Matrix{Float64})
    let k=agent.k,
        new_estimate=update_SCRIBEModel(agent.estimates[k].estimate, ϕₖ),
        new_obs=scribe_observations(X, cwrld, agent.bhv)
        push!(agent.estimates, AgentEnvEstimate(k, new_estimate, new_obs))
    end
end

"""Insert newest information state gained from info fusion into the agent representation.
"""
next_agent_info_state(agent::AgentEnvModel, info::AgentEnvInfo) = push!(agent.information, info)

"""Update agent-internal clock.
"""
next_agent_time(agent::AgentEnvModel) = agent.k+=1

struct SystemEstimators
    system::AgentEnvModel
    A::Function
    ϕ::Function
    Q::Function
    H::Function
    z::Function
    R::Function
    Y::Function
    y::Function

    SystemEstimators(system::AgentEnvModel, A::Function, ϕ::Function, Q::Function,
                     H::Function, z::Function, R::Function,
                     Y::Function, y::Function) = new(system, A, ϕ, Q, H, z, R, Y, y)
end

function simple_LGSF_Estimators(system::AgentEnvModel)
    get_A(k, system) = system.estimates[k].estimate.params.A
    get_ϕ(k, system) = system.estimates[k].estimate.ϕ
    get_Q(k, system) = system.params.w[:Q]
    get_H(k, system) = compute_obs_dynamics(system.estimates[k].estimate, system.estimates[k].observations.X)
    get_z(k, system) = system.estimates[k].observations.z
    get_R(k, system) = system.estimates[k].observations.v[:R]
    get_Y(k, system) = system.information[k].Y
    get_y(k, system) = system.information[k].y

    SystemEstimators(system, kA->get_A(kA, system), kϕ->get_ϕ(kϕ, system), kQ->get_Q(kQ, system),
                     kH->get_H(kH, system), kz->get_z(kz, system), kR->get_R(kR, system),
                     kY->get_Y(kY, system), ky->get_y(ky, system))
end

"""Local computation of the prior update **of the next step** Y⁻(k+1).

Takes two inputs:
* Estimator functions (`Ef::SystemEstimators`)
* The **current** timestep `k`. It will use this along `Ef` to lookup corresponding system information.
"""
function compute_info_priors(Ef::SystemEstimators, k::Integer)
    @unpack _, A, _, Q, H, _, _, Y, y = Ef
    M  = inv(A(k))' * Y(k) * inv(A(k))
    Y⁻ = M - M * inv(M + inv(Q(k))) * M
    y⁻ = Y⁻ * A(k) * Y(k) * y(k)
    return Y⁻, y⁻
end

"""Computes the current innovation gained by observation at time k.

Takes two inputs:
* Estimator functions (`Ef::SystemEstimators`)
* The **current** timestep `k`. It will use this along `Ef` to lookup corresponding system information.
"""
function compute_innov_from_obs(Ef::SystemEstimators, k::Integer)
    @unpack _, _, _, _, H, z, R, _, _ = Ef
    δI = H(k)' * inv(R(k)) * H(k)
    δi = H(k)' * inv(R(k)) * z(k)
    return δI, δi
end

"""Computes the information filter update for y(t=k+1) and Y(t=k+1) from z(k) and y/Y(t=k).
"""
function centralized_fusion(agent_estimators::Vector{SystemEstimators}, k::Integer)
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

    return [AgentEnvInfo(priors[a][2]+nₐ*δī, priors[a][1]+nₐ*δĪ, δī, δĪ) for a in 1:nₐ]
end

"""Recover the state estimate vector from given info state.
"""
recover_estimate_from_info(info::AgentEnvInfo) = inv(info.Y) * info.y

"""Consolidated update process given new fused information estimates y/Y(t=k+1)

Also requires the new state of the world and the new sampling locations.
"""
function progress_agent_env_filter(agent::AgentEnvModel, info::AgentEnvInfo, world::SCRIBEModel, X::Matrix{Float64})
    ϕₖ=recover_estimate_from_info(info) # acquire ϕⱼ(t=k+1)
    next_agent_info_state(agent, info) # set info(t=k+1)
    next_agent_state(agent, ϕₖ, world, X) # set ϕⱼ(t=k+1); acquire and set z(t=k+1)
    next_agent_time(agent) # k ⟵ k+1
end