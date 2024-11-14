export AgentEnvModel, initialize_agent, next_agent_state
export SystemEstimators, simple_LGSF_Estimators

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

function new_agent_estimate(world::SCRIBEModel, k::Integer,
                            estimate::SCRIBEModel, ϕₖ::Vector{Float64},
                            bhv::SCRIBEObserverBehavior, X::Matrix{Float64})
    AgentEnvEstimate(k, update_SCRIBEModel(estimate, ϕₖ), scribe_observations(X, world, bhv))
end

"""Current discrete-time information.

This is separate from the actual model estimate - this is what the Kalman Filter interacts with.

The model estimate is a representation of the system, which is recovered from this.
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
There is no such thing as "inital innovation", so arbitrarily set to zero.
"""
function init_agent_info(nᵩ::Integer)
    AgentEnvInfo(zeros(nᵩ), zeros(nᵩ,nᵩ), zeros(nᵩ), zeros(nᵩ,nᵩ))
end

"""This is the collection of the system over time.

This is a linearized representation.
The mutating elements are the vectors, which are appended to.
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

function next_agent_state(agent::AgentEnvModel, ϕₖ::Vector{Float64}, cwrld::SCRIBEModel, X::Matrix{Float64})
    push!(agent.estimates, new_agent_estimate(cwrld, agent.k+1, agent.estimates[agent.k].estimate, ϕₖ, agent.bhv, X))
end

struct SystemEstimators
    system::AgentEnvModel
    A::Function
    ϕ::Function
    H::Function
    z::Function

    SystemEstimators(system::AgentEnvModel, A::Function, ϕ::Function, H::Function, z::Function) = new(system, A, ϕ, H, z)
end

function simple_LGSF_Estimators(system::AgentEnvModel)
    get_A(k, system) = system.estimates[k].estimate.params.A
    get_ϕ(k, system) = system.estimates[k].estimate.ϕ
    get_H(k, system) = compute_obs_dynamics(system.estimates[k].estimate, system.estimates[k].observations.X)
    get_z(k, system) = system.estimates[k].observations.z

    SystemEstimators(system, kA->get_A(kA, system), kϕ->get_ϕ(kϕ, system), kH->get_H(kH, system), kz->get_z(kz, system))
end