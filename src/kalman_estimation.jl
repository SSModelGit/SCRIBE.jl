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
    AgentEnvEstimate(k, initialize_SCRIBEModel_from_parameters(params),
                     scribe_observations(X,world,bhv))
end

"""Current discrete-time information.

This is separate from the actual model estimate - this is what the Kalman Filter interacts with.

The model estimate is a representation of the system, which is recovered from this.
"""
struct AgentEnvInfo
    y::Vector{Float64}
    Y::Vector{Float64}
    i::Vector{Float64}
    I::Vector{Float64}
end

"""This is the collection of the system over time.

This is a linearized representation.
The mutating elements are the vectors, which are appended to.
"""
mutable struct AgentEnvModel
    params::SCRIBEModelParameters
    bhv::SCRIBEObserverBehavior
    estimates::Vector{AgentEnvEstimate}
    information::Vector{AgentEnvInfo}
end

struct SystemEstimators
    system::AgentEnvModel
    A::Function
    ϕ::Function
    H::Function
    z::Function
end