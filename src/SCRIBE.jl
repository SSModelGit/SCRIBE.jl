module SCRIBE

using Reexport
using LinearAlgebra
using GaussianDistributions: ⊕, Gaussian

include("SCRIBEModels.jl")
@reexport using .SCRIBEModels

abstract type EnvScribe end
abstract type EnvEstimators end

export EnvScribe, EnvEstimators, SCRIBEAgent, initialize_agent

struct SCRIBEAgent
    params::SCRIBEModelParameters
    observer::SCRIBEObserverBehavior
    history::Vector{Matrix{Float64}}
    agent::EnvScribe
    estimators::EnvEstimators

    SCRIBEAgent(params::SCRIBEModelParameters, observer::SCRIBEObserverBehavior,
                history::Vector{Matrix{Float64}}, agent::EnvScribe,
                estimators::EnvEstimators) = new(params, observer, history, agent, estimators)
end

include("kalman_estimation.jl")

"""Creates a SCRIBEAgent appropriate for the KF system, from a KF Estimator.
"""
function initialize_agent(kf_estimators::KFEstimators)
    let ag=kf_estimators.system
        return SCRIBEAgent(ag.params, ag.bhv, [ag.estimates[1].observations.X], ag, kf_estimators)
    end
end

include("CovarianceIntersection.jl")

# Write your package code here.

end