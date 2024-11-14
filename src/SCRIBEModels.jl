module SCRIBEModels

export SCRIBEModel, SCRIBEModelParameters, initialize_SCRIBEModel_from_parameters, update_SCRIBEModel, predict_SCRIBEModel
export SCRIBEObserverBehavior, SCRIBEObserverState, compute_obs_dynamics, scribe_observations

using GaussianDistributions: Gaussian
using LinearAlgebra: norm, I, ⋅

"""Abstract type defined for specialization during model instantiation.

Currently defined types:
    - `LGSFModelParameters`
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

"""Abstract type that defines observer behavior.

This can include:
* Sensors parameters, such as the observation noise covariance
* Details that impact observations, such as a fixed set or function of sensing locations

Currently defined observer behavior types:
    - `LGSFObserverBehavior`
"""
abstract type SCRIBEObserverBehavior end

"""Abstract type that collects the current state of the observer at timestep k.

This is notably the state *after* observations have been collected.

Currently defined observer state types:
    - `LGSFObserverState`
"""
abstract type SCRIBEObserverState end

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

include("lineargaussianscalarfields.jl")

end