module SCRIBEModels

export SCRIBEModel, initialize_SCRIBEModel_from_parameters, update_SCRIBEModel, predict_SCRIBEModel

using GaussianDistributions: Gaussian
using LinearAlgebra: norm, I

"""Abstract type defined for specialization during model instantiation.

Currently defined types:
    - LGSFModelParameters
"""
abstract type SCRIBEModelParameters end

# Define a generic function on the abstract type
function initialize_SCRIBEModel_from_parameters(params::SCRIBEModelParameters)
    # This function has no implementation and is intended to be specialized
    error("`initialize_model_from_parameters` is not implemented for the abstract type SCRIBEModelParameters. Please provide a specific implementation.")
end

"""Abstract type that collects model types. Useful for specialization.

Currently defined model types:
    - LGSFModel
"""
abstract type SCRIBEModel end

# Define a generic function on the abstract type
function update_SCRIBEModel(smodel::SCRIBEModel, ϕ::Vector{Float64})
    # This function has no implementation and is intended to be specialized
    error("`update_model` is not implemented for the abstract type SCRIBEModel. Please provide a specific implementation.")
end

include("SCRIBE_lineargaussianfields.jl")

end