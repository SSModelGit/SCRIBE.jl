module LinearGaussianScalarFieldModels 
export LGSFModelParameters, LGSFModel, initialize_LGSFModel_from_params, update_LGSFModel

using ..SCRIBE: SCRIBEModel
using LinearAlgebra: norm

struct LGSFModelParameters
    nᵩ::Integer # Defines number of model features (size of the model)
    μ::Vector{Union{Vector{Float64},Float64}} # Vector of means per scalar field; index corresponds to scalar field number
    σ::Vector{Float64} # Vector of covs per scalar field; index corresponds to scalar field number
    τ::Vector{Float64} # Vector of lengthscales per scalar field; index corresponds to scalar field number

    function LGSFModelParameters(nᵩ=5, μ=collect(range(-1,1,5)), σ=0.1*ones(5), τ=0.01*ones(5))
        @assert size(μ,1)==nᵩ
        @assert size(σ,1)==nᵩ
        @assert size(τ,1)==nᵩ
        new(nᵩ, μ, σ, τ)
    end
end

struct LGSFModel <: SCRIBEModel
    k::Integer # Timestep associated with model
    params::LGSFModelParameters # Parameters associated with the model
    ψ::Function # Takes two inputs: x (location) and optionally [i_range] to get specific ψᵢ outputs
    ϕ::Vector{Float64} # Coefficient vector
    LGSFModel(k, params, ψ, ϕ) = new(k, params, ψ, ϕ)
end

"""Create the initial LGSFModel based on parameters.

We assume we start the model at discrete k=0.
The ϕ coefficient vector starts at \bm{0}.

Input:
    params::LGSFModelParameters
Output:
    model::LGSFModel
"""
function initialize_LGSFModel_from_params(params::LGSFModelParameters)
    """Defines how ψ(x) is calculated, based on passed parameters.
    
    Use this to fill the LGSFModel field:
        ψ=x->ψ_from_params(x, params)
    """
    function ψ_from_params(x::Union{Vector{Float64}, Float64}, params::LGSFModelParameters)
        p=zeros(params.nᵩ)
        for i in 1:params.nᵩ
            p[i] = (1/params.τ[i]) * exp(-(norm(x-params.μ[i])^2) / params.σ[i])
        end
        return p
    end
    return LGSFModel(0, params, x->ψ_from_params(x, params), zeros(params.nᵩ))
end

function update_LGSFModel(smodel::LGSFModel, ϕ::Vector{Float64})
    LGSFModel(smodel.k+1, smodel.params, smodel.ψ, ϕ)
end

end