export LGSFModelParameters, LGSFModel

struct LGSFModelParameters <: SCRIBEModelParameters
    nᵩ::Integer # Defines number of model features (size of the model)
    μ::Union{Matrix{Float64}, Vector{Float64}} # Vector of means per scalar field; index corresponds to scalar field number
    σ::Vector{Float64} # Vector of covs per scalar field; index corresponds to scalar field number
    τ::Vector{Float64} # Vector of lengthscales per scalar field; index corresponds to scalar field number
    ψ_p::Vector{Dict{Symbol, Any}} # Consolidated vector of ψ defining parameters
    Q::Matrix{Float64} # Covariance matrix of stochastic noise process driving ϕ
    A::Matrix{Float64} # Linear dynamics matrix driving ϕ
    R::Matrix{Float64} # Covariance matrix of stochastic noise process driving z (observations)

    function LGSFModelParameters(μ=hcat(range(-1,1,5), zeros(5)), σ=[0.1], τ=[0.01],
                                 Q::Union{Nothing, Matrix{Float64}}=nothing,
                                 A::Union{Nothing, Matrix{Float64}}=nothing,
                                 R::Union{Nothing, Matrix{Float64}}=nothing)
        # @assert size(μ,1)==nᵩ
        # @assert size(σ,1)==nᵩ
        # @assert size(τ,1)==nᵩ
        let ψ_p=[Dict([(:μ, m),(:σ, s),(:τ, t)]) for m in collect(eachrow(μ)) for s in σ for t in τ], nᵩ=size(ψ_p,1)
            if Q===nothing
                Q=I(nᵩ)
            end
            if A===nothing
                A=I(nᵩ)
            end
            if R===nothing
                R=I(nᵩ)
            end
            new(nᵩ, μ, σ, τ, ψ_p, Q, A, R)
        end
    end
end

struct LGSFModel <: SCRIBEModel
    k::Integer # Timestep associated with model
    params::LGSFModelParameters # Parameters associated with the model
    ψ::Function # Takes two inputs: x (location) # TODO: and optionally [i_range] to get specific ψᵢ outputs
    ϕ::Vector{Float64} # Coefficient vector
    w::Vector{Float64} # Current noise vector of the modeled process
    function LGSFModel(k::Integer, params::LGSFModelParameters, ψ::Function, ϕ::Vector{Float64})
        new(k, params, ψ, ϕ, rand(Gaussian(zeros(params.nᵩ), params.Q)))
    end
end

"""Create the initial LGSFModel based on parameters.

We assume we start the model at discrete k=0.
The ϕ coefficient vector starts at \bm{0}.

Input:
    params::LGSFModelParameters
Output:
    model::LGSFModel
"""
function initialize_SCRIBEModel_from_parameters(params::LGSFModelParameters)
    """Defines how ψ(x) is calculated, based on passed parameters.
    
    Use this to fill the LGSFModel field:
        ψ=x->ψ_from_params(x, params)
    """
    function ψ_from_params(x::Union{Vector{Float64}, Float64}, params::LGSFModelParameters)
        p=zeros(params.nᵩ)
        for (i,k) in enumerate(params.ψ_p)
            p[i] = (1/k[:τ]) * exp(-(norm(x-k[:μ])^2) / k[:σ])
        end
        # for i in 1:params.nᵩ
        #     p[i] = (1/params.τ[i]) * exp(-(norm(x-params.μ[i])^2) / params.σ[i])
        # end
        return p
    end
    return LGSFModel(0, params, x->ψ_from_params(x, params), zeros(params.nᵩ))
end

function update_SCRIBEModel(smodel::LGSFModel, ϕ::Vector{Float64})
    LGSFModel(smodel.k+1, smodel.params, smodel.ψ, ϕ)
end

function predict_SCRIBEModel(smodel::LGSFModel, x::Union{Vector{Float64}, Float64}, k::Integer)
    @assert k==smodel.k "Timestep of prediction does not match the model timestep"
    smodel.ψ(x)'⋅smodel.ϕ
end