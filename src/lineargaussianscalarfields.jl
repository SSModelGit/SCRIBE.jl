export LGSFModelParameters, LGSFModel
export LGSFObserverBehavior, LGSFObserverState

struct LGSFModelParameters <: SCRIBEModelParameters
    nᵩ::Integer # Defines number of model features (size of the model)
    p::Dict{Symbol, Any} # Dict of underlying parameter factors
    ψ_p::Vector{Dict{Symbol, Any}} # Consolidated vector of ψ defining parameters
    ϕ₀::Vector{Float64} # Initial ϕ of the system.
    A::Matrix{Float64} # Linear dynamics matrix driving ϕ
    w::Dict{Symbol, Any} # Definition of stochastic (noise) process driving ϕ - stored in dict

    function LGSFModelParameters(μ::VecOrMat{Float64}, σ::Vector{Float64}, τ::Vector{Float64},
                                 ϕ₀::Union{Nothing, Vector{Float64}}, A::Union{Nothing, Matrix{Float64}},
                                 Q::Union{Nothing, Matrix{Float64}})
        let ψ_p=[Dict([(:μ, m),(:σ, s),(:τ, t)]) for m in collect(eachrow(μ)) for s in σ for t in τ], nᵩ=size(ψ_p,1)
            p=Dict(:μ=>μ, :σ=>σ, :τ=>τ)
            if ϕ₀===nothing
                ϕ₀=zeros(nᵩ)
            else
                @assert size(ϕ₀,1)==nᵩ
            end
            if A===nothing; A=I(nᵩ); end
            if Q===nothing; Q=I(nᵩ); end
            w=Dict(:Q=>Q, :w_dist=>Gaussian(zeros(nᵩ), Q))
            new(nᵩ, p, ψ_p, ϕ₀, A, w)
        end
    end
end

"""Keyword-named constructor for LGSFModelParameters
"""
function LGSFModelParameters(;μ::VecOrMat{Float64}=hcat(range(-1,1,5), zeros(5)),
                              σ::Vector{Float64}=[1.],
                              τ::Vector{Float64}=[1.],
                              ϕ₀::Union{Nothing, Vector{Float64}}=nothing,
                              A::Union{Nothing, Matrix{Float64}}=nothing,
                              Q::Union{Nothing, Matrix{Float64}}=nothing)
    LGSFModelParameters(μ,σ,τ,ϕ₀,A,Q)
end

"""Linear Gaussian Scalar Fields Model type.

Inherits from the SCRIBEModel abstract type. Intended to be a system snapshot, not a global model function.
Will contain information to calculate information about the system state at timestep `k`.

To understand the global evolution of the system, please use dedicated functions from the SCRIBE package.

Fields:\\
`k::Integer`: Timestep that this model will represent. \\
`params::LGSFModelParameters`: The param construct that defines the core of this model. \\
`ψ::Function`: The ψ function for the Gaussian scalar field bases. \\
`ϕ::Vector{Float64}`: Coefficient vector of linear model system, ϕ. \\
`w_k::Vector{Float64}`: Current noise vector of the modeled process. \\

* Constructor information: This is the internal constructor for the LGSF model.

    This is not intended to be used by the end-user, and should only be called within other exposed methods.

    Requires: `k`, `params`, `ψ`, `ϕ`, `w_k`.
"""
struct LGSFModel <: SCRIBEModel
    k::Integer # Timestep associated with model
    params::LGSFModelParameters # Parameters associated with the model
    ψ::Function # Takes two inputs: x (location) # TODO: and optionally [i_range] to get specific ψᵢ outputs
    ϕ::Vector{Float64} # Coefficient vector
    w_k::Vector{Float64} # Current noise vector of the modeled process

    function LGSFModel(k::Integer, params::LGSFModelParameters, ψ::Function, ϕ::Vector{Float64}, w_k::Vector{Float64})
        new(k, params, ψ, ϕ, w_k)
    end
end

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

"""Create the initial LGSFModel based on parameters.

The value of ϕ₀ is drawn from the parameter list.
Initializing implies that we start at discrete k=1.

Input:
    params::LGSFModelParameters
Output:
    model::LGSFModel
"""
function initialize_SCRIBEModel_from_parameters(params::LGSFModelParameters; k=1)
    return LGSFModel(k, params, x->ψ_from_params(x, params), params.ϕ₀, rand(params.w[:w_dist]))
end

"""Computes the stochastic evolution of ϕ for a given timestep of an LGSFModel.

Leverages the Julia-internal `muladd` function for stability of operation.
"""
LGSF_ϕ_dynamics(smodel::LGSFModel) = muladd(smodel.params.A, smodel.ϕ, smodel.w_k)

"""Progresses the discrete-time model through one time step.

This is for a ground-truth model.
This requires an explicit calculation of any involved discrete dynamics.

In the LGSF model, the only dynamic object is ϕ.
We use the simple linear stochastic update.
This is computed via an internal method (not exported) named `LGSF_ϕ_dynamics`.
"""
update_SCRIBEModel(smodel::LGSFModel) = LGSFModel(smodel.k+1, smodel.params, smodel.ψ,
                                                  LGSF_ϕ_dynamics(smodel), rand(smodel.params.w[:w_dist]))

"""Progresses the discrete-time model through one time step.

This is for model estimates. This requires explicit ϕ declarations.
"""
update_SCRIBEModel(smodel::LGSFModel, ϕₖ) = LGSFModel(smodel.k+1, smodel.params, smodel.ψ,
                                                      ϕₖ, rand(smodel.params.w[:w_dist]))

function predict_SCRIBEModel(smodel::LGSFModel, x::Union{Vector{Float64}, Float64}, k::Integer)
    @assert k==smodel.k "Timestep of prediction does not match the model timestep"
    smodel.ψ(x)'⋅smodel.ϕ
end

"""Stores information about the observation process.

Constructor input: `σₛ::Float64`: Scalar covariance value of AWGN perturbing a single observation
"""
struct LGSFObserverBehavior <: SCRIBEObserverBehavior
    v_s::Dict{Symbol, Float64} # Scalar AWGN process impacting z - stored in dict

    LGSFObserverBehavior(σₛ::Float64=0.1) = new(Dict(:μ=>0,:σ=>σₛ))
end

"""Stores information about the current observations.

The following fields define the information stored:\\
`k::Integer`: Discrete timestep associated with observations \\
`nₛ::Integer`: Number of samples gathered in this time step \\
`X::Matrix{Float64}`: Matrix of observation locations (Vector if single observation) \\
\t* Matrix stores the locations vertically, such that each new row represents a new location\\
`H::Matrix{Float64}`: Observation matrix representing taking samples at X \\
`v::Dict{Symbol, AbstractArray{Float64}}`: Dictionary of underlying sample noise factors
`z::Vector{Float64}`: Observations gathered at time step k

Constructor input: List of locations `X`, current system state `lmodel`, observer parameters `o_b`
"""
struct LGSFObserverState <: SCRIBEObserverState
    k::Integer
    nₛ::Integer
    X::Matrix{Float64}
    H::Matrix{Float64}
    v::Dict{Symbol, AbstractArray{Float64}}
    z::Vector{Float64}

    function LGSFObserverState(k::Integer, nₛ::Integer, X::Matrix{Float64},
                               H::Matrix{Float64}, v::Dict{Symbol, AbstractArray{Float64}},
                               z::Vector{Float64})
        new(k,nₛ,X,H,v,z)
    end
end

"""Compute the observation dynamics matrix.

This is also referred to as the **H matrix**.
"""
compute_obs_dynamics(smodel::LGSFModel, X::Matrix{Float64}) = mapslices(smodel.ψ,X,dims=2)

function scribe_observations(X::Matrix{Float64}, smodel::LGSFModel, o_b::LGSFObserverBehavior)
    let nₛ=size(X,1), v_s=o_b.v_s, R=v_s[:σ]*I(nₛ)
        v=Dict(:R=>R, :k=>rand(Gaussian(zeros(nₛ), R)))
        H=compute_obs_dynamics(smodel, X)
        z=muladd(H,smodel.ϕ,v[:k])
        LGSFObserverState(smodel.k, nₛ, X, H, v, z)
    end
end