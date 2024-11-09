module SCRIBE

using Reexport
using LinearAlgebra
using GaussianDistributions: ⊕, Gaussian

include("SCRIBEModels.jl")
@reexport using .SCRIBEModels

include("kalman_estimation.jl")
include("CovarianceIntersection.jl")

# Write your package code here.

end