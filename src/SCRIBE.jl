module SCRIBE

using Reexport
using LinearAlgebra
using GaussianDistributions: ⊕, Gaussian
import Kalman

include("SCRIBEModels.jl")
include("CovarianceIntersection.jl")

@reexport using .SCRIBEModels

# Write your package code here.

end