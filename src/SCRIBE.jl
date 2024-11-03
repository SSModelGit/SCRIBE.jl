module SCRIBE

using Reexport
using LinearAlgebra
using GaussianDistributions: ⊕, Gaussian
import Kalman

abstract type SCRIBEModel end
export SCRIBEModel

include("LinearGaussianScalarFieldModels.jl")
include("CovarianceIntersection.jl")

@reexport using .LinearGaussianScalarFieldModels

# Write your package code here.

end