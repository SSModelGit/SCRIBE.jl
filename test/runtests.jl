using SCRIBE
using Test

# using GaussianProcesses: GPE # this is ONLY for testing

@testset "SCRIBE.jl" begin
    include("test_kalman.jl")
end
