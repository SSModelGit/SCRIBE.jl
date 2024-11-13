using Test
using SCRIBE

function agent_setup()
    ϕ₀=[-1,1,-1,1,-1.]
    gt_params=LGSFModelParameters(μ=hcat(range(-1,1,5), zeros(5)),σ=[1.],τ=[1.],ϕ₀=ϕ₀,A=nothing,Q=nothing)
    gt_model=[initialize_SCRIBEModel_from_parameters(gt_params)]
    @test typeof(gt_model) <: Vector{T} where T<:SCRIBEModel

    v_s=0.1
    observer=LGSFObserverBehavior(v_s)
    @test typeof(observer) <: SCRIBEObserverBehavior
    sample_locations = []
    init_sample_loc = [1 1.]
    push!(sample_locations, init_sample_loc)
    observations=[scribe_observations(init_sample_loc,gt_model[1],observer)]

    @test typeof(observations[1]) <: SCRIBEObserverState
    @test isapprox(observations[1].X,sample_locations[1])
    @test observations[1].k==1
    println("Observatons: ", observations[1])
end

@testset "Individual Agent Setup" begin
    agent_setup()
end