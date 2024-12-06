using Test
using SCRIBE
using LinearAlgebra: I

function test_agent_setup()
    ϕ₀=[-1,1,-1,1,-1.]
    gt_params=LGSFModelParameters(μ=hcat(range(-1,1,5), zeros(5)),σ=[1.],τ=[1.],ϕ₀=ϕ₀,A=nothing,Q=0.1.*I(5))
    gt_model=[initialize_SCRIBEModel_from_parameters(gt_params)]
    @test typeof(gt_model) <: Vector{T} where T<:SCRIBEModel

    v_s=0.1
    observer=LGSFObserverBehavior(v_s)
    @test typeof(observer) <: SCRIBEObserverBehavior
    sample_locations = []
    init_sample_loc = [0. 0.]
    push!(sample_locations, init_sample_loc)
    observations=[scribe_observations(init_sample_loc,gt_model[1],observer)]
    sample_locations[end] = observations[1].X

    @test typeof(observations[1]) <: SCRIBEObserverState
    @test isapprox(observations[1].X,sample_locations[1])
    @test observations[1].k==1
    return gt_params, gt_model, sample_locations, observer, observations
end

"""Rapid-fire (potentially homogeneous multi-agent) setup.

In the case of multiple agents, the only differences will be in initial locations.
All agents will start at [0. 0.].
"""
function quick_setup(nᵩ=2; testing=true, nₐ=1)
    ϕ₀ = (nᵩ==2) ? [-0.5,0.5] : [-0.5,0.5,0.75,0.5,-0.5]

    gt_params=LGSFModelParameters(μ=hcat(range(-1,1,nᵩ), zeros(nᵩ)),σ=[1.],τ=[1.],ϕ₀=ϕ₀,A=nothing,
                                  Q=0.000001*Matrix{Float64}(I(nᵩ)))
    gt_model=[initialize_SCRIBEModel_from_parameters(gt_params)]

    agents = Vector{SCRIBEAgent}(undef, nₐ)
    for a in 1:nₐ
        ag_params=LGSFModelParameters(μ=hcat(range(-1,1,nᵩ), zeros(nᵩ)),σ=[1.],τ=[1.],
                                      ϕ₀=zeros(nᵩ), A=Matrix{Float64}(I(nᵩ)),
                                      Q=0.0001*Matrix{Float64}(I(nᵩ)))
        observer=LGSFObserverBehavior(0.01)
        init_agent_loc=[0. 0.; -0.5 -0.5] + [0. 0.; a-1 a-1]
        lg_Fs = initialize_KF(ag_params, observer, copy(init_agent_loc), gt_model[1])
        agents[a] = initialize_agent(lg_Fs)
    end

    return gt_model, agents
end

function test_estimators(; testing=true)
    (gt_model, agents) = quick_setup(5, testing=testing)
    ag = agents[1].agent
    lg_Fs = agents[1].estimators
    ag_params = agents[1].params
    sample_locations = agents[1].history

    if testing
        @test lg_Fs.ϕ(1)==zeros(5)
        @test lg_Fs.A(1)==Matrix{Float64}(I(5))
        @test lg_Fs.H(1)!==nothing
        @test lg_Fs.z(1)!==nothing

        @test_throws BoundsError lg_Fs.ϕ(2)
        @test_throws BoundsError lg_Fs.A(2)
        @test_throws BoundsError lg_Fs.H(2)
        @test_throws BoundsError lg_Fs.z(2)
    end

    sz = (2,2)
    new_loc = zeros(sz...)
    for i in 1:4
        push!(gt_model, update_SCRIBEModel(gt_model[i]))
        new_loc = rand(sz...)
        next_agent_state(ag,zeros(ag_params.nᵩ), gt_model[i+1], new_loc)
        push!(sample_locations, ag.estimates[i+1].observations.X)
    end
        
    if testing
        @test_throws BoundsError lg_Fs.ϕ(6)
        @test_throws BoundsError lg_Fs.A(6)
        @test_throws BoundsError lg_Fs.H(6)
        @test_throws BoundsError lg_Fs.z(6)

        @test all([isapprox(lg_Fs.ϕ(i), zeros(5)) for i in 1:5])
        @test !all([size(lg_Fs.H(i))==(2,5) for i in 1:5])
        @test all([size(lg_Fs.H(i))==(2,5) for i in 2:5])
        @test all([size(lg_Fs.z(i))==(2,) for i in 2:5])
    end
end

function test_observability(; testing=true)
    (gt_model, agents) = quick_setup(5, testing=testing)
    sample_locations = agents[1].history
    @test !isapprox(sample_locations, zeros(1,2))
end

function single_agent_centralized_KF(nₛ=100; testing=true, tol=0.1,)
    nᵩ=2 # storing for easier debugging
    (gt_model, agents) = quick_setup(nᵩ, testing=testing)
    sagent = agents[1]
    ag = sagent.agent
    lg_Fs = sagent.estimators
    ag_params = sagent.params
    sample_locations = sagent.history

    fused_info=Any[ag.information[1]]
    sz=(1,2) # or make it (1,2)
    new_loc = [0. 0.; 0.5 0.5]
    # new_loc = zeros(sz...)
    for i in 1:nₛ
        push!(fused_info,centralized_fusion([lg_Fs], i)[1])
        push!(gt_model, update_SCRIBEModel(gt_model[i]))
        # new_loc = 3*rand(sz...)
        # progress_agent_env_filter(ag, fused_info[end], gt_model[i+1], new_loc)
        progress_agent_env_filter(ag, fused_info[end], gt_model[i+1], copy(new_loc))
        push!(sample_locations, ag.estimates[i+1].observations.X)
    end

    fvals = Dict(:ϕⱼ=>ag.estimates[end].estimate.ϕ, :ϕ=>gt_model[end].ϕ)

    println("Results:")
    println("k: ", ag.k, "\nϕⱼ(t=final): ", fvals[:ϕⱼ], "\nϕ(t=final):  ", fvals[:ϕ])

    if testing
        @test all(abs.(fvals[:ϕⱼ]-fvals[:ϕ]) .< tol)
    end

    return ag, lg_Fs, gt_model
end

function mul_agent_centralized_KF(nₐ=2, nₛ=100; testing=true, tol=0.1)
    nᵩ = 2
    (gt_model, agents) = quick_setup(nᵩ; testing=testing, nₐ=nₐ)

    estimators = KFEstimators[] # array of pointers to agent estimators
    # numerically iterate to ensure we can always match agent to array element
    for j in eachindex(agents)
        push!(estimators, agents[j].estimators)
    end

    # estimators = KFEstimators[agents[j].estimators for j in eachindex(agents)]

    # Arrays of fused information over time per agent (array of arrays)
    fused_info = Any[[copy(agents[j].agent.information[1]) for j in eachindex(agents)]]

    sz=(1,2) # or make it (1,2)
    new_loc = [[0. 0.; -0.5 -0.5] + [0. 0.; j-1 j-1] for j in eachindex(agents)]
    # new_loc = zeros(sz...)
    for i in 1:nₛ
        push!(fused_info,centralized_fusion(estimators, i))
        push!(gt_model, update_SCRIBEModel(gt_model[i]))
        # new_loc = 3*rand(sz...)
        for j in eachindex(agents)
            progress_agent_env_filter(agents[j].agent, fused_info[end][j], gt_model[i+1], copy(new_loc[j]))
            push!(agents[j].history, agents[j].agent.estimates[i+1].observations.X)
        end
    end

    fests = [(string(j), agents[j].agent.estimates[end].estimate.ϕ) for j in eachindex(agents)]
    fvals = Dict([(:k, agents[1].agent.k), (:ϕ, gt_model[end].ϕ), fests...])

    println("Results:")
    println("k: ", fvals[:k], "\nϕ(t=final):  ", fvals[:ϕ])
    for j in eachindex(agents)
        println("ϕ_", j,"(t=final): ", fvals[string(j)])
    end

    return agents
end

@testset "Single Agent Setup" begin
    test_agent_setup()
end

@testset "Single Agent Estimators" begin
    test_estimators()
end

@testset "Centralized Kalman Filter" begin
    single_agent_centralized_KF(100; tol=0.1)
    mul_agent_centralized_KF(2, 100; tol=0.1)
end