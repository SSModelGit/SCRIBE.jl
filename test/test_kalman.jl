using Test
using SCRIBE
using LinearAlgebra: I

function test_agent_setup()
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
    sample_locations[end] = observations[1].X

    @test typeof(observations[1]) <: SCRIBEObserverState
    @test isapprox(observations[1].X,sample_locations[1])
    @test observations[1].k==1
    return gt_params, gt_model, sample_locations, observer, observations
end

function quick_setup(nᵩ=2; testing=true)
    if nᵩ==2
        ϕ₀=[-0.5,0.5]
    else
        ϕ₀=[-0.5,0.5,0.75,0.5,-0.5]
    end

    gt_params=LGSFModelParameters(μ=hcat(range(-1,1,nᵩ), zeros(nᵩ)),σ=[1.],τ=[1.],ϕ₀=ϕ₀,A=nothing,Q=nothing)
    gt_model=[initialize_SCRIBEModel_from_parameters(gt_params)]
    
    ag_params=LGSFModelParameters(μ=hcat(range(-1,1,nᵩ), zeros(nᵩ)),σ=[1.],τ=[1.],
                                  ϕ₀=zeros(nᵩ), A=Matrix{Float64}(I(nᵩ)), Q=Matrix{Float64}(I(nᵩ)))
    observer=LGSFObserverBehavior(0.3)
    sample_locations=[]
    init_agent_loc=[0. 0.;]
    ag=initialize_agent(ag_params, observer, gt_model[1], init_agent_loc)
    # push!(sample_locations, init_agent_loc)
    push!(sample_locations, ag.estimates[1].observations.X)
    if testing; @test typeof(ag) == AgentEnvModel; end

    lg_Fs=simple_LGSF_Estimators(ag)

    return ϕ₀, gt_params, gt_model, ag_params, observer, sample_locations, ag, lg_Fs
end

function test_estimators(; testing=true)
    (ϕ₀, gt_params, gt_model, ag_params, observer, sample_locations, ag, lg_Fs) = quick_setup(5, testing=testing)

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

function test_centralized_KF(; testing=true)
    (ϕ₀, gt_params, gt_model, ag_params, observer, sample_locations, ag, lg_Fs) = quick_setup(2, testing=testing)
    nᵩ=ag_params.nᵩ # storing for easier debugging

    fused_info=Any[ag.information[1]]
    sz=(2,2)
    new_loc = zeros(sz...)
    for i in 1:100
        push!(fused_info,centralized_fusion([lg_Fs], i)[1])
        push!(gt_model, update_SCRIBEModel(gt_model[i]))
        new_loc = 3*rand(sz...)
        progress_agent_env_filter(ag, fused_info[end], gt_model[i+1], new_loc)
        push!(sample_locations, ag.estimates[i+1].observations.X)
        # next_agent_state(ag,zeros(ag_params.nᵩ), gt_model[i+1], sample_locations[i+1])
        # next_agent_time(ag)
        # next_agent_info_state(ag, centralized_fusion([ag], ag.k)[1])
    end

    return ag, lg_Fs
end

@testset "Single Agent Setup" begin
    test_agent_setup()
end

@testset "Single Agent Estimators" begin
    test_estimators()
end

@testset "Centralized Kalman Filter" begin
    test_centralized_KF()
end