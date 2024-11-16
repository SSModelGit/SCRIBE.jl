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

    @test typeof(observations[1]) <: SCRIBEObserverState
    @test isapprox(observations[1].X,sample_locations[1])
    @test observations[1].k==1
    return gt_params, gt_model, sample_locations, observer, observations
end

function test_estimators(; testing=true)
    ϕ₀=[-0.5,0.5,0.75,0.5,-0.5]
    gt_params=LGSFModelParameters(μ=hcat(range(-1,1,5), zeros(5)),σ=[1.],τ=[1.],ϕ₀=ϕ₀,A=nothing,Q=nothing)
    gt_model=[initialize_SCRIBEModel_from_parameters(gt_params)]
    
    ag_params=LGSFModelParameters(μ=hcat(range(-1,1,5), zeros(5)),σ=[1.],τ=[1.],
                                  ϕ₀=zeros(size(ϕ₀,1)),A=Matrix{Float64}(I(size(ϕ₀,1))),Q=Matrix{Float64}(I(size(ϕ₀,1))))
    observer=LGSFObserverBehavior(0.3)
    sample_locations=[]
    init_agent_loc=[0. 0.;]
    push!(sample_locations, init_agent_loc)
    ag=initialize_agent(ag_params, observer, gt_model[1], init_agent_loc)
    if testing; @test typeof(ag) == AgentEnvModel; end

    lg_Fs=simple_LGSF_Estimators(ag)
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

    # Add temporary lexical scoping for stupid testing
    let gt_model=deepcopy(gt_model), sample_locations=deepcopy(sample_locations), ag=deepcopy(ag)

        for i in 1:4
            push!(gt_model, update_SCRIBEModel(gt_model[i]))
            push!(sample_locations, rand(2,2))
            next_agent_state(ag,zeros(ag_params.nᵩ), gt_model[i+1], sample_locations[i+1])
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

    for i in 1:100
        push!(gt_model, update_SCRIBEModel(gt_model[i]))
        push!(sample_locations, 3*rand(5,2))
        next_agent_state(ag,zeros(ag_params.nᵩ), gt_model[i+1], sample_locations[i+1])
        next_agent_time(ag)
        next_agent_info_state(ag, centralized_fusion([ag], ag.k)[1])
    end

    return ag, lg_Fs
end

@testset "Single Agent Setup" begin
    test_agent_setup()
end

@testset "Single Agent Estimators" begin
    test_estimators()
end