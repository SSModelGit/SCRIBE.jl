using Test
using SCRIBE
using Plots: heatmap, plot, savefig

using JLD2: @save, @load
using LinearAlgebra: I, norm

"""Rapid-fire (potentially homogeneous multi-agent) setup.

In the case of multiple agents, the only differences will be in initial locations.
All agents will start at [0. 0.].
"""
function quick_setup(agent_ids, agent_conns, nᵩ; testing=true)
    ϕ₀ = (nᵩ==2) ? [-0.5,0.5] : [-0.5,0.5,0.75,0.5,-0.5]
    δ_w = 0.001

    gt_params=LGSFModelParameters(μ=hcat(range(-1,1,nᵩ), zeros(nᵩ)),σ=[1.],τ=[1.],ϕ₀=ϕ₀,
                                  A=[1-δ_w 0.; 0. 1-δ_w],
                                  Q=0.000001*Matrix{Float64}(I(nᵩ)))
    gt_model=[initialize_SCRIBEModel_from_parameters(gt_params)]

    ng = init_network_graph(agent_conns)
    a = 0
    for aid in agent_ids
        a += 1
        ag_params=LGSFModelParameters(μ=hcat(range(-1,1,nᵩ), zeros(nᵩ)),σ=[1.],τ=[1.],
                                      ϕ₀=zeros(nᵩ), A=Matrix{Float64}(I(nᵩ)),
                                      Q=0.0001*Matrix{Float64}(I(nᵩ)))
        observer=LGSFObserverBehavior(0.01)
        init_agent_loc=[0. 0.; -0.5 -0.5] + [0. 0.; a-1 a-1]
        lg_Fs = initialize_KF(ag_params, observer, copy(init_agent_loc), gt_model[1])
        ng.vertices[aid] = initialize_agent(aid, lg_Fs, ng)
    end

    return gt_model, ng
end

function mul_agent_distrib_KF(run_name::String, nₐ=3, nₛ=100; testing=true, tol=0.1)
    agent_ids = ["agent1", "agent2", "agent3", "agent4", "agent5"][1:nₐ]
    if nₐ==3
        agent_conns = Dict([("agent1", ["agent2"]),
                            ("agent2", ["agent1", "agent3"]),
                            ("agent3", ["agent2"])])
    elseif nₐ==5
        agent_conns = Dict([("agent1", ["agent2", "agent3"]),
                            ("agent2", ["agent1", "agent3", "agent4"]),
                            ("agent3", ["agent1", "agent2"]),
                            ("agent4", ["agent2", "agent5"]),
                            ("agent5", ["agent4"])])
    else
        return "Wrong number of agents champ."
    end
    nᵩ = 2
    (gt_model, ng) = quick_setup(agent_ids, agent_conns, nᵩ; testing=testing)

    # Arrays of fused information over time per agent (array of arrays)
    # fused_info = Dict([(aid, [copy(ng.vertices[aid].agent.information[1])]) for aid in agent_ids])

    sz=(1,2) # or make it (1,2)
    new_loc = Dict([(agent_ids[j], [0. 0.; -0.5 -0.5] + [0. 0.; j-1 j-1]) for j in eachindex(agent_ids)])
    # new_loc = zeros(sz...)
    for i in 1:nₛ
        print("k: ", i)
        while true
            if all(map(aid->distributed_fusion(i, aid, ng, 0.1, 360), agent_ids))
                print(" ...convergence reached:: ")
                break
            end
        end

        for aid in agent_ids; full_reset_network_connector(ng.vertices[aid].net_conn); end

        push!(gt_model, update_SCRIBEModel(gt_model[i]))
        # new_loc = 3*rand(sz...)
        for aid in agent_ids
            # progress_agent_env_filter(ng.vertices[aid].agent, fused_info[aid][end], gt_model[i+1], copy(new_loc[aid]))
            progress_agent_env_filter(ng.vertices[aid].agent, gt_model[i+1], copy(new_loc[aid]))
            push!(ng.vertices[aid].history, ng.vertices[aid].agent.estimates[i+1].observations.X)
        end
        println("ϕ: ", gt_model[end].ϕ, " | ̂ϕ: ", ng.vertices["agent1"].agent.estimates[end].estimate.ϕ)
    end

    # fests_m = [(aid*"m", ng.vertices[aid].agent.estimates[end].estimate.ϕ) for aid in agent_ids]
    # fests_v = [(aid*"v", inv(ng.vertices[aid].agent.information[end].Y)) for aid in agent_ids]
    # fvals = Dict([(:k, ng.vertices["agent1"].agent.k), (:ϕ, gt_model[end].ϕ), fests_m..., fests_v...])

    # println("Results:")
    # println("k: ", fvals[:k], "\nϕ(t=final):  ", fvals[:ϕ])
    # for aid in agent_ids
    #     println("ϕ_", aid,"(t=final): ", fvals[aid*"m"]," | P_"*aid*"(t_final): ", fvals*"v")
    # end

    simple_print_results(gt_model, ng)

    @save "test/"*run_name*".jld2" gt_model ng
    return gt_model, ng
end

function simple_print_results(gt_model::Vector{T} where T<:SCRIBEModel, ng::NetworkGraph)
    agent_ids = collect(keys(ng.vertices))

    fests_m = [(aid*"m", ng.vertices[aid].agent.estimates[end].estimate.ϕ) for aid in agent_ids]
    fests_v = [(aid*"v", inv(ng.vertices[aid].agent.information[end].Y)) for aid in agent_ids]
    fvals = Dict([(:k, ng.vertices["agent1"].agent.k), (:ϕ, gt_model[end].ϕ), fests_m..., fests_v...])

    println("Results:")
    println("k: ", fvals[:k], "\nϕ(t=final):  ", fvals[:ϕ])
    for aid in agent_ids
        println("ϕ_", aid,"(t=final): ", fvals[aid*"m"]," | P_"*aid*"(t_final): ", fvals[aid*"v"])
    end
end

function error_map_plots(run_name::String)
    @load "test/"*run_name*".jld2" gt ng

    x_range = -2:0.1:2
    y_range = -2:0.1:2

    let gt = gt[end], ag1_mod = ng.vertices["agent1"].agent.estimates[end].estimate,
        png_name = "test/res_plots/"*run_name*"_err_map_agent1.png"
        error_agent1(x::Vector) = norm(predict_SCRIBEModel(gt, x) - predict_SCRIBEModel(ag1_mod, x))

        gt_map_vals = [predict_SCRIBEModel(gt, [x,y]) for y in y_range, x in x_range]
        err1_map_vals = [error_agent1([x,y]) for y in y_range, x in x_range]

        gt_map = heatmap(x_range, y_range, gt_map_vals,
                         color=:viridis, xlabel="X", ylabel="Y",
                         title="Ground truth distribution of phenomena intensity")
        err1_map = heatmap(x_range, y_range, err1_map_vals,
                         color=:viridis, xlabel="X", ylabel="Y",
                         title="Error distribution of \nagent 1's predictions of phenomena intensity")
        
        plot(gt_map, err1_map, layout=(2,1), size=(600,800));
        savefig(png_name);
        println("Plot saved at "*png_name)
    end
end

# mul_agent_distrib_KF();