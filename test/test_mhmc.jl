using Test
using SCRIBE

using Match: @match
using LinearAlgebra: I, norm

using JLD2: @save, @load
using Plots: heatmap, plot, plot!, savefig

"""Rapid-fire (potentially homogeneous multi-agent) setup.

In the case of multiple agents, the only differences will be in initial locations.
All agents will start at [0. 0.].
"""
function quick_setup(agent_ids, agent_conns, nᵩ; testing=true)
    (ϕ₀, μ) = @match nᵩ begin
        2 => ([-0.5,0.5], hcat(range(-1,1,nᵩ), zeros(nᵩ)))
        3 => ([-0.5, 0.5, -0.5], [-2.8 2.6; 0 -2.6; 2.8 1.6])
        5 => ([-0.5,0.5,0.75,0.5,-0.5], hcat(range(-1,1,nᵩ), zeros(nᵩ)))
    end
    δ_w = 0.001 # represents temporal system dynamics (shift from A=identity)

    gt_params=LGSFModelParameters(μ=μ,σ=[0.5],τ=[1.],ϕ₀=ϕ₀,
                                  A=Matrix{Float64}(I(nᵩ) .* (1- δ_w)),
                                  Q=0.000001*Matrix{Float64}(I(nᵩ)))
    gt_model=[initialize_SCRIBEModel_from_parameters(gt_params)]

    ng = init_network_graph(agent_conns)
    a = 0
    for aid in agent_ids
        a += 1
        ag_params=LGSFModelParameters(μ=μ,σ=[0.5],τ=[1.],
                                      ϕ₀=zeros(nᵩ), A=Matrix{Float64}(I(nᵩ)),
                                      Q=0.0001*Matrix{Float64}(I(nᵩ)))
        observer=LGSFObserverBehavior(0.01)
        init_agent_loc=[0. 0.; -0.5 -0.5] + [0. 0.; a-1 a-1]
        lg_Fs = initialize_KF(ag_params, observer, copy(init_agent_loc), gt_model[1])
        ng.vertices[aid] = initialize_agent(aid, lg_Fs, ng)
    end

    return gt_model, ng
end

function generate_agent_wpts(agent_ids, corners = [-5.,5.])
    n = length(agent_ids)
    agent_coords = Dict([(aid, Any[]) for aid in agent_ids])

    bc = [corners[1], corners[1]]
    h = corners[2] - corners[1]
    w = (h - 2*h/10) / n

    for aid in agent_ids
        let ag = agent_coords[aid]
            push!(ag, copy(bc))
            push!(ag, ag[end] + [0.,  h])
            push!(ag, ag[end] + [w/3, 0.])
            push!(ag, ag[end] + [0., -h])
            push!(ag, ag[end] + [w/3, 0.])
            push!(ag, ag[end] + [0.,  h])
            push!(ag, ag[end] + [w/3, 0.])
            push!(ag, ag[end] + [0., -h])

            bc = ag[end] + [h/10, 0.]
        end
    end

    agent_coords
end

function generate_sample_locs_from_wpts(wpts::Vector, k::Integer, d::Float64,)
    let l = length(wpts), start = wpts[(k-1)%l+1], stop = wpts[k%l+1],
        d1 = abs(stop[1] - start[1])/d, d2 = abs(stop[2] - start[2])/d
        if d1 >= d2
            nₛ = max(Integer(round(d1)), 2)
        else
            nₛ = max(Integer(round(d2)), 2)
        end
        # println("\nstart: ", start, " | stop: ", stop, " || d1: ", d1, " | d2: ", d2, " | nₛ: ", nₛ)
        hcat(range(start[1], stop[1], length=nₛ+1),
             range(start[2], stop[2], length=nₛ+1))
    end
end

function mul_agent_perf_conn(run_name::String, nₐ=3, nₛ=100; testing=true, tol=0.1)
    space_corners = [-2., 2.] # corner coordinates

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

    agent_wpts = generate_agent_wpts(agent_ids, space_corners)
    sample_dists = (space_corners[2] - space_corners[1])/20 # ensure distance is small enough for lawnmower pattern

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
        for aid in agent_ids
            progress_agent_env_filter(ng.vertices[aid].agent, gt_model[i+1],
                                      copy(generate_sample_locs_from_wpts(agent_wpts[aid], i, sample_dists)))
            push!(ng.vertices[aid].history, ng.vertices[aid].agent.estimates[i+1].observations.X)
        end
        println("ϕ: ", gt_model[end].ϕ, " | ̂ϕ: ", ng.vertices["agent1"].agent.estimates[end].estimate.ϕ)
    end

    simple_print_results(gt_model, ng)

    @save "test/res_data/"*run_name*".jld2" gt_model ng
    return gt_model, ng
end

function distance_based_conn(ng, agent_ids, conn_dist, i)
    new_conns = Dict{String, Vector{String}}()
    for ag1 in agent_ids
        new_conns[ag1] = []
        for ag2 in agent_ids
            if norm(ng.vertices[ag1].agent.estimates[i].observations.X[end, :] - ng.vertices[ag2].agent.estimates[i].observations.X[end, :]) < conn_dist
                push!(new_conns[ag1], ag2)
            end
        end
    end
    new_conns
end

function no_comm_conns(ng, agent_ids)
    new_conns = Dict{String, Vector{String}}()
    for ag in agent_ids
        new_conns[ag] = String[]
    end
    new_conns
end


function mul_agent_poor_conn(run_name::String, nₐ=3, nₛ=100; testing=true, tol=0.1, conn_dist=10, comm_type=:dist)
    space_corners = [-5., 5.] # corner coordinates

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
    nᵩ = 3
    (gt_model, ng) = quick_setup(agent_ids, agent_conns, nᵩ; testing=testing)

    agent_wpts = generate_agent_wpts(agent_ids, space_corners)
    sample_dists = (space_corners[2] - space_corners[1])/20 # ensure distance is small enough for lawnmower pattern

    for i in 1:nₛ
        new_conns = @match comm_type begin
            :dist => distance_based_conn(ng, agent_ids, conn_dist, i)
            :none => no_comm_conns(ng, agent_ids)
        end
        update_network_graph_edges(new_conns, ng)

        print("k: ", i)
        while true
            if all(map(aid->distributed_fusion(i, aid, ng, 0.1, 360), agent_ids))
                print(" ...convergence reached:: ")
                break
            end
        end

        for aid in agent_ids; full_reset_network_connector(ng.vertices[aid].net_conn); end

        push!(gt_model, update_SCRIBEModel(gt_model[i]))
        for aid in agent_ids
            progress_agent_env_filter(ng.vertices[aid].agent, gt_model[i+1],
                                      copy(generate_sample_locs_from_wpts(agent_wpts[aid], i, sample_dists)))
            push!(ng.vertices[aid].history, ng.vertices[aid].agent.estimates[i+1].observations.X)
        end
        println("ϕ: ", gt_model[end].ϕ, " | ̂ϕ: ", ng.vertices["agent1"].agent.estimates[end].estimate.ϕ)
    end

    simple_print_results(gt_model, ng)

    @save "test/res_data/"*run_name*".jld2" gt_model ng
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

function frechet_dist_eval_plot(run_name::String; layout_size=(800,500))
    png_name = "test/res_plots/"*run_name*"_frechet_dist.png"

    @load "test/res_data/"*run_name*".jld2" gt_model ng

    x_range = -5:0.1:5
    y_range = copy(x_range)
    err_sq(gt, am, v) = norm(predict_SCRIBEModel(gt, v) - predict_SCRIBEModel(am, v))^2
    frechet(gt, am) = sum([err_sq(gt, am, [x, y]) for x in x_range for y in y_range])
    agent_ids = collect(keys(ng.edges))
    frechet_dists = Dict([(aid, [frechet(gt_model[i], ng.vertices[aid].agent.estimates[i].estimate) for i in 1:length(gt_model)]) for aid in agent_ids])

    p1 = plot(size=layout_size, xlabel="Time (in discretized steps)", ylabel="(pseudo-)Frechet distance",
    title="Performance of agent learned environment models \ncompared to ground truth over time", margin=(10, :mm))
    for aid in agent_ids
        plot!(p1, 1:length(gt_model), frechet_dists[aid], label="Agent "*aid[end], lw=2)
    end

    p2 = plot(size=layout_size, xlabel="Time (in discretized steps)",
              title="Close-up of performance of agent learned environment models \nafter gathering initial observations", margin=(10, :mm))
    for aid in agent_ids
        plot!(p2, 1:length(gt_model), frechet_dists[aid], label="Agent "*aid[end], lw=2, yrange=[0., 1.0])
    end

    plot(p1, p2, layout=(1,2), size=(layout_size[1]*2, layout_size[2]))
    savefig(png_name);
    println("Plot saved at "*png_name)
end

function error_mapf(gt::SCRIBEModel, m::SCRIBEModel, x::Vector; mode=:norm)
    @match mode begin
        :norm => norm(predict_SCRIBEModel(gt, x) - predict_SCRIBEModel(m, x))
        :tane => abs(2*atan(predict_SCRIBEModel(gt, x) / predict_SCRIBEModel(m, x)) - π/2)/(π/2)
    end
end

function error_map_plots(run_name::String; layout_size=(1200,1000), mode=:tane)
    png_name = "test/res_plots/"*run_name*"_err_map.png"

    @load "test/res_data/"*run_name*".jld2" gt_model ng
    gt = gt_model[end]

    num_plots = length(ng.vertices) + 1
    needs_padding = num_plots%2==1
    layout_num = (2,Integer(ceil(num_plots/2)))

    x_range = -5:0.1:5
    y_range = copy(x_range)
    gt_map = heatmap(x_range, y_range, [predict_SCRIBEModel(gt, [x,y]) for y in y_range, x in x_range],
                     color=:viridis, xlabel="X", ylabel="Y",
                     title="\nGround truth distribution of phenomena intensity",
                     c = :thermal)

    error_maps = Dict{String, Any}()
    error_mapv = Any[gt_map]

    for aid in keys(ng.vertices)
        let m = ng.vertices[aid].agent.estimates[end].estimate, h=reduce(vcat, ng.vertices[aid].history)
            error_maps[aid] = heatmap(x_range, y_range, [error_mapf(gt, m, [x,y]; mode=mode) for y in y_range, x in x_range],
                                      color=:viridis, xlabel="X", ylabel="Y",
                                      title="Error distribution of \nAgent "*aid[end]*"'s predictions of phenomena intensity",
                                      c = :berlin)
            plot!(error_maps[aid], copy(h[:,1]), copy(h[:,2]),
                  linestyle=:dash, marker=:xcross, linecolor=:red, label="Sampling path and sites")
            push!(error_mapv, error_maps[aid])
        end
    end

    if needs_padding; push!(error_mapv, nothing); end
    plot(error_mapv..., layout=layout_num, size=layout_size)
    savefig(png_name);
    println("Plot saved at "*png_name)
end

# mul_agent_distrib_KF();