using Test
using SCRIBE

using Match: @match
using LinearAlgebra: I, norm
using Combinatorics: combinations

using JLD2: @save, @load
using Plots: heatmap, plot, plot!, savefig

"""Rapid-fire (potentially homogeneous multi-agent) setup.

In the case of multiple agents, the only differences will be in initial locations.
All agents will start at [0. 0.].
"""
function quick_setup(agent_ids, agent_conns;
                     gt_desc=(:gsf, Dict([(:nᵩ, 3), (:τ, 1.), (:σ, 1.)])),
                     ag_desc = Dict([(:gt_same, true), (:σ, 1.), (:τ, 1.), (:μ, [-0.5 0; 0. 0.; 0.5 0.]), (:nᵩ, 3)]),
                     agent_wpts = Dict())
    @match gt_desc[1] begin
        :gsf => return quick_GSF(agent_ids, agent_conns, gt_desc[2]; ag_desc = ag_desc, agent_wpts)
    end
end

function create_grid_gsf_μστ(space_corners=[-5., 5], well_spacing=0.5; τ=1.0, σ=0.5)
    span = space_corners[1]:well_spacing:space_corners[2]
    μ = reduce(vcat, transpose.(map(collect, reshape(collect(Iterators.product(span, span)), :, 1))))
    return [(:gt_same, false), (:σ, 1.), (:τ, 1.), (:μ, μ), (:nᵩ, size(μ,1))]
end

function quick_GSF(agent_ids, agent_conns, gt_desc=Dict([(:nᵩ, 3), (:τ, 1.), (:σ, 1.)]);
                   ag_desc=Dict([(:gt_same, true), (:σ, 1.), (:τ, 1.), (:μ, [-0.5 0; 0. 0.; 0.5 0.]), (:nᵩ, 3)]),
                   agent_wpts=Dict())
    nᵩ = gt_desc[:nᵩ]
    σ = gt_desc[:σ]
    τ = gt_desc[:τ]
    (ϕ₀, μ) = @match nᵩ begin
        2 => ([-0.5,0.5], hcat(range(-1,1,nᵩ), zeros(nᵩ)))
        3 => ([-0.5, 0.5, -0.5], [-2.8 2.6; 0 -2.6; 2.8 1.6])
        5 => ([-0.5,0.5,0.75,0.5,-0.5], hcat(range(-1,1,nᵩ), zeros(nᵩ)))
    end
    δ_w = 0.001 # represents temporal system dynamics (shift from A=identity)

    gt_params=LGSFModelParameters(μ=μ,σ=[σ],τ=[τ],ϕ₀=ϕ₀,
                                  A=Matrix{Float64}(I(nᵩ) .* (1- δ_w)),
                                  Q=0.000001*Matrix{Float64}(I(nᵩ)))
    gt_model=[initialize_SCRIBEModel_from_parameters(gt_params)]

    if !ag_desc[:gt_same]
        nᵩ = ag_desc[:nᵩ]
        σ = ag_desc[:σ]
        τ = ag_desc[:τ]
        μ = ag_desc[:μ]
    end
    ng = init_network_graph(agent_conns)
    a = 0
    for aid in agent_ids
        a += 1
        ag_params=LGSFModelParameters(μ=μ,σ=[σ],τ=[τ],
                                      ϕ₀=zeros(nᵩ), A=Matrix{Float64}(I(nᵩ)),
                                      Q=0.0001*Matrix{Float64}(I(nᵩ)))
        observer=LGSFObserverBehavior(ag_desc[:oₙ])
        init_agent_loc = generate_sample_locs_from_wpts(agent_wpts[aid], 1, 1., size(ag_desc[:μ], 1)) # use some random value for sample distance for now
        # init_agent_loc=[0. 0.; -0.5 -0.5] + [0. 0.; a-1 a-1]
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

function generate_sample_locs_from_wpts(wpts::Vector, k::Integer, d::Float64, μₙ::Integer)
    let l = length(wpts), start = wpts[(k-1)%l+1], stop = wpts[k%l+1],
        d1 = abs(stop[1] - start[1])/d, d2 = abs(stop[2] - start[2])/d
        if d1 >= d2
            nₛ = max(Integer(round(d1)), 2)
        else
            nₛ = max(Integer(round(d2)), 2)
        end
        nₛ = max(nₛ, μₙ+1) # max against the number of environmental wells
        # println("\nstart: ", start, " | stop: ", stop, " || d1: ", d1, " | d2: ", d2, " | nₛ: ", nₛ)
        hcat(range(start[1], stop[1], length=nₛ+1),
             range(start[2], stop[2], length=nₛ+1))
    end
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

function comm_conns(ng, agent_ids, conn_dist, i; comm_type=:dist)
    @match comm_type begin
        :dist => distance_based_conn(ng, agent_ids, conn_dist, i)
        :none => no_comm_conns(ng, agent_ids)
    end
end

function single_run(run_name::String, gt_desc::Tuple, ag_desc::Dict;
                    nₛ=100, space_corners = [-5., 5.], conn_dist=10, comm_type=:dist)
    nₐ = ag_desc[:nₐ]
    μₙ = size(ag_desc[:μ], 1)

    agent_ids = ["agent"*string(i) for i in 1:nₐ]
    agent_conns = @match nₐ begin
        2 => Dict([("agent1", ["agent2"]),
                   ("agent2", ["agent1"])])
        3 => Dict([("agent1", ["agent2"]),
                   ("agent2", ["agent1", "agent3"]),
                   ("agent3", ["agent2"])])
        4 => Dict([("agent1", ["agent2", "agent3"]),
                   ("agent2", ["agent1", "agent4"]),
                   ("agent3", ["agent1", "agent4"]),
                   ("agent4", ["agent2", "agent3"])])
        5 => Dict([("agent1", ["agent2", "agent3"]),
                   ("agent2", ["agent1", "agent3", "agent4"]),
                   ("agent3", ["agent1", "agent2"]),
                   ("agent4", ["agent2", "agent5"]),
                   ("agent5", ["agent4"])])
    end
    if nₐ==3
        println("Wrong number of agents champ.")
    end
    agent_wpts = generate_agent_wpts(agent_ids, space_corners)
    (gt_model, ng) = quick_setup(agent_ids, agent_conns; gt_desc=gt_desc, ag_desc=ag_desc, agent_wpts)

    sample_dists = (space_corners[2] - space_corners[1])/20 # ensure distance is small enough for lawnmower pattern

    for i in 1:nₛ
        new_conns = comm_conns(ng, agent_ids, conn_dist, i; comm_type=comm_type)
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
                                      copy(generate_sample_locs_from_wpts(agent_wpts[aid], i+1, sample_dists, μₙ)))
            push!(ng.vertices[aid].history, ng.vertices[aid].agent.estimates[i+1].observations.X)
        end
        println("ϕ: ", gt_model[end].ϕ, " | ̂ϕ: ", ng.vertices["agent1"].agent.estimates[end].estimate.ϕ)
    end

    # simple_print_results(gt_model, ng)

    @save "test/res_data/"*run_name*".jld2" gt_model ng space_corners
    return gt_model, ng, space_corners
end

function frechet_dist_eval_plot(run_name::String; layout_size=(800,500))
    png_name = "test/res_plots/"*run_name*"_frechet_dist.png"

    @load "test/res_data/"*run_name*".jld2" gt_model ng space_corners

    x_range = space_corners[1]:0.1:space_corners[2]
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

function error_map_plots(run_name::String; layout_size=(2700,1500), mode=:norm)
    png_name = "test/res_plots/"*run_name*"_err_map.png"

    @load "test/res_data/"*run_name*".jld2" gt_model ng space_corners
    gt = gt_model[end]

    num_plots = length(ng.vertices)
    needs_padding = num_plots%2==1
    # layout_num = (2,Integer(ceil(num_plots/2)))
    layout_num = (num_plots, num_plots+1)
    pred_cgrad = :thermal
    err_cgrad = :ice

    x_range = space_corners[1]:0.1:space_corners[2]
    y_range = copy(x_range)
    gt_map = heatmap(x_range, y_range, [predict_SCRIBEModel(gt, [x,y]) for y in y_range, x in x_range],
                     color=:viridis, # xlabel="X", ylabel="Y",
                     title="Ground truth distribution",
                     c = pred_cgrad)

    error_maps = Dict{String, Vector}()
    error_mapv = Any[gt_map]

    for (i, aid) in enumerate(keys(ng.vertices))
        error_maps[aid] = Any[]
        if i≠1
            println("Blank because ground truth is already added, for "*aid[end])
            push!(error_maps[aid], plot(legend=false,grid=false,foreground_color_subplot=:white))
        end
        for (j, a2d) in enumerate(keys(ng.vertices))
            let m1 = ng.vertices[aid].agent.estimates[end].estimate,
                m2 = ng.vertices[a2d].agent.estimates[end].estimate,
                h1=reduce(vcat, ng.vertices[aid].history),
                h2=reduce(vcat, ng.vertices[a2d].history)

                if i==j
                    println("Ground comparison for agent "*aid[end])
                    push!(error_maps[aid], heatmap(x_range, y_range, [error_mapf(gt, m1, [x,y]; mode=mode) for y in y_range, x in x_range],
                                                   color=:viridis, # xlabel="X", ylabel="Y",
                                                   title="% error between \nAgent "*aid[end]*"'s predictions and ground truth",
                                                   c = err_cgrad))
                    plot!(error_maps[aid][end], copy(h1[:,1]), copy(h1[:,2]),
                          linestyle=:dash, marker=:xcross, linecolor=:red, label="Agent "*aid[end]*" sampling sites")
                else
                    println("Comparison for agent "*aid[end]*" with agent "*a2d[end])
                    push!(error_maps[aid], heatmap(x_range, y_range, [error_mapf(m1, m2, [x,y]; mode=mode) for y in y_range, x in x_range],
                                                   color=:viridis, # xlabel="X", ylabel="Y",
                                                   title="% error between \nAgent "*aid[end]*" and Agent "*a2d[end]*" predictions",
                                                   c = err_cgrad))
                    plot!(error_maps[aid][end], copy(h1[:,1]), copy(h1[:,2]),
                          linestyle=:dash, marker=:xcross, linecolor=:red, label="Agent "*aid[end]*" sampling sites")
                    plot!(error_maps[aid][end], copy(h2[:,1]), copy(h2[:,2]),
                          linestyle=:dash, marker=:xcross, linecolor=:red, label="Agent "*a2d[end]*" sampling sites")
                end
            end
        end
        append!(error_mapv, copy(error_maps[aid]))
    end

    # if needs_padding; push!(error_mapv, nothing); end
    plot(error_mapv..., layout=layout_num, size=layout_size)
    savefig(png_name);
    println("Plot saved at "*png_name)
end

function make_run_name(gtd, ad, obn, anum, ctype, nₛ; state_run_parameters=true)
    run_name = string(anum)*"a_"*string(ad[:nᵩ])*"w_"*gtd[3]*"GT_"*string(ctype)*"c_"*obn[2]*"o_"*string(nₛ)*"s_lawnmower"
    if state_run_parameters
        println("================================================")
        println("Starting new run...")
        println("Paramters::")
        println("Ground model type: ", gtd[1], " | Environment character: ", gtd[3])
        println("Number of agents: ", anum)
        println("Number of wells in environment approximation: ", ad[:nᵩ])
        println("Agent communication model: ", ctype)
        println("Observation noise process character: ", obn[2])
        println("Number of samples: ", nₛ)
        println("------------------------------------------------")
        println("Run named: ", run_name)
        println("Agents will follow a lawnmower pattern.")
        println("Commencing run...")
    end
    return run_name
end

function run_scriptor()
    gt_descs = [(:gsf, Dict([(:nᵩ, 3), (:τ, 1.), (:σ, 1.)]), "full"),
                (:gsf, Dict([(:nᵩ, 3), (:τ, 0.5), (:σ, 1.)]), "weak"),
                (:gsf, Dict([(:nᵩ, 3), (:τ, 1.), (:σ, 0.5)]), "small")]
    ag_desc = [[(:gt_same, true), (:σ, 1.), (:τ, 1.), (:μ, [-0.5 0; 0. 0.; 0.5 0.]), (:nᵩ, 3)],
               create_grid_gsf_μστ([-5., 5], 1.0; τ=1.0, σ=0.5)]

    anums = 3:5
    comm_types = [:none, :dist]
    num_samples = 10:10:30
    obs_noises = [(0.01, "low"), (0.05, "med"), (0.1, "high")]

    for gtd in gt_descs
        for agd in ag_desc
            for obn in obs_noises
                for anum in anums
                    for ctype in comm_types
                        for nₛ in num_samples
                            ad = Dict(reduce(vcat, [agd, (:nₐ, anum), (:oₙ, obn[1])]))
                            run_name = make_run_name(gtd, ad, obn, anum, ctype, nₛ; state_run_parameters=false)
                            if size(ad[:μ], 1) > 10
                                println("Commencing run: ", run_name)
                                single_run(run_name, gtd, ad; nₛ=nₛ, space_corners = [-5., 5.], conn_dist=10, comm_type=ctype)
                                println("Plotting run: ", run_name)
                                error_map_plots(run_name)
                            end
                        end
                    end
                end
            end
        end
    end
end