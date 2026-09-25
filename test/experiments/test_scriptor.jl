using SCRIBE

using Match: @match
using LinearAlgebra: I, Symmetric, cond, eigmin, norm
using Random: seed!
using Statistics: mean

using JLD2: @save

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
    return [(:gt_same, false), (:σ, σ), (:τ, τ), (:μ, μ), (:nᵩ, size(μ,1))]
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

function generate_sample_locs_from_wpts(wpts::Vector, k::Integer, d::Float64,
                                        _μₙ::Integer)
    let l = length(wpts), start = wpts[(k-1)%l+1], stop = wpts[k%l+1],
        d1 = abs(stop[1] - start[1])/d, d2 = abs(stop[2] - start[2])/d
        if d1 >= d2
            nₛ = max(Integer(round(d1)), 2)
        else
            nₛ = max(Integer(round(d2)), 2)
        end
        # A finite SPD prior makes per-step full rank unnecessary. Repeating
        # 121+ collinear samples only wastes memory and does not increase rank.
        hcat(range(start[1], stop[1], length=nₛ+1),
             range(start[2], stop[2], length=nₛ+1))
    end
end

function distance_based_conn(ng, agent_ids, conn_dist, i)
    new_conns = Dict{String, Vector{String}}()
    for ag1 in agent_ids
        new_conns[ag1] = []
        for ag2 in agent_ids
            if ag1 != ag2 &&
               norm(ng.vertices[ag1].agent.estimates[i].observations.X[end, :] -
                    ng.vertices[ag2].agent.estimates[i].observations.X[end, :]) < conn_dist
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
    agent_wpts = generate_agent_wpts(agent_ids, space_corners)
    (gt_model, ng) = quick_setup(agent_ids, agent_conns; gt_desc=gt_desc, ag_desc=ag_desc, agent_wpts)

    sample_dists = (space_corners[2] - space_corners[1])/20 # ensure distance is small enough for lawnmower pattern

    for i in 1:nₛ
        new_conns = comm_conns(ng, agent_ids, conn_dist, i; comm_type=comm_type)
        update_network_graph_edges(new_conns, ng)

        print("k: ", i)
        while true
            done = map(aid->distributed_fusion(i, aid, ng, 1e-8, 360), agent_ids)
            deliveries = deliver_consensus_messages(i, ng)
            for (aid, msg) in deliveries
                for nb in ng.edges[aid]
                    if nb != aid && haskey(ng.vertices, nb)
                        consume_consensus_message!(nb, ng.vertices[nb].net_conn,
                                                   msg, i, ng)
                    end
                end
            end
            for aid in keys(deliveries)
                ng.vertices[aid].net_conn.outbox["msg_out"] = nothing
            end
            if all(done)
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
        println(
            " completed | ground coefficient norm=",
            norm(gt_model[end].ϕ),
            " | agent1 coefficient norm=",
            norm(ng.vertices["agent1"].agent.estimates[end].estimate.ϕ),
        )
    end

    # simple_print_results(gt_model, ng)

    metrics = compute_run_metrics(gt_model, ng, space_corners)
    print_run_metrics(metrics)
    mkpath("test/res_data")
    @save "test/res_data/"*run_name*".jld2" gt_model ng space_corners metrics
    return gt_model, ng, space_corners
end

function compute_run_metrics(gt_model, ng, space_corners; grid_step=0.25)
    gt = gt_model[end]
    agent_ids = sort(collect(keys(ng.vertices)))
    x_range = space_corners[1]:grid_step:space_corners[2]
    locations = ([x, y] for x in x_range for y in x_range)
    location_values = collect(locations)

    predictions = Dict(
        aid => [predict_SCRIBEModel(
                    ng.vertices[aid].agent.estimates[end].estimate, x,
                ) for x in location_values]
        for aid in agent_ids
    )
    ground_truth = [predict_SCRIBEModel(gt, x) for x in location_values]

    parameter_consensus_error = maximum(
        norm(ng.vertices[a].agent.estimates[end].estimate.ϕ -
             ng.vertices[b].agent.estimates[end].estimate.ϕ) /
        max(norm(ng.vertices[a].agent.estimates[end].estimate.ϕ),
            norm(ng.vertices[b].agent.estimates[end].estimate.ϕ), 1.0)
        for a in agent_ids for b in agent_ids
    )
    prediction_consensus_rmse = maximum(
        sqrt(mean((predictions[a] .- predictions[b]).^2))
        for a in agent_ids for b in agent_ids
    )

    per_agent = Dict{String, Any}()
    for aid in agent_ids
        errors = predictions[aid] .- ground_truth
        Y = ng.vertices[aid].agent.information[end].Y
        Y_symmetric = Symmetric((Y + Y') / 2)
        per_agent[aid] = Dict(
            :mae => mean(abs.(errors)),
            :rmse => sqrt(mean(errors.^2)),
            :information_min_eigenvalue => eigmin(Y_symmetric),
            :information_condition_number => cond(Matrix(Y_symmetric)),
        )
    end

    Dict(
        :parameter_consensus_error => parameter_consensus_error,
        :prediction_consensus_rmse => prediction_consensus_rmse,
        :per_agent => per_agent,
    )
end

function print_run_metrics(metrics)
    println("Consensus diagnostics:")
    println("  relative parameter consensus error: ",
            metrics[:parameter_consensus_error])
    println("  prediction consensus RMSE: ",
            metrics[:prediction_consensus_rmse])
    println("Model and information diagnostics:")
    for aid in sort(collect(keys(metrics[:per_agent])))
        values = metrics[:per_agent][aid]
        println("  ", aid,
                " | MAE=", values[:mae],
                " | RMSE=", values[:rmse],
                " | λmin(Y)=", values[:information_min_eigenvalue],
                " | cond(Y)=", values[:information_condition_number])
    end
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

"""Run a bounded SCRIBE experiment sweep.

The default executes one ten-step, three-agent run without loading the plotting
backend. Set `make_plots=true` explicitly for plots. `full_sweep=true` exposes
the historical parameter grid, but `max_runs` remains a hard resource guard.
"""
function run_scriptor(; max_runs::Integer=1, make_plots::Bool=false,
                      full_sweep::Bool=false)
    @assert max_runs ≥ 1
    full_ground_truths = [
        (:gsf, Dict([(:nᵩ, 3), (:τ, 1.), (:σ, 1.)]), "full"),
        (:gsf, Dict([(:nᵩ, 3), (:τ, 0.5), (:σ, 1.)]), "weak"),
        (:gsf, Dict([(:nᵩ, 3), (:τ, 1.), (:σ, 0.5)]), "small"),
    ]
    gt_descs = full_sweep ? full_ground_truths : full_ground_truths[1:1]
    agent_description =
        create_grid_gsf_μστ([-5., 5], 1.0; τ=1.0, σ=0.5)

    anums = 3:3
    comm_types = [:dist]
    num_samples = full_sweep ? (10:10:30) : (10:10)
    obs_noises = full_sweep ?
                 [(0.01, "low"), (0.05, "med"), (0.1, "high")] :
                 [(0.01, "low")]
    completed_runs = String[]

    for gtd in gt_descs
        for obn in obs_noises
            for anum in anums
                for ctype in comm_types
                    for nₛ in num_samples
                        length(completed_runs) ≥ max_runs &&
                            return completed_runs
                        ad = Dict(reduce(
                            vcat,
                            [agent_description, (:nₐ, anum),
                             (:oₙ, obn[1])],
                        ))
                        run_name = make_run_name(
                            gtd, ad, obn, anum, ctype, nₛ;
                            state_run_parameters=false,
                        )
                        println("Commencing bounded run: ", run_name)
                        single_run(
                            run_name, gtd, ad;
                            nₛ, space_corners=[-5., 5.],
                            conn_dist=10, comm_type=ctype,
                        )
                        push!(completed_runs, run_name)
                        if make_plots
                            if !isdefined(@__MODULE__, :error_map_plots)
                                include(joinpath(@__DIR__, "plot_results.jl"))
                            end
                            println("Plotting run: ", run_name)
                            Base.invokelatest(error_map_plots, run_name)
                            Base.invokelatest(estimate_map_plots, run_name)
                            Base.invokelatest(spatial_rmse_eval_plot, run_name)
                            GC.gc()
                        end
                    end
                end
            end
        end
    end
    completed_runs
end

"""Run the matched three-agent no-communication/distributed comparison.

Both cases reset the same random seed before setup, so communication topology is
the only experimental difference. Simulations finish before plotting starts.
"""
function run_communication_comparison(; n_steps::Integer=10,
                                      random_seed::Integer=20250725,
                                      make_plots::Bool=true)
    @assert n_steps ≥ 1
    ground_truth =
        (:gsf, Dict([(:nᵩ, 3), (:τ, 1.0), (:σ, 1.0)]), "full")
    agent_description =
        create_grid_gsf_μστ([-5.0, 5.0], 1.0; τ=1.0, σ=0.5)
    observation_noise = (0.01, "low")
    agent_parameters = Dict(reduce(
        vcat,
        [agent_description, (:nₐ, 3), (:oₙ, observation_noise[1])],
    ))
    run_names = Dict{Symbol, String}()

    for communication in (:none, :dist)
        seed!(random_seed)
        run_name = make_run_name(
            ground_truth,
            agent_parameters,
            observation_noise,
            3,
            communication,
            n_steps;
            state_run_parameters=false,
        )
        println("Running matched communication case: ", communication)
        single_run(
            run_name,
            ground_truth,
            agent_parameters;
            nₛ=n_steps,
            space_corners=[-5.0, 5.0],
            conn_dist=10,
            comm_type=communication,
        )
        run_names[communication] = run_name
    end

    plot_paths = String[]
    if make_plots
        if !isdefined(@__MODULE__, :communication_comparison_plots)
            include(joinpath(@__DIR__, "plot_results.jl"))
        end
        append!(
            plot_paths,
            Base.invokelatest(
                communication_comparison_plots,
                run_names[:none],
                run_names[:dist],
            ),
        )
        GC.gc()
    end

    Dict(:runs => run_names, :plots => plot_paths)
end
