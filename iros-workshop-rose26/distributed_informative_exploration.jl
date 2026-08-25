ENV["GKSwstype"] = "100"

include(joinpath(@__DIR__, "informative_exploration_problem.jl"))
include(joinpath(@__DIR__, "filtering_experiments.jl"))

using Plots
using Statistics

const EXPLORATION_BACKENDS = (
    :centralized,
    :scribe,
    :kf_only,
    :ci_only,
    :last_k_observations,
    :independent,
)
const EXPLORATION_COLORS = Dict(
    :centralized => :black,
    :scribe => :royalblue,
    :kf_only => :firebrick,
    :ci_only => :purple,
    :last_k_observations => :seagreen,
    :independent => :darkorange,
)
const EXPLORATION_LABELS = Dict(
    :centralized => "Centralized (Ideal)",
    :scribe => "SCRIBE",
    :kf_only => "KF Only",
    :ci_only => "CI Only",
    :last_k_observations => "Last-2 Observations",
    :independent => "No Communication",
)

Base.@kwdef mutable struct ExplorationBackendState
    ids
    states
    models
    policies
    noise_rngs
    dynamics_rngs
    plans
    observations
    paths
    histories
    oracle
    network
    known_observations
    communication
    cumulative_reward
end

distributed_exploration_settings(profile) = @match profile begin
    :full => Dict(
        :seeds => Tuple(211:2:233),
        :representative_seed => 221,
        :representative_agent_index => 2,
        :n_agents => 4,
        :n_samples => 36,
        :phase_steps => (12, 12, 12),
        :communication_radius => 8.25,
        :last_k => 2,
        :navigation_points => 33,
        :evaluation_points => 41,
        :lookahead => 5,
        :planning_iterations => 64,
        :time_budget => 0.0,
        :noise_variance => 0.30,
        :process_variance => 1e-10,
        :consensus_threshold => 1e-7,
        :consensus_timeline => 180,
        :animation_fps => 3,
    )
    :pilot => Dict(
        :seeds => (211, 213, 215, 217),
        :representative_seed => 211,
        :representative_agent_index => 1,
        :n_agents => 4,
        :n_samples => 36,
        :phase_steps => (12, 12, 12),
        :communication_radius => 8.25,
        :last_k => 2,
        :navigation_points => 33,
        :evaluation_points => 41,
        :lookahead => 5,
        :planning_iterations => 32,
        :time_budget => 0.0,
        :noise_variance => 0.30,
        :process_variance => 1e-10,
        :consensus_threshold => 1e-7,
        :consensus_timeline => 180,
        :animation_fps => 3,
    )
    :smoke => Dict(
        :seeds => (211, 213),
        :representative_seed => 211,
        :representative_agent_index => 1,
        :n_agents => 4,
        :n_samples => 9,
        :phase_steps => (3, 3, 3),
        :communication_radius => 8.25,
        :last_k => 2,
        :navigation_points => 17,
        :evaluation_points => 21,
        :lookahead => 3,
        :planning_iterations => 24,
        :time_budget => 0.0,
        :noise_variance => 0.30,
        :process_variance => 1e-10,
        :consensus_threshold => 1e-6,
        :consensus_timeline => 120,
        :animation_fps => 2,
    )
    _ => throw(ArgumentError(
        "Use the `full`, `pilot`, or `smoke` publication profile.",
    ))
end

function exploration_starts(settings)
    starts = (
        [-4.0 -4.0],
        [4.0 -4.0],
        [-4.0 4.0],
        [4.0 4.0],
    )
    settings[:n_agents] ≤ length(starts) ||
        throw(ArgumentError("Add initial positions for more than four agents."))
    [copy(starts[index]) for index in 1:settings[:n_agents]]
end

function distributed_phase(step, settings)
    step == 0 && return :prior
    limited_end = settings[:phase_steps][1]
    blackout_end = limited_end + settings[:phase_steps][2]
    step ≤ limited_end ? :limited_communication :
    step ≤ blackout_end ? :communication_blackout :
    :limited_recovery
end

function distance_limited_matching(states, communication_radius)
    ids = sort(collect(keys(states)))
    edges = empty_edges(ids)
    candidates = [
        Dict(
            :distance => norm(states[ids[left]] - states[ids[right]]),
            :left => ids[left],
            :right => ids[right],
        )
        for left in 1:(length(ids) - 1)
        for right in (left + 1):length(ids)
        if norm(states[ids[left]] - states[ids[right]]) ≤
            communication_radius
    ]
    used = Set{String}()
    foreach(sort(candidates; by=item -> item[:distance])) do item
        if item[:left] ∉ used && item[:right] ∉ used
            push!(edges[item[:left]], item[:right])
            push!(edges[item[:right]], item[:left])
            push!(used, item[:left])
            push!(used, item[:right])
        end
    end
    edges
end

function distributed_edges(step, settings, states)
    distributed_phase(step, settings) == :communication_blackout &&
        return empty_edges(sort(collect(keys(states))))
    distance_limited_matching(states, settings[:communication_radius])
end

function exploration_policy(mdp, settings, seed)
    solve(
        RiskBoundedInfoMCTS(
            lookahead=settings[:lookahead],
            time_budget=settings[:time_budget],
            quad_order=1,
            risk_budget=1.0,
            alpha=0.0,
            reference_reward=1.0,
            rng=MersenneTwister(seed),
        ),
        mdp,
    )
end

"""Choose an action from a fixed rollout budget for repeatable comparisons."""
function fixed_budget_action(policy, state, model, planning_iterations)
    set_environment_model!(policy, state, model)
    nodekey = VulcanJ.initialize_nodekey(state)
    policy.tree_nodes[nodekey] = TreeNode(
        visits=0,
        action_values=Dict(),
        action_counts=Dict(),
        actions_tried=Set(),
    )
    foreach(1:planning_iterations) do _
        sample_rollout(
            policy,
            nodekey,
            model,
            0,
            policy.solver.lookahead,
            0.0,
            0.0,
            policy.solver.risk_budget,
        )
    end
    root = policy.tree_nodes[nodekey]
    isempty(root.action_values) ?
        first(actions(policy.mdp, state)) :
        argmax(root.action_values)
end

function backend_models(smodel, information, noise_variance)
    Dict(
        aid => SCRIBEModelState(
            smodel,
            copy(information[aid]),
            noise_variance,
        )
        for aid in sort(collect(keys(information)))
    )
end

function field_summary(model, evaluation_locations, truth)
    moments = posterior_model_moments(
        model.smodel,
        model.information,
        evaluation_locations,
    )
    prediction = moments[:μ]
    variance = max.(diag(moments[:Σ]), 0.0)
    error = prediction - truth
    Dict(
        :prediction => prediction,
        :variance => variance,
        :rmse => sqrt(mean(abs2, error)),
        :coverage => mean(abs.(error) .≤ 1.96 .* sqrt.(variance)),
        :integrated_variance => mean(variance),
    )
end

function exploration_consensus_rmse(summaries)
    maximum(
        sqrt(mean(abs2, left[:prediction] - right[:prediction]))
        for left in values(summaries) for right in values(summaries)
    )
end

function exploration_oracle_gap(summaries, oracle_summary)
    mean(
        sqrt(mean(abs2, summary[:prediction] - oracle_summary[:prediction]))
        for summary in values(summaries)
    )
end

function mean_pairwise_distance(states)
    length(states) < 2 && return 0.0
    distances = [
        norm(states[left] - states[right])
        for left in 1:(length(states) - 1)
        for right in (left + 1):length(states)
    ]
    mean(distances)
end

function unique_sample_fraction(paths, step_size)
    all_paths = collect(values(paths))
    isempty(all_paths) && return 0.0
    sampled_paths = filter(!isempty, all_paths)
    isempty(sampled_paths) && return 0.0
    locations = reduce(
        vcat,
        (reduce(vcat, path) for path in sampled_paths),
    )
    cells = Set(
        (
            round(Int, location[1] / step_size),
            round(Int, location[2] / step_size),
        )
        for location in eachrow(locations)
    )
    length(cells) / size(locations, 1)
end

function exploration_metric_row(
    backend,
    seed,
    step,
    settings,
    models,
    oracle,
    mdp,
    truth,
    states,
    paths,
    cumulative_reward,
    communication,
)
    summaries = Dict(
        aid => field_summary(
            model,
            mdp.evaluation_grid.locations,
            truth,
        )
        for (aid, model) in models
    )
    oracle_summary = field_summary(
        oracle,
        mdp.evaluation_grid.locations,
        truth,
    )
    prior_summary = field_summary(
        mdp.initial_model,
        mdp.evaluation_grid.locations,
        truth,
    )
    available_edges = distributed_edges(step, settings, states)
    components = connected_components(available_edges)
    Dict(
        :backend => backend,
        :seed => seed,
        :step => step,
        :phase => distributed_phase(step, settings),
        :samples => step * settings[:n_agents],
        :mean_agent_rmse => mean(summary[:rmse] for summary in values(summaries)),
        :worst_agent_rmse => maximum(summary[:rmse] for summary in values(summaries)),
        :mean_agent_coverage =>
            mean(summary[:coverage] for summary in values(summaries)),
        :mean_integrated_variance =>
            mean(summary[:integrated_variance] for summary in values(summaries)),
        :prediction_consensus_rmse => exploration_consensus_rmse(summaries),
        :trajectory_oracle_gap => exploration_oracle_gap(
            summaries,
            oracle_summary,
        ),
        :oracle_rmse => oracle_summary[:rmse],
        :exploration_rmse_reduction => 100 * (
            prior_summary[:rmse] - oracle_summary[:rmse]
        ) / prior_summary[:rmse],
        :oracle_integrated_variance => oracle_summary[:integrated_variance],
        :cumulative_expected_information => cumulative_reward,
        :mean_pairwise_distance => mean_pairwise_distance(
            collect(values(states)),
        ),
        :unique_sample_fraction => unique_sample_fraction(
            paths,
            mdp.step_size,
        ),
        :available_communication_links =>
            sum(length, values(available_edges)) ÷ 2,
        :largest_communication_component =>
            maximum(length, components),
        :cumulative_messages => communication[:messages],
        :cumulative_bytes => communication[:bytes],
        :cumulative_consensus_iterations => communication[:iterations],
    )
end

function observe_team!(
    plans,
    paths,
    observations,
    mdp,
    states,
    noise_rngs,
)
    foreach(sort(collect(keys(states)))) do aid
        state = states[aid]
        observation = only(mdp.ground_truth(state)) +
            sqrt(mdp.initial_model.R) * randn(noise_rngs[aid])
        item = Dict(
            :X => copy(state),
            :H => prediction_dynamics(mdp.initial_model.smodel, state),
            :z => [observation],
            :R => reshape([mdp.initial_model.R], 1, 1),
        )
        push!(plans[aid], item)
        push!(paths[aid], copy(state))
        push!(observations[aid], observation)
    end
end

function advance_team!(
    policies,
    models,
    states,
    mdp,
    settings,
    dynamics_rngs,
)
    reward = 0.0
    next_states = Dict{String, Matrix{Float64}}()
    foreach(sort(collect(keys(states)))) do aid
        state = states[aid]
        model = models[aid]
        selected_action = fixed_budget_action(
            policies[aid],
            state,
            model,
            settings[:planning_iterations],
        )
        next_state = gen(
            mdp,
            state,
            selected_action,
            dynamics_rngs[aid],
        ).sp
        reward += expected_information_gain(mdp, model, next_state, 1)
        next_states[aid] = next_state
    end
    next_states, reward
end

function initialize_backend_state(backend, problem, settings, seed)
    ids = agent_ids(settings)
    prior = problem[:mdp].initial_model.information
    information = Dict(aid => copy(prior) for aid in ids)
    states = Dict(
        aid => exploration_starts(settings)[index]
        for (index, aid) in enumerate(ids)
    )
    models = backend_models(
        problem[:mdp].initial_model.smodel,
        information,
        problem[:mdp].initial_model.R,
    )
    ExplorationBackendState(
        ids=ids,
        states=states,
        models=models,
        policies=Dict(
            aid => exploration_policy(
                problem[:mdp],
                settings,
                seed + 10_000 * index,
            )
            for (index, aid) in enumerate(ids)
        ),
        noise_rngs=Dict(
            aid => MersenneTwister(seed + 100 * index)
            for (index, aid) in enumerate(ids)
        ),
        dynamics_rngs=Dict(
            aid => MersenneTwister(seed + 1_000 * index)
            for (index, aid) in enumerate(ids)
        ),
        plans=Dict(aid => Any[] for aid in ids),
        observations=Dict(aid => Float64[] for aid in ids),
        paths=Dict(aid => Matrix{Float64}[] for aid in ids),
        histories=Dict(
            aid => SCRIBEModelState[models[aid]]
            for aid in ids
        ),
        oracle=SCRIBEModelState(
            problem[:mdp].initial_model.smodel,
            copy(prior),
            problem[:mdp].initial_model.R,
        ),
        network=nothing,
        known_observations=Dict(
            aid => Set{Tuple{String, Int}}()
            for aid in ids
        ),
        communication=Dict(:messages => 0, :bytes => 0, :iterations => 0),
        cumulative_reward=0.0,
    )
end

function replay_information(
    prior,
    params,
    plans,
    known_observations,
    current_step,
)
    information = copy(prior)
    foreach(1:current_step) do step
        Y_prior, y_prior = publication_prior(
            information,
            params.A,
            params.w[:Q],
        )
        available = filter(
            observation_id -> last(observation_id) == step,
            collect(known_observations),
        )
        innovations = map(available) do (source, observation_step)
            observation = plans[source][observation_step]
            innovation = measurement_information(
                observation[:H],
                observation[:z],
                observation[:R],
            )
            (innovation[:δI], innovation[:δi])
        end
        δI = isempty(innovations) ?
            zeros(size(Y_prior)) :
            sum(first, innovations)
        δi = isempty(innovations) ?
            zeros(length(y_prior)) :
            sum(last, innovations)
        Y = Y_prior + δI
        information = KFEnvInfo(
            y_prior + δi,
            (Y + Y') / 2,
            δi,
            δI,
        )
    end
    information
end

function observation_record_bytes(observation)
    sizeof(Float64) * (
        length(observation[:X]) +
        length(observation[:z]) +
        length(observation[:R])
    )
end

function last_k_observation_step!(state, problem, settings, step)
    foreach(state.ids) do aid
        push!(state.known_observations[aid], (aid, step))
    end
    edges = distributed_edges(step, settings, state.states)
    messages = 0
    bytes = 0
    transfers = Dict(
        aid => Set{Tuple{String, Int}}()
        for aid in state.ids
    )
    foreach(state.ids) do sender
        foreach(edges[sender]) do receiver
            missing = sort(
                collect(
                    setdiff(
                        state.known_observations[sender],
                        state.known_observations[receiver],
                    ),
                );
                by=observation_id -> (last(observation_id), first(observation_id)),
                rev=true,
            )
            selected = missing[1:min(settings[:last_k], length(missing))]
            if !isempty(selected)
                messages += 1
                union!(transfers[receiver], selected)
                bytes += sum(selected) do (source, observation_step)
                    observation_record_bytes(
                        state.plans[source][observation_step],
                    )
                end
            end
        end
    end
    foreach(state.ids) do aid
        union!(state.known_observations[aid], transfers[aid])
    end
    params = problem[:mdp].initial_model.smodel.params
    state.models = Dict(
        aid => SCRIBEModelState(
            state.oracle.smodel,
            replay_information(
                problem[:mdp].initial_model.information,
                params,
                state.plans,
                state.known_observations[aid],
                step,
            ),
            state.oracle.R,
        )
        for aid in state.ids
    )
    Dict(
        :messages => messages,
        :bytes => bytes,
        :iterations => 1,
    )
end

function update_backend_information!(
    backend,
    state,
    problem,
    settings,
    step,
)
    params = problem[:mdp].initial_model.smodel.params
    state.oracle = SCRIBEModelState(
        state.oracle.smodel,
        centralized_update(
            state.oracle.information,
            state.plans,
            step,
            params,
        ),
        state.oracle.R,
    )
    if backend == :centralized
        state.models = Dict(
            aid => SCRIBEModelState(
                state.oracle.smodel,
                copy(state.oracle.information),
                state.oracle.R,
            )
            for aid in state.ids
        )
    elseif backend == :last_k_observations
        step_communication = last_k_observation_step!(
            state,
            problem,
            settings,
            step,
        )
        state.communication = Dict(
            :messages => state.communication[:messages] +
                step_communication[:messages],
            :bytes => state.communication[:bytes] + step_communication[:bytes],
            :iterations => state.communication[:iterations] +
                step_communication[:iterations],
        )
    elseif backend == :independent
        state.models = Dict(
            aid => begin
                estimator = PublicationEstimators(
                    PublicationScribe(
                        step,
                        [state.models[aid].information],
                    ),
                    params.A,
                    params.w[:Q],
                    [state.plans[aid][step]],
                )
                SCRIBEModelState(
                    state.models[aid].smodel,
                    information_filter_update(estimator, 1),
                    state.models[aid].R,
                )
            end
            for aid in state.ids
        )
    elseif backend in (:scribe, :kf_only, :ci_only)
        if isnothing(state.network)
            state.network = initialize_publication_network(
                params,
                problem[:mdp].initial_model.information,
                state.plans,
                settings,
            )
        end
        fusion = backend == :scribe ?
            SCRIBEFusion() :
            backend == :kf_only ?
                KalmanFilterOnlyFusion() :
                CovarianceIntersectionOnlyFusion()
        step_communication = fusion_step!(
            fusion,
            state.network,
            step,
            distributed_edges(step, settings, state.states),
            settings,
        )
        state.communication = Dict(
            :messages => state.communication[:messages] +
                step_communication[:messages],
            :bytes => state.communication[:bytes] + step_communication[:bytes],
            :iterations => state.communication[:iterations] +
                step_communication[:iterations],
        )
        current = Dict(
            aid => state.network.vertices[aid].agent.information[end]
            for aid in state.ids
        )
        state.models = backend_models(
            state.oracle.smodel,
            current,
            state.oracle.R,
        )
    else
        throw(ArgumentError("Unknown exploration backend $backend."))
    end
end

function run_distributed_exploration_backend(
    backend,
    problem,
    settings,
    seed;
    keep_history=false,
)
    state = initialize_backend_state(backend, problem, settings, seed)
    truth = problem[:mdp].ground_truth(problem[:mdp].evaluation_grid.locations)
    rows = Any[
        exploration_metric_row(
            backend,
            seed,
            0,
            settings,
            state.models,
            state.oracle,
            problem[:mdp],
            truth,
            state.states,
            state.paths,
            state.cumulative_reward,
            state.communication,
        ),
    ]

    foreach(1:settings[:n_samples]) do step
        if step > 1
            state.states, reward = advance_team!(
                state.policies,
                state.models,
                state.states,
                problem[:mdp],
                settings,
                state.dynamics_rngs,
            )
            state.cumulative_reward += reward
        else
            state.cumulative_reward += sum(
                expected_information_gain(
                    problem[:mdp],
                    state.models[aid],
                    state.states[aid],
                    1,
                )
                for aid in state.ids
            )
        end
        observe_team!(
            state.plans,
            state.paths,
            state.observations,
            problem[:mdp],
            state.states,
            state.noise_rngs,
        )
        update_backend_information!(
            backend,
            state,
            problem,
            settings,
            step,
        )
        if keep_history
            foreach(state.ids) do aid
                push!(state.histories[aid], state.models[aid])
            end
        end
        push!(
            rows,
            exploration_metric_row(
                backend,
                seed,
                step,
                settings,
                state.models,
                state.oracle,
                problem[:mdp],
                truth,
                state.states,
                state.paths,
                state.cumulative_reward,
                state.communication,
            ),
        )
    end
    Dict(
        :rows => rows,
        :models => state.models,
        :histories => state.histories,
        :paths => state.paths,
        :observations => state.observations,
        :truth => truth,
        :grid => problem[:mdp].evaluation_grid,
    )
end

function write_exploration_rows(rows, output_path)
    fields = keys(first(rows))
    open(output_path, "w") do io
        println(io, join(string.(fields), ","))
        foreach(rows) do row
            println(
                io,
                join(
                    (string(row[field]) for field in fields),
                    ",",
                ),
            )
        end
    end
end

function read_exploration_rows(input_path)
    lines = readlines(input_path)
    fields = Symbol.(split(first(lines), ","))
    integer_fields = Set((
        :seed,
        :step,
        :samples,
        :available_communication_links,
        :largest_communication_component,
        :cumulative_messages,
        :cumulative_bytes,
        :cumulative_consensus_iterations,
    ))
    symbol_fields = Set((:backend, :phase))
    map(Iterators.drop(lines, 1)) do line
        raw_values = split(line, ",")
        values = map(zip(fields, raw_values)) do (field, raw_value)
            field in symbol_fields ? Symbol(raw_value) :
            field in integer_fields ? parse(Int, raw_value) :
            parse(Float64, raw_value)
        end
        Dict(zip(fields, values))
    end
end

function exploration_aggregate(rows, backend, metric)
    selected = filter(row -> row[:backend] == backend, rows)
    steps = sort(unique(getindex.(selected, :step)))
    values = [
        getindex.(
            filter(row -> row[:step] == step, selected),
            metric,
        )
        for step in steps
    ]
    Dict(
        :steps => steps,
        :mean => mean.(values),
        :deviation => map(values) do item
            std(item; corrected=false)
        end,
    )
end

function relative_exploration_aggregate(
    rows,
    backend,
    metric,
    reference_backend,
)
    reference = Dict(
        (row[:seed], row[:step]) => row[metric]
        for row in rows
        if row[:backend] == reference_backend
    )
    selected = filter(row -> row[:backend] == backend, rows)
    steps = sort(unique(getindex.(selected, :step)))
    values = [
        [
            row[metric] / reference[(row[:seed], row[:step])]
            for row in selected
            if row[:step] == step
        ]
        for step in steps
    ]
    Dict(
        :steps => steps,
        :mean => mean.(values),
        :deviation => map(values) do item
            std(item; corrected=false)
        end,
    )
end

function exploration_curve(
    rows,
    metric,
    title,
    ylabel;
    logscale=false,
    axis_logscale=false,
    value_scale=1.0,
    relative_to=nothing,
    ylimits=nothing,
    backends=EXPLORATION_BACKENDS,
)
    plot_object = plot(
        ;
        title,
        xlabel="Sampling Round",
        ylabel,
        yscale=axis_logscale ? :log10 : :identity,
    )
    !isnothing(ylimits) && plot!(plot_object; ylims=ylimits)
    foreach(backends) do backend
        aggregate = isnothing(relative_to) ?
            exploration_aggregate(rows, backend, metric) :
            relative_exploration_aggregate(
                rows,
                backend,
                metric,
                relative_to,
            )
        values = logscale ?
            log10.(max.(aggregate[:mean], eps())) :
            value_scale .* aggregate[:mean]
        deviations = logscale ?
            zeros(length(values)) :
            value_scale .* aggregate[:deviation]
        plot!(
            plot_object,
            aggregate[:steps],
            values;
            ribbon=deviations,
            color=EXPLORATION_COLORS[backend],
            label=EXPLORATION_LABELS[backend],
            linewidth=3.2,
            fillalpha=0.12,
        )
    end
    plot_object
end

function exploration_phase_boundaries!(plot_object, settings)
    blackout_start = settings[:phase_steps][1]
    blackout_end = blackout_start + settings[:phase_steps][2]
    vspan!(
        plot_object,
        [blackout_start, blackout_end];
        color=:gray,
        fillalpha=0.06,
        label=false,
    )
    vline!(
        plot_object,
        [blackout_start, blackout_end];
        color=:gray,
        linestyle=:dash,
        label=false,
    )
    plot_object
end

function save_distributed_exploration_figure(rows, settings, output_dir)
    panels = (
        exploration_curve(
            rows,
            :mean_agent_rmse,
            "Per-Agent Field Reconstruction Error",
            "RMSE [normalized field units]",
        ),
        exploration_curve(
            rows,
            :mean_integrated_variance,
            "Mean Posterior Field Variance",
            "Variance [normalized field units²]",
        ),
        exploration_curve(
            rows,
            :prediction_consensus_rmse,
            "Inter-Agent Model Disagreement",
            "log₁₀(RMSE [normalized field units])";
            logscale=true,
        ),
        exploration_curve(
            rows,
            :trajectory_oracle_gap,
            "Gap to the Pooled-Observation Posterior",
            "log₁₀(RMSE [normalized field units])";
            logscale=true,
        ),
        exploration_curve(
            rows,
            :mean_pairwise_distance,
            "Mean Pairwise Robot Separation",
            "Distance [normalized distance units]",
        ),
        exploration_curve(
            rows,
            :oracle_rmse,
            "Pooled-Data Reconstruction Error",
            "RMSE [normalized field units]",
        ),
    )
    foreach(panels) do panel
        exploration_phase_boundaries!(panel, settings)
    end
    figure = plot(
        panels...;
        layout=(2, 3),
        size=(1350, 800),
        margin=2 * Plots.mm,
    )
    output_path = joinpath(output_dir, "aggregate_metrics.png")
    savefig(figure, output_path)
    println("Plot saved at $output_path")
    output_path
end

function path_panel(result, title)
    grid = result[:grid]
    field_extent = maximum(abs, result[:truth])
    plot_object = heatmap(
        grid.x,
        grid.y,
        reshape(result[:truth], length(grid.x), length(grid.y))';
        color=:balance,
        clims=(-field_extent, field_extent),
        aspect_ratio=:equal,
        title,
        xlabel="x [distance units]",
        ylabel="y [distance units]",
        colorbar=false,
    )
    colors = (:dodgerblue, :darkorange, :limegreen, :magenta)
    foreach(enumerate(sort(collect(keys(result[:paths]))))) do (index, aid)
        path = reduce(vcat, result[:paths][aid])
    plot!(
        plot_object,
        path[:, 1],
            path[:, 2];
            color=colors[index],
            linewidth=2,
            marker=:circle,
            markersize=2,
            label=aid,
        )
    end
    plot_object
end

function posterior_panel(result, aid, frame_index, title)
    model = result[:histories][aid][frame_index]
    moments = posterior_model_moments(
        model.smodel,
        model.information,
        result[:grid].locations,
    )
    heatmap(
        result[:grid].x,
        result[:grid].y,
        reshape(moments[:μ], length(result[:grid].x), length(result[:grid].y))';
        color=:balance,
        clims=(-maximum(abs, result[:truth]), maximum(abs, result[:truth])),
        aspect_ratio=:equal,
        title,
        xlabel="x [km]",
        ylabel="y [km]",
        colorbar=false,
    )
end

function truth_panel(result)
    heatmap(
        result[:grid].x,
        result[:grid].y,
        reshape(
            result[:truth],
            length(result[:grid].x),
            length(result[:grid].y),
        )';
        color=:balance,
        clims=(-maximum(abs, result[:truth]), maximum(abs, result[:truth])),
        aspect_ratio=:equal,
        title="Ground Truth",
        xlabel="x [km]",
        ylabel="y [km]",
        xlims=(-5.4, 5.4),
        ylims=(-5.4, 5.4),
        colorbar=false,
    )
end

function robot_endpoint_marker!(plot_object, path, color)
    endpoint = vec(path[end, :])
    heading = vec(path[end, :] - path[end - 1, :])
    heading ./= max(norm(heading), eps())
    perpendicular = [-heading[2], heading[1]]
    local_vertices = (
        (-0.18, -0.14),
        (0.10, -0.14),
        (0.25, 0.00),
        (0.10, 0.14),
        (-0.18, 0.14),
    )
    vertices = map(local_vertices) do (forward, lateral)
        endpoint + forward .* heading + lateral .* perpendicular
    end
    plot!(
        plot_object,
        Shape(collect(first.(vertices)), collect(last.(vertices)));
        fillcolor=color,
        fillalpha=1.0,
        linecolor=:black,
        linewidth=0.8,
        label=false,
    )
    wheel_center = endpoint - 0.11 .* heading
    wheels = (
        wheel_center + 0.16 .* perpendicular,
        wheel_center - 0.16 .* perpendicular,
    )
    scatter!(
        plot_object,
        collect(first.(wheels)),
        collect(last.(wheels));
        color=:black,
        markersize=2.0,
        markerstrokewidth=0,
        label=false,
    )
end

function trajectory_reconstruction_panel(result, aid, frame_index)
    plot_object = posterior_panel(
        result,
        aid,
        frame_index,
        "Final SCRIBE Reconstruction\nand Robot Trajectories",
    )
    colors = (:dodgerblue, :darkorange, :limegreen, :magenta)
    foreach(enumerate(sort(collect(keys(result[:paths]))))) do (index, path_id)
        path = reduce(vcat, result[:paths][path_id])
        plot!(
            plot_object,
            path[:, 1],
            path[:, 2];
            color=colors[index],
            linewidth=2,
            linestyle=:dashdot,
            label="Robot $index",
        )
        scatter!(
            plot_object,
            [path[1, 1]],
            [path[1, 2]];
            color=colors[index],
            marker=:circle,
            markersize=6,
            markerstrokecolor=:black,
            markerstrokewidth=0.8,
            label=false,
        )
        robot_endpoint_marker!(plot_object, path, colors[index])
    end
    plot!(
        plot_object;
        legend=:bottomleft,
        xlims=(-5.4, 5.4),
        ylims=(-5.4, 5.4),
    )
    plot_object
end

function save_field_reconstruction_figure(results, settings, output_dir)
    scribe = results[:scribe]
    agent_ids = sort(collect(keys(scribe.histories)))
    aid = agent_ids[settings[:representative_agent_index]]
    final_frame = settings[:n_samples] + 1
    figure = plot(
        truth_panel(scribe),
        trajectory_reconstruction_panel(
            scribe,
            aid,
            final_frame,
        );
        layout=(1, 2),
        size=(1080, 500),
        left_margin=4 * Plots.mm,
        right_margin=2 * Plots.mm,
        top_margin=2 * Plots.mm,
        bottom_margin=3 * Plots.mm,
        titlefontsize=15,
        guidefontsize=13,
        tickfontsize=11,
        legendfontsize=9,
    )
    output_path = joinpath(output_dir, "field_reconstruction.png")
    savefig(figure, output_path)
    println("Plot saved at $output_path")
    output_path
end

function save_distributed_performance_figure(rows, settings, output_dir)
    comparison_backends = EXPLORATION_BACKENDS
    panels = (
        exploration_curve(
            rows,
            :mean_agent_rmse,
            "Online Field Reconstruction\nunder Limited Communication",
            "Average Per-Agent\nReconstruction RMSE";
            backends=comparison_backends,
        ),
        exploration_curve(
            rows,
            :prediction_consensus_rmse,
            "Inter-Agent Model Disagreement\nacross Communication Loss",
            "Inter-Agent RMSE";
            ylimits=(0.0, 1.85),
            backends=(
                :scribe,
                :kf_only,
                :ci_only,
                :last_k_observations,
                :independent,
            ),
        ),
        exploration_curve(
            rows,
            :oracle_rmse,
            "Reconstruction from the Team's\nCollected Samples",
            "Team Pooled-Data RMSE";
            ylimits=(0.35, 0.85),
            backends=comparison_backends,
        ),
    )
    annotate!(
        panels[2],
        10.0,
        1.76,
        text("No-communication peak: 2.62", 9, :darkorange),
    )
    foreach(panels) do panel
        exploration_phase_boundaries!(panel, settings)
        plot!(
            panel;
            legend=false,
            left_margin=9 * Plots.mm,
            right_margin=2 * Plots.mm,
            bottom_margin=3 * Plots.mm,
        )
    end
    legend_panel = plot(
        ;
        framestyle=:none,
        showaxis=false,
        grid=false,
        xlims=(0.0, 1.0),
        ylims=(0.0, 1.0),
        legend=:top,
        legend_column=3,
    )
    foreach(comparison_backends) do backend
        plot!(
            legend_panel,
            [NaN],
            [NaN];
            color=EXPLORATION_COLORS[backend],
            label=EXPLORATION_LABELS[backend],
            linewidth=3.2,
        )
    end
    figure = plot(
        panels...,
        legend_panel;
        layout=@layout([grid(1, 3); legend{0.18h}]),
        size=(1560, 540),
        margin=2 * Plots.mm,
        titlefontsize=17,
        guidefontsize=14,
        tickfontsize=12,
        legendfontsize=11,
    )
    output_path = joinpath(output_dir, "distributed_performance.png")
    savefig(figure, output_path)
    println("Plot saved at $output_path")
    output_path
end

function save_representative_figure(results, settings, output_dir)
    scribe = results[:scribe]
    independent = results[:independent]
    frame_indices = (
        settings[:phase_steps][1] + 1,
        sum(settings[:phase_steps][1:2]) + 1,
        settings[:n_samples] + 1,
    )
    aid = first(sort(collect(keys(scribe.histories))))
    figure = plot(
        path_panel(scribe, "SCRIBE paths"),
        path_panel(independent, "Isolated paths"),
        posterior_panel(
            scribe,
            aid,
            frame_indices[1],
            "SCRIBE Agent 1: End of Limited Communication",
        ),
        posterior_panel(
            scribe,
            aid,
            frame_indices[2],
            "SCRIBE Agent 1: End of Communication Blackout",
        ),
        posterior_panel(
            scribe,
            aid,
            frame_indices[3],
            "SCRIBE Agent 1: End of Limited Recovery",
        ),
        posterior_panel(
            independent,
            aid,
            frame_indices[3],
            "No-Communication Agent 1: Final",
        );
        layout=(2, 3),
        size=(1250, 780),
        margin=1.5 * Plots.mm,
    )
    output_path = joinpath(output_dir, "representative_run.png")
    savefig(figure, output_path)
    println("Plot saved at $output_path")
    output_path
end

function save_representative_animation(result, settings, output_dir)
    ids = sort(collect(keys(result[:histories])))
    colors = (:dodgerblue, :darkorange, :limegreen, :magenta)
    grid = result[:grid]
    field_extent = maximum(abs, result[:truth])
    animation = @animate for frame_index in 1:(settings[:n_samples] + 1)
        path_plot = heatmap(
            grid.x,
            grid.y,
            reshape(result[:truth], length(grid.x), length(grid.y))';
            color=:balance,
            clims=(-field_extent, field_extent),
            aspect_ratio=:equal,
            title="Ground Truth and Robot Trajectories",
            xlabel="x [distance units]",
            ylabel="y [distance units]",
            colorbar=false,
        )
        foreach(enumerate(ids)) do (index, aid)
            n_path = min(frame_index - 1, length(result[:paths][aid]))
            if n_path > 0
                path = reduce(vcat, result[:paths][aid][1:n_path])
                plot!(
                    path_plot,
                    path[:, 1],
                    path[:, 2];
                    color=colors[index],
                    linewidth=2,
                    marker=:circle,
                    markersize=2,
                    label=aid,
                )
            end
        end
        posterior_plots = [
            posterior_panel(
                result,
                aid,
                frame_index,
                "$(aid), $(distributed_phase(frame_index - 1, settings))",
            )
            for aid in ids
        ]
        plot(
            path_plot,
            posterior_plots...;
            layout=(2, 3),
            size=(1150, 720),
            margin=1 * Plots.mm,
        )
    end
    output_path = joinpath(output_dir, "scribe_distributed_exploration.gif")
    gif(animation, output_path; fps=settings[:animation_fps])
    println("Animation saved at $output_path")
    output_path
end

function final_exploration_summary(rows)
    final_step = maximum(getindex.(rows, :step))
    map(EXPLORATION_BACKENDS) do backend
        selected = filter(
            row -> row[:backend] == backend && row[:step] == final_step,
            rows,
        )
        Dict(
            :backend => backend,
            :rmse => mean(getindex.(selected, :mean_agent_rmse)),
            :rmse_std => std(
                getindex.(selected, :mean_agent_rmse);
                corrected=false,
            ),
            :uncertainty => mean(
                getindex.(selected, :mean_integrated_variance),
            ),
            :consensus => mean(
                getindex.(selected, :prediction_consensus_rmse),
            ),
            :oracle_gap => mean(
                getindex.(selected, :trajectory_oracle_gap),
            ),
            :unique_fraction => mean(
                getindex.(selected, :unique_sample_fraction),
            ),
            :oracle_rmse => mean(getindex.(selected, :oracle_rmse)),
            :exploration_reduction => mean(
                getindex.(selected, :exploration_rmse_reduction),
            ),
            :oracle_variance => mean(
                getindex.(selected, :oracle_integrated_variance),
            ),
            :pairwise_distance => mean(
                getindex.(selected, :mean_pairwise_distance),
            ),
            :coverage => mean(
                getindex.(selected, :mean_agent_coverage),
            ),
            :bytes => mean(getindex.(selected, :cumulative_bytes)),
        )
    end
end

function write_exploration_summary(rows, output_path)
    summary = final_exploration_summary(rows)
    open(output_path, "w") do io
        println(
            io,
            "backend,rmse_mean,rmse_std,integrated_variance_mean," *
            "consensus_rmse_mean,trajectory_oracle_gap_mean," *
            "unique_sample_fraction_mean,oracle_rmse_mean," *
            "exploration_rmse_reduction_mean," *
            "oracle_integrated_variance_mean," *
            "mean_pairwise_distance,coverage_mean,cumulative_bytes_mean",
        )
        foreach(summary) do row
            println(
                io,
                "$(row[:backend]),$(row[:rmse]),$(row[:rmse_std])," *
                "$(row[:uncertainty]),$(row[:consensus])," *
                "$(row[:oracle_gap]),$(row[:unique_fraction])," *
                "$(row[:oracle_rmse]),$(row[:exploration_reduction])," *
                "$(row[:oracle_variance]),$(row[:pairwise_distance])," *
                "$(row[:coverage]),$(row[:bytes])",
            )
        end
    end
end

function print_distributed_exploration_summary(rows, output_dir)
    println("Distributed informative exploration complete.")
    foreach(final_exploration_summary(rows)) do row
        println(
            "  $(row[:backend]): RMSE=" *
            "$(round(row[:rmse]; digits=4))±" *
            "$(round(row[:rmse_std]; digits=4)), variance=" *
            "$(round(row[:uncertainty]; digits=4)), oracle gap=" *
            "$(round(row[:oracle_gap]; digits=4)), exploration reduction=" *
            "$(round(row[:exploration_reduction]; digits=1))%",
        )
    end
    println("  results: $output_dir")
end

function distributed_exploration_main(
    profile=:full;
    animations=profile == :full,
)
    settings = distributed_exploration_settings(profile)
    output_dir = joinpath(
        @__DIR__,
        "res",
        "experiment_4",
        String(profile),
    )
    mkpath(output_dir)
    rows = Any[]
    representative = Dict{Symbol, Any}()
    foreach(settings[:seeds]) do seed
        problem = exploration_problem(settings)
        foreach(EXPLORATION_BACKENDS) do backend
            keep_history = seed == settings[:representative_seed]
            result = run_distributed_exploration_backend(
                backend,
                problem,
                settings,
                seed;
                keep_history,
            )
            append!(rows, result[:rows])
            keep_history && (representative[backend] = result)
            GC.gc()
        end
        println("Completed exploration seed $seed.")
    end
    write_exploration_rows(
        rows,
        joinpath(output_dir, "metric_history.csv"),
    )
    write_exploration_summary(
        rows,
        joinpath(output_dir, "final_summary.csv"),
    )
    save_distributed_exploration_figure(rows, settings, output_dir)
    save_field_reconstruction_figure(representative, settings, output_dir)
    save_distributed_performance_figure(rows, settings, output_dir)
    save_representative_figure(representative, settings, output_dir)
    animations &&
        save_representative_animation(
            representative[:scribe],
            settings,
            output_dir,
        )
    print_distributed_exploration_summary(rows, output_dir)
    Dict(:rows => rows, :representative => representative)
end

if abspath(PROGRAM_FILE) == @__FILE__
    profile = isempty(ARGS) ? :full : Symbol(first(ARGS))
    animations = length(ARGS) > 1 ?
        ARGS[2] == "animations" :
        profile == :full
    distributed_exploration_main(profile; animations)
end
