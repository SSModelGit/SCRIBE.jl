ENV["GKSwstype"] = "100"

using Distributions: Normal
using LinearAlgebra
using Match: @match
using Plots
using POMDPs
using Random
using SCRIBE
using Serialization
using Statistics
using VulcanJ

include(joinpath(@__DIR__, "asynchronous_model_fusion.jl"))

const SCALING_CHECKPOINT_VERSION = 1
const SCALING_DIRECTIONS = Dict(
    :north => CartesianIndex(1, 0),
    :northeast => CartesianIndex(1, 1),
    :east => CartesianIndex(0, 1),
    :southeast => CartesianIndex(-1, 1),
    :south => CartesianIndex(-1, 0),
    :southwest => CartesianIndex(-1, -1),
    :west => CartesianIndex(0, -1),
    :northwest => CartesianIndex(1, -1),
)

struct ROMSExplorationState
    row
    history
end

struct ROMSExplorationProblem <: MDP{ROMSExplorationState, Symbol}
    sensor
    neighbors
    locations
    grid_locations
    modes
    climatology
    measurement_variance
    variance_gram
    planning_rows
    initial_model
    steps
end

POMDPs.statetype(::ROMSExplorationProblem) = ROMSExplorationState
POMDPs.actiontype(::ROMSExplorationProblem) = Symbol
POMDPs.actions(problem::ROMSExplorationProblem, state) =
    keys(problem.neighbors[state.row])
POMDPs.discount(::ROMSExplorationProblem) = 1.0
POMDPs.isterminal(problem::ROMSExplorationProblem, state) =
    length(state.history) ≥ problem.steps

function POMDPs.gen(problem::ROMSExplorationProblem, state, action, rng)
    row = problem.neighbors[state.row][action]
    location = permutedims(problem.locations[row, :])
    observation = problem.sensor(row, rng)
    next_history = [
        state.history;
        (location=location, observation=observation)
    ]
    (sp=ROMSExplorationState(row, next_history), r=0.0)
end

VulcanJ.extract_location(state::ROMSExplorationState) =
    state.history[end].location
VulcanJ.state_time(::ROMSExplorationProblem, state::ROMSExplorationState) =
    length(state.history)
VulcanJ.observation_history(
    ::VulcanJ.AbstractInfoMCTS,
    ::ROMSExplorationProblem,
    state::ROMSExplorationState,
) = state.history
VulcanJ.get_failure_prob(::ROMSExplorationProblem, _, _) = 0.0
VulcanJ.horizon(problem::ROMSExplorationProblem) = problem.steps
VulcanJ.cellsites(problem::ROMSExplorationProblem) = [
    permutedims(problem.locations[row, :])
    for row in problem.planning_rows
]
VulcanJ.initial_environment_model(
    problem::ROMSExplorationProblem,
    _,
) = problem.initial_model

function coefficient_moments_at(model)
    posterior_coefficient_moments(model.information)
end

function state_measurement_moments(problem, model, state)
    let coefficients=coefficient_moments_at(model),
        H=reshape(problem.modes[state.row, :], 1, :),
        μ=problem.climatology[state.row] + only(H * coefficients[:μ]),
        σ²=only(H * coefficients[:Σ] * H') +
            problem.measurement_variance[state.row]
        (μ=μ, σ²=max(σ², eps(Float64)))
    end
end

function VulcanJ.conditional_observation_distribution(
    problem::ROMSExplorationProblem,
    model::SCRIBEModelState,
    state::ROMSExplorationState,
)
    let moments=state_measurement_moments(problem, model, state)
        Normal(moments.μ, sqrt(moments.σ²))
    end
end

function condition_scaling_model(problem, model, state, observation)
    let H=reshape(problem.modes[state.row, :], 1, :),
        innovation=measurement_information(
            H,
            [observation - problem.climatology[state.row]],
            reshape([problem.measurement_variance[state.row]], 1, 1),
        ),
        Y=model.information.Y + innovation[:δI],
        information=KFEnvInfo(
            model.information.y + innovation[:δi],
            (Y + Y') / 2,
            innovation[:δi],
            innovation[:δI],
        )
        SCRIBEModelState(model.smodel, information, model.R)
    end
end

function VulcanJ.condition_environment_model(
    problem::ROMSExplorationProblem,
    model::SCRIBEModelState,
    state::ROMSExplorationState,
    observation,
)
    condition_scaling_model(problem, model, state, observation)
end

function field_variance(problem, information)
    tr(problem.variance_gram * recover_covariance_from_info(information))
end

function VulcanJ.information_gain(
    ::Val{:variance_reduction},
    problem::ROMSExplorationProblem,
    prior::SCRIBEModelState,
    posterior::SCRIBEModelState,
    _,
    _,
)
    max(
        field_variance(problem, prior.information) -
        field_variance(problem, posterior.information),
        0.0,
    )
end

function VulcanJ.expected_information_gain(
    ::Val{:variance_reduction},
    problem::ROMSExplorationProblem,
    model::SCRIBEModelState,
    state::ROMSExplorationState,
    _,
)
    let coefficients=coefficient_moments_at(model),
        H=reshape(problem.modes[state.row, :], 1, :),
        h=vec(H),
        cross=coefficients[:Σ] * h,
        innovation_variance=dot(h, cross) +
            problem.measurement_variance[state.row]
        dot(cross, problem.variance_gram * cross) / innovation_variance
    end
end

function predictive_scaling_problem(problem, model)
    ROMSExplorationProblem(
        (row, rng) -> let state=ROMSExplorationState(
                row,
                [(location=permutedims(problem.locations[row, :]),
                  observation=0.0)],
            )
            rand(
                rng,
                conditional_observation_distribution(problem, model, state),
            )
        end,
        problem.neighbors,
        problem.locations,
        problem.grid_locations,
        problem.modes,
        problem.climatology,
        problem.measurement_variance,
        problem.variance_gram,
        problem.planning_rows,
        problem.initial_model,
        problem.steps,
    )
end

function VulcanJ.generative_problem(
    problem::ROMSExplorationProblem,
    model::SCRIBEModelState,
    _,
)
    predictive_scaling_problem(problem, model)
end

agent_scaling_settings(profile) = @match profile begin
    :full => Dict(
        :seeds => Tuple(801:2:815),
        :team_sizes => Tuple(2:10),
        :topologies => (:sparse, :moderate, :dense),
        :reliabilities => (:stable, :intermittent),
        :n_samples => 48,
        :lookahead => 4,
        :planning_iterations => 32,
        :planning_stride => 8,
        :noise_variance => 1e-4,
        :link_success_probability => 0.5,
        :consensus_threshold => 1e-7,
        :consensus_timeline => 1000,
    )
    :pilot => Dict(
        :seeds => (801, 803, 805),
        :team_sizes => (2, 6, 10),
        :topologies => (:sparse, :moderate, :dense),
        :reliabilities => (:stable, :intermittent),
        :n_samples => 16,
        :lookahead => 3,
        :planning_iterations => 8,
        :planning_stride => 12,
        :noise_variance => 1e-4,
        :link_success_probability => 0.5,
        :consensus_threshold => 1e-6,
        :consensus_timeline => 120,
    )
    :smoke => Dict(
        :seeds => (801, 803),
        :team_sizes => (2, 3, 4, 8),
        :topologies => (:sparse, :moderate, :dense),
        :reliabilities => (:stable, :intermittent),
        :n_samples => 4,
        :lookahead => 2,
        :planning_iterations => 3,
        :planning_stride => 20,
        :noise_variance => 1e-4,
        :link_success_probability => 0.5,
        :consensus_threshold => 1e-5,
        :consensus_timeline => 80,
    )
    _ => throw(ArgumentError(
        "Use the `full`, `pilot`, or `smoke` agent-scaling profile.",
    ))
end

function navigation_graph(roms)
    grid_rows = wet_row_grid(roms)
    cells = findall(reshape(roms[:wet_mask], roms[:grid_shape]...))
    Dict(
        row => Dict(
            action => grid_rows[cells[row] + direction]
            for (action, direction) in SCALING_DIRECTIONS
            if checkbounds(Bool, grid_rows, cells[row] + direction) &&
                !iszero(grid_rows[cells[row] + direction])
        )
        for row in axes(roms[:locations], 1)
    )
end

function scaling_problem(environment, settings)
    let model=environment[:model],
        roms=environment[:roms],
        modes=eof_modes(model),
        planning_rows=collect(1:settings[:planning_stride]:size(modes, 1)),
        Hₑ=modes[planning_rows, :],
        measurement_variance=settings[:noise_variance] .+
            eof_residual_variance(model),
        prior=SCRIBEModelState(
            model,
            SCRIBE.init_agent_info(environment[:params]),
            settings[:noise_variance],
        )
        ROMSExplorationProblem(
            (row, rng) -> environment[:truth][row] +
                sqrt(settings[:noise_variance]) * randn(rng),
            navigation_graph(roms),
            roms[:locations],
            wet_grid_locations(roms),
            modes,
            eof_mean(model),
            measurement_variance,
            Hₑ' * Hₑ / length(planning_rows),
            planning_rows,
            prior,
            settings[:n_samples],
        )
    end
end

function scaling_policy(problem, settings, seed)
    solve(
        RiskBoundedInfoMCTS(
            lookahead=settings[:lookahead],
            time_budget=0.0,
            risk_budget=Inf,
            alpha_schedule=VulcanJ.fixed_alpha,
            reference_reward=1.0,
            rng=MersenneTwister(seed),
        ),
        problem;
        objective=Val(:variance_reduction),
    )
end

function fixed_rollout_action(policy, state, model, remaining_steps, iterations)
    set_environment_model!(
        policy,
        state,
        model;
        remaining_steps,
    )
    policy.root = VulcanJ.initialize_node(
        policy,
        state,
        model,
        0,
        0.0,
        0.0,
    )
    foreach(1:iterations) do _
        VulcanJ.sample_rollout(policy, policy.root, Inf)
    end
    VulcanJ.cleanup!(policy, policy.root)
    isnothing(policy.root.best) ?
        first(actions(policy.problem, state)) :
        policy.root.branches[policy.root.best].action
end

function farthest_starting_rows(problem, n_agents, seed)
    candidates = collect(keys(problem.neighbors))
    rng = MersenneTwister(seed + 70_000)
    selected = [rand(rng, candidates)]
    while length(selected) < n_agents
        next_row = argmax(candidates) do row
            minimum(
                sum(abs2, problem.grid_locations[row, :] -
                    problem.grid_locations[chosen, :])
                for chosen in selected
            )
        end
        push!(selected, next_row)
        filter!(!=(next_row), candidates)
    end
    selected
end

function add_network_edge!(edges, left, right)
    left == right && return edges
    right ∉ edges[left] && push!(edges[left], right)
    left ∉ edges[right] && push!(edges[right], left)
    edges
end

function ring_edges(ids, distance)
    edges = empty_edges(ids)
    n_agents = length(ids)
    foreach(eachindex(ids)) do index
        foreach(1:min(distance, n_agents - 1)) do offset
            neighbor = mod1(index + offset, n_agents)
            add_network_edge!(edges, ids[index], ids[neighbor])
        end
    end
    foreach(sort!, values(edges))
    edges
end

function topology_edges(ids, topology)
    @match topology begin
        :sparse => ring_edges(ids, 1)
        :moderate => ring_edges(ids, min(2, length(ids) - 1))
        :dense => Dict(aid => filter(!=(aid), ids) for aid in ids)
    end
end

function network_pairs(edges)
    sort([
        (left, right)
        for left in sort(collect(keys(edges)))
        for right in edges[left]
        if agent_number(left) < agent_number(right)
    ])
end

function link_draws(ids, rng)
    Dict(
        (ids[left], ids[right]) => rand(rng)
        for left in 1:(length(ids) - 1)
        for right in (left + 1):length(ids)
    )
end

function reliable_edges(base_edges, reliability, draws, success_probability)
    ids = sort(collect(keys(base_edges)))
    active = empty_edges(ids)
    foreach(network_pairs(base_edges)) do pair
        available = reliability == :stable ||
            draws[pair] < success_probability
        available && add_network_edge!(active, pair...)
    end
    active
end

function observation_record(problem, state)
    observation = last(state.history).observation
    Dict(
        :row => state.row,
        :H => reshape(copy(problem.modes[state.row, :]), 1, :),
        :z => [observation - problem.climatology[state.row]],
        :R => reshape([problem.measurement_variance[state.row]], 1, 1),
        :raw_observation => observation,
    )
end

function model_states(problem, information)
    Dict(
        aid => SCRIBEModelState(
            problem.initial_model.smodel,
            info,
            problem.initial_model.R,
        )
        for (aid, info) in information
    )
end

function coefficient_history_snapshot(information, ids)
    Dict(
        aid => let moments=posterior_coefficient_moments(information[aid])
            (μ=copy(moments[:μ]), Σ=copy(moments[:Σ]))
        end
        for aid in ids
    )
end

function integrated_coefficient_variance(problem, information)
    field_variance(problem, information) +
        mean(eof_residual_variance(problem.initial_model.smodel))
end

function scaling_metrics(
    environment,
    problem,
    information,
    central,
    communication,
    runtime,
    active_links,
    possible_links,
    step,
)
    fields = posterior_fields(problem.initial_model.smodel, information)
    truth = environment[:truth]
    central_field = reconstruct_eof_field(
        problem.initial_model.smodel;
        coefficients=posterior_coefficient_moments(central)[:μ],
    )
    errors = [
        normalized_field_error(field, truth)
        for field in values(fields)
    ]
    Dict(
        :step => step,
        :total_observations => length(information) * step,
        :mean_agent_nrmse => mean(errors),
        :worst_agent_nrmse => maximum(errors),
        :prediction_disagreement => inter_agent_disagreement(fields, truth),
        :centralized_gap => mean(
            normalized_field_error(field, central_field)
            for field in values(fields)
        ),
        :centralized_nrmse => normalized_field_error(central_field, truth),
        :mean_integrated_variance => mean(
            integrated_coefficient_variance(problem, info)
            for info in values(information)
        ),
        :active_links => active_links,
        :possible_links => possible_links,
        :link_availability => possible_links == 0 ? 0.0 :
            active_links / possible_links,
        :cumulative_messages => communication[:messages],
        :cumulative_bytes => communication[:bytes],
        :cumulative_consensus_iterations => communication[:iterations],
        :cumulative_planning_seconds => runtime[:planning],
        :cumulative_fusion_seconds => runtime[:fusion],
    )
end

function info_snapshot(info)
    Dict(:y => copy(info.y), :Y => copy(info.Y))
end

function run_scaling_trial(
    environment,
    problem,
    settings,
    n_agents,
    topology,
    reliability,
    seed,
)
    let ids=agent_ids(n_agents),
        starts=farthest_starting_rows(problem, n_agents, seed),
        states=Dict(
            aid => ROMSExplorationState(starts[index], Any[])
            for (index, aid) in enumerate(ids)
        ),
        policies=Dict(
            aid => scaling_policy(
                problem,
                settings,
                seed + 10_000 * index,
            )
            for (index, aid) in enumerate(ids)
        ),
        noise_rngs=Dict(
            aid => MersenneTwister(seed + 100 * index)
            for (index, aid) in enumerate(ids)
        ),
        link_rng=MersenneTwister(seed + 90_000),
        observations=Dict(aid => Any[] for aid in ids),
        paths=Dict(aid => Int[states[aid].row] for aid in ids),
        prior=problem.initial_model.information,
        network=initialize_asynchronous_network(
            environment[:params],
            prior,
            observations,
            settings[:noise_variance],
        ),
        base_edges=topology_edges(ids, topology),
        possible_links=length(network_pairs(base_edges)),
        metric_history=Any[],
        posterior_history=Any[],
        network_history=Any[],
        communication=Dict(:messages => 0, :bytes => 0, :iterations => 0),
        runtime=Dict(:planning => 0.0, :fusion => 0.0)

        central = copy(prior)
        information = network_information(network)
        push!(metric_history, scaling_metrics(
            environment,
            problem,
            information,
            central,
            communication,
            runtime,
            0,
            possible_links,
            0,
        ))
        push!(posterior_history, coefficient_history_snapshot(information, ids))

        foreach(1:settings[:n_samples]) do step
            models = model_states(problem, information)
            planning_start = time_ns()
            foreach(ids) do aid
                action = fixed_rollout_action(
                    policies[aid],
                    states[aid],
                    models[aid],
                    settings[:n_samples] - step + 1,
                    settings[:planning_iterations],
                )
                states[aid] = gen(
                    problem,
                    states[aid],
                    action,
                    noise_rngs[aid],
                ).sp
                push!(paths[aid], states[aid].row)
                push!(observations[aid], observation_record(problem, states[aid]))
            end
            runtime[:planning] += (time_ns() - planning_start) / 1e9

            central = centralized_update(
                central,
                observations,
                step,
                environment[:params].Q,
            )
            draws = link_draws(ids, link_rng)
            active_edges = reliable_edges(
                base_edges,
                reliability,
                draws,
                settings[:link_success_probability],
            )
            active_pairs = network_pairs(active_edges)
            fusion_start = time_ns()
            step_communication = asynchronous_network_step!(
                network,
                step,
                active_edges,
                settings,
            )
            runtime[:fusion] += (time_ns() - fusion_start) / 1e9
            foreach((:messages, :bytes, :iterations)) do metric
                communication[metric] += step_communication[metric]
            end
            information = network_information(network)

            push!(network_history, (
                step=step,
                possible=copy(network_pairs(base_edges)),
                active=copy(active_pairs),
            ))
            push!(posterior_history, coefficient_history_snapshot(information, ids))
            push!(metric_history, scaling_metrics(
                environment,
                problem,
                information,
                central,
                communication,
                runtime,
                length(active_pairs),
                possible_links,
                step,
            ))
        end

        Dict(
            :checkpoint_version => SCALING_CHECKPOINT_VERSION,
            :condition => Dict(
                :seed => seed,
                :n_agents => n_agents,
                :topology => topology,
                :reliability => reliability,
                :n_samples => settings[:n_samples],
            ),
            :truth_snapshot => ASYNCHRONOUS_TRUTH_SNAPSHOT,
            :metric_history => metric_history,
            :posterior_history => posterior_history,
            :paths => paths,
            :observations => Dict(
                aid => [record[:raw_observation] for record in observations[aid]]
                for aid in ids
            ),
            :sample_rows => Dict(
                aid => [record[:row] for record in observations[aid]]
                for aid in ids
            ),
            :network_history => network_history,
            :final_information => Dict(
                aid => info_snapshot(information[aid])
                for aid in ids
            ),
            :central_information => info_snapshot(central),
            :communication => copy(communication),
            :runtime => copy(runtime),
        )
    end
end

function trial_conditions(settings)
    [
        (
            n_agents=n_agents,
            topology=topology,
            reliability=reliability,
            seed=seed,
        )
        for n_agents in settings[:team_sizes]
        for topology in settings[:topologies]
        for reliability in settings[:reliabilities]
        for seed in settings[:seeds]
    ]
end

function condition_name(condition)
    "n_$(lpad(condition.n_agents, 2, '0'))_" *
    "$(condition.topology)_$(condition.reliability)_" *
    "seed_$(condition.seed)"
end

function scaling_output_directory(profile)
    joinpath(
        @__DIR__,
        "res",
        "agent_number_scaling",
        String(profile),
    )
end

function checkpoint_path(output_dir, condition)
    joinpath(output_dir, "checkpoints", condition_name(condition) * ".jls")
end

function checkpoint_matches(trial, condition, settings)
    saved = trial[:condition]
    trial[:checkpoint_version] == SCALING_CHECKPOINT_VERSION &&
        saved[:seed] == condition.seed &&
        saved[:n_agents] == condition.n_agents &&
        saved[:topology] == condition.topology &&
        saved[:reliability] == condition.reliability &&
        saved[:n_samples] == settings[:n_samples]
end

function completed_checkpoint_names(settings, output_dir)
    Set(
        condition_name(condition)
        for condition in trial_conditions(settings)
        if let path=checkpoint_path(output_dir, condition)
            isfile(path) && checkpoint_matches(
                open(deserialize, path),
                condition,
                settings,
            )
        end
    )
end

function atomic_serialize(output_path, value)
    mkpath(dirname(output_path))
    temporary = output_path * ".tmp.$(getpid())"
    open(temporary, "w") do io
        serialize(io, value)
    end
    mv(temporary, output_path; force=true)
    output_path
end

function atomic_text(writer, output_path)
    mkpath(dirname(output_path))
    temporary = output_path * ".tmp.$(getpid())"
    open(writer, temporary, "w")
    mv(temporary, output_path; force=true)
    output_path
end

function write_progress(settings, output_dir, completed)
    conditions = trial_conditions(settings)
    atomic_text(joinpath(output_dir, "progress.csv")) do io
        println(io, "n_agents,topology,reliability,seed,status")
        foreach(conditions) do condition
            status = condition_name(condition) in completed ?
                "complete" : "pending"
            println(io, join((
                condition.n_agents,
                condition.topology,
                condition.reliability,
                condition.seed,
                status,
            ), ','))
        end
    end
end

function load_scaling_trials(settings, output_dir)
    reduce(vcat, map(trial_conditions(settings)) do condition
        path = checkpoint_path(output_dir, condition)
        if isfile(path)
            trial = open(deserialize, path)
            checkpoint_matches(trial, condition, settings) ? [trial] : Any[]
        else
            Any[]
        end
    end; init=Any[])
end

function trial_metric_row(trial)
    condition = trial[:condition]
    final = last(trial[:metric_history])
    Dict(
        :seed => condition[:seed],
        :n_agents => condition[:n_agents],
        :topology => condition[:topology],
        :reliability => condition[:reliability],
        :n_samples => condition[:n_samples],
        :mean_agent_nrmse => final[:mean_agent_nrmse],
        :worst_agent_nrmse => final[:worst_agent_nrmse],
        :prediction_disagreement => final[:prediction_disagreement],
        :centralized_gap => final[:centralized_gap],
        :centralized_nrmse => final[:centralized_nrmse],
        :mean_integrated_variance => final[:mean_integrated_variance],
        :mean_link_availability => mean(
            row[:link_availability]
            for row in Iterators.drop(trial[:metric_history], 1)
        ),
        :messages => trial[:communication][:messages],
        :bytes => trial[:communication][:bytes],
        :consensus_iterations => trial[:communication][:iterations],
        :planning_seconds => trial[:runtime][:planning],
        :fusion_seconds => trial[:runtime][:fusion],
        :seconds_per_agent_step => (
            trial[:runtime][:planning] + trial[:runtime][:fusion]
        ) / (condition[:n_agents] * condition[:n_samples]),
        :fusion_seconds_per_agent_step => trial[:runtime][:fusion] /
            (condition[:n_agents] * condition[:n_samples]),
        :consensus_iterations_per_agent_step =>
            trial[:communication][:iterations] /
            (condition[:n_agents] * condition[:n_samples]),
        :bytes_per_agent => trial[:communication][:bytes] /
            condition[:n_agents],
        :payload_bytes_per_agent_step => trial[:communication][:bytes] /
            (condition[:n_agents] * condition[:n_samples]),
    )
end

function metric_history_rows(trials)
    reduce(vcat, map(trials) do trial
        condition = trial[:condition]
        [
            merge(row, Dict(
                :seed => condition[:seed],
                :n_agents => condition[:n_agents],
                :topology => condition[:topology],
                :reliability => condition[:reliability],
            ))
            for row in trial[:metric_history]
        ]
    end; init=Any[])
end

function write_dictionary_rows(rows, fields, output_path)
    atomic_text(output_path) do io
        println(io, join(string.(fields), ','))
        foreach(rows) do row
            println(io, join((row[field] for field in fields), ','))
        end
    end
end

function write_saved_paths(trials, output_path)
    atomic_text(output_path) do io
        println(io, "seed,n_agents,topology,reliability,agent,step,row")
        foreach(trials) do trial
            condition = trial[:condition]
            foreach(sort(collect(keys(trial[:paths])))) do aid
                foreach(enumerate(trial[:paths][aid])) do (index, row)
                    println(io, join((
                        condition[:seed],
                        condition[:n_agents],
                        condition[:topology],
                        condition[:reliability],
                        aid,
                        index - 1,
                        row,
                    ), ','))
                end
            end
        end
    end
end

function write_saved_observations(trials, output_path)
    atomic_text(output_path) do io
        println(io, "seed,n_agents,topology,reliability,agent,step,row,observation")
        foreach(trials) do trial
            condition = trial[:condition]
            foreach(sort(collect(keys(trial[:observations])))) do aid
                foreach(eachindex(trial[:observations][aid])) do step
                    println(io, join((
                        condition[:seed],
                        condition[:n_agents],
                        condition[:topology],
                        condition[:reliability],
                        aid,
                        step,
                        trial[:sample_rows][aid][step],
                        trial[:observations][aid][step],
                    ), ','))
                end
            end
        end
    end
end

function aggregate_scaling_results(settings, output_dir)
    trials = load_scaling_trials(settings, output_dir)
    isempty(trials) && return trials, Any[]
    final_rows = trial_metric_row.(trials)
    history_rows = metric_history_rows(trials)
    write_dictionary_rows(
        final_rows,
        (
            :seed,
            :n_agents,
            :topology,
            :reliability,
            :n_samples,
            :mean_agent_nrmse,
            :worst_agent_nrmse,
            :prediction_disagreement,
            :centralized_gap,
            :centralized_nrmse,
            :mean_integrated_variance,
            :mean_link_availability,
            :messages,
            :bytes,
            :consensus_iterations,
            :planning_seconds,
            :fusion_seconds,
            :seconds_per_agent_step,
            :fusion_seconds_per_agent_step,
            :consensus_iterations_per_agent_step,
            :bytes_per_agent,
            :payload_bytes_per_agent_step,
        ),
        joinpath(output_dir, "final_trial_metrics.csv"),
    )
    write_dictionary_rows(
        history_rows,
        (
            :seed,
            :n_agents,
            :topology,
            :reliability,
            :step,
            :total_observations,
            :mean_agent_nrmse,
            :worst_agent_nrmse,
            :prediction_disagreement,
            :centralized_gap,
            :centralized_nrmse,
            :mean_integrated_variance,
            :active_links,
            :possible_links,
            :link_availability,
            :cumulative_messages,
            :cumulative_bytes,
            :cumulative_consensus_iterations,
            :cumulative_planning_seconds,
            :cumulative_fusion_seconds,
        ),
        joinpath(output_dir, "metric_history.csv"),
    )
    write_saved_paths(trials, joinpath(output_dir, "agent_paths.csv"))
    write_saved_observations(
        trials,
        joinpath(output_dir, "observations.csv"),
    )
    expected = length(trial_conditions(settings))
    atomic_text(joinpath(output_dir, "experiment_status.txt")) do io
        println(io, "checkpoint_version=$(SCALING_CHECKPOINT_VERSION)")
        println(io, "completed_trials=$(length(trials))")
        println(io, "expected_trials=$expected")
        println(io, "complete=$(length(trials) == expected)")
        println(io, "truth_snapshot=$(ASYNCHRONOUS_TRUTH_SNAPSHOT)")
    end
    atomic_text(joinpath(output_dir, "experiment_settings.txt")) do io
        println(io, "checkpoint_version=$(SCALING_CHECKPOINT_VERSION)")
        println(io, "truth_snapshot=$(ASYNCHRONOUS_TRUTH_SNAPSHOT)")
        foreach(sort(collect(keys(settings)))) do key
            println(io, "$key=$(settings[key])")
        end
    end
    trials, final_rows
end

function scaling_aggregate(rows, topology, reliability, metric)
    selected = filter(rows) do row
        row[:topology] == topology && row[:reliability] == reliability
    end
    map(sort(unique(getindex.(selected, :n_agents)))) do n_agents
        values = getindex.(
            filter(row -> row[:n_agents] == n_agents, selected),
            metric,
        )
        (
            n_agents=n_agents,
            median=median(values),
            lower=quantile(values, 0.25),
            upper=quantile(values, 0.75),
            trials=length(values),
        )
    end
end

const SCALING_LABELS = Dict(
    :sparse => "Sparse Ring",
    :moderate => "Moderate Degree",
    :dense => "Complete",
)

function stable_scaling_aggregate(rows, metric)
    selected = filter(row -> row[:reliability] == :stable, rows)
    map(sort(unique(getindex.(selected, :n_agents)))) do n_agents
        agent_rows = filter(row -> row[:n_agents] == n_agents, selected)
        values = [
            median(getindex.(
                filter(row -> row[:seed] == seed, agent_rows),
                metric,
            ))
            for seed in sort(unique(getindex.(agent_rows, :seed)))
        ]
        (
            n_agents=n_agents,
            median=median(values),
            lower=quantile(values, 0.25),
            upper=quantile(values, 0.75),
            trials=length(values),
        )
    end
end

function reference_scaling_aggregate(rows, metric)
    map(sort(unique(getindex.(rows, :n_agents)))) do n_agents
        agent_rows = filter(row -> row[:n_agents] == n_agents, rows)
        values = [
            median(getindex.(
                filter(row -> row[:seed] == seed, agent_rows),
                metric,
            ))
            for seed in sort(unique(getindex.(agent_rows, :seed)))
        ]
        (
            n_agents=n_agents,
            median=median(values),
            lower=quantile(values, 0.25),
            upper=quantile(values, 0.75),
            trials=length(values),
        )
    end
end

function stable_topologies_coincide(rows, metrics)
    stable = filter(row -> row[:reliability] == :stable, rows)
    groups = unique((row[:n_agents], row[:seed]) for row in stable)
    all(groups) do (n_agents, seed)
        group = filter(stable) do row
            row[:n_agents] == n_agents && row[:seed] == seed
        end
        all(metrics) do metric
            values = Float64.(getindex.(group, metric))
            isempty(values) || all(
                isapprox(first(values), value; rtol=1e-8, atol=1e-10)
                for value in values
            )
        end
    end
end

function scaling_limits(rows, metrics; show_zero=false)
    selected_metrics = metrics isa Tuple ? metrics : (metrics,)
    values = reduce(vcat, (
        Float64.(getindex.(rows, metric))
        for metric in selected_metrics
    ))
    low, high = extrema(values)
    span = max(high - low, 0.08max(abs(high), 1e-6))
    lower = show_zero ? min(-0.06span, low - 0.08span) :
        max(0.0, low - 0.10span)
    (lower, high + 0.12span)
end

function scaling_panel(
    rows,
    metric,
    title,
    ylabel;
    render_profile=:paper,
    collapse_stable=false,
    centralized_reference=false,
    panel_position=:left,
)
    style = workshop_plot_style(render_profile)
    panel = plot(
        ;
        xlabel="Number of agents",
        ylabel,
        title,
        grid=true,
        legend=false,
        xlims=(1.7, 10.3),
        xticks=2:10,
        ylims=scaling_limits(
            rows,
            centralized_reference ?
                (metric, :centralized_nrmse) : metric;
            show_zero=metric in (
                :prediction_disagreement,
                :centralized_gap,
            ),
        ),
    )
    if collapse_stable
        stable = stable_scaling_aggregate(rows, metric)
        plot!(
            panel,
            getindex.(stable, :n_agents),
            getindex.(stable, :median);
            color=:black,
            linestyle=:solid,
            marker=:circle,
            linewidth=style.linewidth,
            markersize=style.markersize,
            label=false,
        )
        plotted = Vector{Vector{Float64}}()
        foreach((:sparse, :moderate, :dense)) do topology
            summary = scaling_aggregate(
                rows,
                topology,
                :intermittent,
                metric,
            )
            isempty(summary) && return
            values = Float64.(getindex.(summary, :median))
            duplicate = any(plotted) do previous
                length(previous) == length(values) && previous ≈ values
            end
            if !duplicate
                push!(plotted, values)
                plot!(
                    panel,
                    getindex.(summary, :n_agents),
                    values;
                    color=WORKSHOP_TOPOLOGY_COLORS[topology],
                    linestyle=:dash,
                    marker=:diamond,
                    linewidth=style.linewidth,
                    markersize=style.markersize,
                    label=false,
                )
            end
        end
    else
        foreach((:sparse, :moderate, :dense)) do topology
            foreach((:stable, :intermittent)) do reliability
                summary = scaling_aggregate(rows, topology, reliability, metric)
                if !isempty(summary)
                    plot!(
                        panel,
                        getindex.(summary, :n_agents),
                        getindex.(summary, :median);
                        color=WORKSHOP_TOPOLOGY_COLORS[topology],
                        linestyle=reliability == :stable ? :solid : :dash,
                        marker=reliability == :stable ? :circle : :diamond,
                        linewidth=style.linewidth,
                        markersize=style.markersize,
                        label=false,
                    )
                end
            end
        end
    end
    if centralized_reference
        reference = reference_scaling_aggregate(rows, :centralized_nrmse)
        plot!(
            panel,
            getindex.(reference, :n_agents),
            getindex.(reference, :median);
            color=:gray35,
            linestyle=:dot,
            marker=:utriangle,
            linewidth=style.linewidth,
            markersize=style.markersize,
            label=false,
        )
    end
    workshop_panel!(
        panel;
        profile=render_profile,
        left_margin=panel_position == :left ? 8Plots.mm : 12Plots.mm,
        right_margin=panel_position == :left ? 2Plots.mm : 3Plots.mm,
        bottom_margin=6Plots.mm,
        top_margin=4Plots.mm,
    )
end

function scaling_legend_panel(
    topologies;
    render_profile=:paper,
    collapse_stable=false,
    centralized_reference=false,
)
    style = workshop_plot_style(render_profile)
    panel = plot(
        ;
        axis=false,
        grid=false,
        legend=:top,
        legend_columns=3,
        framestyle=:none,
    )
    foreach(topologies) do topology
        plot!(
            panel,
            [NaN],
            [NaN];
            color=WORKSHOP_TOPOLOGY_COLORS[topology],
            linewidth=style.linewidth,
            marker=:diamond,
            markersize=style.markersize,
            label=SCALING_LABELS[topology],
        )
    end
    plot!(
        panel,
        [NaN],
        [NaN];
        color=:black,
        linestyle=:solid,
        marker=:circle,
        linewidth=style.linewidth,
        markersize=style.markersize,
        label=collapse_stable ? "Stable connected" : "Stable",
    )
    plot!(
        panel,
        [NaN],
        [NaN];
        color=:black,
        linestyle=:dash,
        marker=:diamond,
        linewidth=style.linewidth,
        markersize=style.markersize,
        label="50% link availability",
    )
    centralized_reference && plot!(
        panel,
        [NaN],
        [NaN];
        color=:gray35,
        linestyle=:dot,
        marker=:utriangle,
        linewidth=style.linewidth,
        markersize=style.markersize,
        label="Centralized reference",
    )
    plot!(panel; legendfontsize=style.legendfontsize)
    panel
end

function scaling_figure(panels, legend_panel, style; title)
    plot(
        panels...,
        legend_panel;
        layout=@layout([a b; c d; e{0.17h}]),
        size=(style.width, round(Int, 1.32style.spatial_height)),
        plot_title=title,
        plot_titlefontsize=style.titlefontsize,
    )
end

function save_scaling_figures(
    rows,
    output_dir;
    experimental_profile=:full,
)
    if experimental_profile == :smoke
        println("Smoke figures omitted; use the complete full-profile figures.")
        return (performance=nothing, cost=nothing)
    end
    render_profile = :poster
    style = workshop_plot_style(render_profile)
    available_topologies = Set(getindex.(rows, :topology))
    topologies = filter(∈(available_topologies), (:sparse, :moderate, :dense))
    performance_metrics = (
        :mean_agent_nrmse,
        :worst_agent_nrmse,
        :prediction_disagreement,
        :centralized_gap,
    )
    collapse_stable = stable_topologies_coincide(rows, performance_metrics)
    performance = scaling_figure(
        (
          scaling_panel(
            rows,
            :mean_agent_nrmse,
            "(a) Mean reconstruction error",
            "Mean-agent NRMSE";
            render_profile,
            collapse_stable,
            centralized_reference=true,
        ),
          scaling_panel(
            rows,
            :worst_agent_nrmse,
            "(b) Worst-agent reconstruction error",
            "Worst-agent NRMSE";
            render_profile,
            collapse_stable,
            panel_position=:right,
        ),
          scaling_panel(
            rows,
            :prediction_disagreement,
            "(c) Inter-agent agreement",
            "Inter-agent NRMSE";
            render_profile,
            collapse_stable,
        ),
          scaling_panel(
            rows,
            :centralized_gap,
            "(d) Distributed-to-centralized gap",
            "NRMSE from centralized posterior";
            render_profile,
            collapse_stable,
            panel_position=:right,
        )),
        scaling_legend_panel(
            topologies;
            render_profile,
            collapse_stable,
            centralized_reference=true,
        ),
        style,
        title="Agent and Network Scaling of SCRIBE Model Performance",
    )
    performance_paths = save_workshop_figure(
        performance,
        joinpath(output_dir, "agent_number_scaling_performance_poster"),
    )

    operational = scaling_figure(
        (
          scaling_panel(
            rows,
            :mean_integrated_variance,
            "(a) Posterior uncertainty",
            "Integrated posterior variance";
            render_profile,
        ),
          scaling_panel(
            rows,
            :fusion_seconds_per_agent_step,
            "(b) SCRIBE update time",
            "Fusion time / agent-update (s)";
            render_profile,
            panel_position=:right,
        ),
          scaling_panel(
            rows,
            :consensus_iterations_per_agent_step,
            "(c) Consensus effort",
            "Consensus rounds / agent-update";
            render_profile,
        ),
          scaling_panel(
            rows,
            :payload_bytes_per_agent_step,
            "(d) Communication payload",
            "Payload / agent-update (bytes)";
            render_profile,
            panel_position=:right,
        )),
        scaling_legend_panel(topologies; render_profile),
        style;
        title="SCRIBE Uncertainty, Computation, and Communication Scaling",
    )
    cost_paths = save_workshop_figure(
        operational,
        joinpath(output_dir, "agent_number_scaling_cost_poster"),
    )
    (performance=performance_paths, cost=cost_paths)
end

function warmup_scaling_experiment(environment, settings)
    warmup_settings = merge(settings, Dict(
        :n_samples => 1,
        :lookahead => 1,
        :planning_iterations => 1,
    ))
    problem = scaling_problem(environment, warmup_settings)
    n_agents = maximum(settings[:team_sizes])
    foreach((
        (:sparse, :stable),
        (:sparse, :intermittent),
        (:dense, :stable),
        (:dense, :intermittent),
    )) do (topology, reliability)
        run_scaling_trial(
            environment,
            problem,
            warmup_settings,
            n_agents,
            topology,
            reliability,
            -1,
        )
    end
    nothing
end

function run_agent_number_scaling(;
    profile=:full,
    resume=true,
    make_plots=true,
)
    settings = agent_scaling_settings(profile)
    output_dir = scaling_output_directory(profile)
    mkpath(joinpath(output_dir, "checkpoints"))
    conditions = trial_conditions(settings)
    completed = resume ?
        completed_checkpoint_names(settings, output_dir) :
        Set{String}()
    pending = filter(conditions) do condition
        condition_name(condition) ∉ completed
    end
    write_progress(settings, output_dir, completed)

    if !isempty(pending)
        println(
            "$(length(pending)) of $(length(conditions)) scaling trials pending.",
        )
        environment = load_asynchronous_environment()
        problem = scaling_problem(environment, settings)
        warmup_scaling_experiment(environment, settings)
        foreach(enumerate(pending)) do (index, condition)
            println(
                "[$index/$(length(pending))] " * condition_name(condition),
            )
            flush(stdout)
            trial = run_scaling_trial(
                environment,
                problem,
                settings,
                condition.n_agents,
                condition.topology,
                condition.reliability,
                condition.seed,
            )
            atomic_serialize(checkpoint_path(output_dir, condition), trial)
            push!(completed, condition_name(condition))
            write_progress(settings, output_dir, completed)
            final = last(trial[:metric_history])
            println(
                "  NRMSE=$(round(final[:mean_agent_nrmse]; digits=4)), " *
                "disagreement=$(round(final[:prediction_disagreement]; digits=4)), " *
                "messages=$(trial[:communication][:messages])",
            )
            flush(stdout)
            GC.gc()
        end
    else
        println("All $(length(conditions)) scaling trials are already checkpointed.")
    end

    trials, final_rows = aggregate_scaling_results(settings, output_dir)
    make_plots && !isempty(final_rows) &&
        save_scaling_figures(
            final_rows,
            output_dir;
            experimental_profile=profile,
        )
    println(
        "Saved $(length(trials))/$(length(conditions)) trials at $output_dir",
    )
    Dict(:trials => trials, :final_rows => final_rows, :settings => settings)
end

function plot_saved_agent_scaling(profile=:full)
    settings = agent_scaling_settings(profile)
    output_dir = scaling_output_directory(profile)
    trials, final_rows = aggregate_scaling_results(settings, output_dir)
    isempty(final_rows) || save_scaling_figures(
        final_rows,
        output_dir;
        experimental_profile=profile,
    )
    println(
        "Rebuilt aggregate files and figures from $(length(trials)) checkpoints.",
    )
    Dict(:trials => trials, :final_rows => final_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    let profile=isempty(ARGS) ? :full : Symbol(first(ARGS)),
        plots_only="plots_only" in ARGS,
        resume="rerun" ∉ ARGS,
        make_plots="no_plots" ∉ ARGS
        plots_only ?
            plot_saved_agent_scaling(profile) :
            run_agent_number_scaling(;
                profile,
                resume,
                make_plots,
            )
    end
end
