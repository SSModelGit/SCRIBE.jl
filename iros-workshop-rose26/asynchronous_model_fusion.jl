ENV["GKSwstype"] = "100"

using LinearAlgebra
using Match: @match
using Plots
using Random
using SCRIBE
using SCRIBE.ROMSTools
using Serialization
using Statistics

include(joinpath(@__DIR__, "ram_head_eof_experiment.jl"))
include(joinpath(@__DIR__, "workshop_plotting.jl"))

const ASYNCHRONOUS_TRUTH_SNAPSHOT = 5534
const ASYNCHRONOUS_CHECKPOINT_VERSION = 1

mutable struct AsynchronousScribe <: EnvScribe
    k
    information
end

struct AsynchronousEstimators <: EnvEstimators
    system
    Q
    observations
end

function inverse_positive_definite(M)
    let factor=cholesky(Symmetric((M + M') / 2))
        Matrix(factor \ Matrix{Float64}(I, size(M)...))
    end
end

function asynchronous_prior(info, Q)
    let P=inverse_positive_definite(info.Y),
        μ=P * info.y,
        P⁻=P + Q,
        Y⁻=inverse_positive_definite(P⁻)
        ((Y⁻ + Y⁻') / 2, Y⁻ * μ)
    end
end

function SCRIBE.compute_info_priors(
    estimators::AsynchronousEstimators,
    _::Integer,
)
    asynchronous_prior(last(estimators.system.information), estimators.Q)
end

function SCRIBE.compute_innov_from_obs(
    estimators::AsynchronousEstimators,
    k::Integer,
)
    let observation=estimators.observations[k],
        innovation=measurement_information(
            observation[:H],
            observation[:z],
            observation[:R],
        )
        (innovation[:δI], innovation[:δi])
    end
end

SCRIBE.next_agent_info_state(system::AsynchronousScribe, info::KFEnvInfo) =
    push!(system.information, info)

asynchronous_fusion_settings(profile) = @match profile begin
    :full => Dict(
        :seeds => Tuple(701:2:723),
        :team_sizes => Tuple(3:10),
        :path_cycles => 3,
        :mission_steps => nothing,
        :metric_points => 101,
        :communication_radius => 7.0,
        :noise_variance => 1e-4,
        :consensus_threshold => 1e-7,
        :consensus_timeline => 120,
        :representative_team_size => 6,
        :representative_seed => 723,
        :animation_frames => 72,
        :animation_fps => 5,
    )
    :pilot => Dict(
        :seeds => (701, 703, 705),
        :team_sizes => (3, 6, 10),
        :path_cycles => 1,
        :mission_steps => nothing,
        :metric_points => 61,
        :communication_radius => 7.0,
        :noise_variance => 1e-4,
        :consensus_threshold => 1e-6,
        :consensus_timeline => 100,
        :representative_team_size => 6,
        :representative_seed => 703,
        :animation_frames => 48,
        :animation_fps => 5,
    )
    :smoke => Dict(
        :seeds => (701,),
        :team_sizes => (3, 6),
        :path_cycles => 1,
        :mission_steps => 1200,
        :metric_points => 21,
        :communication_radius => 7.0,
        :noise_variance => 1e-4,
        :consensus_threshold => 1e-5,
        :consensus_timeline => 80,
        :representative_team_size => 6,
        :representative_seed => 701,
        :animation_frames => 20,
        :animation_fps => 4,
    )
    _ => throw(ArgumentError(
        "Use the `full`, `pilot`, or `smoke` asynchronous-fusion profile.",
    ))
end

function zero_mean_eof_parameters(params)
    EOFClimateModelParameters(
        params.decomposition;
        locations=params.locations,
        ϕ₀=zeros(params.nᵩ),
        prior_covariance=params.P₀,
        process_covariance=zeros(params.nᵩ, params.nᵩ),
        interpolation=params.interpolation,
        interpolation_neighbors=params.interpolation_neighbors,
        metadata=copy(params.metadata),
    )
end

function load_asynchronous_environment()
    params = zero_mean_eof_parameters(
        load_eof_model_parameters(RAM_HEAD_EOF_MODEL),
    )
    stride = Int(metadata_value(params.metadata["temporal_stride"]))
    roms = let archive=read_roms_velocity(RAM_HEAD_ARCHIVE, :u)
        prepare_roms_velocity(archive; temporal_stride=stride)
    end
    GC.gc()
    truth = roms[:data][:, ASYNCHRONOUS_TRUTH_SNAPSHOT]
    coefficients = eof_coefficients(params, truth)
    standardized = coefficients ./ sqrt.(diag(params.P₀))
    Dict(
        :params => params,
        :model => initialize_SCRIBEModel_from_parameters(params),
        :roms => roms,
        :truth => truth,
        :truth_coefficients => coefficients,
        :standardized_coefficients => standardized,
    )
end

function shifted_regions(regions, row_shift, column_shift)
    [
        (
            rows=(first(region[:rows]) + row_shift):
                (last(region[:rows]) + row_shift),
            columns=(first(region[:columns]) + column_shift):
                (last(region[:columns]) + column_shift),
        )
        for region in regions
    ]
end

function region_layout_score(regions)
    let aspect_ratios=[
            max(
                length(region[:rows]) / length(region[:columns]),
                length(region[:columns]) / length(region[:rows]),
            )
            for region in regions
        ],
        areas=[
            length(region[:rows]) * length(region[:columns])
            for region in regions
        ]
        maximum(aspect_ratios) + 0.1std(areas; corrected=false) / mean(areas)
    end
end

function balanced_rectangles(n_rows, n_columns, n_regions, cache)
    n_regions == 1 && return [(
        rows=1:n_rows,
        columns=1:n_columns,
    )]
    key = (n_rows, n_columns, n_regions)
    haskey(cache, key) && return cache[key]
    candidates = mapreduce(vcat, 1:(n_regions ÷ 2)) do n_left
        n_right = n_regions - n_left
        layouts = Any[]
        if n_rows > 1
            row_cut = clamp(
                round(Int, n_rows * n_left / n_regions),
                1,
                n_rows - 1,
            )
            push!(layouts, vcat(
                balanced_rectangles(row_cut, n_columns, n_left, cache),
                shifted_regions(
                    balanced_rectangles(
                        n_rows - row_cut,
                        n_columns,
                        n_right,
                        cache,
                    ),
                    row_cut,
                    0,
                ),
            ))
        end
        if n_columns > 1
            column_cut = clamp(
                round(Int, n_columns * n_left / n_regions),
                1,
                n_columns - 1,
            )
            push!(layouts, vcat(
                balanced_rectangles(n_rows, column_cut, n_left, cache),
                shifted_regions(
                    balanced_rectangles(
                        n_rows,
                        n_columns - column_cut,
                        n_right,
                        cache,
                    ),
                    0,
                    column_cut,
                ),
            ))
        end
        layouts
    end
    cache[key] = candidates[argmin(region_layout_score.(candidates))]
end

function operational_regions(grid_shape, n_agents)
    regions = balanced_rectangles(
        grid_shape[1],
        grid_shape[2],
        n_agents,
        Dict(),
    )
    sort!(regions; by=region -> (
        first(region[:columns]),
        first(region[:rows]),
    ))
end

range_overlap(left, right) =
    max(first(left), first(right)) ≤ min(last(left), last(right))

function adjacent_regions(left, right)
    row_contact = (
        last(left[:rows]) + 1 == first(right[:rows]) ||
        last(right[:rows]) + 1 == first(left[:rows])
    ) && range_overlap(left[:columns], right[:columns])
    column_contact = (
        last(left[:columns]) + 1 == first(right[:columns]) ||
        last(right[:columns]) + 1 == first(left[:columns])
    ) && range_overlap(left[:rows], right[:rows])
    row_contact || column_contact
end

function wet_row_grid(roms)
    rows = zeros(Int, prod(roms[:grid_shape]))
    rows[roms[:wet_mask]] = axes(roms[:locations], 1)
    reshape(rows, roms[:grid_shape]...)
end

function wet_grid_locations(roms)
    cells = findall(reshape(roms[:wet_mask], roms[:grid_shape]...))
    hcat(getindex.(cells, 1), getindex.(cells, 2))
end

function regional_lawnmower_path(grid_rows, region)
    reduce(vcat, map(enumerate(region[:columns])) do (index, column)
        rows = grid_rows[region[:rows], column]
        filter(!iszero, isodd(index) ? rows : reverse(rows))
    end)
end

function regional_paths(roms, regions)
    let grid_rows=wet_row_grid(roms)
        Dict(
            "agent$index" => regional_lawnmower_path(grid_rows, region)
            for (index, region) in enumerate(regions)
        )
    end
end

lawnmower_period(path) = max(2length(path) - 2, 1)

function lawnmower_row(path, step, offset)
    length(path) == 1 && return only(path)
    position = mod(step - 1 + offset, lawnmower_period(path)) + 1
    path[position ≤ length(path) ? position : 2length(path) - position]
end

function experiment_mission_steps(settings, roms)
    !isnothing(settings[:mission_steps]) && return settings[:mission_steps]
    regions = operational_regions(roms[:grid_shape], minimum(settings[:team_sizes]))
    paths = regional_paths(roms, regions)
    settings[:path_cycles] * maximum(lawnmower_period, values(paths))
end

agent_ids(n_agents) = ["agent$index" for index in 1:n_agents]
agent_number(aid) = parse(Int, replace(aid, "agent" => ""))

function adjacent_agent_pairs(regions)
    [
        ("agent$left", "agent$right")
        for left in 1:(length(regions) - 1)
        for right in (left + 1):length(regions)
        if adjacent_regions(regions[left], regions[right])
    ]
end

function empty_edges(ids)
    Dict(aid => String[] for aid in ids)
end

function contact_edges(states, adjacent_pairs, active_contacts, radius)
    near = Set(
        pair for pair in adjacent_pairs
        if norm(states[first(pair)] - states[last(pair)]) ≤ radius
    )
    candidates = sort(
        collect(setdiff(near, active_contacts));
        by=pair -> norm(states[first(pair)] - states[last(pair)]),
    )
    selected = Tuple{String, String}[]
    used = Set{String}()
    foreach(candidates) do pair
        if first(pair) ∉ used && last(pair) ∉ used
            push!(selected, pair)
            push!(used, first(pair), last(pair))
        end
    end
    next_active = union(intersect(active_contacts, near), Set(selected))
    edges = empty_edges(sort(collect(keys(states))))
    foreach(selected) do (left, right)
        push!(edges[left], right)
        push!(edges[right], left)
    end
    edges, next_active, selected
end

function observation_plan(
    model,
    truth,
    paths,
    n_steps,
    noise_variance,
    seed,
)
    let rng=MersenneTwister(seed),
        residual=eof_residual_variance(model),
        modes=eof_modes(model),
        climatology=eof_mean(model)
        offsets = Dict(
            aid => rand(rng, 0:(lawnmower_period(path) - 1))
            for (aid, path) in paths
        )
        observations = Dict(
            aid => map(1:n_steps) do step
                row = lawnmower_row(path, step, offsets[aid])
                Dict(
                    :row => row,
                    :H => reshape(copy(modes[row, :]), 1, :),
                    :z => [
                        truth[row] - climatology[row] +
                        sqrt(noise_variance) * randn(rng),
                    ],
                    :R => reshape(
                        [noise_variance + residual[row]],
                        1,
                        1,
                    ),
                )
            end
            for (aid, path) in paths
        )
        Dict(:observations => observations, :offsets => offsets)
    end
end

function network_outbox()
    Dict{String, Any}(
        "lc" => nothing,
        "ln" => nothing,
        "lv" => 0,
        "cvc" => 0,
        "prior" => nothing,
        "innov" => nothing,
        "stage" => :prior,
        "seq" => 0,
        "cache" => Dict{String, Any}(),
        "msg_out" => nothing,
        "complete" => false,
        "n_eff" => 1,
        "stage_ready" => false,
    )
end

function initialize_asynchronous_network(params, prior, observations, noise_variance)
    let ids=sort(collect(keys(observations))),
        edges=empty_edges(ids),
        network=init_network_graph(edges),
        observer=EOFObserverBehavior(noise_variance)
        foreach(ids) do aid
            system = AsynchronousScribe(1, [copy(prior)])
            estimators = AsynchronousEstimators(
                system,
                params.Q,
                observations[aid],
            )
            network.vertices[aid] = SCRIBEAgent(
                aid,
                params,
                observer,
                [zeros(1, size(params.locations, 2))],
                system,
                NetworkConnector(String[], network_outbox()),
                estimators,
            )
        end
        network
    end
end

function deliver_messages!(k, network)
    let outgoing=deliver_consensus_messages(k, network),
        messages=0
        foreach(outgoing) do (aid, message)
            foreach(network.edges[aid]) do neighbor
                consume_consensus_message!(
                    neighbor,
                    network.vertices[neighbor].net_conn,
                    message,
                    k,
                    network,
                )
                messages += 1
            end
        end
        foreach(keys(outgoing)) do aid
            network.vertices[aid].net_conn.outbox["msg_out"] = nothing
        end
        messages
    end
end

function finish_network_step!(network)
    foreach(values(network.vertices)) do vertex
        vertex.agent.k += 1
        full_reset_network_connector(vertex.net_conn)
    end
end

function local_information_update(info, observation, Q)
    let (Y⁻, y⁻)=asynchronous_prior(info, Q),
        innovation=measurement_information(
            observation[:H],
            observation[:z],
            observation[:R],
        ),
        Y=Y⁻ + innovation[:δI]
        KFEnvInfo(
            y⁻ + innovation[:δi],
            (Y + Y') / 2,
            innovation[:δi],
            innovation[:δI],
        )
    end
end

function local_network_step!(network, k)
    foreach(values(network.vertices)) do vertex
        push!(
            vertex.agent.information,
            local_information_update(
                last(vertex.agent.information),
                vertex.estimators.observations[k],
                vertex.estimators.Q,
            ),
        )
    end
    finish_network_step!(network)
    Dict(:messages => 0, :bytes => 0, :iterations => 1)
end

function asynchronous_network_step!(network, k, edges, settings)
    all(isempty, values(edges)) && return local_network_step!(network, k)
    update_network_graph_edges(edges, network)
    messages = 0
    iterations = 0
    ids = sort(collect(keys(network.vertices)))
    while true
        iterations += 1
        complete = map(ids) do aid
            distributed_fusion(
                k,
                aid,
                network,
                settings[:consensus_threshold],
                settings[:consensus_timeline],
            )
        end
        messages += deliver_messages!(k, network)
        all(complete) && break
    end
    payload_size = let info=last(first(values(network.vertices)).agent.information)
        sizeof(Float64) * (length(info.Y) + length(info.y))
    end
    finish_network_step!(network)
    Dict(
        :messages => messages,
        :bytes => messages * payload_size,
        :iterations => iterations,
    )
end

function centralized_update(info, observations, k, Q)
    let (Y⁻, y⁻)=asynchronous_prior(info, Q),
        innovations=map(values(observations)) do plan
            measurement_information(plan[k][:H], plan[k][:z], plan[k][:R])
        end,
        δI=sum(getindex.(innovations, :δI)),
        δi=sum(getindex.(innovations, :δi)),
        Y=Y⁻ + δI
        KFEnvInfo(y⁻ + δi, (Y + Y') / 2, δi, δI)
    end
end

function network_information(network)
    Dict(
        aid => last(network.vertices[aid].agent.information)
        for aid in sort(collect(keys(network.vertices)))
    )
end

function posterior_fields(model, information)
    Dict(
        aid => reconstruct_eof_field(
            model;
            coefficients=posterior_coefficient_moments(info)[:μ],
        )
        for (aid, info) in information
    )
end

normalized_field_error(estimate, truth) =
    sqrt(mean(abs2, estimate - truth)) / std(truth; corrected=false)

function field_error_terms(model, truth)
    let modes=eof_modes(model),
        offset=eof_mean(model) - truth
        (
            quadratic=modes' * modes,
            linear=modes' * offset,
            constant=dot(offset, offset),
            normalization=length(truth) * var(truth; corrected=false),
        )
    end
end

function coefficient_field_error(coefficients, terms)
    sqrt(max(
        dot(coefficients, terms.quadratic * coefficients) +
            2dot(coefficients, terms.linear) + terms.constant,
        0.0,
    ) / terms.normalization)
end

function capture_agent_errors!(history, information, terms, step, contacts)
    push!(history[:steps], step)
    foreach(history[:ids]) do aid
        coefficients = posterior_coefficient_moments(information[aid])[:μ]
        push!(
            history[:errors][aid],
            coefficient_field_error(coefficients, terms),
        )
    end
    foreach(contacts) do pair
        foreach(pair) do aid
            push!(history[:communication_steps][aid], step)
        end
    end
    history
end

function inter_agent_disagreement(fields, truth)
    ids = sort(collect(keys(fields)))
    maximum(
        normalized_field_error(fields[ids[left]], fields[ids[right]])
        for left in 1:(length(ids) - 1)
        for right in (left + 1):length(ids)
    )
end

function trial_metrics(
    model,
    truth,
    scribe_information,
    independent_information,
    central,
)
    scribe_fields = posterior_fields(model, scribe_information)
    independent_fields = posterior_fields(model, independent_information)
    central_field = reconstruct_eof_field(
        model;
        coefficients=posterior_coefficient_moments(central)[:μ],
    )
    scribe_errors = Dict(
        aid => normalized_field_error(field, truth)
        for (aid, field) in scribe_fields
    )
    Dict(
        :team_nrmse => mean(values(scribe_errors)),
        :worst_nrmse => maximum(values(scribe_errors)),
        :disagreement => inter_agent_disagreement(scribe_fields, truth),
        :centralized_gap => mean(
            normalized_field_error(field, central_field)
            for field in values(scribe_fields)
        ),
        :independent_nrmse => mean(
            normalized_field_error(field, truth)
            for field in values(independent_fields)
        ),
        :centralized_nrmse => normalized_field_error(central_field, truth),
        :agent_nrmse => scribe_errors,
    )
end

function metric_row(metrics, seed, n_agents, step, communication)
    Dict(
        :seed => seed,
        :n_agents => n_agents,
        :step => step,
        :observations_per_agent => step,
        :total_observations => n_agents * step,
        :team_nrmse => metrics[:team_nrmse],
        :worst_nrmse => metrics[:worst_nrmse],
        :disagreement => metrics[:disagreement],
        :centralized_gap => metrics[:centralized_gap],
        :independent_nrmse => metrics[:independent_nrmse],
        :centralized_nrmse => metrics[:centralized_nrmse],
        :contacts => communication[:contacts],
        :messages => communication[:messages],
        :bytes => communication[:bytes],
        :consensus_iterations => communication[:iterations],
    )
end

function coefficient_snapshot(information, ids)
    reduce(
        hcat,
        (posterior_coefficient_moments(information[aid])[:μ] for aid in ids),
    )
end

function capture_frame!(history, step, information, contacts)
    push!(history[:steps], step)
    push!(
        history[:coefficients],
        coefficient_snapshot(information, history[:ids]),
    )
    push!(history[:contacts], copy(contacts))
    empty!(contacts)
end

function run_asynchronous_trial(
    environment,
    settings,
    n_agents,
    n_steps,
    seed;
    capture_history=false,
)
    let params=environment[:params],
        model=environment[:model],
        roms=environment[:roms],
        truth=environment[:truth],
        ids=agent_ids(n_agents),
        regions=operational_regions(roms[:grid_shape], n_agents),
        paths=regional_paths(roms, regions),
        plan=observation_plan(
            model,
            truth,
            paths,
            n_steps,
            settings[:noise_variance],
            seed,
        ),
        observations=plan[:observations],
        prior=SCRIBE.init_agent_info(params),
        network=initialize_asynchronous_network(
            params,
            prior,
            observations,
            settings[:noise_variance],
        ),
        independent=Dict(aid => copy(prior) for aid in ids),
        adjacency=adjacent_agent_pairs(regions),
        grid_locations=wet_grid_locations(roms),
        metric_steps=Set(round.(Int, range(
            0,
            n_steps;
            length=min(settings[:metric_points], n_steps + 1),
        ))),
        animation_steps=Set(round.(Int, range(
            0,
            n_steps;
            length=min(settings[:animation_frames], n_steps + 1),
        ))),
        history=Dict(
            :ids => ids,
            :steps => Int[],
            :coefficients => Matrix{Float64}[],
            :contacts => Vector{Tuple{String, String}}[],
        ),
        error_terms=field_error_terms(model, truth),
        agent_error_history=Dict(
            :ids => ids,
            :steps => Int[],
            :errors => Dict(aid => Float64[] for aid in ids),
            :communication_steps => Dict(aid => Int[] for aid in ids),
        ),
        pending_contacts=Tuple{String, String}[],
        contact_log=Any[],
        rows=Any[],
        communication=Dict(
            :contacts => 0,
            :messages => 0,
            :bytes => 0,
            :iterations => 0,
        )

        central = copy(prior)
        active_contacts = Set{Tuple{String, String}}()
        initial_metrics = trial_metrics(
            model,
            truth,
            network_information(network),
            independent,
            central,
        )
        push!(rows, metric_row(
            initial_metrics,
            seed,
            n_agents,
            0,
            communication,
        ))
        capture_history && capture_frame!(
            history,
            0,
            network_information(network),
            pending_contacts,
        )
        capture_history && capture_agent_errors!(
            agent_error_history,
            network_information(network),
            error_terms,
            0,
            Tuple{String, String}[],
        )

        foreach(1:n_steps) do step
            states = Dict(
                aid => vec(grid_locations[
                    observations[aid][step][:row],
                    :,
                ])
                for aid in ids
            )
            edges, active_contacts, contacts = contact_edges(
                states,
                adjacency,
                active_contacts,
                settings[:communication_radius],
            )
            append!(pending_contacts, contacts)
            foreach(contacts) do (left, right)
                push!(contact_log, (
                    seed=seed,
                    n_agents=n_agents,
                    step=step,
                    left=left,
                    right=right,
                ))
            end
            central = centralized_update(
                central,
                observations,
                step,
                params.Q,
            )
            foreach(ids) do aid
                independent[aid] = local_information_update(
                    independent[aid],
                    observations[aid][step],
                    params.Q,
                )
            end
            step_communication = asynchronous_network_step!(
                network,
                step,
                edges,
                settings,
            )
            communication[:contacts] += length(contacts)
            foreach((:messages, :bytes, :iterations)) do metric
                communication[metric] += step_communication[metric]
            end
            capture_history && capture_agent_errors!(
                agent_error_history,
                network_information(network),
                error_terms,
                step,
                contacts,
            )

            if step in metric_steps
                metrics = trial_metrics(
                    model,
                    truth,
                    network_information(network),
                    independent,
                    central,
                )
                push!(rows, metric_row(
                    metrics,
                    seed,
                    n_agents,
                    step,
                    communication,
                ))
            end
            if capture_history && step in animation_steps
                capture_frame!(
                    history,
                    step,
                    network_information(network),
                    pending_contacts,
                )
            end
        end

        final_information = network_information(network)
        final_metrics = trial_metrics(
            model,
            truth,
            final_information,
            independent,
            central,
        )
        Dict(
            :checkpoint_version => ASYNCHRONOUS_CHECKPOINT_VERSION,
            :rows => rows,
            :seed => seed,
            :n_agents => n_agents,
            :mission_steps => n_steps,
            :regions => regions,
            :paths => paths,
            :sample_rows => Dict(
                aid => getindex.(observations[aid], :row)
                for aid in ids
            ),
            :offsets => plan[:offsets],
            :final_coefficients => Dict(
                aid => posterior_coefficient_moments(info)[:μ]
                for (aid, info) in final_information
            ),
            :final_covariances => Dict(
                aid => posterior_coefficient_moments(info)[:Σ]
                for (aid, info) in final_information
            ),
            :final_metrics => final_metrics,
            :communication => communication,
            :contact_log => contact_log,
            :history => capture_history ? history : nothing,
            :agent_error_history => capture_history ?
                agent_error_history : nothing,
        )
    end
end

function write_contact_history(trials, output_path)
    open(output_path, "w") do io
        println(io, "seed,n_agents,step,left,right")
        foreach(trials) do trial
            foreach(trial[:contact_log]) do contact
                println(io, join((
                    contact.seed,
                    contact.n_agents,
                    contact.step,
                    contact.left,
                    contact.right,
                ), ','))
            end
        end
    end
end

function write_final_agent_metrics(trials, output_path)
    open(output_path, "w") do io
        println(io, "seed,n_agents,agent,nrmse")
        foreach(trials) do trial
            foreach(sort(collect(keys(trial[:final_metrics][:agent_nrmse])))) do aid
                println(io, join((
                    trial[:seed],
                    trial[:n_agents],
                    aid,
                    trial[:final_metrics][:agent_nrmse][aid],
                ), ','))
            end
        end
    end
end

function write_rows(rows, output_path)
    fields = (
        :seed,
        :n_agents,
        :step,
        :observations_per_agent,
        :total_observations,
        :team_nrmse,
        :worst_nrmse,
        :disagreement,
        :centralized_gap,
        :independent_nrmse,
        :centralized_nrmse,
        :contacts,
        :messages,
        :bytes,
        :consensus_iterations,
    )
    open(output_path, "w") do io
        println(io, join(string.(fields), ','))
        foreach(rows) do row
            println(io, join((row[field] for field in fields), ','))
        end
    end
end

function final_summary(trials)
    map(sort(unique(getindex.(trials, :n_agents)))) do n_agents
        selected = filter(trial -> trial[:n_agents] == n_agents, trials)
        values(metric) = [
            trial[:final_metrics][metric]
            for trial in selected
        ]
        Dict(
            :n_agents => n_agents,
            :trials => length(selected),
            :scribe_nrmse => median(values(:team_nrmse)),
            :scribe_worst_nrmse => median(values(:worst_nrmse)),
            :disagreement => median(values(:disagreement)),
            :centralized_gap => median(values(:centralized_gap)),
            :independent_nrmse => median(values(:independent_nrmse)),
            :centralized_nrmse => median(values(:centralized_nrmse)),
            :contacts => median([
                trial[:communication][:contacts]
                for trial in selected
            ]),
            :messages => median([
                trial[:communication][:messages]
                for trial in selected
            ]),
            :bytes => median([
                trial[:communication][:bytes]
                for trial in selected
            ]),
        )
    end
end

function write_final_summary(trials, output_path)
    summary = final_summary(trials)
    fields = (
        :n_agents,
        :trials,
        :scribe_nrmse,
        :scribe_worst_nrmse,
        :disagreement,
        :centralized_gap,
        :independent_nrmse,
        :centralized_nrmse,
        :contacts,
        :messages,
        :bytes,
    )
    open(output_path, "w") do io
        println(io, join(string.(fields), ','))
        foreach(summary) do row
            println(io, join((row[field] for field in fields), ','))
        end
    end
end

function curve_summary(rows, n_agents)
    selected = filter(row -> row[:n_agents] == n_agents, rows)
    map(sort(unique(getindex.(selected, :step)))) do step
        step_rows = filter(row -> row[:step] == step, selected)
        values = getindex.(step_rows, :team_nrmse)
        (
            step=step,
            median=median(values),
        )
    end
end

function rmse_panel(rows; render_profile=:paper, smoke=false)
    style = workshop_plot_style(render_profile)
    team_sizes = sort(unique(getindex.(rows, :n_agents)))
    gradient = cgrad(:blues)
    colors = [
        get(gradient, shade)
        for shade in range(0.44, 0.92; length=length(team_sizes))
    ]
    panel = plot(
        ;
        xlabel="Elapsed timesteps",
        ylabel="Whole-domain NRMSE",
        title=smoke ?
            "(a) Team RMSE comparison (smoke check)" :
            "(a) Team RMSE comparison",
        grid=true,
        legend=:outertop,
        legend_columns=min(4, length(team_sizes)),
    )
    foreach(zip(team_sizes, colors)) do (n_agents, color)
        curve = curve_summary(rows, n_agents)
        plot!(
            panel,
            getindex.(curve, :step),
            getindex.(curve, :median);
            color,
            linewidth=style.linewidth,
            label="n = $n_agents",
        )
    end
    workshop_panel!(
        panel;
        profile=render_profile,
        bottom_margin=9Plots.mm,
    )
end

function selected_asynchronous_trial(trials)
    recorded = filter(
        trial -> trial[:n_agents] >= 6 &&
            !isnothing(trial[:agent_error_history]),
        trials,
    )
    argmin(recorded) do trial
        trial[:final_metrics][:team_nrmse]
    end
end

function agent_rmse_panel(trial; render_profile=:paper)
    style = workshop_plot_style(render_profile)
    history = trial[:agent_error_history]
    ids = history[:ids]
    gradient = cgrad(:viridis)
    colors = [
        get(gradient, shade)
        for shade in range(0.08, 0.92; length=length(ids))
    ]
    panel = plot(
        ;
        xlabel="Elapsed timesteps",
        ylabel="Agent NRMSE",
        title="(b) Agent RMSE, n=$(trial[:n_agents]) (communication dots)",
        grid=true,
        legend=:outertop,
        legend_columns=-1,
        ylims=(0.38, 1.1),
        yticks=0.4:0.2:1.0,
    )
    foreach(zip(ids, colors)) do (aid, color)
        plot!(
            panel,
            history[:steps],
            history[:errors][aid];
            color,
            linewidth=style.linewidth,
            label=replace(aid, "agent" => "A"),
        )
        communication_steps = history[:communication_steps][aid]
        scatter!(
            panel,
            communication_steps,
            history[:errors][aid][communication_steps .+ 1];
            color,
            marker=:circle,
            markersize=style.markersize,
            markerstrokecolor=:black,
            markerstrokewidth=0.8,
            label=false,
        )
    end
    workshop_panel!(
        panel;
        profile=render_profile,
        left_margin=6Plots.mm,
        right_margin=3Plots.mm,
        bottom_margin=9Plots.mm,
        top_margin=4Plots.mm,
    )
    foreach(enumerate(zip(ids, colors))) do (index, (aid, color))
        plot!(
            panel,
            history[:steps],
            history[:errors][aid];
            inset=index == 1 ? (
                1,
                bbox(0.04, 0.25, 0.46, 0.43, :top, :right),
            ) : nothing,
            subplot=2,
            color,
            linewidth=max(style.linewidth - 1.0, 1.5),
            label=false,
        )
    end
    plot!(
        panel;
        subplot=2,
        xlims=(0, 650),
        ylims=(0.38, 4.1),
        xticks=([0, 300, 600], ["0", "300", "600"]),
        yticks=([0.5, 2.0, 4.0], ["0.5", "2", "4"]),
        title="Initial transient",
        titlefontsize=max(style.tickfontsize - 2, 10),
        tickfontsize=max(style.tickfontsize - 4, 9),
        grid=true,
        gridalpha=0.13,
        foreground_color_grid=:gray75,
        background_color_inside=:white,
        legend=false,
        margin=1Plots.mm,
    )
    panel
end

function region_outline!(panel, region; color=:yellow, linewidth=2.0)
    x₀, x₁ = first(region[:rows]) - 0.5, last(region[:rows]) + 0.5
    y₀, y₁ = first(region[:columns]) - 0.5, last(region[:columns]) + 0.5
    plot!(
        panel,
        [x₀, x₁, x₁, x₀, x₀],
        [y₀, y₀, y₁, y₁, y₀];
        color,
        linewidth,
        label=false,
    )
    panel
end

function selected_reconstruction(environment, trial)
    errors = trial[:final_metrics][:agent_nrmse]
    team_median = median(collect(values(errors)))
    aid = argmin(collect(keys(errors))) do candidate
        abs(errors[candidate] - team_median)
    end
    coefficients = trial[:final_coefficients][aid]
    Dict(
        :trial => trial,
        :agent => aid,
        :field => reconstruct_eof_field(
            environment[:model];
            coefficients,
        ),
        :nrmse => errors[aid],
    )
end

function sampling_trajectory(environment, trial, aid)
    wet_grid_locations(environment[:roms])[trial[:sample_rows][aid], :]
end

function sampling_trajectory!(panel, trajectory, aid; render_profile=:paper)
    style = workshop_plot_style(render_profile)
    plot!(
        panel,
        trajectory[:, 1],
        trajectory[:, 2];
        color="#CC79A7",
        linewidth=max(style.linewidth - 0.6, 1.5),
        alpha=0.72,
        label=false,
    )
    scatter!(
        panel,
        [trajectory[1, 1]],
        [trajectory[1, 2]];
        color="#CC79A7",
        marker=:circle,
        markersize=style.markersize,
        markerstrokecolor=:black,
        label=false,
    )
    scatter!(
        panel,
        [trajectory[end, 1]],
        [trajectory[end, 2]];
        color="#F0E442",
        marker=:diamond,
        markersize=style.markersize,
        markerstrokecolor=:black,
        label=false,
    )
    panel
end

function roms_coordinate_ticks(roms; n_ticks=3)
    longitude = fill(NaN, roms[:grid_shape]...)
    latitude = fill(NaN, roms[:grid_shape]...)
    longitude[roms[:wet_mask]] = roms[:locations][:, 1]
    latitude[roms[:wet_mask]] = roms[:locations][:, 2]
    x_ticks = round.(Int, range(
        0.2roms[:grid_shape][1],
        0.8roms[:grid_shape][1];
        length=n_ticks,
    ))
    y_ticks = round.(Int, range(
        0.2roms[:grid_shape][2],
        0.8roms[:grid_shape][2];
        length=n_ticks,
    ))
    (
        x=(x_ticks, string.(round.([
            median(filter(isfinite, longitude[row, :]))
            for row in x_ticks
        ]; digits=4))),
        y=(y_ticks, string.(round.([
            median(filter(isfinite, latitude[:, column]))
            for column in y_ticks
        ]; digits=4))),
    )
end

function spatial_panel(
    values,
    environment,
    title,
    limit;
    render_profile=:paper,
    left_margin=7Plots.mm,
    right_margin=2Plots.mm,
    colorbar=false,
    show_latitude=true,
)
    style = workshop_plot_style(render_profile)
    has_colorbar = colorbar != false
    ticks = roms_coordinate_ticks(environment[:roms])
    panel = plot_roms_field(
        values,
        environment[:roms];
        title,
        color=:balance,
        clims=(-limit, limit),
    )
    plot!(
        panel;
        axis=true,
        colorbar,
        colorbar_title=has_colorbar ? "m s⁻¹" : "",
        colorbar_titlefontsize=style.guidefontsize,
        xlabel="Longitude (°)",
        ylabel=show_latitude ? "Latitude (°)" : "",
        xticks=ticks.x,
        yticks=show_latitude ? ticks.y : false,
        legend=false,
    )
    workshop_panel!(
        panel;
        profile=render_profile,
        left_margin,
        right_margin,
        bottom_margin=6Plots.mm,
        top_margin=5Plots.mm,
    )
end

function save_asynchronous_figure(
    environment,
    rows,
    trials,
    settings,
    output_dir;
    smoke=false,
)
    style = workshop_plot_style(:poster)
    trial = selected_asynchronous_trial(trials)
    example = selected_reconstruction(environment, trial)
    aid = example[:agent]
    truth = environment[:truth]
    limit = maximum(abs, truth)
    truth_panel = spatial_panel(
        truth,
        environment,
        "(c) Ground truth",
        limit;
        render_profile=:poster,
        left_margin=12Plots.mm,
        right_margin=0Plots.mm,
        colorbar=true,
    )
    reconstruction_panel = spatial_panel(
        example[:field],
        environment,
        "(d) Final SCRIBE posterior",
        limit;
        render_profile=:poster,
        left_margin=0Plots.mm,
        right_margin=4Plots.mm,
        colorbar=true,
        show_latitude=false,
    )
    sampling_trajectory!(
        reconstruction_panel,
        sampling_trajectory(environment, trial, aid),
        aid;
        render_profile=:poster,
    )
    plot!(reconstruction_panel; legend=false)
    grid_shape = environment[:roms][:grid_shape]
    annotation = "n=$(trial[:n_agents]), seed=$(trial[:seed])\n" *
        "agent NRMSE=$(round(example[:nrmse]; digits=3))\n" *
        "team NRMSE=$(round(trial[:final_metrics][:team_nrmse]; digits=3))\n" *
        "pink path: $aid samples"
    annotate!(
        reconstruction_panel,
        0.96grid_shape[1],
        0.07grid_shape[2],
        text(
            annotation,
            style.annotationfontsize,
            :right,
            :bottom,
            :white,
        ),
    )
    figure = plot(
        rmse_panel(rows; render_profile=:poster, smoke),
        agent_rmse_panel(trial; render_profile=:poster),
        truth_panel,
        reconstruction_panel;
        layout=grid(2, 2; heights=[0.46, 0.54]),
        size=(1900, 1180),
        plot_title="Asynchronous SCRIBE Model Fusion During Continuous Sensing",
        plot_titlefontsize=style.titlefontsize,
    )
    output_base = joinpath(output_dir, "asynchronous_model_fusion_poster")
    output_paths = save_workshop_figure(figure, output_base)
    open(joinpath(output_dir, "illustrative_trial.txt"), "w") do io
        println(io, "seed=$(trial[:seed])")
        println(io, "n_agents=$(trial[:n_agents])")
        println(io, "agent=$aid")
        println(io, "agent_nrmse=$(example[:nrmse])")
        println(io, "team_nrmse=$(trial[:final_metrics][:team_nrmse])")
        println(io, "selection=lowest team NRMSE among n >= 6; agent nearest team-median NRMSE")
    end
    output_paths
end

function posterior_animation_panel(
    environment,
    trial,
    coefficients,
    aid,
    step,
    contacts,
    limit,
)
    index = findfirst(==(aid), trial[:history][:ids])
    field = reconstruct_eof_field(
        environment[:model];
        coefficients=view(coefficients, :, index),
    )
    partners = [
        first(pair) == aid ? last(pair) : first(pair)
        for pair in contacts
        if aid in pair
    ]
    title = isempty(partners) ?
        "$aid, observation $step" :
        "$aid ↔ $(join(partners, ", "))"
    panel = plot_roms_field(
        field,
        environment[:roms];
        title,
        color=:balance,
        clims=(-limit, limit),
    )
    region_outline!(
        panel,
        trial[:regions][agent_number(aid)];
        color=:black,
        linewidth=1.2,
    )
    plot!(
        panel;
        colorbar=false,
        foreground_color_subplot=isempty(partners) ? :black : :firebrick,
        titlefontcolor=isempty(partners) ? :black : :firebrick,
        margin=0.5Plots.mm,
    )
    panel
end

function save_asynchronous_animation(environment, trial, settings, output_dir)
    history = trial[:history]
    limit = maximum(abs, environment[:truth])
    n_agents = trial[:n_agents]
    n_columns = ceil(Int, sqrt(n_agents))
    n_rows = ceil(Int, n_agents / n_columns)
    animation = Animation()
    foreach(eachindex(history[:steps])) do frame_index
        panels = [
            posterior_animation_panel(
                environment,
                trial,
                history[:coefficients][frame_index],
                aid,
                history[:steps][frame_index],
                history[:contacts][frame_index],
                limit,
            )
            for aid in history[:ids]
        ]
        figure = plot(
            panels...;
            layout=(n_rows, n_columns),
            size=(340n_columns, 300n_rows),
            plot_title="Asynchronous SCRIBE model propagation",
            titlefontsize=11,
            plot_titlefontsize=14,
            margin=1Plots.mm,
        )
        frame(animation, figure)
    end
    output_path = joinpath(output_dir, "asynchronous_model_fusion.gif")
    gif(animation, output_path; fps=settings[:animation_fps])
    println("Animation saved at $output_path")
    output_path
end

function write_experiment_metadata(environment, settings, n_steps, output_path)
    open(output_path, "w") do io
        println(io, "truth_snapshot=$(ASYNCHRONOUS_TRUTH_SNAPSHOT)")
        println(io, "eof_rank=$(environment[:params].nᵩ)")
        println(io, "truth_coefficients=$(join(environment[:truth_coefficients], ','))")
        println(io, "standardized_truth_coefficients=" *
            join(environment[:standardized_coefficients], ','))
        println(io, "mission_steps=$n_steps")
        println(io, "team_sizes=$(join(settings[:team_sizes], ','))")
        println(io, "seeds=$(join(settings[:seeds], ','))")
        println(io, "communication_radius=$(settings[:communication_radius])")
        println(io, "sensor_variance=$(settings[:noise_variance])")
    end
end

asynchronous_output_directory(profile) = joinpath(
    @__DIR__,
    "res",
    "asynchronous_model_fusion",
    String(profile),
)

asynchronous_condition_name(n_agents, seed) =
    "n_$(lpad(n_agents, 2, '0'))_seed_$seed"

asynchronous_checkpoint_path(output_dir, n_agents, seed) = joinpath(
    output_dir,
    "checkpoints",
    asynchronous_condition_name(n_agents, seed) * ".jls",
)

function atomic_asynchronous_serialize(output_path, value)
    mkpath(dirname(output_path))
    temporary = output_path * ".tmp.$(getpid())"
    open(temporary, "w") do io
        serialize(io, value)
    end
    mv(temporary, output_path; force=true)
    output_path
end

function asynchronous_checkpoint_matches(trial, n_agents, seed, n_steps)
    get(trial, :checkpoint_version, 0) == ASYNCHRONOUS_CHECKPOINT_VERSION &&
        trial[:n_agents] == n_agents &&
        trial[:seed] == seed &&
        trial[:mission_steps] == n_steps
end

function load_asynchronous_trials(settings, output_dir, n_steps)
    reduce(vcat, map(settings[:team_sizes]) do n_agents
        reduce(vcat, map(settings[:seeds]) do seed
            path = asynchronous_checkpoint_path(output_dir, n_agents, seed)
            if isfile(path)
                trial = open(deserialize, path)
                asynchronous_checkpoint_matches(
                    trial,
                    n_agents,
                    seed,
                    n_steps,
                ) ? [trial] : Any[]
            else
                Any[]
            end
        end; init=Any[])
    end; init=Any[])
end

function write_asynchronous_progress(settings, output_dir, trials)
    completed = Set(
        (trial[:n_agents], trial[:seed])
        for trial in trials
    )
    open(joinpath(output_dir, "progress.csv"), "w") do io
        println(io, "n_agents,seed,status")
        foreach(settings[:team_sizes]) do n_agents
            foreach(settings[:seeds]) do seed
                status = (n_agents, seed) in completed ? "complete" : "pending"
                println(io, "$n_agents,$seed,$status")
            end
        end
    end
end

function asynchronous_plot_context(environment, n_steps)
    Dict(
        :truth => environment[:truth],
        :model => environment[:model],
        :roms => Dict(
            :grid_shape => environment[:roms][:grid_shape],
            :wet_mask => environment[:roms][:wet_mask],
            :locations => environment[:roms][:locations],
        ),
        :truth_coefficients => environment[:truth_coefficients],
        :standardized_coefficients => environment[:standardized_coefficients],
        :mission_steps => n_steps,
        :truth_snapshot => ASYNCHRONOUS_TRUTH_SNAPSHOT,
    )
end

function aggregate_asynchronous_results(
    environment,
    trials,
    settings,
    n_steps,
    output_dir,
)
    rows = reduce(vcat, getindex.(trials, :rows); init=Any[])
    write_rows(rows, joinpath(output_dir, "metric_history.csv"))
    write_final_summary(trials, joinpath(output_dir, "final_summary.csv"))
    write_contact_history(trials, joinpath(output_dir, "contact_history.csv"))
    write_final_agent_metrics(
        trials,
        joinpath(output_dir, "final_agent_metrics.csv"),
    )
    write_experiment_metadata(
        environment,
        settings,
        n_steps,
        joinpath(output_dir, "experiment_metadata.txt"),
    )
    rows
end

function print_asynchronous_summary(environment, trials, output_dir)
    println("Asynchronous EOF fusion experiment complete.")
    println(
        "  snapshot $(ASYNCHRONOUS_TRUTH_SNAPSHOT), standardized " *
        "coefficients $(round.(environment[:standardized_coefficients]; digits=3))",
    )
    foreach(final_summary(trials)) do row
        println(
            "  n=$(row[:n_agents]): SCRIBE NRMSE=" *
            "$(round(row[:scribe_nrmse]; digits=4)), " *
            "no communication=$(round(row[:independent_nrmse]; digits=4)), " *
            "centralized=$(round(row[:centralized_nrmse]; digits=4)), " *
            "disagreement=$(round(row[:disagreement]; digits=4)), " *
            "contacts=$(row[:contacts])",
        )
    end
    println("  results: $output_dir")
end

function run_asynchronous_model_fusion(;
    profile=:full,
    make_animation=true,
    make_plots=true,
    resume=true,
)
    let settings=asynchronous_fusion_settings(profile),
        environment=load_asynchronous_environment(),
        n_steps=experiment_mission_steps(settings, environment[:roms]),
        output_dir=asynchronous_output_directory(profile)

        mkpath(joinpath(output_dir, "checkpoints"))
        atomic_asynchronous_serialize(
            joinpath(output_dir, "plot_context.jls"),
            asynchronous_plot_context(environment, n_steps),
        )
        saved_trials = resume ?
            load_asynchronous_trials(settings, output_dir, n_steps) : Any[]
        trials = Dict(
            (trial[:n_agents], trial[:seed]) => trial
            for trial in saved_trials
        )
        println(
            "Running fixed $(n_steps)-observation missions for team sizes " *
            "$(join(settings[:team_sizes], ", ")).",
        )
        foreach(settings[:team_sizes]) do n_agents
            foreach(settings[:seeds]) do seed
                capture = n_agents == settings[:representative_team_size] &&
                    seed == settings[:representative_seed]
                key = (n_agents, seed)
                reusable = haskey(trials, key) &&
                    (!capture || (
                        !isnothing(trials[key][:history]) &&
                        haskey(trials[key], :agent_error_history) &&
                        !isnothing(trials[key][:agent_error_history])
                    ))
                trial = reusable ? trials[key] : run_asynchronous_trial(
                        environment,
                        settings,
                        n_agents,
                        n_steps,
                        seed;
                        capture_history=capture,
                    )
                if !reusable
                    trials[key] = trial
                    atomic_asynchronous_serialize(
                        asynchronous_checkpoint_path(
                            output_dir,
                            n_agents,
                            seed,
                        ),
                        trial,
                    )
                    write_asynchronous_progress(
                        settings,
                        output_dir,
                        collect(values(trials)),
                    )
                end
                println(
                    "  n=$n_agents, seed=$seed" *
                    (reusable ? " (cached)" : "") * ": final NRMSE=" *
                    "$(round(trial[:final_metrics][:team_nrmse]; digits=4)), " *
                    "contacts=$(trial[:communication][:contacts])",
                )
                flush(stdout)
                GC.gc()
            end
        end

        ordered_trials = [
            trials[(n_agents, seed)]
            for n_agents in settings[:team_sizes]
            for seed in settings[:seeds]
        ]
        rows = aggregate_asynchronous_results(
            environment,
            ordered_trials,
            settings,
            n_steps,
            output_dir,
        )
        make_plots && save_asynchronous_figure(
            environment,
            rows,
            ordered_trials,
            settings,
            output_dir;
            smoke=profile == :smoke,
        )
        if make_animation
            representative = trials[(
                settings[:representative_team_size],
                settings[:representative_seed],
            )]
            save_asynchronous_animation(
                environment,
                representative,
                settings,
                output_dir,
            )
        end
        print_asynchronous_summary(environment, ordered_trials, output_dir)
        Dict(:rows => rows, :trials => ordered_trials, :settings => settings)
    end
end

function plot_saved_asynchronous_model_fusion(; profile=:full)
    settings = asynchronous_fusion_settings(profile)
    output_dir = asynchronous_output_directory(profile)
    environment = open(
        deserialize,
        joinpath(output_dir, "plot_context.jls"),
    )
    trials = load_asynchronous_trials(
        settings,
        output_dir,
        environment[:mission_steps],
    )
    rows = reduce(vcat, getindex.(trials, :rows); init=Any[])
    save_asynchronous_figure(
        environment,
        rows,
        trials,
        settings,
        output_dir;
        smoke=profile == :smoke,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    let profile=isempty(ARGS) ? :full : Symbol(first(ARGS)),
        make_animation="no_animation" ∉ ARGS,
        make_plots="no_plots" ∉ ARGS,
        resume="rerun" ∉ ARGS
        "plots_only" in ARGS ?
            plot_saved_asynchronous_model_fusion(; profile) :
            run_asynchronous_model_fusion(;
                profile,
                make_animation,
                make_plots,
                resume,
            )
    end
end
