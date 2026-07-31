using SCRIBE

using LinearAlgebra
using Match: @match
using Random
using Statistics

abstract type PublicationFusion end

struct SCRIBEFusion <: PublicationFusion end
struct IndependentFusion <: PublicationFusion end
struct KalmanFilterOnlyFusion <: PublicationFusion end
struct CovarianceIntersectionOnlyFusion <: PublicationFusion end

fusion_name(::SCRIBEFusion) = :scribe
fusion_name(::IndependentFusion) = :independent
fusion_name(::KalmanFilterOnlyFusion) = :kf_only
fusion_name(::CovarianceIntersectionOnlyFusion) = :ci_only

"""A minimal information-filter state used by the publication experiments."""
mutable struct PublicationScribe <: EnvScribe
    k
    information
end

"""Deterministic estimator inputs consumed by SCRIBE's consensus implementation."""
struct PublicationEstimators <: EnvEstimators
    system
    A
    Q
    observations
end

function inverse_positive_definite(M)
    let factor=cholesky(Symmetric((M + M') / 2))
        Matrix(factor \ Matrix{Float64}(I, size(M)...))
    end
end

function publication_prior(info, A, Q)
    let P=inverse_positive_definite(info.Y),
        μ=P * info.y,
        P⁻=(A * P * A' + Q),
        Y⁻=inverse_positive_definite(P⁻),
        y⁻=Y⁻ * A * μ
        ((Y⁻ + Y⁻') / 2, y⁻)
    end
end

function SCRIBE.compute_info_priors(
    estimators::PublicationEstimators,
    k::Integer,
)
    publication_prior(
        estimators.system.information[k],
        estimators.A,
        estimators.Q,
    )
end

function SCRIBE.compute_innov_from_obs(
    estimators::PublicationEstimators,
    k::Integer,
)
    let observation=estimators.observations[k],
        innovation=measurement_information(
            observation.H,
            observation.z,
            observation.R,
        )
        (innovation.δI, innovation.δi)
    end
end

function SCRIBE.next_agent_info_state(
    system::PublicationScribe,
    info::KFEnvInfo,
)
    push!(system.information, info)
end

publication_settings(profile) = @match profile begin
    :full => (
        seeds=Tuple(101:2:159),
        n_agents=4,
        n_steps=24,
        phase_steps=(8, 8, 8),
        communication_radius=8.25,
        basis_points=5,
        evaluation_points=31,
        noise_variance=0.04,
        process_variance=1e-6,
        consensus_threshold=1e-7,
        consensus_timeline=180,
    )
    :smoke => (
        seeds=(101, 103),
        n_agents=4,
        n_steps=6,
        phase_steps=(2, 2, 2),
        communication_radius=8.25,
        basis_points=4,
        evaluation_points=15,
        noise_variance=0.04,
        process_variance=1e-6,
        consensus_threshold=1e-6,
        consensus_timeline=120,
    )
    _ => throw(ArgumentError("Use the `full` or `smoke` publication profile."))
end

function publication_locations(n_points)
    let axis=collect(range(-5.0, 5.0; length=n_points))
        reduce(vcat, ([x y] for y in axis for x in axis))
    end
end

function publication_parameters(settings)
    let centers=publication_locations(settings.basis_points),
        nᵩ=size(centers, 1)
        LGSFModelParameters(
            μ=centers,
            σ=[2.1],
            τ=[1.0],
            ϕ₀=zeros(nᵩ),
            A=Matrix{Float64}(I, nᵩ, nᵩ),
            Q=settings.process_variance *
                Matrix{Float64}(I, nᵩ, nᵩ),
        )
    end
end

function coefficient_covariance(centers)
    let Δ²=[
            sum(abs2, μᵢ - μⱼ)
            for μᵢ in eachrow(centers), μⱼ in eachrow(centers)
        ],
        correlation=exp.(-Δ² / (2 * 2.2^2)),
        nᵩ=size(centers, 1)
        1.4 * correlation +
            1e-5 * Matrix{Float64}(I, nᵩ, nᵩ)
    end
end

function information_from_covariance(covariance)
    let Y=inverse_positive_definite(covariance),
        nᵩ=size(covariance, 1)
        KFEnvInfo(
            zeros(nᵩ),
            Y,
            zeros(nᵩ),
            zeros(nᵩ, nᵩ),
        )
    end
end

function publication_prior_information(smodel, evaluation_locations)
    let covariance=coefficient_covariance(smodel.params.p[:μ]),
        H=prediction_dynamics(smodel, evaluation_locations),
        represented_variance=vec(sum((H * covariance) .* H; dims=2)),
        scale=1.0 / mean(represented_variance)
        information_from_covariance(scale * covariance)
    end
end

function matched_coefficients(smodel, evaluation_locations)
    let coefficients=map(eachrow(smodel.params.p[:μ])) do μ
            let x=μ[1],
                y=μ[2]
                0.75 * sin(0.55 * x) * cos(0.45 * y) +
                    0.55 * exp(
                        -((x - 2.0)^2 + (y + 1.6)^2) / 4.0,
                    ) -
                    0.45 * exp(
                        -((x + 2.4)^2 + (y - 1.8)^2) / 3.0,
                    )
            end
        end,
        H=prediction_dynamics(smodel, evaluation_locations),
        field=H * coefficients
        coefficients / std(field)
    end
end

function misspecified_truth(X)
    map(eachrow(X)) do location
        let x=location[1],
            y=location[2],
            broad=0.75 * sin(0.62 * x) * cos(0.48 * y),
            local_structure=0.30 * sin(1.45 * x + 0.35 * y),
            positive=1.75 * exp(
                -((x + 2.35)^2 / 1.2 + (y - 2.0)^2 / 0.8),
            ),
            negative=-1.55 * exp(
                -((x - 2.15)^2 / 0.7 + (y + 1.8)^2 / 1.3),
            ),
            ridge=0.65 *
                exp(-(y - 0.8 * sin(0.9 * x))^2 / 0.22) *
                cos(0.5 * x)
            broad + local_structure + positive + negative + ridge
        end
    end
end

function truth_definition(kind, smodel, evaluation_locations)
    @match kind begin
        :matched => let coefficients=matched_coefficients(
                smodel,
                evaluation_locations,
            )
            (
                coefficients=coefficients,
                evaluate=X -> prediction_dynamics(smodel, X) * coefficients,
            )
        end
        :misspecified => (
            coefficients=nothing,
            evaluate=misspecified_truth,
        )
    end
end

function agent_ids(settings)
    ["agent$(i)" for i in 1:settings.n_agents]
end

function sample_location(agent_index, k, settings)
    let progress=(k - 1) / max(settings.n_steps - 1, 1),
        forward=isodd(agent_index),
        x=forward ? -4.6 + 9.2 * progress : 4.6 - 9.2 * progress,
        y_base=-4.2 +
            8.4 * (agent_index - 1) / max(settings.n_agents - 1, 1),
        y=clamp(
            y_base + 0.28 * sin(2π * progress + agent_index),
            -4.8,
            4.8,
        )
        reshape([x, y], 1, :)
    end
end

function observation_plan(
    smodel,
    truth,
    settings,
    seed,
)
    let rng=MersenneTwister(seed),
        R=reshape([settings.noise_variance], 1, 1)
        Dict(
            aid => map(1:settings.n_steps) do k
                let agent_index=parse(Int, replace(aid, "agent" => "")),
                    X=sample_location(agent_index, k, settings),
                    H=prediction_dynamics(smodel, X),
                    z=truth.evaluate(X) .+
                        sqrt(settings.noise_variance) .* randn(rng, 1)
                    (X=X, H=H, z=z, R=R)
                end
            end
            for aid in agent_ids(settings)
        )
    end
end

function complete_edges(ids)
    Dict(aid => filter(!=(aid), ids) for aid in ids)
end

function component_edges(components)
    reduce(merge, (
        Dict(aid => filter(!=(aid), component) for aid in component)
        for component in components
    ))
end

function split_edges(ids)
    let midpoint=length(ids) ÷ 2
        component_edges((ids[1:midpoint], ids[midpoint + 1:end]))
    end
end

function empty_edges(ids)
    Dict(aid => String[] for aid in ids)
end

function communication_phase(k, settings)
    let limited_end=settings.phase_steps[1],
        blackout_end=limited_end + settings.phase_steps[2]
        k ≤ limited_end ? :limited_communication :
        k ≤ blackout_end ? :communication_blackout :
        :limited_recovery
    end
end

function publication_distance_limited_matching(k, ids, settings)
    positions = Dict(
        aid => sample_location(
            parse(Int, replace(aid, "agent" => "")),
            k,
            settings,
        )
        for aid in ids
    )
    candidates = [
        (
            distance=norm(positions[ids[left]] - positions[ids[right]]),
            left=ids[left],
            right=ids[right],
        )
        for left in 1:(length(ids) - 1)
        for right in (left + 1):length(ids)
        if norm(positions[ids[left]] - positions[ids[right]]) ≤
            settings.communication_radius
    ]
    edges = empty_edges(ids)
    used = Set{String}()
    foreach(sort(candidates; by=item -> item.distance)) do item
        if item.left ∉ used && item.right ∉ used
            push!(edges[item.left], item.right)
            push!(edges[item.right], item.left)
            push!(used, item.left)
            push!(used, item.right)
        end
    end
    edges
end

function scheduled_edges(experiment, k, ids, settings)
    @match experiment begin
        1 => complete_edges(ids)
        2 => communication_phase(k, settings) == :communication_blackout ?
            empty_edges(ids) :
            publication_distance_limited_matching(k, ids, settings)
        3 => communication_phase(k, settings) == :communication_blackout ?
            empty_edges(ids) :
            publication_distance_limited_matching(k, ids, settings)
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

function initialize_publication_network(
    params,
    prior,
    observations,
    settings,
)
    let ids=agent_ids(settings),
        edges=complete_edges(ids),
        ng=init_network_graph(edges),
        observer=LGSFObserverBehavior(settings.noise_variance)

        foreach(ids) do aid
            let system=PublicationScribe(1, [copy(prior)]),
                estimators=PublicationEstimators(
                    system,
                    params.A,
                    params.w[:Q],
                    observations[aid],
                ),
                connector=NetworkConnector(
                    copy(edges[aid]),
                    network_outbox(),
                )
                ng.vertices[aid] = SCRIBEAgent(
                    aid,
                    params,
                    observer,
                    [copy(observations[aid][1].X)],
                    system,
                    connector,
                    estimators,
                )
            end
        end
        ng
    end
end

function connected_components(edges)
    let remaining=Set(keys(edges)),
        components=Vector{Vector{String}}()
        while !isempty(remaining)
            start=first(remaining)
            component=String[]
            frontier=[start]
            while !isempty(frontier)
                aid=pop!(frontier)
                aid ∉ remaining && continue
                delete!(remaining, aid)
                push!(component, aid)
                append!(frontier, edges[aid])
            end
            push!(components, sort(component))
        end
        components
    end
end

function payload_bytes(ng)
    let info=first(values(ng.vertices)).agent.information[end],
        scalar_count=length(info.Y) + length(info.y)
        sizeof(Float64) * scalar_count
    end
end

function deliver_publication_messages!(k, ng)
    let deliveries=deliver_consensus_messages(k, ng),
        messages=0
        foreach(deliveries) do delivery
            aid, message=delivery
            foreach(ng.edges[aid]) do neighbor
                consume_consensus_message!(
                    neighbor,
                    ng.vertices[neighbor].net_conn,
                    message,
                    k,
                    ng,
                )
                messages += 1
            end
        end
        foreach(keys(deliveries)) do aid
            ng.vertices[aid].net_conn.outbox["msg_out"] = nothing
        end
        messages
    end
end

function finish_publication_step!(ng)
    foreach(values(ng.vertices)) do vertex
        vertex.agent.k += 1
        full_reset_network_connector(vertex.net_conn)
    end
end

function fusion_step!(
    ::SCRIBEFusion,
    ng,
    k,
    edges,
    settings,
)
    update_network_graph_edges(edges, ng)
    messages=0
    iterations=0
    while true
        iterations += 1
        complete=map(agent_ids(settings)) do aid
            distributed_fusion(
                k,
                aid,
                ng,
                settings.consensus_threshold,
                settings.consensus_timeline,
            )
        end
        messages += deliver_publication_messages!(k, ng)
        all(complete) && break
    end
    finish_publication_step!(ng)
    (
        messages=messages,
        bytes=messages * payload_bytes(ng),
        iterations=iterations,
    )
end

function local_information_update(estimators, k)
    let (Y⁻, y⁻)=compute_info_priors(estimators, k),
        (δI, δi)=compute_innov_from_obs(estimators, k),
        Y=(Y⁻ + δI)
        KFEnvInfo(
            y⁻ + δi,
            (Y + Y') / 2,
            δi,
            δI,
        )
    end
end

function fusion_step!(
    ::IndependentFusion,
    ng,
    k,
    _,
    _settings,
)
    foreach(values(ng.vertices)) do vertex
        next_agent_info_state(
            vertex.agent,
            local_information_update(vertex.estimators, k),
        )
    end
    finish_publication_step!(ng)
    (messages=0, bytes=0, iterations=1)
end

function kalman_filter_component_update!(component, ng, k)
    innovations = map(component) do aid
        compute_innov_from_obs(ng.vertices[aid].estimators, k)
    end
    δI = sum(first, innovations)
    δi = sum(last, innovations)
    foreach(component) do aid
        Y⁻, y⁻ = compute_info_priors(ng.vertices[aid].estimators, k)
        Y = Y⁻ + δI
        next_agent_info_state(
            ng.vertices[aid].agent,
            KFEnvInfo(
                y⁻ + δi,
                (Y + Y') / 2,
                δi,
                δI,
            ),
        )
    end
end

function fusion_step!(
    ::KalmanFilterOnlyFusion,
    ng,
    k,
    edges,
    _settings,
)
    update_network_graph_edges(edges, ng)
    foreach(connected_components(edges)) do component
        kalman_filter_component_update!(component, ng, k)
    end
    messages=sum(length, values(edges))
    finish_publication_step!(ng)
    (
        messages=messages,
        bytes=messages * payload_bytes(ng),
        iterations=1,
    )
end

function covariance_intersection_component_update!(component, ng, k)
    local_posteriors = [
        local_information_update(ng.vertices[aid].estimators, k)
        for aid in component
    ]
    weights = SCRIBE.covariance_intersection_weights(
        [info.Y for info in local_posteriors],
    )
    Y = sum(
        weights[index] * local_posteriors[index].Y
        for index in eachindex(weights)
    )
    y = sum(
        weights[index] * local_posteriors[index].y
        for index in eachindex(weights)
    )
    δI = sum(
        weights[index] * local_posteriors[index].I
        for index in eachindex(weights)
    )
    δi = sum(
        weights[index] * local_posteriors[index].i
        for index in eachindex(weights)
    )
    fused = KFEnvInfo(y, (Y + Y') / 2, δi, (δI + δI') / 2)
    foreach(component) do aid
        next_agent_info_state(ng.vertices[aid].agent, copy(fused))
    end
end

function fusion_step!(
    ::CovarianceIntersectionOnlyFusion,
    ng,
    k,
    edges,
    _settings,
)
    update_network_graph_edges(edges, ng)
    foreach(connected_components(edges)) do component
        covariance_intersection_component_update!(component, ng, k)
    end
    messages=sum(length, values(edges))
    finish_publication_step!(ng)
    (
        messages=messages,
        bytes=messages * payload_bytes(ng),
        iterations=1,
    )
end

function centralized_update(info, plans, k, params)
    let (Y⁻, y⁻)=publication_prior(info, params.A, params.w[:Q]),
        innovations=map(values(plans)) do observations
            let observation=observations[k],
                innovation=measurement_information(
                    observation.H,
                    observation.z,
                    observation.R,
                )
                (innovation.δI, innovation.δi)
            end
        end,
        δI=sum(first, innovations),
        δi=sum(last, innovations),
        Y=(Y⁻ + δI)
        KFEnvInfo(
            y⁻ + δi,
            (Y + Y') / 2,
            δi,
            δI,
        )
    end
end

function field_moments(info, H)
    let coefficients=posterior_coefficient_moments(info),
        μ=H * coefficients.μ,
        σ²=vec(sum((H * coefficients.Σ) .* H; dims=2))
        (μ=μ, σ²=max.(σ², 0.0), coefficients=coefficients)
    end
end

relative_distance(left, right) =
    norm(left - right) / max(norm(left), norm(right), 1.0)

function prediction_consensus_error(moments)
    maximum(
        sqrt(mean(abs2, left.μ - right.μ))
        for left in moments for right in moments
    )
end

function information_consensus_error(information)
    maximum(
        max(
            relative_distance(left.Y, right.Y),
            relative_distance(left.y, right.y),
        )
        for left in information for right in information
    )
end

function predictive_log_likelihood(error, σ², noise_variance)
    let variance=σ² .+ noise_variance
        mean(
            -0.5 .* (
                log.(2π .* variance) .+
                abs2.(error) ./ variance
            ),
        )
    end
end

interval_coverage(error, σ²) =
    mean(abs.(error) .≤ 1.96 .* sqrt.(σ²))

function normalized_nees(information, coefficients)
    isnothing(coefficients) && return NaN
    mean(information) do info
        let moments=posterior_coefficient_moments(info),
            error=moments.μ - coefficients
            (error ⋅ (info.Y * error)) / length(error)
        end
    end
end

function minimum_conservatism_eigenvalue(information, central)
    let Pₒ=recover_covariance_from_info(central)
        minimum(information) do info
            eigmin(Symmetric(recover_covariance_from_info(info) - Pₒ))
        end
    end
end

function publication_metric_row(
    experiment,
    truth_kind,
    backend,
    seed,
    step,
    phase,
    information,
    central,
    H,
    truth,
    coefficients,
    noise_variance,
    messages,
    bytes,
    iterations,
)
    let moments=map(info -> field_moments(info, H), information),
        central_moments=field_moments(central, H),
        errors=map(model -> model.μ - truth, moments),
        truth_deviation=std(truth),
        rmses=map(error -> sqrt(mean(abs2, error)), errors),
        central_gaps=map(
            model -> sqrt(mean(abs2, model.μ - central_moments.μ)),
            moments,
        ),
        log_likelihoods=map(
            (error, model) -> predictive_log_likelihood(
                error,
                model.σ²,
                noise_variance,
            ),
            errors,
            moments,
        ),
        coverages=map(
            (error, model) -> interval_coverage(error, model.σ²),
            errors,
            moments,
        )
        (
            experiment=experiment,
            truth=truth_kind,
            backend=backend,
            seed=seed,
            step=step,
            phase=phase,
            rmse=mean(rmses),
            normalized_rmse=mean(rmses) / truth_deviation,
            predictive_log_likelihood=mean(log_likelihoods),
            interval_coverage=mean(coverages),
            integrated_uncertainty=mean(
                mean(model.σ²) for model in moments
            ),
            prediction_consensus_rmse=prediction_consensus_error(moments),
            information_consensus_error=
                information_consensus_error(information),
            centralized_prediction_gap=mean(central_gaps),
            normalized_nees=normalized_nees(
                information,
                coefficients,
            ),
            minimum_conservatism_eigenvalue=
                minimum_conservatism_eigenvalue(
                    information,
                    central,
                ),
            cumulative_messages=messages,
            cumulative_bytes=bytes,
            consensus_iterations=iterations,
        )
    end
end

function centralized_metric_row(
    experiment,
    truth_kind,
    seed,
    step,
    phase,
    central,
    H,
    truth,
    coefficients,
    noise_variance,
)
    publication_metric_row(
        experiment,
        truth_kind,
        :centralized,
        seed,
        step,
        phase,
        [central],
        central,
        H,
        truth,
        coefficients,
        noise_variance,
        0,
        0,
        0,
    )
end

function experiment_backends(experiment)
    @match experiment begin
        1 => (SCRIBEFusion(), IndependentFusion())
        2 => (
            SCRIBEFusion(),
            IndependentFusion(),
            KalmanFilterOnlyFusion(),
            CovarianceIntersectionOnlyFusion(),
        )
        3 => (
            SCRIBEFusion(),
            IndependentFusion(),
            KalmanFilterOnlyFusion(),
            CovarianceIntersectionOnlyFusion(),
        )
    end
end

experiment_truth(experiment) =
    experiment == 3 ? :misspecified : :matched

function trial_phase(experiment, k, settings)
    experiment == 1 ? :continuous_sharing : communication_phase(k, settings)
end

function current_information(ng)
    [
        ng.vertices[aid].agent.information[end]
        for aid in sort(collect(keys(ng.vertices)))
    ]
end

function run_publication_trial(experiment, settings, seed)
    let truth_kind=experiment_truth(experiment),
        params=publication_parameters(settings),
        smodel=initialize_SCRIBEModel_from_parameters(params),
        evaluation_locations=publication_locations(
            settings.evaluation_points,
        ),
        H=prediction_dynamics(smodel, evaluation_locations),
        truth=truth_definition(
            truth_kind,
            smodel,
            evaluation_locations,
        ),
        truth_values=truth.evaluate(evaluation_locations),
        prior=publication_prior_information(
            smodel,
            evaluation_locations,
        ),
        plans=observation_plan(smodel, truth, settings, seed),
        central=copy(prior),
        backends=experiment_backends(experiment),
        networks=Dict(
            fusion_name(backend) => initialize_publication_network(
                params,
                prior,
                plans,
                settings,
            )
            for backend in backends
        ),
        communication=Dict(
            fusion_name(backend) => (
                messages=0,
                bytes=0,
                iterations=0,
            )
            for backend in backends
        ),
        rows=Any[]

        push!(
            rows,
            centralized_metric_row(
                experiment,
                truth_kind,
                seed,
                0,
                :prior,
                central,
                H,
                truth_values,
                truth.coefficients,
                settings.noise_variance,
            ),
        )
        foreach(backends) do backend
            let name=fusion_name(backend)
                push!(
                    rows,
                    publication_metric_row(
                        experiment,
                        truth_kind,
                        name,
                        seed,
                        0,
                        :prior,
                        current_information(networks[name]),
                        central,
                        H,
                        truth_values,
                        truth.coefficients,
                        settings.noise_variance,
                        0,
                        0,
                        0,
                    ),
                )
            end
        end

        foreach(1:settings.n_steps) do k
            central=centralized_update(central, plans, k, params)
            phase=trial_phase(experiment, k, settings)
            edges=scheduled_edges(
                experiment,
                k,
                agent_ids(settings),
                settings,
            )

            push!(
                rows,
                centralized_metric_row(
                    experiment,
                    truth_kind,
                    seed,
                    k,
                    phase,
                    central,
                    H,
                    truth_values,
                    truth.coefficients,
                    settings.noise_variance,
                ),
            )

            foreach(backends) do backend
                let name=fusion_name(backend),
                    step_communication=fusion_step!(
                        backend,
                        networks[name],
                        k,
                        edges,
                        settings,
                    ),
                    previous=communication[name],
                    cumulative=(
                        messages=previous.messages +
                            step_communication.messages,
                        bytes=previous.bytes +
                            step_communication.bytes,
                        iterations=previous.iterations +
                            step_communication.iterations,
                    )
                    communication[name] = cumulative
                    push!(
                        rows,
                        publication_metric_row(
                            experiment,
                            truth_kind,
                            name,
                            seed,
                            k,
                            phase,
                            current_information(networks[name]),
                            central,
                            H,
                            truth_values,
                            truth.coefficients,
                            settings.noise_variance,
                            cumulative.messages,
                            cumulative.bytes,
                            cumulative.iterations,
                        ),
                    )
                end
            end
        end
        rows
    end
end

function write_publication_rows(rows, output_path)
    let fields=propertynames(first(rows))
        open(output_path, "w") do io
            println(io, join(string.(fields), ","))
            foreach(rows) do row
                println(
                    io,
                    join((string(getproperty(row, field)) for field in fields), ","),
                )
            end
        end
    end
end

function final_metric_summary(rows)
    let final_step=maximum(getproperty.(rows, :step)),
        final_rows=filter(row -> row.step == final_step, rows),
        backends=sort(unique(getproperty.(final_rows, :backend)); by=string)
        map(backends) do backend
            let selected=filter(row -> row.backend == backend, final_rows)
                (
                    backend=backend,
                    rmse=mean(getproperty.(selected, :rmse)),
                    rmse_deviation=std(
                        getproperty.(selected, :rmse);
                        corrected=false,
                    ),
                    centralized_gap=mean(
                        getproperty.(
                            selected,
                            :centralized_prediction_gap,
                        ),
                    ),
                    coverage=mean(
                        getproperty.(selected, :interval_coverage),
                    ),
                    normalized_nees=let values=filter(
                            isfinite,
                            getproperty.(selected, :normalized_nees),
                        )
                        isempty(values) ? NaN : mean(values)
                    end,
                )
            end
        end
    end
end

function write_final_metric_summary(rows, output_path)
    let summary=final_metric_summary(rows)
        open(output_path, "w") do io
            println(
                io,
                "backend,rmse_mean,rmse_std,centralized_gap_mean," *
                "coverage_mean,normalized_nees_mean",
            )
            foreach(summary) do row
                println(
                    io,
                    "$(row.backend),$(row.rmse),$(row.rmse_deviation)," *
                    "$(row.centralized_gap),$(row.coverage)," *
                    "$(row.normalized_nees)",
                )
            end
        end
    end
end

function print_publication_summary(experiment, rows, output_dir)
    println("Publication experiment $experiment complete.")
    foreach(final_metric_summary(rows)) do row
        println(
            "  $(row.backend): RMSE=" *
            "$(round(row.rmse; digits=4))±" *
            "$(round(row.rmse_deviation; digits=4)), " *
            "central gap=$(round(row.centralized_gap; digits=6)), " *
            "coverage=$(round(row.coverage; digits=4))",
        )
    end
    println("  results: $output_dir")
end

function run_filtering_publication_experiment(
    experiment;
    profile=:full,
    make_plots=true,
)
    let settings=publication_settings(profile),
        rows=Any[]
        foreach(settings.seeds) do seed
            append!(
                rows,
                run_publication_trial(experiment, settings, seed),
            )
            GC.gc()
        end

        output_dir=joinpath(
            @__DIR__,
            "res",
            "experiment_$(experiment)",
            String(profile),
        )
        mkpath(output_dir)
        write_publication_rows(
            rows,
            joinpath(output_dir, "metric_history.csv"),
        )
        write_final_metric_summary(
            rows,
            joinpath(output_dir, "final_summary.csv"),
        )
        if make_plots
            if !isdefined(@__MODULE__, :plot_publication_experiment)
                include(joinpath(@__DIR__, "filtering_plots.jl"))
            end
            Base.invokelatest(
                plot_publication_experiment,
                experiment,
                rows,
                settings,
                output_dir,
            )
        end
        print_publication_summary(experiment, rows, output_dir)
        rows
    end
end

function run_filtering_publication_experiments(;
    profile=:full,
    experiments=(1, 2, 3),
    make_plots=true,
)
    map(experiments) do experiment
        run_filtering_publication_experiment(
            experiment;
            profile,
            make_plots,
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    let profile=isempty(ARGS) ? :full : Symbol(first(ARGS)),
        experiments=length(ARGS) < 2 ?
            (1, 2, 3) :
            Tuple(parse.(Int, split(ARGS[2], ","))),
        make_plots=length(ARGS) < 3 || ARGS[3] != "no_plots"
        run_filtering_publication_experiments(
            ;
            profile,
            experiments,
            make_plots,
        )
    end
end
