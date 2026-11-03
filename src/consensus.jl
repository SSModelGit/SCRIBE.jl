export reset_consensus_count, full_reset_network_connector, progress_agent_env_filter
export network_precheck, network_postcheck, network_prior_update, network_averaging_update
export distributed_fusion, consume_consensus_message!, deliver_consensus_messages

symmetrize(M::AbstractMatrix) = Matrix(Symmetric((M + M') / 2))

function reset_consensus_count(nc::NetworkConnector)
    nc.outbox["lv"] = 0
    nc.outbox["cvc"] = 0
end

function full_reset_network_connector(nc::NetworkConnector)
    nc.outbox["lc"] = nothing
    nc.outbox["ln"] = nothing
    nc.outbox["lv"] = 0
    nc.outbox["cvc"] = 0
    nc.outbox["prior"] = nothing
    nc.outbox["innov"] = nothing
    nc.outbox["stage"] = :prior
    nc.outbox["seq"] = 0
    nc.outbox["cache"] = Dict{Tuple{Symbol, Int}, Dict{String, ConsensusMessage}}() # Dict{String, Any}()
    nc.outbox["msg_out"] = nothing
    nc.outbox["complete"] = false
    nc.outbox["n_eff"] = 1
    nc.outbox["participants"] = Set{String}()
end

function queue_consensus_message!(agent_id::String, k::Integer, nc::NetworkConnector)
    if isnothing(nc.outbox["lc"])
        nc.outbox["msg_out"] = nothing
        return
    end
    nc.outbox["seq"] += 1
    nc.outbox["msg_out"] = ConsensusMessage(
        agent_id,
        k,
        nc.outbox["stage"],
        nc.outbox["lv"],
        nc.outbox["seq"],
        length(nc.neighbors),
        sort!(collect(nc.outbox["participants"])),
        copy.(nc.outbox["lc"]),
    )
end

# """Accept a consensus message only from a current neighbor and for the exact round."""
# function consume_consensus_message!(nc::NetworkConnector, msg::ConsensusMessage,
#                                     k::Integer)
#     if msg.sender ∉ nc.neighbors ||
#        msg.k != k ||
#        msg.stage != nc.outbox["stage"] ||
#        msg.lv != nc.outbox["lv"]
#         return false
#     end
#     cache = nc.outbox["cache"]
#     prev = get(cache, msg.sender, nothing)
#     if isnothing(prev) || msg.seq > prev.seq
#         cache[msg.sender] = msg
#         return true
#     end
#     return false
# end

function consume_consensus_message!(nc::NetworkConnector, msg::ConsensusMessage, k::Integer)
    out = nc.outbox

    # validate the stage of consensus
    if msg.sender ∉ nc.neighbors
        return false
    end
    if !(msg.k == k && !out["complete"])
        return false
    end
    if !(msg.stage ∈ (:prior, :innov) && msg.lv ≥ 1)
        return false
    end

    if msg.stage == out["stage"]
        if msg.lv < max(out["lv"], 1)
            return false
        end
    else
        if !(out["stage"] == :prior && msg.stage == :innov)
            return false
        end
    end

    inbox = get!(out["cache"], (msg.stage, Int(msg.lv))) do
        Dict{String, ConsensusMessage}()
    end

    if haskey(inbox, msg.sender)
        return false
    end

    inbox[msg.sender] = ConsensusMessage(
        msg.sender,
        msg.k,
        msg.stage,
        msg.lv,
        msg.seq,
        msg.degree,
        copy(msg.participants),
        copy.(msg.payload),
    )
    return true
end

function deliver_consensus_messages(k::Integer, ng::NetworkGraph)
    outgoing = Dict{String, ConsensusMessage}()
    for (aid, agent) in ng.vertices
        msg = agent.net_conn.outbox["msg_out"]
        if !isnothing(msg) && msg.k == k
            outgoing[aid] = msg
        end
    end
    outgoing
end

function network_precheck(
    k::Integer,
    agent_id::String,
    estimators::EnvEstimators,
    net_conn::NetworkConnector,
)
    stage = net_conn.outbox["stage"]
    if stage == :prior
        net_conn.outbox["lc"] = compute_info_priors(estimators, k)
        net_conn.outbox["participants"] = Set([agent_id])
    elseif stage == :innov
        net_conn.outbox["lc"] = compute_innov_from_obs(estimators, k)
    else
        error("Unknown consensus stage $stage")
    end
    net_conn.outbox["ln"] = nothing
    net_conn.outbox["lv"] = 1
    net_conn.outbox["cvc"] = 0
    # empty!(net_conn.outbox["cache"])
end

function relative_shift(a, b)
    norm(a - b) / max(norm(a), norm(b), 1.0)
end

function network_postcheck(nc::NetworkConnector, threshold::Float64; agent_id = nothing)
    shifts = map(i -> relative_shift(nc.outbox["lc"][i], nc.outbox["ln"][i]), (1, 2))
    nc.outbox["cvc"] = all(shifts .≤ threshold) ? nc.outbox["cvc"] + 1 : 0

    if !isnothing(agent_id)
        println(
            agent_id,
            " relative consensus shifts: ",
            shifts,
            " | round: ",
            nc.outbox["lv"],
            " | stable rounds: ",
            nc.outbox["cvc"],
        )
    end

    delete!(nc.outbox["cache"], (nc.outbox["stage"], Int(nc.outbox["lv"])))

    nc.outbox["lc"] = copy.(nc.outbox["ln"])
    nc.outbox["ln"] = nothing
    nc.outbox["lv"] += 1
    nc.outbox["cvc"]
end

# function valid_neighbor_messages(nc::NetworkConnector)
#     cache = nc.outbox["cache"]
#     Dict(
#         nb => cache[nb]
#         for nb in nc.neighbors
#         if haskey(cache, nb) &&
#            cache[nb].lv == nc.outbox["lv"] &&
#            cache[nb].stage == nc.outbox["stage"]
#     )
# end

# function neighbor_messages_ready(nc::NetworkConnector)
#     cache = nc.outbox["cache"]
#     all(
#         haskey(cache, nb) &&
#         cache[nb].lv == nc.outbox["lv"] &&
#         cache[nb].stage == nc.outbox["stage"]
#         for nb in nc.neighbors
#     )
# end

function valid_neighbor_messages(nc::NetworkConnector)
    out = nc.outbox
    get!(out["cache"], (out["stage"], Int(out["lv"]))) do
        Dict{String, ConsensusMessage}()
    end
end

neighbor_messages_ready(nc::NetworkConnector) =
    all(id->haskey(valid_neighbor_messages(nc), id), nc.neighbors)

function merge_participants!(nc::NetworkConnector, messages)
    foreach(values(messages)) do message
        union!(nc.outbox["participants"], message.participants)
    end
    nc.outbox["n_eff"] = length(nc.outbox["participants"])
end

function project_simplex(v::Vector{Float64})
    u = sort(v; rev = true)
    cssv = cumsum(u) .- 1.0
    ρ = findlast(i -> u[i] - cssv[i] / i > 0.0, eachindex(u))
    isnothing(ρ) && return fill(1.0 / length(v), length(v))
    θ = cssv[ρ] / ρ
    max.(v .- θ, 0.0)
end

function trace_covariance_and_gradient(
    weights::Vector{Float64},
    information_matrices::Vector{Matrix{Float64}},
)
    Ȳ = symmetrize(sum(weights[i] * information_matrices[i] for i in eachindex(weights)))
    factor = cholesky(Symmetric(Ȳ); check = true)
    P̄ = Matrix(factor \ Matrix{Float64}(I, size(Ȳ)...))
    objective = tr(P̄)
    gradient = [-tr(P̄ * information_matrices[i] * P̄) for i in eachindex(weights)]
    objective, gradient
end

"""Solve the thesis trace-covariance CI subproblem on the probability simplex."""
function covariance_intersection_weights(
    information_matrices::Vector{Matrix{Float64}};
    tolerance::Float64 = 1e-10,
    max_iterations::Integer = 200,
)
    n = length(information_matrices)
    n == 1 && return [1.0]
    weights = fill(1.0 / n, n)
    objective, gradient = trace_covariance_and_gradient(weights, information_matrices)

    for _ in 1:max_iterations
        step = 1.0 / max(norm(gradient), 1.0)
        accepted = false
        candidate = weights
        candidate_objective = objective

        for _ in 1:30
            candidate = project_simplex(weights .- step .* gradient)
            direction = candidate - weights
            if norm(direction) ≤ tolerance
                return candidate
            end
            candidate_objective, _ = trace_covariance_and_gradient(candidate, information_matrices)
            if candidate_objective ≤ objective + 1e-4 * dot(gradient, direction)
                accepted = true
                break
            end
            step *= 0.5
        end

        accepted || return weights
        if norm(candidate - weights) ≤ tolerance ||
           abs(candidate_objective - objective) ≤ tolerance * max(abs(objective), 1.0)
            return candidate
        end

        weights = candidate
        objective, gradient = trace_covariance_and_gradient(weights, information_matrices)
    end
    weights
end

function network_prior_update(nc::NetworkConnector, threshold::Float64 = 0.0)
    cYᵢ, cyᵢ = copy.(nc.outbox["lc"])
    messages = valid_neighbor_messages(nc)
    neighbor_ids = sort(collect(keys(messages)))

    if all(
        relative_shift(cYᵢ, messages[id].payload[1]) ≤ threshold &&
            relative_shift(cyᵢ, messages[id].payload[2]) ≤ threshold for id in neighbor_ids
    )
        nc.outbox["ln"] = (cYᵢ, cyᵢ)
        return
    end

    information_matrices =
        Matrix{Float64}[symmetrize(cYᵢ), (symmetrize(messages[id].payload[1]) for id in neighbor_ids)...]
    information_vectors = Vector{Float64}[cyᵢ, (messages[id].payload[2] for id in neighbor_ids)...]
    weights = covariance_intersection_weights(information_matrices)

    Y_next = symmetrize(sum(weights[i] * information_matrices[i] for i in eachindex(weights)))
    y_next = sum(weights[i] * information_vectors[i] for i in eachindex(weights))
    cholesky(Symmetric(Y_next); check = true)
    nc.outbox["ln"] = (Y_next, y_next)
end

function network_averaging_update(nc::NetworkConnector)
    xᵢ = copy.(nc.outbox["lc"])
    messages = valid_neighbor_messages(nc)
    degree = length(nc.neighbors)

    δI = copy(xᵢ[1])
    δi = copy(xᵢ[2])
    for message in values(messages)
        xⱼ = message.payload
        γᵢⱼ = 1.0 / (1.0 + max(degree, message.degree))
        δI .+= γᵢⱼ .* (xⱼ[1] .- xᵢ[1])
        δi .+= γᵢⱼ .* (xⱼ[2] .- xᵢ[2])
    end
    nc.outbox["ln"] = (symmetrize(δI), δi)
end

function begin_innovation_stage!(nc::NetworkConnector)
    outbox = nc.outbox
    outbox["prior"] = copy.(outbox["lc"])
    reset_consensus_count(nc)
    outbox["stage"] = :innov
    # empty!(outbox["cache"])
    outbox["lc"] = nothing
    outbox["ln"] = nothing
    outbox["msg_out"] = nothing
end

function info_state_update_post_consensus(scribe::SCRIBEAgent)
    @unpack agent, net_conn = scribe
    @unpack outbox = net_conn
    δĪ, δī = outbox["innov"]
    Y⁻, y⁻ = outbox["prior"]
    nₐ = length(outbox["participants"])
    outbox["n_eff"] = nₐ
    Y_next = symmetrize(Y⁻ + nₐ * δĪ)
    y_next = y⁻ + nₐ * δī
    cholesky(Symmetric(Y_next); check = true)
    next_agent_info_state(agent, KFEnvInfo(y_next, Y_next, δī, δĪ))
end

function complete_innovation_stage!(scribe::SCRIBEAgent)
    outbox = scribe.net_conn.outbox
    outbox["innov"] = copy.(outbox["lc"])
    info_state_update_post_consensus(scribe)
    outbox["complete"] = true
    outbox["msg_out"] = nothing
end

"""Perform one agent-local round of SCRIBE distributed fusion.

`consensus_rounds` is the shared finite horizon used to approximate the thesis'
"while not converged" loop. The agent reads only its own state and messages
accepted from its current neighbors. Participant identifiers propagate with
those messages and locally determine the innovation multiplier.
"""
function distributed_fusion(k::Integer, scribe::SCRIBEAgent, threshold::Float64, consensus_rounds::Integer)
    @assert consensus_rounds > 0 "consensus_rounds must be positive"
    @unpack id, estimators, net_conn = scribe
    outbox = net_conn.outbox
    if outbox["complete"]; return true; end

    if outbox["lv"] == 0
        network_precheck(k, id, estimators, net_conn)
        queue_consensus_message!(id, k, net_conn)
        return false
    end

    stage = outbox["stage"]
    if !neighbor_messages_ready(net_conn)
        queue_consensus_message!(id, k, net_conn)
        return false
    end

    messages = valid_neighbor_messages(net_conn)
    merge_participants!(net_conn, messages)
    if stage == :prior
        network_prior_update(net_conn, threshold)
    elseif stage == :innov
        network_averaging_update(net_conn)
    else
        error("Unknown consensus stage $stage")
    end
    network_postcheck(net_conn, threshold)

    if outbox["lv"] > consensus_rounds
        if stage == :prior
            begin_innovation_stage!(net_conn)
            return false
        end

        complete_innovation_stage!(scribe)
        return true
    end

    queue_consensus_message!(id, k, net_conn)
    false
end

"""Advance the environment filter after distributed fusion has set information at `k+1`."""
function progress_agent_env_filter(agent::KFEnvScribe, world::SCRIBEModel, X::Matrix{Float64})
    ϕₖ = recover_estimate_from_info(agent, agent.k + 1)
    next_agent_state(agent, ϕₖ, world, X)
    next_agent_time(agent)
end

"""Advance a distributed data-backed filter and pull its next observation."""
function progress_agent_env_filter(agent::KFEnvScribe, X::Matrix{Float64})
    @assert (agent.bhv isa DataObserver) "Location-only progression requires a DataObserver"
    ϕₖ = recover_estimate_from_info(agent, agent.k + 1)
    next_agent_state(agent, ϕₖ, X)
    next_agent_time(agent)
end

"""Advance a distributed filter with a pre-collected sensor measurement."""
function progress_agent_env_filter(agent::KFEnvScribe, measurement::SensorObservation)
    @assert (get_model_time(agent.estimates[agent.k].estimate) + 1 == measurement.k) "Measurement time does not match expected time."
    ϕₖ = recover_estimate_from_info(agent, agent.k + 1)
    next_agent_state(agent, ϕₖ, measurement)
    next_agent_time(agent)
end
