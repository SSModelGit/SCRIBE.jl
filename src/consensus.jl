export reset_consensus_count, full_reset_network_connector, progress_agent_env_filter
export network_precheck, network_postcheck, network_prior_update, network_averaging_update
export distributed_fusion, consume_consensus_message!, deliver_consensus_messages

const CONSENSUS_STABLE_ROUNDS = 3

symmetrize(M::AbstractMatrix) = Matrix(Symmetric((M + M') / 2))

function reset_consensus_count(nc::NetworkConnector)
    nc.outbox["lv"] = 0
    nc.outbox["cvc"] = 0
    nc.outbox["stage_ready"] = false
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
    nc.outbox["cache"] = Dict{String, Any}()
    nc.outbox["msg_out"] = nothing
    nc.outbox["complete"] = false
    nc.outbox["n_eff"] = 1
    nc.outbox["stage_ready"] = false
end

current_neighbors(agent_id::String, ng::NetworkGraph) =
    get(ng.edges, agent_id, String[])

function component_agents(agent_id::String, ng::NetworkGraph)
    visited = Set{String}()
    stack = String[agent_id]
    while !isempty(stack)
        aid = pop!(stack)
        aid in visited && continue
        push!(visited, aid)
        append!(stack, (nb for nb in current_neighbors(aid, ng) if nb ∉ visited))
    end
    sort!(collect(visited))
end

function queue_consensus_message!(agent_id::String, k::Integer,
                                  nc::NetworkConnector, ng::NetworkGraph)
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
        length(current_neighbors(agent_id, ng)),
        copy.(nc.outbox["lc"]),
    )
end

"""Accept a consensus message only from a current neighbor and for the exact round."""
function consume_consensus_message!(receiver_id::String, nc::NetworkConnector,
                                    msg::ConsensusMessage, k::Integer,
                                    ng::NetworkGraph)
    if msg.sender ∉ current_neighbors(receiver_id, ng) ||
       msg.k != k ||
       msg.stage != nc.outbox["stage"] ||
       msg.lv != nc.outbox["lv"]
        return false
    end

    cache = nc.outbox["cache"]
    prev = get(cache, msg.sender, nothing)
    if isnothing(prev) || msg.seq > prev.seq
        cache[msg.sender] = msg
        return true
    end
    return false
end

"""Compatibility overload. Prefer the graph-aware method above."""
function consume_consensus_message!(nc::NetworkConnector, msg::ConsensusMessage,
                                    k::Integer)
    if msg.sender ∉ nc.neighbors ||
       msg.k != k ||
       msg.stage != nc.outbox["stage"] ||
       msg.lv != nc.outbox["lv"]
        return false
    end
    cache = nc.outbox["cache"]
    prev = get(cache, msg.sender, nothing)
    if isnothing(prev) || msg.seq > prev.seq
        cache[msg.sender] = msg
        return true
    end
    return false
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

function network_precheck(k::Integer, estimators::EnvEstimators,
                          net_conn::NetworkConnector)
    stage = net_conn.outbox["stage"]
    if stage == :prior
        net_conn.outbox["lc"] = compute_info_priors(estimators, k)
    elseif stage == :innov
        net_conn.outbox["lc"] = compute_innov_from_obs(estimators, k)
    else
        error("Unknown consensus stage $stage")
    end
    net_conn.outbox["ln"] = nothing
    net_conn.outbox["lv"] = 1
    net_conn.outbox["cvc"] = 0
    net_conn.outbox["stage_ready"] = false
    empty!(net_conn.outbox["cache"])
end

function relative_shift(a, b)
    norm(a - b) / max(norm(a), norm(b), 1.0)
end

function network_postcheck(nc::NetworkConnector, threshold::Float64;
                           agent_id=nothing)
    shifts = map(i -> relative_shift(nc.outbox["lc"][i],
                                     nc.outbox["ln"][i]), (1, 2))
    nc.outbox["cvc"] = all(shifts .≤ threshold) ?
                       nc.outbox["cvc"] + 1 : 0
    nc.outbox["stage_ready"] =
        nc.outbox["cvc"] ≥ CONSENSUS_STABLE_ROUNDS

    if !isnothing(agent_id)
        println(agent_id, " relative consensus shifts: ", shifts,
                " | round: ", nc.outbox["lv"],
                " | stable rounds: ", nc.outbox["cvc"])
    end

    nc.outbox["lc"] = copy.(nc.outbox["ln"])
    nc.outbox["ln"] = nothing
    nc.outbox["lv"] += 1
    nc.outbox["cvc"]
end

function valid_neighbor_payloads(agent_id::String, nc::NetworkConnector,
                                 ng::NetworkGraph)
    cache = nc.outbox["cache"]
    Dict(
        nb => copy.(cache[nb].payload)
        for nb in current_neighbors(agent_id, ng)
        if haskey(cache, nb) &&
           cache[nb].lv == nc.outbox["lv"] &&
           cache[nb].stage == nc.outbox["stage"]
    )
end

function neighbor_messages_ready(agent_id::String, nc::NetworkConnector,
                                 ng::NetworkGraph)
    cache = nc.outbox["cache"]
    all(
        haskey(cache, nb) &&
        cache[nb].lv == nc.outbox["lv"] &&
        cache[nb].stage == nc.outbox["stage"]
        for nb in current_neighbors(agent_id, ng)
    )
end

function project_simplex(v::Vector{Float64})
    u = sort(v; rev=true)
    cssv = cumsum(u) .- 1.0
    ρ = findlast(i -> u[i] - cssv[i] / i > 0.0, eachindex(u))
    isnothing(ρ) && return fill(1.0 / length(v), length(v))
    θ = cssv[ρ] / ρ
    max.(v .- θ, 0.0)
end

function trace_covariance_and_gradient(weights::Vector{Float64},
                                       information_matrices::Vector{Matrix{Float64}})
    Ȳ = symmetrize(sum(weights[i] * information_matrices[i]
                       for i in eachindex(weights)))
    factor = cholesky(Symmetric(Ȳ); check=true)
    P̄ = Matrix(factor \ Matrix{Float64}(I, size(Ȳ)...))
    objective = tr(P̄)
    gradient = [
        -tr(P̄ * information_matrices[i] * P̄)
        for i in eachindex(weights)
    ]
    objective, gradient
end

"""Solve the thesis trace-covariance CI subproblem on the probability simplex."""
function covariance_intersection_weights(
    information_matrices::Vector{Matrix{Float64}};
    tolerance::Float64=1e-10,
    max_iterations::Integer=200,
)
    n = length(information_matrices)
    n == 1 && return [1.0]
    weights = fill(1.0 / n, n)
    objective, gradient =
        trace_covariance_and_gradient(weights, information_matrices)

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
            candidate_objective, _ =
                trace_covariance_and_gradient(candidate,
                                              information_matrices)
            if candidate_objective ≤
               objective + 1e-4 * dot(gradient, direction)
                accepted = true
                break
            end
            step *= 0.5
        end

        accepted || return weights
        if norm(candidate - weights) ≤ tolerance ||
           abs(candidate_objective - objective) ≤
           tolerance * max(abs(objective), 1.0)
            return candidate
        end

        weights = candidate
        objective, gradient =
            trace_covariance_and_gradient(weights, information_matrices)
    end
    weights
end

function network_prior_update(agent_id::String, nc::NetworkConnector,
                              ng::NetworkGraph)
    cYᵢ, cyᵢ = copy.(nc.outbox["lc"])
    neighbor_values = valid_neighbor_payloads(agent_id, nc, ng)
    neighbor_ids = sort(collect(keys(neighbor_values)))

    information_matrices =
        Matrix{Float64}[symmetrize(cYᵢ),
                        (symmetrize(neighbor_values[id][1])
                         for id in neighbor_ids)...]
    information_vectors =
        Vector{Float64}[cyᵢ,
                        (neighbor_values[id][2]
                         for id in neighbor_ids)...]
    weights = covariance_intersection_weights(information_matrices)

    Y_next = symmetrize(sum(weights[i] * information_matrices[i]
                            for i in eachindex(weights)))
    y_next = sum(weights[i] * information_vectors[i]
                 for i in eachindex(weights))
    cholesky(Symmetric(Y_next); check=true)
    nc.outbox["ln"] = (Y_next, y_next)
end

function network_averaging_update(agent_id::String, nc::NetworkConnector,
                                  ng::NetworkGraph)
    xᵢ = copy.(nc.outbox["lc"])
    neighbors = valid_neighbor_payloads(agent_id, nc, ng)
    degree = length(current_neighbors(agent_id, ng))

    δI = copy(xᵢ[1])
    δi = copy(xᵢ[2])
    for (nb, xⱼ) in neighbors
        γᵢⱼ = 1.0 /
              (1.0 + max(degree,
                         length(current_neighbors(nb, ng))))
        δI .+= γᵢⱼ .* (xⱼ[1] .- xᵢ[1])
        δi .+= γᵢⱼ .* (xⱼ[2] .- xᵢ[2])
    end
    nc.outbox["ln"] = (symmetrize(δI), δi)
end

function component_spread(component::Vector{String}, ng::NetworkGraph)
    max_spread = 0.0
    for i in eachindex(component), j in (i + 1):length(component)
        left = ng.vertices[component[i]].net_conn.outbox["lc"]
        right = ng.vertices[component[j]].net_conn.outbox["lc"]
        if isnothing(left) || isnothing(right)
            return Inf
        end
        max_spread = max(max_spread,
                         relative_shift(left[1], right[1]),
                         relative_shift(left[2], right[2]))
    end
    max_spread
end

function component_ready(component::Vector{String}, stage::Symbol,
                         ng::NetworkGraph, threshold::Float64)
    all(
        let outbox = ng.vertices[aid].net_conn.outbox
            outbox["stage"] == stage && outbox["stage_ready"]
        end
        for aid in component
    ) && component_spread(component, ng) ≤ threshold
end

function begin_innovation_stage!(component::Vector{String}, ng::NetworkGraph)
    for aid in component
        outbox = ng.vertices[aid].net_conn.outbox
        outbox["prior"] = copy.(outbox["lc"])
        reset_consensus_count(ng.vertices[aid].net_conn)
        outbox["stage"] = :innov
        empty!(outbox["cache"])
        outbox["lc"] = nothing
        outbox["ln"] = nothing
        outbox["msg_out"] = nothing
    end
end

function info_state_update_post_consensus(agent_id::String, ng::NetworkGraph)
    @unpack agent, net_conn = ng.vertices[agent_id]
    @unpack outbox = net_conn
    δĪ, δī = outbox["innov"]
    Y⁻, y⁻ = outbox["prior"]
    nₐ = length(component_agents(agent_id, ng))
    outbox["n_eff"] = nₐ
    Y_next = symmetrize(Y⁻ + nₐ * δĪ)
    y_next = y⁻ + nₐ * δī
    cholesky(Symmetric(Y_next); check=true)
    next_agent_info_state(
        agent,
        KFEnvInfo(y_next, Y_next, δī, δĪ),
    )
end

function complete_innovation_stage!(component::Vector{String},
                                    ng::NetworkGraph)
    for aid in component
        outbox = ng.vertices[aid].net_conn.outbox
        outbox["innov"] = copy.(outbox["lc"])
    end
    for aid in component
        info_state_update_post_consensus(aid, ng)
        outbox = ng.vertices[aid].net_conn.outbox
        outbox["complete"] = true
        outbox["msg_out"] = nothing
    end
end

"""Perform one round of SCRIBE prior-CI or innovation-average consensus.

Completion requires three stable local rounds and agreement across the entire
current connected component. `timeline` is a hard safety limit, not a substitute
for convergence.
"""
function distributed_fusion(k::Integer, agent_id::String, ng::NetworkGraph,
                            threshold::Float64, timeline::Integer)
    @unpack estimators, net_conn = ng.vertices[agent_id]
    outbox = net_conn.outbox

    outbox["complete"] && return true

    if outbox["lv"] == 0
        network_precheck(k, estimators, net_conn)
        queue_consensus_message!(agent_id, k, net_conn, ng)
        return false
    end

    stage = outbox["stage"]
    if !neighbor_messages_ready(agent_id, net_conn, ng)
        # Do not advance the local round using a partial neighborhood. Re-send
        # the current value so a delayed neighbor can still synchronize.
        queue_consensus_message!(agent_id, k, net_conn, ng)
        return false
    end
    if stage == :prior
        network_prior_update(agent_id, net_conn, ng)
    elseif stage == :innov
        network_averaging_update(agent_id, net_conn, ng)
    else
        error("Unknown consensus stage $stage")
    end
    network_postcheck(net_conn, threshold)

    component = component_agents(agent_id, ng)
    if component_ready(component, stage, ng, threshold)
        if stage == :prior
            begin_innovation_stage!(component, ng)
            return false
        else
            complete_innovation_stage!(component, ng)
            return true
        end
    end
    if outbox["lv"] > timeline
        spread = component_spread(component, ng)
        error("Connected component $(join(component, ", ")) failed to converge " *
              "in stage $stage within $timeline rounds (relative spread=$spread).")
    end

    queue_consensus_message!(agent_id, k, net_conn, ng)
    false
end

"""Advance the environment filter after distributed fusion has set information at `k+1`."""
function progress_agent_env_filter(agent::KFEnvScribe, world::SCRIBEModel,
                                   X::Matrix{Float64})
    ϕₖ = recover_estimate_from_info(agent, agent.k + 1)
    next_agent_state(agent, ϕₖ, world, X)
    next_agent_time(agent)
end
