using JuMP
using Ipopt

export reset_consensus_count, full_reset_network_connector, progress_agent_env_filter
export network_precheck, network_postcheck, network_prior_update, network_averaging_update
export distributed_fusion

reset_consensus_count(nc::NetworkConnector) = nc.outbox["lv"] = 0

function full_reset_network_connector(nc::NetworkConnector)
    nc.outbox["lc"] = nothing
    nc.outbox["ln"] = nothing
    nc.outbox["lv"] = 0
    nc.outbox["cvc"] = 0
    nc.outbox["prior"] = nothing
    nc.outbox["innov"] = nothing
end

function network_precheck(k::Integer, estimators::EnvEstimators, net_conn::NetworkConnector)
    if isnothing(net_conn.outbox["prior"])
        net_conn.outbox["lc"] = compute_info_priors(estimators, k)
    elseif isnothing(net_conn.outbox["innov"])
        net_conn.outbox["lc"] = compute_innov_from_obs(estimators, k)
    end
    net_conn.outbox["ln"] = nothing
    net_conn.outbox["lv"] = 1
    net_conn.outbox["cvc"] = 0
end

function network_postcheck(nc::NetworkConnector, threshold::Float64; agent_id=nothing)
    let cvc = nc.outbox["cvc"]
        shifts = map(i->norm(abs.(nc.outbox["lc"][i] - nc.outbox["ln"][i])), [1,2])
        nc.outbox["cvc"] = (all(shifts .< threshold) ? cvc+1 : 0)
        if !isnothing(agent_id)
            println(agent_id*"LC: ", nc.outbox["lc"])
            println(agent_id*"LN: ", nc.outbox["ln"])
            println(agent_id*"LV: ", nc.outbox["lv"], " | CVC: ", nc.outbox["cvc"])
        end
        nc.outbox["lc"] = (copy(nc.outbox["ln"][1]), copy(nc.outbox["ln"][2]))
        nc.outbox["ln"] = nothing
        nc.outbox["lv"] += 1

        return nc.outbox["cvc"]
        # return nc.outbox["cvc"] > length(ng.vertices)
    end
end

function network_prior_update(nc::NetworkConnector, ng::NetworkGraph)
    (cYᵢ, cyᵢ) = copy.(nc.outbox["lc"])
    cYyⱼ = Dict([(nb, copy.(ng.vertices[nb].net_conn.outbox["lc"])) for nb in nc.neighbors])

    sY = [cYᵢ, [Y[1] for Y in values(cYyⱼ)]...]
    sYinv = [inv(Y) for Y in sY]
    sy = [cyᵢ, [Y[2] for Y in values(cYyⱼ)]...]
    Nᵢ = size(sY, 1)

    model = Model(Ipopt.Optimizer); set_silent(model);
    @variable(model, ω[1:Nᵢ]>=0.0)
    @constraint(model, sum(ω)==1.0)
    @objective(model, Min, tr(sum(ω[i] * sYinv[i] for i in 1:Nᵢ)))
    optimize!(model)
    ω_best = value.(ω)

    # Complete network prior update
    nc.outbox["ln"] = (sum(ω_best .* sY), sum(ω_best .* sy))
end

function network_averaging_update(nc::NetworkConnector, ng::NetworkGraph)
    xᵢ = copy.(nc.outbox["lc"])
    xⱼ = Dict([(nb, copy.(ng.vertices[nb].net_conn.outbox["lc"])) for nb in nc.neighbors])

    # acquire differential updates from neighbors
    deg = size(nc.neighbors, 1)
    nb_degs = Dict([(nb, size(ng.edges[nb], 1)) for nb in nc.neighbors])
    γᵢ = Dict([(nb, 1/(1 + max(deg, nb_degs[nb]))) for nb in nc.neighbors])
    scaled_diffs = Dict([(nb, γᵢ[nb] .* (xⱼ[nb] .- xᵢ)) for nb in nc.neighbors])

    # Complete network averaging
    δI = xᵢ[1] + sum(map(x->x[1], values(scaled_diffs)))
    δi = xᵢ[2] + sum(map(x->x[2], values(scaled_diffs)))
    nc.outbox["ln"] = (δI, δi)
end

"""Perform one round of distributed fusion.

Return true if complete. Otherwise, return false.

Note that convergence on priors is not completion.
"""
function distributed_fusion(k::Integer, agent_id::String, ng::NetworkGraph, threshold::Float64, timeline::Integer)
    @unpack estimators, net_conn = ng.vertices[agent_id]

    # Compute priors from current system state at t=k and previous information state at t=k-1
    nₐ=length(ng.vertices)

    if net_conn.outbox["lv"] == 0
        network_precheck(k, estimators, net_conn)
        return false
    else
        if isnothing(net_conn.outbox["prior"])
            network_prior_update(net_conn, ng)
            # TODO: come up with better convergence conditions
            cvc = network_postcheck(net_conn, threshold) # network_postcheck(net_conn, threshold; agent_id) # debugging mode
            if net_conn.outbox["lv"] > timeline
                net_conn.outbox["prior"] = copy.(net_conn.outbox["lc"])
                reset_consensus_count(net_conn)
            end
            return false
        elseif isnothing(net_conn.outbox["innov"])
            network_averaging_update(net_conn, ng)
            # TODO: come up with better convergence conditions
            cvc = network_postcheck(net_conn, threshold) # network_postcheck(net_conn, threshold; agent_id) # debugging mode
            if net_conn.outbox["lv"] > timeline
                # Compute innovations for t=k+1 from current observation at t=k
                net_conn.outbox["innov"] = (copy(net_conn.outbox["lc"][1]), copy(net_conn.outbox["lc"][2]))
                info_state_update_post_consensus(agent_id, ng)
                return true
            else
                return false
            end
        else
            @error "clear out the outbox properly..."
        end
    end
end

function info_state_update_post_consensus(agent_id::String, ng::NetworkGraph)
    @unpack agent, net_conn = ng.vertices[agent_id]
    @unpack outbox = net_conn
    let δĪ=outbox["innov"][1], δī=outbox["innov"][2], Y⁻=outbox["prior"][1], y⁻=outbox["prior"][2], nₐ = ng.connectivity[agent_id]
        next_agent_info_state(agent, KFEnvInfo(y⁻+nₐ*δī, Y⁻+nₐ*δĪ, δī, δĪ)) # set info(t=k+1)
    end
end

"""Consolidated update process. Assumes information state is updated during distributed fusion.

Requires the new state of the world and the new sampling locations.
"""
function progress_agent_env_filter(agent::KFEnvScribe, world::SCRIBEModel, X::Matrix{Float64})
    ϕₖ=recover_estimate_from_info(agent, agent.k+1) # acquire ϕⱼ(t=k+1)
    next_agent_state(agent, ϕₖ, world, X) # set ϕⱼ(t=k+1); acquire and set z(t=k+1)
    next_agent_time(agent) # k ⟵ k+1
end