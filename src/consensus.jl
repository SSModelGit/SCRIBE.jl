export network_averaging_precheck, network_averaging_update, network_averaging_postcheck
export distributed_fusion

function network_averaging_precheck(k::Integer, agent_id::String, ng::NetworkGraph)
    @unpack estimators, net_conn = ng.vertices[agent_id]
    net_conn.outbox["lc"] = compute_innov_from_obs(estimators, k)
    net_conn.outbox["ln"] = nothing
    net_conn.outbox["lv"] = 1
    net_conn.outbox["cvc"] = 0
end

function network_averaging_update(agent_id::String, ng::NetworkGraph)
    nc = ng.vertices[agent_id].net_conn
    xᵢ = nc.outbox["lc"]
    xⱼ = Dict([(nb, ng.vertices[nb].net_conn.outbox["lc"]) for nb in nc.neighbors])

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

function network_averaging_postcheck(agent_id::String, ng::NetworkGraph, threshold::Float64)
    let nc = ng.vertices[agent_id].net_conn, cvc = nc.outbox["cvc"]
        shifts = map(i->norm(abs.(nc.outbox["lc"][i] - nc.outbox["ln"][i])), [1,2])
        nc.outbox["cvc"] = (all(shifts .< threshold) ? cvc+1 : 0)
        nc.outbox["lc"] = (copy(nc.outbox["ln"][1]), copy(nc.outbox["ln"][2]))
        nc.outbox["ln"] = nothing
        nc.outbox["lv"] += 1

        return nc.outbox["cvc"] > length(ng.vertices)
    end
end

function distributed_fusion(k::Integer, agent_id::String, ng::NetworkGraph)
    @unpack estimators, net_conn = ng.vertices[agent_id]
    # Compute priors from current system state at t=k and previous information state at t=k-1
    prior = compute_info_priors(estimators, k)
    nₐ=length(ng.vertices)

    # Compute innovations for t=k+1 from current observation at t=k
    δĪ=net_conn.outbox["lc"][1]
    δī=net_conn.outbox["lc"][2]

    return KFEnvInfo(prior[2]+nₐ*δī, prior[1]+nₐ*δĪ, δī, δĪ)
end