module SCRIBE

using Reexport
using LinearAlgebra
using GaussianDistributions: ⊕, Gaussian
using Parameters: @unpack

include("SCRIBEModels.jl")
@reexport using .SCRIBEModels

struct ConsensusMessage
    sender::String
    k::Integer
    stage::Symbol
    lv::Integer
    seq::Integer
    degree::Integer
    payload::Tuple{Matrix{Float64}, Vector{Float64}}
end

"""Network information stored locally by agent.

Knows its immediate neighbors, and thus who to check in the network graph.

Outbox: Messages to be provided to other agents.
- Structure: Dictionary of specific values.
    - "lc" => Local (current) consensus message payload for this agent. A tuple `(Matrix, Vector)` representing the information matrix and vector. Initialize to `nothing`.
    - "ln" => Next (proposed) consensus message payload for this agent. Same shape as `"lc"`. Initialize to `nothing`.
    - "lv" => The local iteration counter (l-value) for consensus rounds. Integer, initialize to `0`.
    - "cvc" => Consecutive valid convergence count: number of sequential rounds where `lc` and `ln` were within threshold. Integer, initialize to `0`.
    - "prior" => Stored prior information tuple used between stages. Initialize to `nothing`.
    - "innov" => Stored innovation tuple after averaging stage. Initialize to `nothing`.
    - "stage" => Symbol controlling current consensus stage; one of `:prior` or `:innov`. Initialize to `:prior`.
    - "seq" => Local sequence number for outgoing messages. Integer, incremented each time an outgoing message is queued. Initialize to `0`.
    - "cache" => Dictionary mapping neighbor id (String) => `ConsensusMessage`. Represents messages received from neighbors for the current round; acts as an incoming mailbox. Initialize to empty `Dict{String,Any}()`.
    - "msg_out" => The `ConsensusMessage` this agent has queued for delivery (or `nothing`). The communication/test harness should read this to simulate sending. Initialize to `nothing`.
    - "complete" => Boolean indicating whether the agent completed the full `:prior`→`:innov` fusion cycle for the current timestep. Initialize to `false`.
    - "n_eff" => Number of agents in the participating connected component. Integer, initialize to `1`.
    - "stage_ready" => Boolean indicating local convergence of the current consensus stage. Initialize to `false`.
"""
struct NetworkConnector
    neighbors::Vector
    outbox::Dict

    NetworkConnector(neighbors::Vector, outbox::Dict) = new(neighbors, outbox)
end

abstract type EnvScribe end
abstract type EnvEstimators end

export EnvScribe, EnvEstimators, SCRIBEAgent, initialize_agent
export ConsensusMessage, NetworkGraph, NetworkConnector, init_network_graph, connected_agents, update_network_graph_edges

struct SCRIBEAgent
    id::String
    params::SCRIBEModelParameters
    observer::SCRIBEObserverBehavior
    history::Vector{Matrix{Float64}}
    agent::EnvScribe
    net_conn::NetworkConnector
    estimators::EnvEstimators

    SCRIBEAgent(id::String, params::SCRIBEModelParameters, observer::SCRIBEObserverBehavior,
                history::Vector{Matrix{Float64}}, agent::EnvScribe, net_conn::NetworkConnector,
                estimators::EnvEstimators) = new(id, params, observer, history, agent, net_conn, estimators)
end

include("kalman_estimation.jl")
include("model_prediction.jl")
include("model_visualization.jl")

struct NetworkGraph
    vertices::Dict{String, SCRIBEAgent}
    edges::Dict{String, Vector{String}}
    connectivity::Dict{String, Integer}

    NetworkGraph(vertices::Dict, edges::Dict, connectivity::Dict) = new(vertices, edges, connectivity)
end

function connected_agents(edges::Dict{String, Vector{String}})
    vertex_list = keys(edges)
    visited = Set{String}()
    d_nₐ = Dict{String, Integer}()
    for aid in vertex_list; d_nₐ[aid] = 0; end

    function dfs(vertex)
        push!(visited, vertex)
        n = 1
        for nb in edges[vertex]
            if nb ∉ visited
                n += dfs(nb)
            end
        end
        return n
    end

    for aid in vertex_list
        if d_nₐ[aid] == 0
            d_nₐ[aid] = dfs(aid)
            for nb_id in visited
                d_nₐ[nb_id] = d_nₐ[aid]
            end
            empty!(visited)
        end
    end

    return d_nₐ
end

function update_network_graph_edges(new_edges::Dict, ng::NetworkGraph)
    vertex_ids = Set(keys(ng.edges))
    normalized_edges = Dict{String, Vector{String}}()
    for aid in keys(ng.edges)
        neighbors = unique(String.(get(new_edges, aid, String[])))
        filter!(nb -> nb != aid && nb in vertex_ids, neighbors)
        normalized_edges[aid] = neighbors
    end
    for (aid, neighbors) in normalized_edges
        for nb in neighbors
            @assert aid in normalized_edges[nb] "SCRIBE consensus requires an undirected communication graph; edge $aid→$nb has no reverse edge."
        end
        ng.edges[aid] = copy(neighbors)
        if haskey(ng.vertices, aid)
            # Compatibility mirror only. Consensus logic reads `ng.edges`.
            empty!(ng.vertices[aid].net_conn.neighbors)
            append!(ng.vertices[aid].net_conn.neighbors, neighbors)
        end
    end
    
    new_connectivity = connected_agents(ng.edges)
    for (k,v) in new_connectivity
        ng.connectivity[k] = v
    end
end

function init_network_graph(edges::Dict{String, Vector{String}})
    normalized_edges = Dict{String, Vector{String}}(
        aid => unique(filter(nb -> nb != aid, String.(neighbors)))
        for (aid, neighbors) in edges
    )
    for (aid, neighbors) in normalized_edges
        for nb in neighbors
            @assert haskey(normalized_edges, nb) "Unknown neighbor $nb for agent $aid."
            @assert aid in normalized_edges[nb] "SCRIBE consensus requires an undirected communication graph; edge $aid→$nb has no reverse edge."
        end
    end
    NetworkGraph(Dict{String, SCRIBEAgent}(), normalized_edges,
                 connected_agents(normalized_edges))
end

"""Creates a SCRIBEAgent appropriate for the KF system, from a KF Estimator.
"""
function initialize_agent(id::String, kf_estimators::KFEstimators, net_graph::NetworkGraph)
    let ag=kf_estimators.system
        nc=NetworkConnector(net_graph.edges[id], Dict{String, Any}("lc"=>nothing, "ln"=>nothing,
                                                                   "lv"=>0, "cvc"=>0,
                                                                   "prior"=>nothing, "innov"=>nothing,
                                                                   "stage"=>:prior, "seq"=>0,
                                                                   "cache"=>Dict{String, Any}(),
                                                                   "msg_out"=>nothing,
                                                                   "complete"=>false,
                                                                   "n_eff"=>1,
                                                                   "stage_ready"=>false))
        return SCRIBEAgent(id, ag.params, ag.bhv, [ag.estimates[1].observations.X], ag, nc, kf_estimators)
    end
end

"""Compatibility constructor for an isolated agent."""
function initialize_agent(kf_estimators::KFEstimators)
    agent_id = "agent1"
    net_graph = init_network_graph(Dict(agent_id => String[]))
    agent = initialize_agent(agent_id, kf_estimators, net_graph)
    net_graph.vertices[agent_id] = agent
    agent
end

include("consensus.jl")

end
