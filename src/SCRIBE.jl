module SCRIBE

using Reexport
using LinearAlgebra
using GaussianDistributions: ⊕, Gaussian
using Parameters: @unpack

include("SCRIBEModels.jl")
@reexport using .SCRIBEModels

"""Network information stored locally by agent.

Knows its immediate neighbors, and thus who to check in the network graph.

Outbox: Messages to be provided to other agents.
    - Structure: Dictionary of specific values.
      - "lc" => Message for the current timepoint. Initialize to nothing.
      - "ln" => Message for the next timepoint. Initialize to nothing.
      - "lv" => The iteration cycle number (l-value). Initialize to zero.
      - "cvc" => Number of iterations of the "lc" and "ln" being within threshold of each other. Initialize to zero.
"""
struct NetworkConnector
    neighbors::Vector
    outbox::Dict

    NetworkConnector(neighbors::Vector, outbox::Dict) = new(neighbors, outbox)
end

abstract type EnvScribe end
abstract type EnvEstimators end

export EnvScribe, EnvEstimators, SCRIBEAgent, initialize_agent
export NetworkGraph, NetworkConnector, init_network_graph, connected_agents, update_network_graph_edges

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
    for (k,v) in new_edges
        ng.edges[k] = copy(v)
    end
    
    new_connectivity = connected_agents(new_edges)
    for (k,v) in new_connectivity
        ng.connectivity[k] = v
    end
end

function init_network_graph(edges::Dict{String, Vector{String}})
    NetworkGraph(Dict{String, SCRIBEAgent}(), edges, connected_agents(edges))
end

"""Creates a SCRIBEAgent appropriate for the KF system, from a KF Estimator.
"""
function initialize_agent(id::String, kf_estimators::KFEstimators, net_graph::NetworkGraph)
    let ag=kf_estimators.system
        nc=NetworkConnector(net_graph.edges[id], Dict{String, Any}("lc"=>nothing, "ln"=>nothing,
                                                                   "lv"=>0, "cvc"=>0,
                                                                   "prior"=>nothing, "innov"=>nothing))
        return SCRIBEAgent(id, ag.params, ag.bhv, [ag.estimates[1].observations.X], ag, nc, kf_estimators)
    end
end

include("consensus.jl")

include("CovarianceIntersection.jl")

end