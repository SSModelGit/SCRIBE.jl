module R2SCRIBE

using SCRIBE

using ROS2
using JSON

mutable struct ROSAgent
    scribe::SCRIBEAgent
    config::ROSConnector
    node::ROS2.Node

    publishers::Dict
    subscriptions::Dict
    services::Dict

    session::Union{Nothing, String}
    k::Int
    rounds::Int
    threshold::Float64
    active::Bool
end

function SCRIBE.initialize_ros_agent(scribe::SCRIBEAgent, ros_config::ROSConnector;
                                     node, rounds, threshold)
    let r2node=ROSNode(node),
        ros_agent = ROSAgent(scribe, ros_config, r2node,
                        Dict{String, ROS2.Publisher}(),
                        Dict{String, ROS2.Subscription}(),
                        Dict{String, ROS2.Service}(),
                        nothing, 0, rounds, threshold, true)
        
        return ros_agent
    end
end

function write_consensus_msg_json(ros_agent::ROSAgent, msg::ConsensusMessage)
    let ros_msg=create_msg("std_msgs.msg.String")
        msg_json = JSON.json(Dict(
            "sender" => msg.sender,
            "receiver" => msg.receiver,
            "stage" => string(msg.stage),
            "seq" => msg.seq,
            "lv" => msg.lv,
            "cvc" => msg.cvc,
            "prior" => msg.prior,
            "innov" => msg.innov
        ))
        ros_msg.data = msg_json
        return ros_msg
    end
end

end