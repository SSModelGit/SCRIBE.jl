ENV["GKSwstype"] = "100"

using Plots
using Serialization

include(joinpath(@__DIR__, "agent_number_scaling.jl"))

function save_paper_results_figure()
    asynchronous_settings = asynchronous_fusion_settings(:full)
    asynchronous_output = asynchronous_output_directory(:full)
    environment = open(
        deserialize,
        joinpath(asynchronous_output, "plot_context.jls"),
    )
    asynchronous_trials = load_asynchronous_trials(
        asynchronous_settings,
        asynchronous_output,
        environment[:mission_steps],
    )
    asynchronous_rows = reduce(
        vcat,
        getindex.(asynchronous_trials, :rows);
        init=Any[],
    )
    representative = selected_asynchronous_trial(asynchronous_trials)

    scaling_settings = agent_scaling_settings(:full)
    scaling_output = scaling_output_directory(:full)
    _, scaling_rows = aggregate_scaling_results(
        scaling_settings,
        scaling_output,
    )
    topologies = filter(
        ∈(Set(getindex.(scaling_rows, :topology))),
        (:sparse, :moderate, :dense),
    )

    figure = plot(
        rmse_panel(asynchronous_rows; render_profile=:paper),
        agent_rmse_panel(representative; render_profile=:paper),
        scaling_panel(
            scaling_rows,
            :fusion_seconds_per_agent_step,
            "(c) SCRIBE update time",
            "Fusion time / agent-update (s)";
            render_profile=:paper,
        ),
        scaling_panel(
            scaling_rows,
            :payload_bytes_per_agent_step,
            "(d) Communication payload",
            "Payload / agent-update (bytes)";
            render_profile=:paper,
            panel_position=:right,
        ),
        scaling_legend_panel(topologies; render_profile=:paper);
        layout=@layout([a b; c d; e{0.10h}]),
        size=(1400, 680),
    )
    output_dir = joinpath(@__DIR__, "res", "paper")
    mkpath(output_dir)
    save_workshop_figure(
        figure,
        joinpath(output_dir, "asynchronous_fusion_and_scaling"),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    save_paper_results_figure()
end
