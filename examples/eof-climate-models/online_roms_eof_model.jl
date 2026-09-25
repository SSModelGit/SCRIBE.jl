using Plots
using SCRIBE
using Statistics: mean

include("_roms_data.jl")

const ONLINE_EOF_SCENARIOS = [
    Dict(:prior_snapshot => 1, :truth_snapshot => 4545),
    Dict(:prior_snapshot => 1515, :truth_snapshot => 4923),
    Dict(:prior_snapshot => 3030, :truth_snapshot => 5302),
    Dict(:prior_snapshot => 4544, :truth_snapshot => 5680),
]

function exploration_trajectory(roms, n_samples)
    wet_row = zeros(Int, prod(roms[:grid_shape]))
    wet_row[roms[:wet_mask]] = axes(roms[:locations], 1)
    grid_rows = reshape(wet_row, roms[:grid_shape]...)
    serpentine = Int[]
    for column in axes(grid_rows, 2)
        rows = grid_rows[:, column]
        append!(serpentine, column % 2 == 1 ? rows : reverse(rows))
    end
    filter!(row -> row != 0, serpentine)
    selected = round.(Int, range(1, length(serpentine); length=n_samples))
    data_rows = serpentine[selected]
    wet_cells = findall(reshape(roms[:wet_mask], roms[:grid_shape]...))
    grid_locations = hcat(
        getindex.(wet_cells[data_rows], 1),
        getindex.(wet_cells[data_rows], 2),
    )
    Dict(
        :locations => roms[:locations][data_rows, :],
        :grid_locations => grid_locations,
    )
end

function simulate_online_learning(
    params,
    roms,
    scenario;
    n_samples=400,
    sensor_variance=1e-4,
)
    truth = roms[:data][:, scenario[:truth_snapshot]]
    decomposition = params.decomposition
    truth_coefficients = eof_coefficients(params, truth)
    model = eof_model_at_coefficients(
        params,
        decomposition.coefficients[:, scenario[:prior_snapshot]],
    )
    scenario_params = model.params
    trajectory = exploration_trajectory(roms, n_samples)
    observer = DataObserver(
        roms_snapshot_sensor(truth, roms[:locations]),
        EOFObserverBehavior(sensor_variance),
    )
    first_location = permutedims(trajectory[:locations][1, :])
    estimators = initialize_KF(scenario_params, observer, first_location)
    agent = initialize_agent(estimators)

    predictions = [reconstruct_eof_field(model)]
    coefficients = [copy(scenario_params.ϕ₀)]
    field_rmse = [sqrt(mean(abs2, predictions[1] - truth))]

    for sample in 1:n_samples
        information = information_filter_update(agent.estimators, sample)
        posterior = posterior_coefficient_moments(information)[:μ]
        prediction = reconstruct_eof_field(model; coefficients=posterior)
        push!(coefficients, posterior)
        push!(predictions, prediction)
        push!(field_rmse, sqrt(mean(abs2, prediction - truth)))

        if sample < n_samples
            next_location = permutedims(trajectory[:locations][sample + 1, :])
            progress_agent_env_filter(agent.agent, information, next_location)
            push!(agent.history, next_location)
        end
    end

    Dict(
        :scenario => scenario,
        :truth => truth,
        :truth_coefficients => truth_coefficients,
        :trajectory => trajectory,
        :predictions => predictions,
        :coefficients => coefficients,
        :field_rmse => field_rmse,
        :agent => agent,
    )
end

function scenario_directory(scenario)
    name = "prior_$(lpad(scenario[:prior_snapshot], 4, '0'))_" *
        "truth_$(lpad(scenario[:truth_snapshot], 4, '0'))"
    joinpath(eof_result_dir(:online), name)
end

function save_learning_animation(result, roms; frame_stride=5, fps=10)
    scenario = result[:scenario]
    output_dir = scenario_directory(scenario)
    mkpath(output_dir)
    samples = unique(vcat(
        collect(0:frame_stride:(length(result[:predictions]) - 1)),
        length(result[:predictions]) - 1,
    ))
    color_limit = maximum(maximum(abs, field) for field in (
        result[:truth],
        result[:predictions][1],
    ))

    animation = @animate for sample in samples
        truth_panel = plot_roms_field(
            result[:truth],
            roms;
            title="Ground truth snapshot $(scenario[:truth_snapshot])",
            color=:balance,
            clims=(-color_limit, color_limit),
        )
        plot!(
            truth_panel;
            colorbar=false,
        )
        posterior_panel = plot_roms_field(
            result[:predictions][sample + 1],
            roms;
            title="Posterior after $sample samples\n" *
                "RMSE=$(round(result[:field_rmse][sample + 1]; digits=4))",
            color=:balance,
            clims=(-color_limit, color_limit),
        )
        plot!(
            posterior_panel;
            colorbar=false,
        )
        if sample > 0
            path = result[:trajectory][:grid_locations][1:sample, :]
            plot!(
                posterior_panel,
                path[:, 1],
                path[:, 2];
                color=:black,
                linewidth=0.7,
                linestyle=:dash,
                marker=:circle,
                markersize=1.2,
                markerstrokewidth=0,
                alpha=0.7,
                label=false,
            )
            scatter!(
                posterior_panel,
                [path[end, 1]],
                [path[end, 2]];
                color=:yellow,
                markerstrokecolor=:black,
                markersize=5,
                label=false,
            )
        end
        plot(
            truth_panel,
            posterior_panel;
            layout=(1, 2),
            size=(1100, 500),
            plot_title="Wrong prior snapshot $(scenario[:prior_snapshot]) " *
                "→ truth snapshot $(scenario[:truth_snapshot])",
        )
    end
    output_path = joinpath(output_dir, "posterior_learning.gif")
    gif(animation, output_path; fps)
    output_path
end

function save_rmse_plot(result)
    scenario = result[:scenario]
    output_dir = scenario_directory(scenario)
    mkpath(output_dir)
    figure = plot(
        0:(length(result[:field_rmse]) - 1),
        result[:field_rmse];
        xlabel="Samples gathered",
        ylabel="Full-field RMSE",
        title="EOF model correction: prior $(scenario[:prior_snapshot]) " *
            "→ truth $(scenario[:truth_snapshot])",
        linewidth=3,
        color=:navy,
        label=false,
        grid=true,
    )
    output_path = joinpath(output_dir, "field_rmse.png")
    savefig(figure, output_path)
    output_path
end

function save_reconstruction_comparison(result, roms)
    scenario = result[:scenario]
    fields = (
        result[:truth],
        first(result[:predictions]),
        last(result[:predictions]),
    )
    titles = (
        "Ground truth snapshot $(scenario[:truth_snapshot])",
        "Prior reconstruction\nfrom snapshot $(scenario[:prior_snapshot])",
        "Posterior reconstruction\nafter $(length(result[:field_rmse]) - 1) samples",
    )
    color_limit = maximum(maximum(abs, field) for field in fields)
    panels = map(zip(fields, titles)) do (field, title)
        panel = plot_roms_field(
            field,
            roms;
            title,
            color=:balance,
            clims=(-color_limit, color_limit),
        )
        plot!(
            panel;
            colorbar=false,
        )
    end
    figure = plot(
        panels...;
        layout=(1, 3),
        size=(1500, 470),
        plot_title="EOF correction: prior $(scenario[:prior_snapshot]) " *
            "→ truth $(scenario[:truth_snapshot])",
    )
    output_path = joinpath(
        scenario_directory(scenario),
        "reconstruction_comparison.png",
    )
    savefig(figure, output_path)
    output_path
end

"""Run four wrong-prior, fixed-truth ROMS assimilation demonstrations."""
function operate_roms_eof_model(;
    artifact=roms_eof_artifact(:u),
    scenarios=ONLINE_EOF_SCENARIOS,
    n_samples=400,
    sensor_variance=1e-4,
    frame_stride=5,
    animation_fps=10,
)
    params = load_eof_model_parameters(artifact)
    component = Symbol(String(metadata_scalar(params.metadata["component"])))
    temporal_stride = Int(metadata_scalar(params.metadata["temporal_stride"]))
    archive_data = read_roms_velocity(RAMS_HEAD_ARCHIVE, component)
    roms = prepare_roms_velocity(archive_data; temporal_stride)

    map(scenarios) do scenario
        println("Simulating prior snapshot $(scenario[:prior_snapshot]) " *
                "against truth snapshot $(scenario[:truth_snapshot])...")
        result = simulate_online_learning(
            params,
            roms,
            scenario;
            n_samples,
            sensor_variance,
        )
        animation = save_learning_animation(
            result,
            roms;
            frame_stride,
            fps=animation_fps,
        )
        rmse_plot = save_rmse_plot(result)
        comparison_plot = save_reconstruction_comparison(result, roms)
        coefficient_error = sqrt(mean(
            abs2,
            last(result[:coefficients]) - result[:truth_coefficients],
        ))
        println("  RMSE: $(round(first(result[:field_rmse]); digits=4)) → " *
                "$(round(last(result[:field_rmse]); digits=4))")
        println("  Final coefficient RMSE: " *
                "$(round(coefficient_error; digits=4))")
        println("  Saved $animation")
        println("  Saved $rmse_plot")
        println("  Saved $comparison_plot")
        result
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && operate_roms_eof_model()
