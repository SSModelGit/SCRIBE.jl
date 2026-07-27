using SCRIBE
using JLD2: @load, load
using LinearAlgebra: norm
using Plots: heatmap, plot, plot!, savefig
using Statistics: mean

function spatial_rmse_eval_plot(run_name::String; layout_size=(900, 500),
                                grid_step=0.2)
    png_name = "test/res_plots/" * run_name * "_spatial_rmse.png"
    mkpath(dirname(png_name))
    @load "test/res_data/" * run_name * ".jld2" gt_model ng space_corners

    x_range = space_corners[1]:grid_step:space_corners[2]
    y_range = copy(x_range)
    err_sq(gt, estimate, location) =
        abs2(predict_SCRIBEModel(gt, location) -
             predict_SCRIBEModel(estimate, location))
    model_rmse(gt, estimate) =
        sqrt(mean(err_sq(gt, estimate, [x, y])
                  for x in x_range for y in y_range))
    agent_ids = sort(collect(keys(ng.edges)))
    model_rmses = Dict(
        aid => [
            model_rmse(
                gt_model[i],
                ng.vertices[aid].agent.estimates[i].estimate,
            )
            for i in eachindex(gt_model)
        ]
        for aid in agent_ids
    )

    rmse_plot = plot(
        size=layout_size,
        xlabel="Time (discrete steps)",
        ylabel="Spatial RMSE",
        title="Agent model error against ground truth",
    )
    for aid in agent_ids
        plot!(rmse_plot, eachindex(gt_model), model_rmses[aid];
              label="Agent " * aid[end], lw=2)
    end
    savefig(rmse_plot, png_name)
    println("Plot saved at " * png_name)
    png_name
end

frechet_dist_eval_plot(run_name::String; kwargs...) =
    spatial_rmse_eval_plot(run_name; kwargs...)

function error_mapf(gt::SCRIBEModel, model::SCRIBEModel, x::Vector;
                    mode=:norm)
    if mode === :norm
        return abs(predict_SCRIBEModel(gt, x) -
                   predict_SCRIBEModel(model, x))
    elseif mode === :tane
        return abs(2 * atan(predict_SCRIBEModel(gt, x) /
                            predict_SCRIBEModel(model, x)) -
                   π / 2) / (π / 2)
    end
    throw(ArgumentError("Unknown error-map mode: $mode"))
end

function error_map_plots(run_name::String; layout_size=(1200, 800),
                         mode=:norm, grid_step=0.2)
    png_name = "test/res_plots/err_maps/" * run_name * "_err_map.png"
    mkpath(dirname(png_name))
    @load "test/res_data/" * run_name * ".jld2" gt_model ng space_corners
    gt = gt_model[end]

    agent_ids = sort(collect(keys(ng.vertices)))
    num_cols = Integer(ceil(sqrt(length(agent_ids))))
    num_rows = Integer(ceil(length(agent_ids) / num_cols))
    x_range = space_corners[1]:grid_step:space_corners[2]
    y_range = copy(x_range)
    error_values = Dict(
        aid => [
            error_mapf(
                gt,
                ng.vertices[aid].agent.estimates[end].estimate,
                [x, y];
                mode,
            )
            for y in y_range, x in x_range
        ]
        for aid in agent_ids
    )
    shared_clims = (0.0, maximum(maximum, values(error_values)))
    error_plots = Any[]
    for aid in agent_ids
        history = reduce(vcat, ng.vertices[aid].history)
        agent_plot = heatmap(
            x_range,
            y_range,
            error_values[aid];
            color=:ice,
            clims=shared_clims,
            title="Agent " * aid[end] * " absolute error",
        )
        plot!(agent_plot, copy(history[:, 1]), copy(history[:, 2]);
              linestyle=:dash, marker=:xcross, linecolor=:red,
              label="Sampling sites")
        push!(error_plots, agent_plot)
    end

    combined_plot = plot(error_plots...; layout=(num_rows, num_cols),
                         size=layout_size)
    savefig(combined_plot, png_name)
    println("Plot saved at " * png_name)
    png_name
end

function estimate_map_plots(run_name::String; layout_size=(1200, 800),
                            grid_step=0.2)
    png_name = "test/res_plots/est_maps/" * run_name * "_est_map.png"
    mkpath(dirname(png_name))
    @load "test/res_data/" * run_name * ".jld2" gt_model ng space_corners
    gt = gt_model[end]

    agent_ids = sort(collect(keys(ng.vertices)))
    num_plots = length(agent_ids) + 1
    num_cols = Integer(ceil(sqrt(num_plots)))
    num_rows = Integer(ceil(num_plots / num_cols))
    x_range = space_corners[1]:grid_step:space_corners[2]
    y_range = copy(x_range)
    all_models = [
        gt;
        [ng.vertices[aid].agent.estimates[end].estimate for aid in agent_ids]
    ]
    shared_clims = extrema(
        predict_SCRIBEModel(model, [x, y])
        for model in all_models for x in x_range for y in y_range
    )

    estimate_plots = Any[
        heatmap(
            x_range,
            y_range,
            [predict_SCRIBEModel(gt, [x, y])
             for y in y_range, x in x_range];
            color=:viridis,
            clims=shared_clims,
            title="Ground truth distribution",
        ),
    ]
    for aid in agent_ids
        model = ng.vertices[aid].agent.estimates[end].estimate
        history = reduce(vcat, ng.vertices[aid].history)
        agent_plot = heatmap(
            x_range,
            y_range,
            [predict_SCRIBEModel(model, [x, y])
             for y in y_range, x in x_range];
            color=:viridis,
            clims=shared_clims,
            title="Agent " * aid[end] * " estimate",
        )
        plot!(agent_plot, copy(history[:, 1]), copy(history[:, 2]);
              linestyle=:dash, marker=:xcross, linecolor=:red,
              label="Sampling sites", lw=2)
        push!(estimate_plots, agent_plot)
    end

    combined_plot = plot(estimate_plots...; layout=(num_rows, num_cols),
                         size=layout_size)
    savefig(combined_plot, png_name)
    println("Plot saved at " * png_name)
    png_name
end

function communication_comparison_plots(no_comm_run::String,
                                        distributed_run::String;
                                        grid_step=0.25)
    run_specs = [
        ("No communication", no_comm_run, :dash),
        ("Distributed communication", distributed_run, :solid),
    ]
    run_data = [
        let data = load("test/res_data/" * run_name * ".jld2")
            (
                label=label,
                run_name=run_name,
                linestyle=linestyle,
                gt_model=data["gt_model"],
                ng=data["ng"],
                space_corners=data["space_corners"],
            )
        end
        for (label, run_name, linestyle) in run_specs
    ]

    @assert length(run_data[1].gt_model) == length(run_data[2].gt_model)
    @assert all(
        isapprox(
            run_data[1].gt_model[k].ϕ,
            run_data[2].gt_model[k].ϕ;
            rtol=0,
            atol=0,
        )
        for k in eachindex(run_data[1].gt_model)
    ) "Matched comparison did not reproduce identical ground-truth states."

    agent_ids = sort(collect(keys(run_data[1].ng.vertices)))
    agent_colors = [:royalblue, :darkorange, :seagreen]
    x_range =
        run_data[1].space_corners[1]:grid_step:run_data[1].space_corners[2]
    locations = [[x, y] for x in x_range for y in x_range]

    rmse_plot = plot(
        ylabel="Spatial RMSE against ground truth",
        title="Model accuracy: no communication vs distributed SCRIBE",
    )
    consensus_plot = plot(
        xlabel="Time (discrete steps)",
        ylabel="Maximum pairwise prediction RMSE",
        title="Inter-agent model disagreement",
    )

    for data in run_data
        prediction_cache = Dict(
            (aid, k) => [
                predict_SCRIBEModel(
                    data.ng.vertices[aid].agent.estimates[k].estimate,
                    location,
                )
                for location in locations
            ]
            for aid in agent_ids
            for k in eachindex(data.gt_model)
        )
        ground_truth_cache = Dict(
            k => [
                predict_SCRIBEModel(data.gt_model[k], location)
                for location in locations
            ]
            for k in eachindex(data.gt_model)
        )

        for (agent_index, aid) in enumerate(agent_ids)
            rmses = [
                sqrt(mean(
                    (prediction_cache[(aid, k)] .-
                     ground_truth_cache[k]).^2,
                ))
                for k in eachindex(data.gt_model)
            ]
            plot!(
                rmse_plot,
                eachindex(data.gt_model),
                rmses;
                color=agent_colors[agent_index],
                linestyle=data.linestyle,
                linewidth=2,
                label=data.label * " / " * aid,
            )
        end

        disagreement = [
            maximum(
                sqrt(mean(
                    (prediction_cache[(left, k)] .-
                     prediction_cache[(right, k)]).^2,
                ))
                for left in agent_ids
                for right in agent_ids
            )
            for k in eachindex(data.gt_model)
        ]
        plot!(
            consensus_plot,
            eachindex(data.gt_model),
            disagreement;
            linestyle=data.linestyle,
            linewidth=3,
            label=data.label,
        )
    end

    comparison_stem =
        "test/res_plots/3a_121w_fullGT_none_vs_dist_lowo_" *
        string(length(run_data[1].gt_model) - 1) * "s"
    performance_path = comparison_stem * "_performance.png"
    mkpath(dirname(performance_path))
    performance_figure = plot(
        rmse_plot,
        consensus_plot;
        layout=(2, 1),
        size=(1100, 900),
    )
    savefig(performance_figure, performance_path)
    println("Plot saved at ", performance_path)

    final_error_values = Dict{Tuple{Int, String}, Matrix{Float64}}()
    for (case_index, data) in enumerate(run_data)
        ground_truth = data.gt_model[end]
        for aid in agent_ids
            estimate = data.ng.vertices[aid].agent.estimates[end].estimate
            final_error_values[(case_index, aid)] = [
                abs(
                    predict_SCRIBEModel(ground_truth, [x, y]) -
                    predict_SCRIBEModel(estimate, [x, y]),
                )
                for y in x_range, x in x_range
            ]
        end
    end
    shared_clims = (0.0, maximum(maximum, values(final_error_values)))
    error_maps = Any[]
    for (case_index, data) in enumerate(run_data)
        for aid in agent_ids
            history = reduce(vcat, data.ng.vertices[aid].history)
            error_plot = heatmap(
                x_range,
                x_range,
                final_error_values[(case_index, aid)];
                color=:ice,
                clims=shared_clims,
                colorbar=aid == last(agent_ids),
                title=data.label * " — " * aid,
                titlefontsize=11,
            )
            plot!(
                error_plot,
                copy(history[:, 1]),
                copy(history[:, 2]);
                linestyle=:dash,
                marker=:xcross,
                linecolor=:red,
                label=aid == first(agent_ids) ? "Sampling sites" : "",
            )
            push!(error_maps, error_plot)
        end
    end
    error_map_path = comparison_stem * "_final_error_maps.png"
    error_map_figure = plot(
        error_maps...;
        layout=(length(run_data), length(agent_ids)),
        size=(1400, 850),
    )
    savefig(error_map_figure, error_map_path)
    println("Plot saved at ", error_map_path)

    [performance_path, error_map_path]
end
