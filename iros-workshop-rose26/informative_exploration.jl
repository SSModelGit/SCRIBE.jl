ENV["GKSwstype"] = "100"

include(joinpath(@__DIR__, "informative_exploration_problem.jl"))

using Plots
using Statistics

publication_exploration_settings(profile) = @match profile begin
    :full => (
        seed=18,
        navigation_points=33,
        evaluation_points=51,
        n_samples=108,
        lookahead=8,
        time_budget=0.10,
        noise_variance=0.08,
        process_variance=1e-10,
        animation_fps=4,
    )
    :smoke => (
        seed=18,
        navigation_points=17,
        evaluation_points=21,
        n_samples=16,
        lookahead=3,
        time_budget=0.02,
        noise_variance=0.08,
        process_variance=1e-10,
        animation_fps=2,
    )
    _ => throw(ArgumentError("Use the `full` or `smoke` publication profile."))
end

function publication_frame_indices(visualization)
    unique([
        1,
        2,
        cld(length(visualization), 2),
        length(visualization),
    ])
end

function publication_truth_panel(visualization)
    let grid=visualization.grid,
        values=visualization.ground_truth,
        field_extent=maximum(abs, [
            values;
            reduce(
                vcat,
                getproperty.(
                    getproperty.(visualization.frames, :summary),
                    :prediction,
                ),
            )
        ])
        heatmap(
            grid.x,
            grid.y,
            reshape(values, length(grid.x), length(grid.y))';
            aspect_ratio=:equal,
            color=:balance,
            clims=(-field_extent, field_extent),
            xlabel="x",
            ylabel="y",
            title="Ground truth",
            titlefontsize=11,
            guidefontsize=9,
            tickfontsize=8,
            margin=0.5 * Plots.mm,
            left_margin=2.5 * Plots.mm,
            bottom_margin=2.5 * Plots.mm,
        )
    end
end

function publication_error_panel(visualization)
    let samples=getproperty.(visualization.frames, :n_samples),
        summaries=getproperty.(visualization.frames, :summary),
        plot_object=plot(
            samples,
            getproperty.(summaries, :rmse);
            label="RMSE",
            linewidth=2,
            xlabel="samples",
            ylabel="field error",
            title="Prediction error",
        )
        plot!(
            plot_object,
            samples,
            getproperty.(summaries, :mae);
            label="MAE",
            linewidth=2,
        )
        plot_object
    end
end

function publication_information_panels(visualization)
    let samples=getproperty.(visualization.frames, :n_samples),
        summaries=getproperty.(visualization.frames, :summary),
        uncertainty_plot=plot(
            samples,
            getproperty.(summaries, :mean_uncertainty);
            label="mean posterior σ",
            linewidth=2,
            xlabel="samples",
            ylabel="posterior σ",
            title="Posterior uncertainty",
        ),
        information_plot=plot(
            samples,
            getproperty.(visualization.frames, :expected_information);
            label="expected MI",
            linewidth=2,
            xlabel="samples",
            ylabel="nats",
            title="Expected information",
        )
        plot!(
            information_plot,
            samples,
            getproperty.(visualization.frames, :realized_information);
            label="realized KL",
            linewidth=2,
        )
        (uncertainty_plot, information_plot)
    end
end

function save_publication_exploration_figure(visualization, output_dir)
    let frames=publication_frame_indices(visualization),
        posterior_plots=map(
            frame_index -> plot_posterior_mean_map(
                visualization;
                frame_index,
            ),
            frames[2:end],
        ),
        (uncertainty_curve, information_curve)=
            publication_information_panels(visualization),
        figure=plot(
            publication_truth_panel(visualization),
            posterior_plots...,
            plot_posterior_uncertainty_map(visualization),
            publication_error_panel(visualization),
            uncertainty_curve,
            information_curve;
            layout=(2, 4),
            size=(1500, 660),
            margin=1 * Plots.mm,
        ),
        output_path=joinpath(
            output_dir,
            "informative_exploration_summary.png",
        )
        savefig(figure, output_path)
        println("Plot saved at $output_path")
        output_path
    end
end

function save_publication_exploration_metrics(
    result,
    visualization,
    output_dir,
)
    open(joinpath(output_dir, "metric_history.csv"), "w") do io
        println(
            io,
            "samples,x,y,observation,rmse,normalized_rmse,mae," *
            "predictive_log_likelihood,interval_coverage," *
            "integrated_uncertainty,mean_uncertainty," *
            "maximum_uncertainty,integrated_variance_reduction," *
            "expected_mutual_information,realized_kl",
        )
        foreach(visualization.frames) do evaluated
            summary=evaluated.summary
            sample=evaluated.n_samples
            location=sample == 0 ?
                first(result.sampling_locations) :
                result.sampling_locations[sample]
            observation=sample == 0 ? "" : result.observations[sample]
            println(
                io,
                "$(sample),$(location[1]),$(location[2])," *
                "$(observation),$(summary.rmse)," *
                "$(summary.normalized_rmse),$(summary.mae)," *
                "$(summary.predictive_log_likelihood)," *
                "$(summary.interval_coverage)," *
                "$(summary.integrated_uncertainty)," *
                "$(summary.mean_uncertainty)," *
                "$(summary.maximum_uncertainty)," *
                "$(evaluated.integrated_variance_reduction)," *
                "$(evaluated.expected_information)," *
                "$(evaluated.realized_information)",
            )
        end
    end
end

function save_publication_exploration(
    result,
    output_dir,
    settings;
    animations=false,
)
    let visualization=result.visualization
        save_static_visualizations(
            visualization;
            output_dir,
            metrics=[
                :posterior_mean,
                :posterior_uncertainty,
                :posterior_against_ground_truth,
                :absolute_error,
                :metric_history,
                :predictions_against_ground_truth,
                :posterior_summary,
            ],
        )
        if animations
            save_animated_visualizations(
                visualization;
                output_dir,
                metrics=[
                    :posterior_mean,
                    :posterior_uncertainty,
                    :posterior_against_ground_truth,
                    :posterior_summary,
                ],
                fps=settings.animation_fps,
            )
        end
        save_publication_exploration_metrics(
            result,
            visualization,
            output_dir,
        )
        save_publication_exploration_figure(visualization, output_dir)
        visualization
    end
end

function print_experiment_summary(visualization, output_dir)
    let initial=first(visualization.frames).summary,
        final=last(visualization.frames).summary
        println("SCRIBE-backed VulcanJ exploration complete.")
        println("  samples: $(last(visualization.frames).n_samples)")
        println(
            "  RMSE: $(round(initial.rmse; digits=4)) → " *
            "$(round(final.rmse; digits=4))",
        )
        println(
            "  mean posterior σ: " *
            "$(round(initial.mean_uncertainty; digits=4)) → " *
            "$(round(final.mean_uncertainty; digits=4))",
        )
        println("  results: $output_dir")
    end
end

function planning_publication_main(
    profile=:full;
    animations=false,
)
    let settings=publication_exploration_settings(profile),
        problem=exploration_problem(settings),
        result=run_exploration(
            problem.mdp,
            problem.start,
            settings,
        ),
        output_dir=joinpath(
            @__DIR__,
            "res",
            "single_agent_diagnostic",
            String(profile),
        ),
        visualization=save_publication_exploration(
            result,
            output_dir,
            settings;
            animations,
        )
        print_experiment_summary(visualization, output_dir)
        (result=result, visualization=visualization)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    let profile=isempty(ARGS) ? :full : Symbol(first(ARGS)),
        animations=length(ARGS) > 1 && ARGS[2] == "animations"
        planning_publication_main(profile; animations)
    end
end
