using Plots
using Match: @match
using Statistics

publication_backend_title(backend) = @match backend begin
    :centralized => "Centralized (Ideal)"
    :scribe => "SCRIBE"
    :kf_only => "KF Only"
    :ci_only => "CI Only"
    :independent => "No Communication"
end

publication_backend_color(backend) = @match backend begin
    :centralized => :black
    :scribe => :royalblue
    :kf_only => :firebrick
    :ci_only => :purple
    :independent => :darkorange
end

function metric_aggregate(rows, backend, metric)
    let selected=filter(row -> row[:backend] == backend, rows),
        steps=sort(unique(getindex.(selected, :step)))
        values=map(steps) do step
            filter(
                isfinite,
                getindex.(
                    filter(row -> row[:step] == step, selected),
                    metric,
                ),
            )
        end
        Dict(
            :steps => steps,
            :mean => map(mean, values),
            :deviation => map(
                value -> std(value; corrected=false),
                values,
            ),
        )
    end
end

metric_transform(value, transform) = @match transform begin
    :identity => value
    :log10 => log10(max(value, eps()))
    :kibibytes => value / 1024
    :percent => 100 * value
end

function publication_metric_plot(
    rows,
    backends,
    metric,
    title,
    ylabel;
    transform=:identity,
)
    let plot_object=plot(
            ;
            title,
            xlabel="Measurement Round",
            ylabel,
        )
        foreach(backends) do backend
            aggregate=metric_aggregate(rows, backend, metric)
            mean_values=metric_transform.(aggregate[:mean], transform)
            deviation_values=@match transform begin
                :identity => aggregate[:deviation]
                :kibibytes => aggregate[:deviation] ./ 1024
                :percent => 100 .* aggregate[:deviation]
                :log10 => zeros(length(aggregate[:deviation]))
            end
            plot!(
                plot_object,
                aggregate[:steps],
                mean_values;
                ribbon=deviation_values,
                linewidth=2,
                color=publication_backend_color(backend),
                label=publication_backend_title(backend),
                fillalpha=0.12,
            )
        end
        plot_object
    end
end

function phase_boundaries!(plot_object, settings)
    let first_boundary=settings[:phase_steps][1],
        second_boundary=first_boundary + settings[:phase_steps][2]
        vline!(
            plot_object,
            [first_boundary, second_boundary];
            color=:gray,
            linestyle=:dash,
            label=false,
        )
        plot_object
    end
end

function exactness_figure(rows)
    let backends=(:centralized, :scribe, :independent),
        rmse=publication_metric_plot(
            rows,
            backends,
            :rmse,
            "Field Reconstruction under Continuous Sharing",
            "RMSE [normalized field units]",
        ),
        consensus=publication_metric_plot(
            rows,
            (:scribe, :independent),
            :prediction_consensus_rmse,
            "Inter-Agent Disagreement",
            "log₁₀(Prediction RMSE [normalized field units])";
            transform=:log10,
        ),
        centralized_gap=publication_metric_plot(
            rows,
            (:scribe, :independent),
            :centralized_prediction_gap,
            "Gap to Pooled-Observation Posterior",
            "log₁₀(Prediction RMSE [normalized field units])";
            transform=:log10,
        ),
        nees=publication_metric_plot(
            rows,
            backends,
            :normalized_nees,
            "Coefficient-State Consistency",
            "Normalized NEES [unitless]",
        )
        hline!(nees, [1.0]; color=:gray, linestyle=:dash, label=false)
        plot(
            rmse,
            consensus,
            centralized_gap,
            nees;
            layout=(2, 2),
            size=(1000, 760),
            margin=2 * Plots.mm,
            titlefontsize=10,
            guidefontsize=9,
            tickfontsize=8,
            legendfontsize=8,
        )
    end
end

function reconnection_figure(rows, settings)
    let backends=(:centralized, :scribe, :kf_only, :ci_only, :independent),
        rmse=publication_metric_plot(
            rows,
            backends,
            :rmse,
            "Field Reconstruction during Communication Loss",
            "RMSE [normalized field units]",
        ),
        consensus=publication_metric_plot(
            rows,
            (:scribe, :kf_only, :ci_only, :independent),
            :prediction_consensus_rmse,
            "Inter-Agent Disagreement",
            "log₁₀(Prediction RMSE [normalized field units])";
            transform=:log10,
        ),
        coverage=publication_metric_plot(
            rows,
            backends,
            :interval_coverage,
            "Posterior Calibration",
            "95% Interval Coverage [%]";
            transform=:percent,
        ),
        centralized_gap=publication_metric_plot(
            rows,
            (:scribe, :kf_only, :ci_only, :independent),
            :centralized_prediction_gap,
            "Gap to Pooled-Observation Posterior",
            "log₁₀(Prediction RMSE [normalized field units])";
            transform=:log10,
        ),
        conservatism=publication_metric_plot(
            rows,
            (:scribe, :kf_only, :ci_only, :independent),
            :minimum_conservatism_eigenvalue,
            "Covariance Conservatism",
            "Minimum eigenvalue of P − Pₒ [field units²]",
        ),
        communication=publication_metric_plot(
            rows,
            (:scribe, :kf_only, :ci_only),
            :cumulative_bytes,
            "Cumulative Model-Exchange Payload",
            "Data Transmitted [KiB]";
            transform=:kibibytes,
        )
        hline!(coverage, [95.0]; color=:gray, linestyle=:dash, label=false)
        hline!(
            conservatism,
            [0.0];
            color=:gray,
            linestyle=:dash,
            label=false,
        )
        foreach(
            plot_object -> phase_boundaries!(plot_object, settings),
            (
                rmse,
                consensus,
                coverage,
                centralized_gap,
                conservatism,
                communication,
            ),
        )
        plot(
            rmse,
            consensus,
            coverage,
            centralized_gap,
            conservatism,
            communication;
            layout=(3, 2),
            size=(1200, 1200),
            margin=2 * Plots.mm,
            titlefontsize=10,
            guidefontsize=9,
            tickfontsize=8,
            legendfontsize=8,
        )
    end
end

function misspecification_figure(rows, settings)
    let backends=(:centralized, :scribe, :kf_only, :ci_only, :independent),
        rmse=publication_metric_plot(
            rows,
            backends,
            :rmse,
            "Field Reconstruction under Model Mismatch",
            "RMSE [normalized field units]",
        ),
        normalized_rmse=publication_metric_plot(
            rows,
            backends,
            :normalized_rmse,
            "Normalized Field Reconstruction Error",
            "NRMSE [unitless]",
        ),
        log_likelihood=publication_metric_plot(
            rows,
            backends,
            :predictive_log_likelihood,
            "Predictive Log Likelihood",
            "Mean log p(y) [nats]",
        ),
        coverage=publication_metric_plot(
            rows,
            backends,
            :interval_coverage,
            "Posterior Calibration",
            "95% Interval Coverage [%]";
            transform=:percent,
        ),
        uncertainty=publication_metric_plot(
            rows,
            backends,
            :integrated_uncertainty,
            "Posterior Uncertainty",
            "Mean Field Variance [normalized field units²]",
        ),
        consensus=publication_metric_plot(
            rows,
            (:scribe, :kf_only, :ci_only, :independent),
            :prediction_consensus_rmse,
            "Inter-Agent Disagreement",
            "log₁₀(Prediction RMSE [normalized field units])";
            transform=:log10,
        )
        hline!(coverage, [95.0]; color=:gray, linestyle=:dash, label=false)
        foreach(
            plot_object -> phase_boundaries!(plot_object, settings),
            (
                rmse,
                normalized_rmse,
                log_likelihood,
                coverage,
                uncertainty,
                consensus,
            ),
        )
        plot(
            rmse,
            normalized_rmse,
            log_likelihood,
            coverage,
            uncertainty,
            consensus;
            layout=(3, 2),
            size=(1200, 1200),
            margin=2 * Plots.mm,
            titlefontsize=10,
            guidefontsize=9,
            tickfontsize=8,
            legendfontsize=8,
        )
    end
end

function plot_publication_experiment(
    experiment,
    rows,
    settings,
    output_dir,
)
    let figure=(@match experiment begin
            1 => exactness_figure(rows)
            2 => reconnection_figure(rows, settings)
            3 => misspecification_figure(rows, settings)
        end),
        output_path=joinpath(output_dir, "aggregate_metrics.png")
        savefig(figure, output_path)
        println("Plot saved at $output_path")
        output_path
    end
end
