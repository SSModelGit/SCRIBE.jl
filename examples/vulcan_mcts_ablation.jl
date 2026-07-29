include("vulcan_model_comparison.jl")

abstract type AblationBackend end

struct MultiscaleSCRIBE <: AblationBackend end
struct LocalSCRIBE <: AblationBackend end
struct SquaredExponentialGP <: AblationBackend end

ablation_name(::MultiscaleSCRIBE) = "multiscale_scribe"
ablation_name(::LocalSCRIBE) = "local_scribe"
ablation_name(::SquaredExponentialGP) = "se_gp"

ablation_title(::MultiscaleSCRIBE) = "Multiscale SCRIBE"
ablation_title(::LocalSCRIBE) = "Local-only SCRIBE"
ablation_title(::SquaredExponentialGP) = "SE GP"

ablation_settings(profile) = @match profile begin
    :full => (
        seeds=(41, 53, 67, 79, 97),
        n_samples=36,
        navigation_points=33,
        evaluation_points=41,
        reward_points=17,
        lookahead=4,
        planning_iterations=12,
        time_budget=0.001,
        animation_fps=4,
    )
    :smoke => (
        seeds=(41, 53),
        n_samples=6,
        navigation_points=17,
        evaluation_points=21,
        reward_points=11,
        lookahead=3,
        planning_iterations=6,
        time_budget=0.001,
        animation_fps=2,
    )
    _ => throw(ArgumentError("Use the `full` or `smoke` ablation profile."))
end

function local_scribe_model(scenario, noise_variance, evaluation_locations)
    let centers=complicated_local_centers(),
        params=comparison_scribe_parameters(centers, [1.35]),
        smodel=initialize_SCRIBEModel_from_parameters(params),
        covariance=coefficient_prior_covariance(
            centers;
            coefficient_variance=1.0,
            correlation_length=1.3,
        ),
        information=normalized_basis_information(
            smodel,
            evaluation_locations,
            covariance;
            field_variance=prior_field_deviation(scenario)^2,
        )
        SCRIBEModelState(smodel, information, noise_variance)
    end
end

function ablation_model(
    ::MultiscaleSCRIBE,
    scenario,
    noise_variance,
    evaluation_locations,
)
    comparison_scribe_model(
        scenario,
        noise_variance,
        evaluation_locations,
    )
end

function ablation_model(
    ::LocalSCRIBE,
    scenario,
    noise_variance,
    evaluation_locations,
)
    local_scribe_model(
        scenario,
        noise_variance,
        evaluation_locations,
    )
end

function ablation_model(
    ::SquaredExponentialGP,
    scenario,
    noise_variance,
    _,
)
    comparison_gp_model(scenario, noise_variance)
end

function ablation_run_settings(settings, seed)
    (
        seed=seed,
        navigation_points=settings.navigation_points,
        evaluation_points=settings.evaluation_points,
        reward_points=settings.reward_points,
        n_samples=settings.n_samples,
        lookahead=settings.lookahead,
        planning_iterations=settings.planning_iterations,
        time_budget=settings.time_budget,
        animation_fps=settings.animation_fps,
    )
end

function ablation_records(records)
    map(records) do record
        summary=record.summary
        (
            n_samples=record.n_samples,
            planning_reward=record.planning_reward,
            summary=(
                rmse=summary.rmse,
                normalized_rmse=summary.normalized_rmse,
                predictive_log_likelihood=summary.predictive_log_likelihood,
                interval_coverage=summary.interval_coverage,
                integrated_uncertainty=summary.integrated_uncertainty,
            ),
        )
    end
end

function run_ablation(
    backend,
    scenario,
    settings,
    seed,
    grid,
)
    let run_settings=ablation_run_settings(settings, seed),
        reward_locations=comparison_reward_locations(run_settings),
        initial_model=ablation_model(
            backend,
            scenario,
            0.08,
            reward_locations,
        ),
        problem=comparison_problem(
            run_settings,
            scenario,
            initial_model,
            reward_locations,
        ),
        result=run_model_comparison(
            problem.mdp,
            problem.start,
            run_settings,
            grid,
        )
        (
            backend=backend,
            seed=seed,
            records=ablation_records(result.records),
            runtime=result.runtime,
        )
    end
end

function ablation_metric_series(run, metric)
    metric == :planning_reward ?
        getproperty.(run.records, :planning_reward) :
        metric_series(run.records, metric)
end

function aggregate_ablation_metric(runs, backend, metric)
    let selected=filter(run -> run.backend isa typeof(backend), runs),
        values=reduce(
            hcat,
            map(run -> ablation_metric_series(run, metric), selected),
        )
        (
            samples=getproperty.(first(selected).records, :n_samples),
            mean=vec(mean(values; dims=2)),
            deviation=vec(std(values; dims=2, corrected=false)),
        )
    end
end

function ablation_metric_plot(runs, backends, metric, title, ylabel)
    let plot_object=plot(
            ;
            xlabel="samples",
            ylabel,
            title,
        )
        foreach(backends) do backend
            aggregate=aggregate_ablation_metric(runs, backend, metric)
            plot!(
                plot_object,
                aggregate.samples,
                aggregate.mean;
                ribbon=aggregate.deviation,
                fillalpha=0.15,
                linewidth=2,
                label=ablation_title(backend),
            )
        end
        plot_object
    end
end

function save_ablation_metrics(runs, backends, output_dir)
    let rmse=ablation_metric_plot(
            runs,
            backends,
            :rmse,
            "Field RMSE",
            "RMSE",
        ),
        normalized_rmse=ablation_metric_plot(
            runs,
            backends,
            :normalized_rmse,
            "Normalized field RMSE",
            "NRMSE",
        ),
        predictive_log_likelihood=ablation_metric_plot(
            runs,
            backends,
            :predictive_log_likelihood,
            "Mean predictive log likelihood",
            "mean log p(y)",
        ),
        interval_coverage=ablation_metric_plot(
            runs,
            backends,
            :interval_coverage,
            "95% latent interval coverage",
            "coverage",
        ),
        integrated_uncertainty=ablation_metric_plot(
            runs,
            backends,
            :integrated_uncertainty,
            "Integrated posterior uncertainty",
            "mean variance",
        ),
        planning_reward=ablation_metric_plot(
            runs,
            backends,
            :planning_reward,
            "Planning reward",
            "mean variance reduction",
        )
        hline!(
            interval_coverage,
            [0.95];
            color=:black,
            linestyle=:dash,
            label=false,
        )
        figure=plot(
            rmse,
            normalized_rmse,
            predictive_log_likelihood,
            interval_coverage,
            integrated_uncertainty,
            planning_reward;
            layout=(3, 2),
            size=(1200, 1200),
        )
        savefig(figure, joinpath(output_dir, "aggregate_metrics.png"))
    end
end

function save_ablation_runs(runs, output_dir)
    open(joinpath(output_dir, "final_metrics.csv"), "w") do io
        println(
            io,
            "backend,seed,rmse,normalized_rmse,predictive_log_likelihood," *
            "interval_coverage,integrated_uncertainty,runtime",
        )
        foreach(runs) do run
            summary=last(run.records).summary
            println(
                io,
                "$(ablation_name(run.backend)),$(run.seed)," *
                "$(summary.rmse),$(summary.normalized_rmse)," *
                "$(summary.predictive_log_likelihood)," *
                "$(summary.interval_coverage)," *
                "$(summary.integrated_uncertainty),$(run.runtime)",
            )
        end
    end
end

function final_ablation_summary(runs, backend)
    let selected=filter(run -> run.backend isa typeof(backend), runs),
        summaries=getproperty.(last.(getproperty.(selected, :records)), :summary)
        (
            rmse=getproperty.(summaries, :rmse),
            normalized_rmse=getproperty.(summaries, :normalized_rmse),
            predictive_log_likelihood=getproperty.(
                summaries,
                :predictive_log_likelihood,
            ),
            interval_coverage=getproperty.(summaries, :interval_coverage),
            integrated_uncertainty=getproperty.(
                summaries,
                :integrated_uncertainty,
            ),
        )
    end
end

function save_aggregate_ablation_summary(runs, backends, output_dir)
    open(joinpath(output_dir, "aggregate_summary.csv"), "w") do io
        println(
            io,
            "backend,rmse_mean,rmse_std,normalized_rmse_mean," *
            "predictive_log_likelihood_mean,interval_coverage_mean," *
            "integrated_uncertainty_mean",
        )
        foreach(backends) do backend
            metrics=final_ablation_summary(runs, backend)
            println(
                io,
                "$(ablation_name(backend)),$(mean(metrics.rmse))," *
                "$(std(metrics.rmse; corrected=false))," *
                "$(mean(metrics.normalized_rmse))," *
                "$(mean(metrics.predictive_log_likelihood))," *
                "$(mean(metrics.interval_coverage))," *
                "$(mean(metrics.integrated_uncertainty))",
            )
        end
    end
end

function print_ablation_summary(runs, backends, output_dir)
    println("Complicated-field MCTS ablation complete.")
    foreach(backends) do backend
        let metrics=final_ablation_summary(runs, backend)
            println(
                "  $(ablation_title(backend)): " *
                "RMSE=$(round(mean(metrics.rmse); digits=4))±" *
                "$(round(std(metrics.rmse; corrected=false); digits=4)), " *
                "NRMSE=$(round(mean(metrics.normalized_rmse); digits=4)), " *
                "log p=$(round(mean(metrics.predictive_log_likelihood); digits=4)), " *
                "coverage=$(round(mean(metrics.interval_coverage); digits=4)), " *
                "IV=$(round(mean(metrics.integrated_uncertainty); digits=4))",
            )
        end
    end
    println("  results: $output_dir")
end

function ablation_main(profile=:full)
    let settings=ablation_settings(profile),
        scenario=ComplicatedComparison(),
        grid=surface_grid(settings.evaluation_points),
        backends=(
            MultiscaleSCRIBE(),
            LocalSCRIBE(),
            SquaredExponentialGP(),
        ),
        runs=vec(
            map(
                Iterators.product(backends, settings.seeds),
            ) do experiment
                backend, seed=experiment
                result=run_ablation(
                    backend,
                    scenario,
                    settings,
                    seed,
                    grid,
                )
                GC.gc()
                result
            end,
        ),
        output_dir=joinpath(
            @__DIR__,
            "res",
            "vulcan_model_comparison",
            "ablation",
            scenario_name(scenario),
            String(profile),
        )
        mkpath(output_dir)
        save_ablation_metrics(runs, backends, output_dir)
        save_ablation_runs(runs, output_dir)
        save_aggregate_ablation_summary(runs, backends, output_dir)
        print_ablation_summary(runs, backends, output_dir)
        runs
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    ablation_main(isempty(ARGS) ? :full : Symbol(first(ARGS)))
end
