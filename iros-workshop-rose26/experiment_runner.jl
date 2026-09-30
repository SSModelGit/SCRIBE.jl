using Match: @match

"""
    experiment_runner(experiments; profile=:full)

Run the selected workshop experiments sequentially, using their existing output paths.

Available symbols:
- `:experiment_1`, `:experiment_2`, `:experiment_3`: filtering comparisons.
- `:experiment_4`: distributed informative exploration.
- `:single_agent_diagnostic`: single-agent informative exploration.
- `:asynchronous_model_fusion`: opportunistic fusion during lawnmower surveys.
- `:agent_number_scaling`: agent count and communication scaling.
- `:opportunistic_exploration`: informative exploration with opportunistic sharing.
- `:eof_snapshot_comparison`: EOF reconstruction snapshot comparison (no profile).
- `:paper_results`: assemble the paper figure from existing full results (no profile).

The asynchronous fusion and scaling experiments retain their default checkpoint
resumption behavior. Snapshot comparison and paper figure generation ignore `profile`.

```julia
include("iros-workshop-rose26/experiment_runner.jl")
experiment_runner([:experiment_1, :experiment_4, :opportunistic_exploration])
experiment_runner([:opportunistic_exploration]; profile=:smoke)
```
"""
function experiment_runner(experiments; profile=:full)
    foreach(experiments) do experiment
        script, run = @match experiment begin
            :experiment_1 => ("filtering_experiments.jl",
                m -> m.run_filtering_publication_experiment(1; profile))
            :experiment_2 => ("filtering_experiments.jl",
                m -> m.run_filtering_publication_experiment(2; profile))
            :experiment_3 => ("filtering_experiments.jl",
                m -> m.run_filtering_publication_experiment(3; profile))
            :experiment_4 => ("distributed_informative_exploration.jl",
                m -> m.distributed_exploration_main(profile))
            :single_agent_diagnostic => ("informative_exploration.jl",
                m -> m.planning_publication_main(profile))
            :asynchronous_model_fusion => ("asynchronous_model_fusion.jl",
                m -> m.run_asynchronous_model_fusion(; profile))
            :agent_number_scaling => ("agent_number_scaling.jl",
                m -> m.run_agent_number_scaling(; profile))
            :opportunistic_exploration => ("opportunistic_comms_distributed_exploration.jl",
                m -> m.opportunistic_main(profile))
            :eof_snapshot_comparison => ("compare_ram_head_snapshots.jl",
                m -> m.compare_ram_head_snapshots())
            :paper_results => ("paper_results_figure.jl",
                m -> m.save_paper_results_figure())
        end

        # Isolate experiment definitions, including their nested includes.
        let scope=Module(gensym(:WorkshopExperiment))
            Core.eval(scope, :(include(path) = Base.include(@__MODULE__, path)))
            Base.include(scope, joinpath(@__DIR__, script))
            println("Running $(experiment) ($(profile))")
            Base.invokelatest(run, scope)
        end
        GC.gc()
    end
    nothing
end
