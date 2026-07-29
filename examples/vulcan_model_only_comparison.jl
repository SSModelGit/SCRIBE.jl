include("vulcan_model_comparison.jl")

function farthest_sample_locations(settings)
    let axis=collect(
            range(-5.0, 5.0; length=settings.navigation_points),
        ),
        candidates=regular_locations(axis, axis),
        center=argmin(map(eachrow(candidates)) do x
            sum(abs2, x)
        end),
        selected=[center],
        minimum_distance=map(eachrow(candidates)) do x
            sum(abs2, x - view(candidates, center, :))
        end

        foreach(2:settings.n_samples) do _
            index=argmax(minimum_distance)
            push!(selected, index)
            minimum_distance=min.(
                minimum_distance,
                map(eachrow(candidates)) do x
                    sum(abs2, x - view(candidates, index, :))
                end,
            )
        end
        candidates[selected, :]
    end
end

function shared_sample_measurements(
    scenario,
    locations,
    noise_variance,
    seed,
)
    let rng=MersenneTwister(seed),
        truth=comparison_ground_truth(scenario, locations)
        truth .+ sqrt(noise_variance) .* randn(rng, length(truth))
    end
end

function model_only_state(index, location)
    ComparisonState(index, Tuple(location))
end

function run_model_only_comparison(scenario, settings, grid)
    let reward_locations=comparison_reward_locations(settings),
        scribe_problem=make_comparison_mdp(
            settings,
            scenario,
            comparison_scribe_model(
                scenario,
                0.08,
                reward_locations,
            ),
            reward_locations,
        ),
        gp_problem=make_comparison_mdp(
            settings,
            scenario,
            comparison_gp_model(scenario, 0.08),
            reward_locations,
        ),
        locations=farthest_sample_locations(settings),
        measurements=shared_sample_measurements(
            scenario,
            locations,
            0.08,
            settings.seed,
        ),
        truth=comparison_ground_truth(scenario, grid.locations),
        first_state=model_only_state(1, view(locations, 1, :)),
        scribe_history=Any[],
        gp_history=Any[],
        scribe_model=scribe_problem.mdp.initial_model,
        gp_model=gp_problem.mdp.initial_model,
        scribe_runtime=0.0,
        gp_runtime=0.0

        push!(
            scribe_history,
            comparison_snapshot(
                scribe_problem.mdp,
                scribe_model,
                grid,
                truth;
                n_samples=0,
                state=first_state,
            ),
        )
        push!(
            gp_history,
            comparison_snapshot(
                gp_problem.mdp,
                gp_model,
                grid,
                truth;
                n_samples=0,
                state=first_state,
            ),
        )

        foreach(enumerate(zip(eachrow(locations), measurements))) do sample
            index, (location, measurement)=sample
            state=model_only_state(index, location)

            start_time=time()
            scribe_model=condition_environment_model(
                scribe_problem.mdp,
                scribe_model,
                state,
                measurement,
            )
            scribe_runtime += time() - start_time

            start_time=time()
            gp_model=condition_environment_model(
                gp_problem.mdp,
                gp_model,
                state,
                measurement,
            )
            gp_runtime += time() - start_time

            push!(
                scribe_history,
                comparison_snapshot(
                    scribe_problem.mdp,
                    scribe_model,
                    grid,
                    truth;
                    n_samples=index,
                    state,
                    observation=measurement,
                ),
            )
            push!(
                gp_history,
                comparison_snapshot(
                    gp_problem.mdp,
                    gp_model,
                    grid,
                    truth;
                    n_samples=index,
                    state,
                    observation=measurement,
                ),
            )
        end

        (
            scribe=(
                history=scribe_history,
                truth=truth,
                scenario=scenario,
                runtime=scribe_runtime,
            ),
            gp=(
                history=gp_history,
                truth=truth,
                scenario=scenario,
                runtime=gp_runtime,
            ),
            locations=locations,
            measurements=measurements,
        )
    end
end

function save_shared_samples(result, output_dir)
    open(joinpath(output_dir, "shared_samples.csv"), "w") do io
        println(io, "sample,x,y,measurement")
        foreach(enumerate(zip(eachrow(result.locations), result.measurements))) do sample
            index, (location, measurement)=sample
            println(
                io,
                "$(index),$(location[1]),$(location[2]),$(measurement)",
            )
        end
    end
end

function save_model_only_comparison(result, grid, output_dir)
    mkpath(output_dir)
    save_final_model_comparison(
        result.scribe,
        result.gp,
        grid,
        output_dir;
        shared_samples=true,
    )
    save_comparison_metrics(
        result.scribe,
        result.gp,
        output_dir;
        include_planning=false,
    )
    save_final_metrics(result, output_dir)
    save_shared_samples(result, output_dir)
end

function model_only_comparison(scenario, profile)
    let settings=comparison_settings(scenario, profile),
        grid=surface_grid(settings.evaluation_points),
        result=run_model_only_comparison(scenario, settings, grid),
        output_dir=joinpath(
            @__DIR__,
            "res",
            "vulcan_model_comparison",
            "model_only",
            scenario_name(scenario),
            String(profile),
        )
        save_model_only_comparison(result, grid, output_dir)
        print_model_comparison(result, output_dir)
        result
    end
end

function model_only_main(profile=:full)
    map(
        scenario -> model_only_comparison(scenario, profile),
        (SimpleComparison(), ComplicatedComparison()),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    model_only_main(isempty(ARGS) ? :full : Symbol(first(ARGS)))
end
