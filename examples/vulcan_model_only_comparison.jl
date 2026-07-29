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

function run_model_only_comparison(scenario, settings, grid)
    let reward_locations=comparison_reward_locations(settings),
        scribe_problem=comparison_problem(
            settings,
            scenario,
            comparison_scribe_model(
                scenario,
                0.08,
                reward_locations,
            ),
            reward_locations,
        ),
        gp_problem=comparison_problem(
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
        states=map(location -> reshape(collect(location), 1, :), eachrow(locations)),
        first_state=first(states),
        scribe_records=Any[],
        gp_records=Any[],
        scribe_model=scribe_problem.mdp.initial_model,
        gp_model=gp_problem.mdp.initial_model,
        scribe_states=SCRIBEModelState[scribe_model],
        scribe_runtime=0.0,
        gp_runtime=0.0

        push!(
            scribe_records,
            comparison_record(
                scribe_problem.mdp,
                scribe_model,
                grid,
                truth;
                n_samples=0,
                state=first_state,
            ),
        )
        push!(
            gp_records,
            comparison_record(
                gp_problem.mdp,
                gp_model,
                grid,
                truth;
                n_samples=0,
                state=first_state,
            ),
        )

        foreach(enumerate(zip(states, measurements))) do sample
            index, (state, measurement)=sample
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
            push!(scribe_states, scribe_model)

            push!(
                scribe_records,
                comparison_record(
                    scribe_problem.mdp,
                    scribe_model,
                    grid,
                    truth;
                    n_samples=index,
                    state,
                ),
            )
            push!(
                gp_records,
                comparison_record(
                    gp_problem.mdp,
                    gp_model,
                    grid,
                    truth;
                    n_samples=index,
                    state,
                ),
            )
        end

        (
            scribe=(
                records=scribe_records,
                model_states=scribe_states,
                sampling_locations=states,
                observations=measurements,
                truth=truth,
                scenario=scenario,
                runtime=scribe_runtime,
            ),
            gp=(
                records=gp_records,
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
