ENV["GKSwstype"] = "100"

using Distributions: Normal
using LinearAlgebra
using Match: @match
using POMDPs
using Random
using SCRIBE
using VulcanJ

const EXPLORATION_ACTIONS = (
    :north,
    :northeast,
    :east,
    :southeast,
    :south,
    :southwest,
    :west,
    :northwest,
)

struct InformativeExplorationMDP <: MDP{Matrix, Symbol}
    bounds
    step_size
    planning_dynamics
    ground_truth
    initial_model
    evaluation_grid
    n_steps
end

experiment_settings(profile) = @match profile begin
    :full => (
        seed=18,
        navigation_points=33,
        evaluation_points=51,
        n_samples=108,
        lookahead=8,
        time_budget=0.10,
        animation_fps=4,
    )
    :smoke => (
        seed=18,
        navigation_points=17,
        evaluation_points=21,
        n_samples=16,
        lookahead=3,
        time_budget=0.02,
        animation_fps=2,
    )
    _ => throw(ArgumentError("Use the `full` or `smoke` experiment profile."))
end

POMDPs.statetype(::InformativeExplorationMDP) = Matrix
POMDPs.actiontype(::InformativeExplorationMDP) = Symbol
POMDPs.actions(::InformativeExplorationMDP, _) = EXPLORATION_ACTIONS
POMDPs.discount(::InformativeExplorationMDP) = 1.0
POMDPs.isterminal(::InformativeExplorationMDP, _) = false

function POMDPs.gen(mdp::InformativeExplorationMDP, state, action, _)
    let direction=(@match action begin
            :north => (0, 1)
            :northeast => (1, 1)
            :east => (1, 0)
            :southeast => (1, -1)
            :south => (0, -1)
            :southwest => (-1, -1)
            :west => (-1, 0)
            :northwest => (-1, 1)
        end),
        next_state=clamp.(
            state .+ mdp.step_size .* reshape(collect(direction), 1, :),
            first(mdp.bounds),
            last(mdp.bounds),
        )
        (sp=next_state, r=0.0)
    end
end

VulcanJ.get_failure_prob(::InformativeExplorationMDP, _, _) = 0.0
VulcanJ.horizon(mdp::InformativeExplorationMDP) = mdp.n_steps
VulcanJ.initial_environment_model(
    mdp::InformativeExplorationMDP,
    _,
) = mdp.initial_model

function VulcanJ.expected_information_gain(
    mdp::InformativeExplorationMDP,
    model::SCRIBEModelState,
    location,
    ::Integer,
)
    integrated_variance_reduction(
        model.information,
        prediction_dynamics(model.smodel, location),
        mdp.planning_dynamics,
        model.R,
    )
end

function VulcanJ.conditional_observation_distribution(
    ::InformativeExplorationMDP,
    model::SCRIBEModelState,
    location,
)
    let moments=posterior_measurement_moments(
            model.smodel,
            model.information,
            location,
            model.R,
        )
        Normal(only(moments.μ), sqrt(only(moments.Σ)))
    end
end

function VulcanJ.condition_environment_model(
    ::InformativeExplorationMDP,
    model::SCRIBEModelState,
    location,
    observation,
)
    SCRIBEModelState(
        model.smodel,
        condition_on_measurement(
            model.smodel,
            model.information,
            location,
            observation,
            model.R,
        ),
        model.R,
    )
end

grid_locations(x, y) =
    reduce(vcat, ([xᵢ yⱼ] for yⱼ in y for xᵢ in x))

function exploration_problem(settings)
    let model_axis=collect(range(-5.5, 5.5; length=6)),
        centers=grid_locations(model_axis, model_axis),
        nᵩ=size(centers, 1),
        parameters=LGSFModelParameters(
            μ=centers,
            σ=[2.5],
            τ=[1.0],
            ϕ₀=zeros(nᵩ),
            A=Matrix{Float64}(I, nᵩ, nᵩ),
            Q=1e-10 * Matrix{Float64}(I, nᵩ, nᵩ),
        ),
        smodel=initialize_SCRIBEModel_from_parameters(parameters),
        Δ²=[
            sum(abs2, μᵢ - μⱼ)
            for μᵢ in eachrow(centers), μⱼ in eachrow(centers)
        ],
        covariance=0.45 * exp.(-Δ² / (2 * 2.4^2)) +
            1e-6 * Matrix{Float64}(I, nᵩ, nᵩ),
        covariance_factor=cholesky(Symmetric(covariance)),
        Y=Matrix(
            covariance_factor \
            Matrix{Float64}(I, nᵩ, nᵩ),
        ),
        initial_model=SCRIBEModelState(
            smodel,
            KFEnvInfo(zeros(nᵩ), Y, zeros(nᵩ), zeros(nᵩ, nᵩ)),
            0.08,
        ),
        evaluation_axis=collect(
            range(-5.0, 5.0; length=settings.evaluation_points),
        ),
        evaluation_grid=SCRIBEVisualizationGrid(
            evaluation_axis,
            evaluation_axis,
        ),
        planning_axis=collect(range(-5.0, 5.0; length=21)),
        planning_dynamics=prediction_dynamics(
            smodel,
            grid_locations(planning_axis, planning_axis),
        ),
        ground_truth=X -> map(eachrow(X)) do x
            let x₁=x[1],
                x₂=x[2],
                broad=0.75 * sin(0.72 * x₁) * cos(0.47 * x₂),
                variation=0.35 * sin(1.20 * x₁ + 0.35 * x₂),
                positive=1.65 * exp(
                    -((x₁ - 2.10)^2 / 2.40 +
                      (x₂ + 1.65)^2 / 1.10),
                ),
                negative=-1.50 * exp(
                    -((x₁ + 2.35)^2 / 1.30 +
                      (x₂ - 1.45)^2 / 2.00),
                )
                broad + variation + positive + negative
            end
        end,
        mdp=InformativeExplorationMDP(
            (-5.0, 5.0),
            10.0 / (settings.navigation_points - 1),
            planning_dynamics,
            ground_truth,
            initial_model,
            evaluation_grid,
            settings.n_samples,
        )
        (mdp=mdp, start=zeros(1, 2))
    end
end

function run_exploration(mdp, start, settings)
    let rng=MersenneTwister(settings.seed),
        policy=solve(
            RiskBoundedInfoMCTS(
                lookahead=settings.lookahead,
                time_budget=settings.time_budget,
                quad_order=1,
                risk_budget=1.0,
                alpha=0.0,
                reference_reward=1.0,
                rng=MersenneTwister(settings.seed + 1),
            ),
            mdp,
        ),
        model=initial_environment_model(mdp, start),
        model_states=SCRIBEModelState[model],
        sampling_locations=Matrix[],
        observations=Float64[],
        state=copy(start)

        foreach(1:settings.n_samples) do sample
            if sample > 1
                set_environment_model!(policy, state, model)
                state=gen(
                    mdp,
                    state,
                    action(policy, state),
                    rng,
                ).sp
            end
            observation=only(mdp.ground_truth(state)) +
                sqrt(model.R) * randn(rng)
            model=condition_environment_model(
                mdp,
                model,
                state,
                observation,
            )
            push!(sampling_locations, copy(state))
            push!(observations, observation)
            push!(model_states, model)
        end

        visualization=scribe_model_history(
            model_states;
            sampling_locations,
            grid=mdp.evaluation_grid,
            ground_truth=mdp.ground_truth,
        )
        (
            model_states=model_states,
            sampling_locations=sampling_locations,
            observations=observations,
            visualization=visualization,
        )
    end
end

function save_experiment(result, output_dir, settings)
    save_static_visualizations(
        result.visualization;
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
    save_animated_visualizations(
        result.visualization;
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

function print_experiment_summary(result, output_dir)
    let initial=first(result.visualization.frames).summary,
        final=last(result.visualization.frames).summary
        println("SCRIBE-backed VulcanJ exploration complete.")
        println("  samples: $(length(result.observations))")
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

function main(profile=:full)
    let settings=experiment_settings(profile),
        problem=exploration_problem(settings),
        result=run_exploration(problem.mdp, problem.start, settings),
        output_dir=normpath(joinpath(
            @__DIR__,
            "..",
            "res",
            "vulcan-planner-integration",
            String(profile),
        ))
        save_experiment(result, output_dir, settings)
        print_experiment_summary(result, output_dir)
        result
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? :full : Symbol(first(ARGS)))
end
