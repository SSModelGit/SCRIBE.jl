ENV["GKSwstype"] = "100"

const SCRIBE_ROOT = normpath(joinpath(@__DIR__, ".."))
const VULCANJ_ROOT = get(
    ENV,
    "VULCANJ_ROOT",
    "/home/shashank/cbase/secondary/jbase/VulcanJ",
)

pushfirst!(LOAD_PATH, SCRIBE_ROOT)
pushfirst!(LOAD_PATH, VULCANJ_ROOT)

using LinearAlgebra
using Match: @match
using Plots
using POMDPs
using Random
using SCRIBE
using Statistics
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

"""The opaque environment model carried through VulcanJ's search tree."""
struct VulcanSCRIBEModel
    smodel::LGSFModel
    information::KFEnvInfo
    observer::LGSFObserverBehavior
end

"""A predictive measurement whose tree identity is its sampling state."""
struct StatePlanningObservation
    state
    value
end

Base.:(==)(a::StatePlanningObservation, b::StatePlanningObservation) =
    a.state == b.state

Base.isequal(a::StatePlanningObservation, b::StatePlanningObservation) =
    isequal(a.state, b.state)

Base.hash(observation::StatePlanningObservation, h::UInt) =
    hash(observation.state, hash(StatePlanningObservation, h))

"""A predictive distribution whose samples merge by sampling-state history."""
struct StatePlanningDistribution
    state
    distribution
end

function Random.rand(rng::AbstractRNG, distribution::StatePlanningDistribution)
    StatePlanningObservation(
        distribution.state,
        rand(rng, distribution.distribution),
    )
end

"""A small grid-navigation problem whose reward comes entirely from SCRIBE."""
struct SCRIBEExplorationMDP <: MDP{Tuple{Int, Int}, Symbol}
    x_grid
    y_grid
    reward_dynamics
    ground_truth::LGSFModel
    initial_model::VulcanSCRIBEModel
    n_steps
end

"""A regular grid used only to evaluate and visualize model surfaces."""
struct SurfaceGrid
    x
    y
    locations
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

state_step(action) = @match action begin
    :north => (0, 1)
    :northeast => (1, 1)
    :east => (1, 0)
    :southeast => (1, -1)
    :south => (0, -1)
    :southwest => (-1, -1)
    :west => (-1, 0)
    :northwest => (-1, 1)
end

function successor_state(mdp, state, action)
    let Δ=state_step(action)
        (state[1] + Δ[1], state[2] + Δ[2])
    end
end

function state_is_valid(mdp, state)
    1 ≤ state[1] ≤ length(mdp.x_grid) &&
        1 ≤ state[2] ≤ length(mdp.y_grid)
end

function state_location(mdp, state)
    [mdp.x_grid[state[1]] mdp.y_grid[state[2]]]
end

POMDPs.statetype(::Type{SCRIBEExplorationMDP}) = Tuple{Int, Int}
POMDPs.actiontype(::Type{SCRIBEExplorationMDP}) = Symbol
POMDPs.discount(::SCRIBEExplorationMDP) = 1.0
POMDPs.isterminal(::SCRIBEExplorationMDP, _) = false

function POMDPs.actions(mdp::SCRIBEExplorationMDP, state)
    filter(EXPLORATION_ACTIONS) do action
        state_is_valid(mdp, successor_state(mdp, state, action))
    end
end

function POMDPs.gen(mdp::SCRIBEExplorationMDP, state, action, _)
    (sp=successor_state(mdp, state, action), r=0.0)
end

VulcanJ.get_failure_prob(::SCRIBEExplorationMDP, _, _) = 0.0
VulcanJ.horizon(mdp::SCRIBEExplorationMDP) = mdp.n_steps

"""Supply SCRIBE's initial posterior without exposing it to VulcanJ."""
function VulcanJ.initial_environment_model(
    mdp::SCRIBEExplorationMDP,
    _,
)
    mdp.initial_model
end

"""Use integrated field-variance reduction as the planning reward."""
function VulcanJ.expected_information_gain(
    mdp::SCRIBEExplorationMDP,
    model::VulcanSCRIBEModel,
    state,
    ::Integer,
)
    let X=state_location(mdp, state),
        R=model.observer.v_s[:σ],
        Hₛ=prediction_dynamics(model.smodel, X)
        integrated_variance_reduction(
            model.information,
            Hₛ,
            mdp.reward_dynamics,
            R,
        )
    end
end

"""Give VulcanJ the SCRIBE posterior predictive measurement distribution."""
function VulcanJ.conditional_observation_distribution(
    mdp::SCRIBEExplorationMDP,
    model::VulcanSCRIBEModel,
    state,
)
    let X=state_location(mdp, state),
        R=model.observer.v_s[:σ]
        StatePlanningDistribution(
            state,
            posterior_measurement_distribution(
                model.smodel,
                model.information,
                X,
                R,
            ),
        )
    end
end

measurement_value(observation::StatePlanningObservation) = observation.value
measurement_value(observation) = observation

"""Condition the SCRIBE information state on a rollout or realized sample."""
function VulcanJ.condition_environment_model(
    mdp::SCRIBEExplorationMDP,
    model::VulcanSCRIBEModel,
    state,
    observation,
)
    let X=state_location(mdp, state),
        R=model.observer.v_s[:σ],
        information=condition_on_measurement(
            model.smodel,
            model.information,
            X,
            measurement_value(observation),
            R,
        )
        VulcanSCRIBEModel(model.smodel, information, model.observer)
    end
end

function regular_locations(x, y)
    reduce(vcat, ([xᵢ yⱼ] for yⱼ in y for xᵢ in x))
end

function surface_grid(n_points)
    let x=collect(range(-5.0, 5.0; length=n_points)),
        y=collect(range(-5.0, 5.0; length=n_points))
        SurfaceGrid(x, y, regular_locations(x, y))
    end
end

function basis_centers()
    regular_locations(
        collect(range(-5.5, 5.5; length=6)),
        collect(range(-5.5, 5.5; length=6)),
    )
end

function transform_locations(locations, θ, translation)
    let rotation=[
            cos(θ) -sin(θ)
            sin(θ) cos(θ)
        ]
        locations * rotation' .+ reshape(translation, 1, :)
    end
end

function ground_truth_basis_centers()
    let centers=regular_locations(
            collect(range(-4.6, 4.6; length=5)),
            collect(range(-4.6, 4.6; length=5)),
        )
        transform_locations(centers, π / 14, [0.35, -0.25])
    end
end

function ground_truth_coefficients(centers)
    map(eachrow(centers)) do μ
        let x=μ[1],
            y=μ[2],
            broad_structure=0.75 * sin(0.72 * x) * cos(0.47 * y),
            local_variation=0.35 * sin(1.20 * x + 0.35 * y),
            positive_well=1.65 * exp(
                -((x - 2.10)^2 / 2.40 + (y + 1.65)^2 / 1.10),
            ),
            negative_well=-1.50 * exp(
                -((x + 2.35)^2 / 1.30 + (y - 1.45)^2 / 2.00),
            )
            broad_structure + local_variation +
                positive_well + negative_well
        end
    end
end

function scalar_field_parameters(
    centers,
    coefficients;
    basis_width=2.5,
)
    let nᵩ=length(coefficients)
        LGSFModelParameters(
            μ=centers,
            σ=[basis_width],
            τ=[1.0],
            ϕ₀=coefficients,
            A=Matrix{Float64}(I, nᵩ, nᵩ),
            Q=1e-10 * Matrix{Float64}(I, nᵩ, nᵩ),
        )
    end
end

function coefficient_prior_covariance(
    centers;
    coefficient_variance=0.45,
    correlation_length=2.4,
)
    let Δ²=[
            sum(abs2, μᵢ - μⱼ)
            for μᵢ in eachrow(centers), μⱼ in eachrow(centers)
        ],
        correlation=exp.(-Δ² / (2 * correlation_length^2)),
        nᵩ=size(centers, 1)
        coefficient_variance * correlation +
            1e-6 * Matrix{Float64}(I, nᵩ, nᵩ)
    end
end

function initial_information(centers; covariance_parameters...)
    let covariance=coefficient_prior_covariance(
            centers;
            covariance_parameters...,
        ),
        covariance_factor=cholesky(Symmetric(covariance)),
        nᵩ=size(covariance, 1),
        Y=Matrix(
            covariance_factor \
            Matrix{Float64}(I, nᵩ, nᵩ),
        )
        KFEnvInfo(zeros(nᵩ), Y, zeros(nᵩ), zeros(nᵩ, nᵩ))
    end
end

function make_exploration_problem(settings)
    let model_centers=basis_centers(),
        truth_centers=ground_truth_basis_centers(),
        truth_coefficients=ground_truth_coefficients(truth_centers),
        truth_params=scalar_field_parameters(
            truth_centers,
            truth_coefficients;
            basis_width=1.7,
        ),
        model_params=scalar_field_parameters(
            model_centers,
            zeros(size(model_centers, 1)),
        ),
        ground_truth=initialize_SCRIBEModel_from_parameters(truth_params),
        smodel=initialize_SCRIBEModel_from_parameters(model_params),
        observer=LGSFObserverBehavior(0.08),
        information=initial_information(model_centers),
        model=VulcanSCRIBEModel(smodel, information, observer),
        navigation_grid=collect(
            range(-5.0, 5.0; length=settings.navigation_points),
        ),
        reward_axis=collect(range(-5.0, 5.0; length=21)),
        reward_locations=regular_locations(reward_axis, reward_axis),
        reward_dynamics=prediction_dynamics(smodel, reward_locations),
        mdp=SCRIBEExplorationMDP(
            navigation_grid,
            navigation_grid,
            reward_dynamics,
            ground_truth,
            model,
            settings.n_samples,
        ),
        start=(
            cld(length(navigation_grid), 2),
            cld(length(navigation_grid), 2),
        )
        (mdp=mdp, start=start)
    end
end

function make_planner(mdp, settings)
    let solver=RiskBoundedInfoMCTS(
            lookahead=settings.lookahead,
            time_budget=settings.time_budget,
            quad_order=1,
            risk_budget=1.0,
            alpha=0.0,
            reference_reward=1.0,
            rng=MersenneTwister(settings.seed + 1),
        )
        solve(solver, mdp)
    end
end

function latent_truth(mdp, X)
    prediction_dynamics(mdp.ground_truth, X) * mdp.ground_truth.ϕ
end

function sample_environment(mdp, state, observer, rng)
    let X=state_location(mdp, state),
        μ=only(latent_truth(mdp, X)),
        σ²=observer.v_s[:σ]
        μ + sqrt(σ²) * randn(rng)
    end
end

function observe_environment(mdp, model, state, rng)
    let planning_reward=expected_information_gain(mdp, model, state, 1),
        X=state_location(mdp, state),
        expected_information=mutual_information(
            model.smodel,
            model.information,
            X,
            model.observer.v_s[:σ],
        ),
        observation=sample_environment(mdp, state, model.observer, rng),
        posterior=condition_environment_model(
            mdp,
            model,
            state,
            observation,
        ),
        realized_information=D_KL(posterior.information, model.information)
        (
            model=posterior,
            observation=observation,
            planning_reward=planning_reward,
            expected_information=expected_information,
            realized_information=realized_information,
        )
    end
end

function model_summary(mdp, model, grid, truth)
    let prediction=posterior_model_moments(
            model.smodel,
            model.information,
            grid.locations,
        ).μ,
        uncertainty=predict_model_uncertainty(
            model.smodel,
            model.information,
            grid.locations;
            metric=:standard_deviation,
        ),
        error=prediction - truth,
        coefficient_mean=posterior_coefficient_moments(
            model.information,
        ).μ
        (
            prediction=prediction,
            uncertainty=uncertainty,
            absolute_error=abs.(error),
            rmse=sqrt(mean(abs2, error)),
            mae=mean(abs, error),
            mean_uncertainty=mean(uncertainty),
            maximum_uncertainty=maximum(uncertainty),
            coefficient_magnitude=norm(coefficient_mean) /
                sqrt(length(coefficient_mean)),
            mean_entropy=evaluate_information_metric(
                model.information;
                metric=:differential_entropy,
            ) / length(coefficient_mean),
        )
    end
end

function exploration_snapshot(
    mdp,
    model,
    grid,
    truth;
    n_samples,
    state,
    observation=missing,
    planning_reward=0.0,
    expected_information=0.0,
    realized_information=0.0,
)
    (
        n_samples=n_samples,
        state=state,
        location=vec(state_location(mdp, state)),
        observation=observation,
        planning_reward=planning_reward,
        expected_information=expected_information,
        realized_information=realized_information,
        model=model,
        summary=model_summary(mdp, model, grid, truth),
    )
end

function run_exploration(mdp, start, settings, grid)
    let rng=MersenneTwister(settings.seed),
        truth=latent_truth(mdp, grid.locations),
        policy=make_planner(mdp, settings),
        history=Any[],
        model=initial_environment_model(mdp, start),
        state=start

        push!(
            history,
            exploration_snapshot(
                mdp,
                model,
                grid,
                truth;
                n_samples=0,
                state,
            ),
        )

        observation=observe_environment(mdp, model, state, rng)
        model=observation.model
        push!(
            history,
            exploration_snapshot(
                mdp,
                model,
                grid,
                truth;
                n_samples=1,
                state,
                observation=observation.observation,
                planning_reward=observation.planning_reward,
                expected_information=observation.expected_information,
                realized_information=observation.realized_information,
            ),
        )

        for n_samples in 2:settings.n_samples
            set_environment_model!(policy, state, model)
            selected_action=action(policy, state)
            state=successor_state(mdp, state, selected_action)
            observation=observe_environment(mdp, model, state, rng)
            model=observation.model

            push!(
                history,
                exploration_snapshot(
                    mdp,
                    model,
                    grid,
                    truth;
                    n_samples,
                    state,
                    observation=observation.observation,
                    planning_reward=observation.planning_reward,
                    expected_information=observation.expected_information,
                    realized_information=observation.realized_information,
                ),
            )
        end

        (history=history, truth=truth)
    end
end

surface_matrix(grid, values) =
    reshape(values, length(grid.x), length(grid.y))'

function sampled_path(history, frame_index)
    let sampled=history[2:frame_index]
        isempty(sampled) ?
            zeros(0, 2) :
            reduce(vcat, (reshape(frame.location, 1, :) for frame in sampled))
    end
end

function overlay_path!(plot_object, path)
    if !isempty(path)
        plot!(
            plot_object,
            path[:, 1],
            path[:, 2];
            color=:white,
            linewidth=2,
            marker=:circle,
            markersize=2,
            label=false,
        )
        scatter!(
            plot_object,
            path[end:end, 1],
            path[end:end, 2];
            color=:black,
            markersize=4,
            label=false,
        )
    end
    plot_object
end

function surface_panel(
    grid,
    values,
    title;
    color=:viridis,
    color_limits=nothing,
)
    heatmap(
        grid.x,
        grid.y,
        surface_matrix(grid, values);
        aspect_ratio=:equal,
        color,
        clims=color_limits,
        xlabel="x",
        ylabel="y",
        title,
        titlefontsize=11,
        guidefontsize=9,
        tickfontsize=8,
        margin=1 * Plots.mm,
        left_margin=3 * Plots.mm,
        bottom_margin=3 * Plots.mm,
    )
end

function posterior_frame(history, frame_index, grid, truth, limits)
    let frame=history[frame_index],
        summary=frame.summary,
        path=sampled_path(history, frame_index),
        truth_plot=surface_panel(
            grid,
            truth,
            "Ground truth";
            color=:balance,
            color_limits=limits.field,
        ),
        posterior_plot=surface_panel(
            grid,
            summary.prediction,
            "Posterior mean";
            color=:balance,
            color_limits=limits.field,
        ),
        uncertainty_plot=surface_panel(
            grid,
            summary.uncertainty,
            "Posterior standard deviation";
            color=:viridis,
            color_limits=limits.uncertainty,
        ),
        error_plot=surface_panel(
            grid,
            summary.absolute_error,
            "Absolute error";
            color=:thermal,
            color_limits=limits.error,
        )

        foreach(
            plot_object -> overlay_path!(plot_object, path),
            (truth_plot, posterior_plot, uncertainty_plot, error_plot),
        )

        plot(
            truth_plot,
            posterior_plot,
            uncertainty_plot,
            error_plot;
            layout=(1, 4),
            size=(1400, 310),
            margin=1 * Plots.mm,
            plot_titlefontsize=11,
            plot_title="Samples: $(frame.n_samples)   RMSE: " *
                "$(round(summary.rmse; digits=3))   Expected IVR: " *
                "$(round(frame.planning_reward; digits=3))",
        )
    end
end

function surface_limits(history, truth)
    let predictions=reduce(vcat, (frame.summary.prediction for frame in history)),
        uncertainty=maximum(
            frame.summary.maximum_uncertainty for frame in history
        ),
        error=maximum(
            maximum(frame.summary.absolute_error) for frame in history
        ),
        field_extent=maximum(abs, [truth; predictions])
        (
            field=(-field_extent, field_extent),
            uncertainty=(0.0, uncertainty),
            error=(0.0, error),
        )
    end
end

function standalone_posterior_specification(frame, limits, field)
    @match field begin
        :mean => (
            values=frame.summary.prediction,
            title="Posterior mean — $(frame.n_samples) samples\n" *
                "RMSE $(round(frame.summary.rmse; digits=3))",
            color=:balance,
            limits=limits.field,
        )
        :uncertainty => (
            values=frame.summary.uncertainty,
            title="Posterior uncertainty — $(frame.n_samples) samples\n" *
                "mean σ $(round(frame.summary.mean_uncertainty; digits=3))",
            color=:viridis,
            limits=limits.uncertainty,
        )
    end
end

function standalone_posterior_frame(
    history,
    frame_index,
    grid,
    limits,
    field,
)
    let frame=history[frame_index],
        path=sampled_path(history, frame_index),
        specification=standalone_posterior_specification(
            frame,
            limits,
            field,
        ),
        plot_object=surface_panel(
            grid,
            specification.values,
            specification.title;
            color=specification.color,
            color_limits=specification.limits,
        )
        overlay_path!(plot_object, path)
        plot!(
            plot_object;
            size=(480, 390),
            margin=0.5 * Plots.mm,
            left_margin=2 * Plots.mm,
            bottom_margin=2 * Plots.mm,
            right_margin=0.5 * Plots.mm,
            top_margin=0.5 * Plots.mm,
        )
    end
end

standalone_animation_name(field) = @match field begin
    :mean => "posterior_mean.gif"
    :uncertainty => "posterior_uncertainty.gif"
end

function save_standalone_posterior_animations(
    history,
    grid,
    truth,
    output_dir,
    settings,
)
    let limits=surface_limits(history, truth)
        foreach((:mean, :uncertainty)) do field
            animation=Animation()
            foreach(eachindex(history)) do frame_index
                frame(
                    animation,
                    standalone_posterior_frame(
                        history,
                        frame_index,
                        grid,
                        limits,
                        field,
                    ),
                )
            end
            gif(
                animation,
                joinpath(output_dir, standalone_animation_name(field));
                fps=settings.animation_fps,
            )
        end
    end
end

function ground_truth_posterior_frame(
    history,
    frame_index,
    grid,
    truth,
    limits,
)
    let frame=history[frame_index],
        path=sampled_path(history, frame_index),
        truth_plot=surface_panel(
            grid,
            truth,
            "Ground truth";
            color=:balance,
            color_limits=limits.field,
        ),
        posterior_plot=surface_panel(
            grid,
            frame.summary.prediction,
            "Posterior mean — $(frame.n_samples) samples\n" *
                "RMSE $(round(frame.summary.rmse; digits=3))";
            color=:balance,
            color_limits=limits.field,
        )
        overlay_path!(posterior_plot, path)
        plot(
            truth_plot,
            posterior_plot;
            layout=(1, 2),
            size=(900, 380),
            margin=1 * Plots.mm,
        )
    end
end

function save_ground_truth_posterior_animation(
    history,
    grid,
    truth,
    output_dir,
    settings,
)
    let animation=Animation(),
        limits=surface_limits(history, truth)
        foreach(eachindex(history)) do frame_index
            frame(
                animation,
                ground_truth_posterior_frame(
                    history,
                    frame_index,
                    grid,
                    truth,
                    limits,
                ),
            )
        end
        gif(
            animation,
            joinpath(output_dir, "ground_truth_vs_posterior.gif");
            fps=settings.animation_fps,
        )
    end
end

function save_posterior_animation(history, grid, truth, output_dir, settings)
    let animation=Animation(),
        limits=surface_limits(history, truth)
        foreach(eachindex(history)) do frame_index
            frame(
                animation,
                posterior_frame(history, frame_index, grid, truth, limits),
            )
        end
        gif(
            animation,
            joinpath(output_dir, "posterior_exploration.gif");
            fps=settings.animation_fps,
        )
    end
end

function save_final_surfaces(history, grid, truth, output_dir)
    let limits=surface_limits(history, truth),
        final_plot=posterior_frame(
            history,
            lastindex(history),
            grid,
            truth,
            limits,
        )
        savefig(final_plot, joinpath(output_dir, "final_surfaces.png"))
    end
end

function save_metric_history(history, output_dir)
    let samples=getproperty.(history, :n_samples),
        summaries=getproperty.(history, :summary),
        rmse=getproperty.(summaries, :rmse),
        mae=getproperty.(summaries, :mae),
        mean_uncertainty=getproperty.(summaries, :mean_uncertainty),
        maximum_uncertainty=getproperty.(summaries, :maximum_uncertainty),
        expected_information=getproperty.(history, :expected_information),
        realized_information=getproperty.(history, :realized_information),
        coefficient_magnitude=getproperty.(summaries, :coefficient_magnitude),
        mean_entropy=getproperty.(summaries, :mean_entropy),
        entropy_reduction=first(mean_entropy) .- mean_entropy

        error_plot=plot(
            samples,
            rmse;
            label="RMSE",
            marker=:circle,
            ylabel="field error",
            xlabel="samples",
        )
        plot!(error_plot, samples, mae; label="MAE", marker=:circle)

        uncertainty_plot=plot(
            samples,
            mean_uncertainty;
            label="mean",
            marker=:circle,
            ylabel="posterior σ",
            xlabel="samples",
        )
        plot!(
            uncertainty_plot,
            samples,
            maximum_uncertainty;
            label="maximum",
            marker=:circle,
        )

        information_plot=plot(
            samples,
            expected_information;
            label="expected MI",
            marker=:circle,
            ylabel="information (nats)",
            xlabel="samples",
        )
        plot!(
            information_plot,
            samples,
            realized_information;
            label="realized KL",
            marker=:circle,
        )

        model_plot=plot(
            samples,
            coefficient_magnitude;
            label="RMS posterior coefficient",
            marker=:circle,
            ylabel="coefficient magnitude",
            xlabel="samples",
        )
        plot!(
            model_plot,
            samples,
            entropy_reduction;
            label="entropy reduction / coefficient",
            color=:orange,
            linestyle=:dash,
        )

        diagnostics=plot(
            error_plot,
            uncertainty_plot,
            information_plot,
            model_plot;
            layout=(2, 2),
            size=(900, 680),
            margin=2 * Plots.mm,
        )
        savefig(diagnostics, joinpath(output_dir, "exploration_metrics.png"))
    end
end

function comparison_frames(history)
    unique([1, cld(length(history), 2), length(history)])
end

function save_prediction_comparison(history, truth, output_dir)
    let comparison=plot(
            truth,
            truth;
            color=:black,
            linestyle=:dash,
            label="identity",
            xlabel="ground truth",
            ylabel="posterior prediction",
            aspect_ratio=:equal,
        )
        foreach(comparison_frames(history)) do frame_index
            frame=history[frame_index]
            scatter!(
                comparison,
                truth,
                frame.summary.prediction;
                markersize=2,
                markerstrokewidth=0,
                alpha=0.45,
                label="$(frame.n_samples) samples",
            )
        end
        plot!(
            comparison;
            size=(520, 470),
            margin=2 * Plots.mm,
            left_margin=4 * Plots.mm,
            bottom_margin=4 * Plots.mm,
        )
        savefig(
            comparison,
            joinpath(output_dir, "predicted_vs_ground_truth.png"),
        )
    end
end

function save_experiment(result, grid, output_dir, settings)
    mkpath(output_dir)
    save_posterior_animation(
        result.history,
        grid,
        result.truth,
        output_dir,
        settings,
    )
    save_standalone_posterior_animations(
        result.history,
        grid,
        result.truth,
        output_dir,
        settings,
    )
    save_ground_truth_posterior_animation(
        result.history,
        grid,
        result.truth,
        output_dir,
        settings,
    )
    save_final_surfaces(result.history, grid, result.truth, output_dir)
    save_metric_history(result.history, output_dir)
    save_prediction_comparison(result.history, result.truth, output_dir)
end

function print_experiment_summary(history, output_dir)
    let initial=first(history).summary,
        final=last(history).summary
        println("SCRIBE-backed VulcanJ exploration complete.")
        println("  samples: $(last(history).n_samples)")
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
        problem=make_exploration_problem(settings),
        grid=surface_grid(settings.evaluation_points),
        result=run_exploration(
            problem.mdp,
            problem.start,
            settings,
            grid,
        ),
        output_dir=joinpath(
            @__DIR__,
            "res",
            "vulcan_scribe",
            String(profile),
        )
        save_experiment(result, grid, output_dir, settings)
        print_experiment_summary(result.history, output_dir)
        result
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? :full : Symbol(first(ARGS)))
end
