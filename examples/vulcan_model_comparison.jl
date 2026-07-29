include("vulcan_scribe_exploration.jl")

using Distributions: Normal
using GaussianProcesses

"""A grid state that also carries the physical location expected by a GP."""
struct ComparisonState
    index
    location
end

abstract type ComparisonScenario end

struct SimpleComparison <: ComparisonScenario end
struct ComplicatedComparison <: ComparisonScenario end

scenario_name(::SimpleComparison) = "simple"
scenario_name(::ComplicatedComparison) = "complicated"

scenario_title(::SimpleComparison) = "Simple"
scenario_title(::ComplicatedComparison) = "Complicated multiscale"

"""The shared exploration problem used by both modeling backends."""
struct ModelComparisonMDP{M} <: MDP{ComparisonState, Symbol}
    x_grid
    y_grid
    reward_locations
    reward_dynamics
    scenario::ComparisonScenario
    initial_model::M
    noise_variance
    n_steps
    planning_iterations
end

comparison_settings(::SimpleComparison, profile) = @match profile begin
    :full => (
        seed=27,
        navigation_points=33,
        evaluation_points=51,
        reward_points=21,
        n_samples=40,
        lookahead=4,
        planning_iterations=12,
        time_budget=0.001,
        animation_fps=4,
    )
    :smoke => (
        seed=27,
        navigation_points=17,
        evaluation_points=21,
        reward_points=11,
        n_samples=6,
        lookahead=3,
        planning_iterations=8,
        time_budget=0.001,
        animation_fps=2,
    )
    _ => throw(ArgumentError("Use the `full` or `smoke` comparison profile."))
end

comparison_settings(::ComplicatedComparison, profile) = @match profile begin
    :full => (
        seed=38,
        navigation_points=33,
        evaluation_points=51,
        reward_points=17,
        n_samples=48,
        lookahead=4,
        planning_iterations=12,
        time_budget=0.001,
        animation_fps=4,
    )
    :smoke => (
        seed=38,
        navigation_points=17,
        evaluation_points=21,
        reward_points=11,
        n_samples=6,
        lookahead=3,
        planning_iterations=8,
        time_budget=0.001,
        animation_fps=2,
    )
    _ => throw(ArgumentError("Use the `full` or `smoke` comparison profile."))
end

function comparison_state(mdp, index)
    ComparisonState(
        index,
        (mdp.x_grid[index[1]], mdp.y_grid[index[2]]),
    )
end

function state_location(::ModelComparisonMDP, state::ComparisonState)
    reshape(collect(state.location), 1, :)
end

function successor_index(state::ComparisonState, action)
    let Δ=state_step(action)
        (state.index[1] + Δ[1], state.index[2] + Δ[2])
    end
end

function successor_state(mdp::ModelComparisonMDP, state::ComparisonState, action)
    comparison_state(mdp, successor_index(state, action))
end

function state_is_valid(mdp::ModelComparisonMDP, index)
    1 ≤ index[1] ≤ length(mdp.x_grid) &&
        1 ≤ index[2] ≤ length(mdp.y_grid)
end

POMDPs.statetype(::Type{ModelComparisonMDP{M}}) where {M} = ComparisonState
POMDPs.actiontype(::Type{ModelComparisonMDP{M}}) where {M} = Symbol
POMDPs.discount(::ModelComparisonMDP) = 1.0
POMDPs.isterminal(::ModelComparisonMDP, _) = false

function POMDPs.actions(mdp::ModelComparisonMDP, state)
    filter(EXPLORATION_ACTIONS) do action
        state_is_valid(mdp, successor_index(state, action))
    end
end

function POMDPs.gen(mdp::ModelComparisonMDP, state, action, _)
    (sp=successor_state(mdp, state, action), r=0.0)
end

VulcanJ.get_failure_prob(::ModelComparisonMDP, _, _) = 0.0
VulcanJ.horizon(mdp::ModelComparisonMDP) = mdp.n_steps

function VulcanJ.initial_environment_model(mdp::ModelComparisonMDP, _)
    mdp.initial_model
end

function VulcanJ.extract_location(state::ComparisonState)
    reshape(collect(state.location), 1, :)
end

function gp_cross_covariance(gp, Xₑ, Xₛ)
    let prior=GaussianProcesses.cov(gp.kernel, Xₑ, Xₛ)
        isempty(gp.y) ?
            prior :
            prior -
            GaussianProcesses.cov(gp.kernel, Xₑ, gp.x) *
            (gp.cK \ GaussianProcesses.cov(gp.kernel, gp.x, Xₛ))
    end
end

function gp_integrated_variance_reduction(gp, Xₛ, Xₑ)
    let cross_covariance=gp_cross_covariance(gp, Xₑ, Xₛ),
        sample_variance=only(last(predict_f(gp, Xₛ))),
        measurement_variance=sample_variance + noise_variance(gp)
        mean(abs2, cross_covariance) / measurement_variance
    end
end

function VulcanJ.expected_information_gain(
    mdp::ModelComparisonMDP,
    model::VulcanSCRIBEModel,
    state,
    ::Integer,
)
    let X=state_location(mdp, state),
        Hₛ=prediction_dynamics(model.smodel, X),
        R=model.observer.v_s[:σ]
        integrated_variance_reduction(
            model.information,
            Hₛ,
            mdp.reward_dynamics,
            R,
        )
    end
end

function VulcanJ.expected_information_gain(
    mdp::ModelComparisonMDP,
    model::GPE,
    state,
    ::Integer,
)
    gp_integrated_variance_reduction(
        model,
        state_location(mdp, state)',
        mdp.reward_locations',
    )
end

function VulcanJ.conditional_observation_distribution(
    mdp::ModelComparisonMDP,
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

function VulcanJ.conditional_observation_distribution(
    ::ModelComparisonMDP,
    model::GPE,
    state,
)
    let (μ, σ²)=VulcanJ.gp_predict(model, state)
        StatePlanningDistribution(
            state,
            Normal(μ, sqrt(max(σ² + noise_variance(model), eps()))),
        )
    end
end

function VulcanJ.condition_environment_model(
    mdp::ModelComparisonMDP,
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

measurement_scalar(measurement::Number) = Float64(measurement)
measurement_scalar(measurement) = Float64(only(measurement))

function VulcanJ.add_obs_to_gp(state::ComparisonState, observation, gp::GPE)
    let X=VulcanJ.extract_location(state)',
        z=measurement_scalar(measurement_value(observation))
        GPE(
            hcat(gp.x, X),
            [gp.y; z],
            gp.mean,
            gp.kernel,
            gp.logNoise,
        )
    end
end

"""A smooth two-lobe field used for the introductory comparison."""
function comparison_ground_truth(::SimpleComparison, X)
    map(eachrow(X)) do x
        let x₁=x[1],
            x₂=x[2],
            positive_lobe=2.5 * exp(
                -((x₁ - 2.3)^2 / 4.0 + (x₂ + 0.8)^2 / 3.0),
            ),
            negative_lobe=-2.2 * exp(
                -((x₁ + 2.2)^2 / 3.2 + (x₂ - 0.9)^2 / 4.2),
            ),
            background=0.25 * sin(0.45 * x₁ - 0.30 * x₂)
            positive_lobe + negative_lobe + background
        end
    end
end

"""A multiscale field that lies in neither backend's exact model class."""
function comparison_ground_truth(::ComplicatedComparison, X)
    map(eachrow(X)) do x
        let x₁=x[1],
            x₂=x[2],
            broad_structure=0.95 * sin(0.58 * x₁) * cos(0.64 * x₂),
            fine_structure=0.48 *
                sin(1.45 * x₁ + 0.28 * x₂) *
                cos(1.12 * x₂),
            positive_peak=2.2 * exp(
                -((x₁ + 2.6)^2 / 1.2 + (x₂ - 2.3)^2 / 0.75),
            ),
            negative_peak=-2.0 * exp(
                -((x₁ - 2.4)^2 / 0.65 + (x₂ + 2.0)^2 / 1.4),
            ),
            curved_ridge=0.85 *
                exp(-(x₂ - 0.9 * sin(0.95 * x₁))^2 / 0.20) *
                cos(0.55 * x₁)
            broad_structure + fine_structure + positive_peak +
                negative_peak + curved_ridge
        end
    end
end

function comparison_basis_centers(n_centers)
    regular_locations(
        collect(range(-5.5, 5.5; length=n_centers)),
        collect(range(-5.5, 5.5; length=n_centers)),
    )
end

function comparison_scribe_parameters(centers, basis_widths)
    let nᵩ=length(centers[:, 1]) * length(basis_widths)
        LGSFModelParameters(
            μ=centers,
            σ=basis_widths,
            τ=[1.0],
            ϕ₀=zeros(nᵩ),
            A=Matrix{Float64}(I, nᵩ, nᵩ),
            Q=1e-10 * Matrix{Float64}(I, nᵩ, nᵩ),
        )
    end
end

function information_from_covariance(covariance)
    let covariance_factor=cholesky(Symmetric(covariance)),
        nᵩ=size(covariance, 1),
        Y=Matrix(
            covariance_factor \
            Matrix{Float64}(I, nᵩ, nᵩ),
        )
        KFEnvInfo(zeros(nᵩ), Y, zeros(nᵩ), zeros(nᵩ, nᵩ))
    end
end

function normalized_basis_information(
    smodel,
    evaluation_locations,
    base_covariance;
    field_variance=4.0,
)
    let H=prediction_dynamics(smodel, evaluation_locations),
        represented_variance=vec(sum(
            (H * base_covariance) .* H;
            dims=2,
        )),
        covariance_scale=field_variance / mean(represented_variance),
        nᵩ=size(base_covariance, 1),
        covariance=covariance_scale * base_covariance +
            1e-8 * Matrix{Float64}(I, nᵩ, nᵩ)
        information_from_covariance(covariance)
    end
end

prior_field_deviation(::SimpleComparison) = 2.0
prior_field_deviation(::ComplicatedComparison) = 1.0

function multiscale_prior_parameters(basis)
    basis[:σ] < 2.0 ?
        (variance=1.0, correlation_length=1.3) :
        (variance=0.18, correlation_length=2.5)
end

function multiscale_prior_covariance(params)
    let basis=params.ψ_p,
        covariance=[
            if ψᵢ[:σ] == ψⱼ[:σ]
                let prior=multiscale_prior_parameters(ψᵢ),
                    Δ²=sum(abs2, ψᵢ[:μ] - ψⱼ[:μ])
                    prior.variance * exp(
                        -Δ² / (2 * prior.correlation_length^2),
                    )
                end
            else
                0.0
            end
            for ψᵢ in basis, ψⱼ in basis
        ]
        covariance +
            1e-6 * Matrix{Float64}(I, params.nᵩ, params.nᵩ)
    end
end

function complicated_scribe_parameters()
    let local_centers=comparison_basis_centers(9),
        broad_centers=comparison_basis_centers(4),
        centers=vcat(local_centers, broad_centers),
        basis_widths=vcat(
            fill(1.35, size(local_centers, 1)),
            fill(3.0, size(broad_centers, 1)),
        ),
        params=comparison_scribe_parameters(centers, [1.0])
        foreach(zip(params.ψ_p, basis_widths)) do (basis, width)
            basis[:σ] = width
        end
        params
    end
end

function comparison_scribe_model(
    scenario::SimpleComparison,
    noise_variance,
    evaluation_locations,
)
    let centers=comparison_basis_centers(7),
        params=comparison_scribe_parameters(centers, [2.1]),
        smodel=initialize_SCRIBEModel_from_parameters(params),
        covariance=coefficient_prior_covariance(
            centers;
            coefficient_variance=1.0,
            correlation_length=1.5,
        ),
        information=normalized_basis_information(
            smodel,
            evaluation_locations,
            covariance,
            field_variance=prior_field_deviation(scenario)^2,
        ),
        observer=LGSFObserverBehavior(noise_variance)
        VulcanSCRIBEModel(smodel, information, observer)
    end
end

function comparison_scribe_model(
    scenario::ComplicatedComparison,
    noise_variance,
    evaluation_locations,
)
    let params=complicated_scribe_parameters(),
        smodel=initialize_SCRIBEModel_from_parameters(params),
        covariance=multiscale_prior_covariance(params),
        information=normalized_basis_information(
            smodel,
            evaluation_locations,
            covariance,
            field_variance=prior_field_deviation(scenario)^2,
        ),
        observer=LGSFObserverBehavior(noise_variance)
        VulcanSCRIBEModel(smodel, information, observer)
    end
end

gp_length_scale(::SimpleComparison) = 1.8
gp_length_scale(::ComplicatedComparison) = 1.25

function comparison_gp_model(scenario, noise_variance)
    let length_scale=gp_length_scale(scenario),
        signal_deviation=prior_field_deviation(scenario),
        observation_deviation=sqrt(noise_variance)
        GPE(
            Matrix{Float64}(undef, 2, 0),
            Float64[],
            MeanZero(),
            SE(
                fill(log(length_scale), 2),
                log(signal_deviation),
            ),
            log(observation_deviation),
        )
    end
end

function comparison_reward_locations(settings)
    let reward_axis=collect(
            range(-5.0, 5.0; length=settings.reward_points),
        )
        regular_locations(reward_axis, reward_axis)
    end
end

function make_comparison_mdp(
    settings,
    scenario,
    initial_model,
    reward_locations,
)
    let navigation_grid=collect(
            range(-5.0, 5.0; length=settings.navigation_points),
        ),
        reward_dynamics=initial_model isa VulcanSCRIBEModel ?
            prediction_dynamics(initial_model.smodel, reward_locations) :
            nothing,
        mdp=ModelComparisonMDP(
            navigation_grid,
            navigation_grid,
            reward_locations,
            reward_dynamics,
            scenario,
            initial_model,
            0.08,
            settings.n_samples,
            settings.planning_iterations,
        ),
        center=cld(length(navigation_grid), 2)
        (
            mdp=mdp,
            start=comparison_state(mdp, (center, center)),
        )
    end
end

function scribe_surface_moments(model, X)
    let H=prediction_dynamics(model.smodel, X),
        coefficients=posterior_coefficient_moments(model.information),
        μ=H * coefficients.μ,
        σ²=vec(sum((H * coefficients.Σ) .* H; dims=2))
        (μ=μ, σ²=max.(σ², 0.0))
    end
end

function gp_surface_moments(gp, X)
    let Xₚ=X',
        prior_variance=map(eachcol(Xₚ)) do x
            GaussianProcesses.cov(gp.kernel, x, x)
        end
        if isempty(gp.y)
            (μ=zeros(size(X, 1)), σ²=prior_variance)
        else
            let cross_covariance=GaussianProcesses.cov(
                    gp.kernel,
                    gp.x,
                    Xₚ,
                ),
                weighted_cross_covariance=gp.cK \ cross_covariance,
                μ=cross_covariance' * gp.alpha,
                σ²=prior_variance - vec(sum(
                    cross_covariance .* weighted_cross_covariance;
                    dims=1,
                ))
                (μ=μ, σ²=max.(σ², 0.0))
            end
        end
    end
end

model_surface_moments(model::VulcanSCRIBEModel, X) =
    scribe_surface_moments(model, X)

model_surface_moments(model::GPE, X) =
    gp_surface_moments(model, X)

function mean_predictive_log_likelihood(error, σ², noise_variance)
    let predictive_variance=σ² .+ noise_variance
        mean(
            -0.5 .* (
                log.(2π .* predictive_variance) .+
                abs2.(error) ./ predictive_variance
            ),
        )
    end
end

function prediction_interval_coverage(error, σ²; deviation=1.96)
    mean(abs.(error) .≤ deviation .* sqrt.(σ²))
end

function comparison_model_summary(model, grid, truth, noise_variance)
    let moments=model_surface_moments(model, grid.locations),
        uncertainty=sqrt.(moments.σ²),
        error=moments.μ - truth,
        rmse=sqrt(mean(abs2, error)),
        truth_deviation=std(truth)
        (
            prediction=moments.μ,
            uncertainty=uncertainty,
            absolute_error=abs.(error),
            rmse=rmse,
            normalized_rmse=rmse / truth_deviation,
            mae=mean(abs, error),
            mean_uncertainty=mean(uncertainty),
            integrated_uncertainty=mean(moments.σ²),
            maximum_uncertainty=maximum(uncertainty),
            predictive_log_likelihood=mean_predictive_log_likelihood(
                error,
                moments.σ²,
                noise_variance,
            ),
            interval_coverage=prediction_interval_coverage(
                error,
                moments.σ²,
            ),
        )
    end
end

function comparison_snapshot(
    mdp,
    model,
    grid,
    truth;
    n_samples,
    state,
    observation=missing,
    planning_reward=0.0,
)
    (
        n_samples=n_samples,
        state=state,
        location=vec(state_location(mdp, state)),
        observation=observation,
        planning_reward=planning_reward,
        expected_information=0.0,
        realized_information=0.0,
        model=model,
        summary=comparison_model_summary(
            model,
            grid,
            truth,
            mdp.noise_variance,
        ),
    )
end

function sample_comparison_environment(mdp, state, rng)
    let μ=only(comparison_ground_truth(
            mdp.scenario,
            state_location(mdp, state),
        ))
        μ + sqrt(mdp.noise_variance) * randn(rng)
    end
end

function observe_comparison_environment(mdp, model, state, rng)
    let planning_reward=expected_information_gain(mdp, model, state, 1),
        observation=sample_comparison_environment(mdp, state, rng),
        posterior=condition_environment_model(
            mdp,
            model,
            state,
            observation,
        )
        (
            model=posterior,
            observation=observation,
            planning_reward=planning_reward,
        )
    end
end

"""Plan from a fixed number of VulcanJ rollouts for repeatable comparisons."""
function comparison_action(policy, state, planning_iterations)
    let nodekey=VulcanJ.initialize_nodekey(state),
        solver=policy.solver,
        model=policy.node_models[nodekey]

        policy.tree_nodes[nodekey] = TreeNode(
            visits=0,
            action_values=Dict(),
            action_counts=Dict(),
            actions_tried=Set(),
        )

        foreach(1:planning_iterations) do _
            sample_rollout(
                policy,
                nodekey,
                model,
                0,
                solver.lookahead,
                0.0,
                0.0,
                solver.risk_budget,
            )
        end

        root=policy.tree_nodes[nodekey]
        selected_action=argmax(root.action_values)
        policy.best_action[state] = selected_action
        selected_action
    end
end

function run_model_comparison(mdp, start, settings, grid)
    let rng=MersenneTwister(settings.seed),
        truth=comparison_ground_truth(mdp.scenario, grid.locations),
        policy=make_planner(mdp, settings),
        history=Any[],
        model=initial_environment_model(mdp, start),
        state=start,
        start_time=time()

        push!(
            history,
            comparison_snapshot(
                mdp,
                model,
                grid,
                truth;
                n_samples=0,
                state,
            ),
        )

        observation=observe_comparison_environment(mdp, model, state, rng)
        model=observation.model
        push!(
            history,
            comparison_snapshot(
                mdp,
                model,
                grid,
                truth;
                n_samples=1,
                state,
                observation=observation.observation,
                planning_reward=observation.planning_reward,
            ),
        )

        for n_samples in 2:settings.n_samples
            set_environment_model!(policy, state, model)
            selected_action=comparison_action(
                policy,
                state,
                settings.planning_iterations,
            )
            state=successor_state(mdp, state, selected_action)
            observation=observe_comparison_environment(
                mdp,
                model,
                state,
                rng,
            )
            model=observation.model
            push!(
                history,
                comparison_snapshot(
                    mdp,
                    model,
                    grid,
                    truth;
                    n_samples,
                    state,
                    observation=observation.observation,
                    planning_reward=observation.planning_reward,
                ),
            )
        end

        (
            history=history,
            truth=truth,
            scenario=mdp.scenario,
            runtime=time() - start_time,
        )
    end
end

function comparison_path_plot(scribe_history, gp_history, grid, truth)
    let plot_object=surface_panel(
            grid,
            truth,
            "Exploration paths";
            color=:balance,
        ),
        scribe_path=sampled_path(scribe_history, lastindex(scribe_history)),
        gp_path=sampled_path(gp_history, lastindex(gp_history))
        if scribe_path == gp_path
            scatter!(
                plot_object,
                scribe_path[:, 1],
                scribe_path[:, 2];
                color=:white,
                markerstrokecolor=:black,
                markersize=3,
                label="shared samples",
            )
        else
            plot!(
                plot_object,
                scribe_path[:, 1],
                scribe_path[:, 2];
                color=:white,
                linewidth=2,
                label="SCRIBE",
            )
            plot!(
                plot_object,
                gp_path[:, 1],
                gp_path[:, 2];
                color=:yellow,
                linewidth=2,
                label="GP",
            )
        end
        plot_object
    end
end

function overlay_samples!(plot_object, path)
    if !isempty(path)
        scatter!(
            plot_object,
            path[:, 1],
            path[:, 2];
            color=:white,
            markerstrokecolor=:black,
            markersize=3,
            label=false,
        )
    end
    plot_object
end

function final_model_panel(
    history,
    grid,
    field,
    title;
    color,
    limits,
    shared_samples=false,
)
    let plot_object=surface_panel(
            grid,
            field,
            title;
            color,
            color_limits=limits,
        ),
        path=sampled_path(history, lastindex(history))
        shared_samples ?
            overlay_samples!(plot_object, path) :
            overlay_path!(plot_object, path)
    end
end

function final_prediction_scatter(scribe_history, gp_history, truth)
    let scribe_prediction=last(scribe_history).summary.prediction,
        gp_prediction=last(gp_history).summary.prediction,
        plot_object=plot(
            truth,
            truth;
            color=:black,
            linestyle=:dash,
            label="identity",
            xlabel="ground truth",
            ylabel="prediction",
            title="Final predictions",
            aspect_ratio=:equal,
        )
        scatter!(
            plot_object,
            truth,
            scribe_prediction;
            label="SCRIBE",
            alpha=0.35,
            markersize=2,
            markerstrokewidth=0,
        )
        scatter!(
            plot_object,
            truth,
            gp_prediction;
            label="GP",
            alpha=0.35,
            markersize=2,
            markerstrokewidth=0,
        )
        plot_object
    end
end

function save_final_model_comparison(
    scribe_result,
    gp_result,
    grid,
    output_dir,
    ;
    shared_samples=false,
)
    let truth=scribe_result.truth,
        scribe_final=last(scribe_result.history).summary,
        gp_final=last(gp_result.history).summary,
        field_extent=maximum(abs, [
            truth;
            scribe_final.prediction;
            gp_final.prediction;
        ]),
        uncertainty_extent=max(
            maximum(scribe_final.uncertainty),
            maximum(gp_final.uncertainty),
        ),
        error_extent=max(
            maximum(scribe_final.absolute_error),
            maximum(gp_final.absolute_error),
        ),
        truth_plot=surface_panel(
            grid,
            truth,
            "$(scenario_title(scribe_result.scenario)) ground truth";
            color=:balance,
            color_limits=(-field_extent, field_extent),
        ),
        scribe_mean=final_model_panel(
            scribe_result.history,
            grid,
            scribe_final.prediction,
            "SCRIBE posterior mean";
            color=:balance,
            limits=(-field_extent, field_extent),
            shared_samples,
        ),
        gp_mean=final_model_panel(
            gp_result.history,
            grid,
            gp_final.prediction,
            "GP posterior mean";
            color=:balance,
            limits=(-field_extent, field_extent),
            shared_samples,
        ),
        paths=comparison_path_plot(
            scribe_result.history,
            gp_result.history,
            grid,
            truth,
        ),
        scribe_uncertainty=final_model_panel(
            scribe_result.history,
            grid,
            scribe_final.uncertainty,
            "SCRIBE posterior σ";
            color=:viridis,
            limits=(0.0, uncertainty_extent),
            shared_samples,
        ),
        gp_uncertainty=final_model_panel(
            gp_result.history,
            grid,
            gp_final.uncertainty,
            "GP posterior σ";
            color=:viridis,
            limits=(0.0, uncertainty_extent),
            shared_samples,
        ),
        predictions=final_prediction_scatter(
            scribe_result.history,
            gp_result.history,
            truth,
        ),
        scribe_error=final_model_panel(
            scribe_result.history,
            grid,
            scribe_final.absolute_error,
            "SCRIBE absolute error";
            color=:thermal,
            limits=(0.0, error_extent),
            shared_samples,
        ),
        gp_error=final_model_panel(
            gp_result.history,
            grid,
            gp_final.absolute_error,
            "GP absolute error";
            color=:thermal,
            limits=(0.0, error_extent),
            shared_samples,
        ),
        comparison=plot(
            truth_plot,
            scribe_mean,
            gp_mean,
            paths,
            scribe_uncertainty,
            gp_uncertainty,
            predictions,
            scribe_error,
            gp_error;
            layout=(3, 3),
            size=(1500, 1250),
        )
        savefig(comparison, joinpath(output_dir, "final_model_comparison.png"))
    end
end

function metric_series(history, metric)
    getproperty.(getproperty.(history, :summary), metric)
end

function comparison_metric_plot(
    scribe_history,
    gp_history,
    metric,
    title,
    ylabel,
)
    let samples=getproperty.(scribe_history, :n_samples),
        plot_object=plot(
            samples,
            metric_series(scribe_history, metric);
            label="SCRIBE",
            linewidth=2,
            xlabel="samples",
            ylabel,
            title,
        )
        plot!(
            plot_object,
            samples,
            metric_series(gp_history, metric);
            label="GP",
            linewidth=2,
        )
        plot_object
    end
end

function save_comparison_metrics(
    scribe_result,
    gp_result,
    output_dir;
    include_planning=true,
)
    let scribe_history=scribe_result.history,
        gp_history=gp_result.history,
        rmse=comparison_metric_plot(
            scribe_history,
            gp_history,
            :rmse,
            "Field RMSE",
            "RMSE",
        ),
        normalized_rmse=comparison_metric_plot(
            scribe_history,
            gp_history,
            :normalized_rmse,
            "Normalized field RMSE",
            "NRMSE",
        ),
        predictive_log_likelihood=comparison_metric_plot(
            scribe_history,
            gp_history,
            :predictive_log_likelihood,
            "Mean predictive log likelihood",
            "mean log p(y)",
        ),
        interval_coverage=comparison_metric_plot(
            scribe_history,
            gp_history,
            :interval_coverage,
            "95% latent interval coverage",
            "coverage",
        ),
        integrated_uncertainty=comparison_metric_plot(
            scribe_history,
            gp_history,
            :integrated_uncertainty,
            "Integrated posterior uncertainty",
            "mean variance",
        ),
        final_metric=include_planning ?
            let samples=getproperty.(scribe_history, :n_samples),
                reward=plot(
                    samples,
                    getproperty.(scribe_history, :planning_reward);
                    label="SCRIBE",
                    linewidth=2,
                    xlabel="samples",
                    ylabel="mean variance reduction",
                    title="Planning reward",
                )
                plot!(
                    reward,
                    samples,
                    getproperty.(gp_history, :planning_reward);
                    label="GP",
                    linewidth=2,
                )
            end :
            comparison_metric_plot(
                scribe_history,
                gp_history,
                :mae,
                "Field MAE",
                "MAE",
            )
        hline!(interval_coverage, [0.95]; color=:black, linestyle=:dash, label=false)
        metrics=plot(
            rmse,
            normalized_rmse,
            predictive_log_likelihood,
            interval_coverage,
            integrated_uncertainty,
            final_metric;
            layout=(3, 2),
            size=(1100, 1200),
        )
        savefig(metrics, joinpath(output_dir, "model_comparison_metrics.png"))
    end
end

function save_final_metrics(result, output_dir)
    open(joinpath(output_dir, "final_metrics.csv"), "w") do io
        println(
            io,
            "model,rmse,normalized_rmse,mae,predictive_log_likelihood," *
            "interval_coverage,integrated_uncertainty,runtime",
        )
        foreach((("scribe", result.scribe), ("gp", result.gp))) do entry
            name, model_result=entry
            summary=last(model_result.history).summary
            println(
                io,
                "$(name),$(summary.rmse),$(summary.normalized_rmse)," *
                "$(summary.mae),$(summary.predictive_log_likelihood)," *
                "$(summary.interval_coverage)," *
                "$(summary.integrated_uncertainty),$(model_result.runtime)",
            )
        end
    end
end

function save_model_comparison(result, grid, output_dir, settings)
    let scribe_dir=joinpath(output_dir, "scribe"),
        gp_dir=joinpath(output_dir, "gp")
        mkpath(scribe_dir)
        mkpath(gp_dir)
        save_posterior_animation(
            result.scribe.history,
            grid,
            result.scribe.truth,
            scribe_dir,
            settings,
        )
        save_posterior_animation(
            result.gp.history,
            grid,
            result.gp.truth,
            gp_dir,
            settings,
        )
        save_final_model_comparison(
            result.scribe,
            result.gp,
            grid,
            output_dir,
        )
        save_comparison_metrics(result.scribe, result.gp, output_dir)
        save_final_metrics(result, output_dir)
    end
end

function print_model_comparison(result, output_dir)
    let scribe_final=last(result.scribe.history).summary,
        gp_final=last(result.gp.history).summary
        println(
            "$(scenario_title(result.scribe.scenario)) " *
            "VulcanJ model comparison complete.",
        )
        println("  samples: $(last(result.scribe.history).n_samples)")
        println(
            "  SCRIBE: RMSE=$(round(scribe_final.rmse; digits=4)), " *
            "NRMSE=$(round(scribe_final.normalized_rmse; digits=4)), " *
            "log p=$(round(scribe_final.predictive_log_likelihood; digits=4)), " *
            "coverage=$(round(scribe_final.interval_coverage; digits=4)), " *
            "IV=$(round(scribe_final.integrated_uncertainty; digits=4)), " *
            "runtime=$(round(result.scribe.runtime; digits=2)) s",
        )
        println(
            "  GP:     RMSE=$(round(gp_final.rmse; digits=4)), " *
            "NRMSE=$(round(gp_final.normalized_rmse; digits=4)), " *
            "log p=$(round(gp_final.predictive_log_likelihood; digits=4)), " *
            "coverage=$(round(gp_final.interval_coverage; digits=4)), " *
            "IV=$(round(gp_final.integrated_uncertainty; digits=4)), " *
            "runtime=$(round(result.gp.runtime; digits=2)) s",
        )
        println("  results: $output_dir")
    end
end

function comparison_main(scenario::ComparisonScenario, profile=:full)
    let settings=comparison_settings(scenario, profile),
        grid=surface_grid(settings.evaluation_points),
        reward_locations=comparison_reward_locations(settings),
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
        scribe_result=run_model_comparison(
            scribe_problem.mdp,
            scribe_problem.start,
            settings,
            grid,
        ),
        gp_result=run_model_comparison(
            gp_problem.mdp,
            gp_problem.start,
            settings,
            grid,
        ),
        result=(scribe=scribe_result, gp=gp_result),
        output_dir=joinpath(
            @__DIR__,
            "res",
            "vulcan_model_comparison",
            "planner",
            scenario_name(scenario),
            String(profile),
        )
        save_model_comparison(result, grid, output_dir, settings)
        print_model_comparison(result, output_dir)
        result
    end
end
