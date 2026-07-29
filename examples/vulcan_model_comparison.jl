ENV["GKSwstype"] = "100"

using Distributions: Normal
using GaussianProcesses
using LinearAlgebra
using Match: @match
using POMDPs
using Plots
using Random
using SCRIBE
using Statistics
using VulcanJ

const COMPARISON_ACTIONS = (
    :north,
    :northeast,
    :east,
    :southeast,
    :south,
    :southwest,
    :west,
    :northwest,
)

regular_locations(x, y) =
    reduce(vcat, ([xᵢ yⱼ] for yⱼ in y for xᵢ in x))

function surface_grid(n_points)
    let axis=collect(range(-5.0, 5.0; length=n_points))
        SCRIBEVisualizationGrid(axis, axis)
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

surface_matrix(grid, values) =
    reshape(values, length(grid.x), length(grid.y))'

function sampled_path(records, frame_index)
    let sampled=records[2:frame_index]
        isempty(sampled) ?
            zeros(0, 2) :
            reduce(
                vcat,
                (reshape(record.location, 1, :) for record in sampled),
            )
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

function comparison_surface_limits(records, truth)
    let predictions=reduce(
            vcat,
            (record.summary.prediction for record in records),
        ),
        uncertainty=maximum(
            record.summary.maximum_uncertainty for record in records
        ),
        error=maximum(
            maximum(record.summary.absolute_error) for record in records
        ),
        field_extent=maximum(abs, [truth; predictions])
        (
            field=(-field_extent, field_extent),
            uncertainty=(0.0, uncertainty),
            error=(0.0, error),
        )
    end
end

function comparison_posterior_frame(
    records,
    frame_index,
    grid,
    truth,
    limits,
)
    let record=records[frame_index],
        path=sampled_path(records, frame_index),
        truth_plot=surface_panel(
            grid,
            truth,
            "Ground truth";
            color=:balance,
            color_limits=limits.field,
        ),
        posterior_plot=surface_panel(
            grid,
            record.summary.prediction,
            "Posterior mean";
            color=:balance,
            color_limits=limits.field,
        ),
        uncertainty_plot=surface_panel(
            grid,
            record.summary.uncertainty,
            "Posterior standard deviation";
            color=:viridis,
            color_limits=limits.uncertainty,
        ),
        error_plot=surface_panel(
            grid,
            record.summary.absolute_error,
            "Absolute error";
            color=:thermal,
            color_limits=limits.error,
        )
        foreach(
            plot_object -> overlay_path!(plot_object, path),
            (posterior_plot, uncertainty_plot, error_plot),
        )
        plot(
            truth_plot,
            posterior_plot,
            uncertainty_plot,
            error_plot;
            layout=(2, 2),
            size=(900, 720),
            margin=0.5 * Plots.mm,
            plot_titlefontsize=11,
            plot_title="Samples: $(record.n_samples)",
        )
    end
end

function save_comparison_animation(
    records,
    grid,
    truth,
    output_path,
    settings,
)
    let animation=Animation(),
        limits=comparison_surface_limits(records, truth)
        foreach(eachindex(records)) do frame_index
            frame(
                animation,
                comparison_posterior_frame(
                    records,
                    frame_index,
                    grid,
                    truth,
                    limits,
                ),
            )
        end
        gif(animation, output_path; fps=settings.animation_fps)
    end
end

abstract type ComparisonScenario end

struct SimpleComparison <: ComparisonScenario end
struct ComplicatedComparison <: ComparisonScenario end

scenario_name(::SimpleComparison) = "simple"
scenario_name(::ComplicatedComparison) = "complicated"

scenario_title(::SimpleComparison) = "Simple"
scenario_title(::ComplicatedComparison) = "Complicated multiscale"

"""The shared planning problem used to compare modeling backends."""
struct BackendComparisonMDP <: MDP{Matrix, Symbol}
    bounds
    step_size
    reward_locations
    planning_dynamics
    scenario::ComparisonScenario
    initial_model
    noise_variance
    n_steps
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

POMDPs.statetype(::BackendComparisonMDP) = Matrix
POMDPs.actiontype(::BackendComparisonMDP) = Symbol
POMDPs.actions(::BackendComparisonMDP, _) = COMPARISON_ACTIONS
POMDPs.discount(::BackendComparisonMDP) = 1.0
POMDPs.isterminal(::BackendComparisonMDP, _) = false

function POMDPs.gen(mdp::BackendComparisonMDP, state, action, _)
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

VulcanJ.get_failure_prob(::BackendComparisonMDP, _, _) = 0.0
VulcanJ.horizon(mdp::BackendComparisonMDP) = mdp.n_steps
VulcanJ.initial_environment_model(
    mdp::BackendComparisonMDP,
    _,
) = mdp.initial_model

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
    mdp::BackendComparisonMDP,
    model::SCRIBEModelState,
    state,
    ::Integer,
)
    let Hₛ=prediction_dynamics(model.smodel, state),
        R=model.R
        integrated_variance_reduction(
            model.information,
            Hₛ,
            mdp.planning_dynamics,
            R,
        )
    end
end

function VulcanJ.expected_information_gain(
    mdp::BackendComparisonMDP,
    model::GPE,
    state,
    ::Integer,
)
    gp_integrated_variance_reduction(
        model,
        state',
        mdp.reward_locations',
    )
end

function VulcanJ.conditional_observation_distribution(
    mdp::BackendComparisonMDP,
    model::SCRIBEModelState,
    state,
)
    let moments=posterior_measurement_moments(
            model.smodel,
            model.information,
            state,
            model.R,
        )
        Normal(only(moments.μ), sqrt(only(moments.Σ)))
    end
end

function VulcanJ.conditional_observation_distribution(
    ::BackendComparisonMDP,
    model::GPE,
    state,
)
    let (μ, σ²)=VulcanJ.gp_predict(model, state)
        Normal(μ, sqrt(max(σ² + noise_variance(model), eps())))
    end
end

function VulcanJ.condition_environment_model(
    ::BackendComparisonMDP,
    model::SCRIBEModelState,
    state,
    observation,
)
    let R=model.R,
        information=condition_on_measurement(
            model.smodel,
            model.information,
            state,
            observation,
            R,
        )
        SCRIBEModelState(model.smodel, information, R)
    end
end

measurement_scalar(measurement::Number) = Float64(measurement)
measurement_scalar(measurement) = Float64(only(measurement))

function VulcanJ.add_obs_to_gp(state::Matrix, observation, gp::GPE)
    let X=state',
        z=measurement_scalar(observation)
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

function rotated_quadratic_distance(x₁, x₂, μ, θ, λ)
    let δ₁=x₁ - μ[1],
        δ₂=x₂ - μ[2],
        c=cos(θ),
        s=sin(θ),
        u=muladd(c, δ₁, s * δ₂),
        v=muladd(-s, δ₁, c * δ₂)
        u^2 / λ[1] + v^2 / λ[2]
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
                -rotated_quadratic_distance(
                    x₁,
                    x₂,
                    (-2.45, 2.15),
                    π / 7,
                    (1.4, 0.52),
                ),
            ),
            negative_peak=-2.0 * exp(
                -rotated_quadratic_distance(
                    x₁,
                    x₂,
                    (2.15, -2.25),
                    -π / 5,
                    (0.62, 1.55),
                ),
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

function staggered_basis_centers(n_centers, offset, stagger)
    let x_axis=collect(range(-5.5, 5.5; length=n_centers)),
        y_axis=collect(range(-5.5, 5.5; length=n_centers))
        reduce(
            vcat,
            map(enumerate(y_axis)) do row
                index, y=row
                x_shift=offset[1] + (isodd(index) ? stagger : -stagger)
                hcat(
                    x_axis .+ x_shift,
                    fill(y + offset[2], n_centers),
                )
            end,
        )
    end
end

complicated_local_centers() =
    staggered_basis_centers(9, (0.2, -0.25), 0.32)

complicated_broad_centers() =
    staggered_basis_centers(4, (-0.35, 0.4), 0.3)

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
    let local_centers=complicated_local_centers(),
        broad_centers=complicated_broad_centers(),
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
        )
        SCRIBEModelState(smodel, information, noise_variance)
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
        )
        SCRIBEModelState(smodel, information, noise_variance)
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

function comparison_problem(
    settings,
    scenario,
    initial_model,
    reward_locations,
)
    let planning_dynamics=initial_model isa SCRIBEModelState ?
            prediction_dynamics(initial_model.smodel, reward_locations) :
            nothing,
        mdp=BackendComparisonMDP(
            (-5.0, 5.0),
            10.0 / (settings.navigation_points - 1),
            reward_locations,
            planning_dynamics,
            scenario,
            initial_model,
            0.08,
            settings.n_samples,
        )
        (mdp=mdp, start=zeros(1, 2))
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

model_surface_moments(model::SCRIBEModelState, X) =
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

function comparison_record(
    mdp,
    model,
    grid,
    truth;
    n_samples,
    state,
    planning_reward=0.0,
)
    (
        n_samples=n_samples,
        location=vec(state),
        planning_reward=planning_reward,
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
            state,
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
        records=Any[],
        model=initial_environment_model(mdp, start),
        model_states=model isa SCRIBEModelState ?
            SCRIBEModelState[model] :
            Any[model],
        sampling_locations=Matrix[],
        observations=Float64[],
        state=copy(start),
        start_time=time()

        push!(
            records,
            comparison_record(
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
        push!(model_states, model)
        push!(sampling_locations, copy(state))
        push!(observations, observation.observation)
        push!(
            records,
            comparison_record(
                mdp,
                model,
                grid,
                truth;
                n_samples=1,
                state,
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
            state=gen(mdp, state, selected_action, rng).sp
            observation=observe_comparison_environment(
                mdp,
                model,
                state,
                rng,
            )
            model=observation.model
            push!(model_states, model)
            push!(sampling_locations, copy(state))
            push!(observations, observation.observation)
            push!(
                records,
                comparison_record(
                    mdp,
                    model,
                    grid,
                    truth;
                    n_samples,
                    state,
                    planning_reward=observation.planning_reward,
                ),
            )
        end

        (
            records=records,
            model_states=model_states,
            sampling_locations=sampling_locations,
            observations=observations,
            truth=truth,
            scenario=mdp.scenario,
            runtime=time() - start_time,
        )
    end
end

function comparison_path_plot(scribe_records, gp_records, grid, truth)
    let plot_object=surface_panel(
            grid,
            truth,
            "Exploration paths";
            color=:balance,
        ),
        scribe_path=sampled_path(scribe_records, lastindex(scribe_records)),
        gp_path=sampled_path(gp_records, lastindex(gp_records))
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
    records,
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
        path=sampled_path(records, lastindex(records))
        shared_samples ?
            overlay_samples!(plot_object, path) :
            overlay_path!(plot_object, path)
    end
end

function final_prediction_scatter(scribe_records, gp_records, truth)
    let scribe_prediction=last(scribe_records).summary.prediction,
        gp_prediction=last(gp_records).summary.prediction,
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
        scribe_final=last(scribe_result.records).summary,
        gp_final=last(gp_result.records).summary,
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
            scribe_result.records,
            grid,
            scribe_final.prediction,
            "SCRIBE posterior mean";
            color=:balance,
            limits=(-field_extent, field_extent),
            shared_samples,
        ),
        gp_mean=final_model_panel(
            gp_result.records,
            grid,
            gp_final.prediction,
            "GP posterior mean";
            color=:balance,
            limits=(-field_extent, field_extent),
            shared_samples,
        ),
        paths=comparison_path_plot(
            scribe_result.records,
            gp_result.records,
            grid,
            truth,
        ),
        scribe_uncertainty=final_model_panel(
            scribe_result.records,
            grid,
            scribe_final.uncertainty,
            "SCRIBE posterior σ";
            color=:viridis,
            limits=(0.0, uncertainty_extent),
            shared_samples,
        ),
        gp_uncertainty=final_model_panel(
            gp_result.records,
            grid,
            gp_final.uncertainty,
            "GP posterior σ";
            color=:viridis,
            limits=(0.0, uncertainty_extent),
            shared_samples,
        ),
        predictions=final_prediction_scatter(
            scribe_result.records,
            gp_result.records,
            truth,
        ),
        scribe_error=final_model_panel(
            scribe_result.records,
            grid,
            scribe_final.absolute_error,
            "SCRIBE absolute error";
            color=:thermal,
            limits=(0.0, error_extent),
            shared_samples,
        ),
        gp_error=final_model_panel(
            gp_result.records,
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

function metric_series(records, metric)
    getproperty.(getproperty.(records, :summary), metric)
end

function comparison_metric_plot(
    scribe_records,
    gp_records,
    metric,
    title,
    ylabel,
)
    let samples=getproperty.(scribe_records, :n_samples),
        plot_object=plot(
            samples,
            metric_series(scribe_records, metric);
            label="SCRIBE",
            linewidth=2,
            xlabel="samples",
            ylabel,
            title,
        )
        plot!(
            plot_object,
            samples,
            metric_series(gp_records, metric);
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
    let scribe_records=scribe_result.records,
        gp_records=gp_result.records,
        rmse=comparison_metric_plot(
            scribe_records,
            gp_records,
            :rmse,
            "Field RMSE",
            "RMSE",
        ),
        normalized_rmse=comparison_metric_plot(
            scribe_records,
            gp_records,
            :normalized_rmse,
            "Normalized field RMSE",
            "NRMSE",
        ),
        predictive_log_likelihood=comparison_metric_plot(
            scribe_records,
            gp_records,
            :predictive_log_likelihood,
            "Mean predictive log likelihood",
            "mean log p(y)",
        ),
        interval_coverage=comparison_metric_plot(
            scribe_records,
            gp_records,
            :interval_coverage,
            "95% latent interval coverage",
            "coverage",
        ),
        integrated_uncertainty=comparison_metric_plot(
            scribe_records,
            gp_records,
            :integrated_uncertainty,
            "Integrated posterior uncertainty",
            "mean variance",
        ),
        final_metric=include_planning ?
            let samples=getproperty.(scribe_records, :n_samples),
                reward=plot(
                    samples,
                    getproperty.(scribe_records, :planning_reward);
                    label="SCRIBE",
                    linewidth=2,
                    xlabel="samples",
                    ylabel="mean variance reduction",
                    title="Planning reward",
                )
                plot!(
                    reward,
                    samples,
                    getproperty.(gp_records, :planning_reward);
                    label="GP",
                    linewidth=2,
                )
            end :
            comparison_metric_plot(
                scribe_records,
                gp_records,
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
            summary=last(model_result.records).summary
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

function comparison_visualization(result, grid)
    let model_states=SCRIBEModelState[result.model_states...]
        scribe_model_history(
            model_states;
            sampling_locations=result.sampling_locations,
            grid,
            ground_truth=result.truth,
        )
    end
end

function save_model_comparison(result, grid, output_dir, settings)
    let scribe_dir=joinpath(output_dir, "scribe"),
        gp_dir=joinpath(output_dir, "gp"),
        scribe_visualization=comparison_visualization(result.scribe, grid)
        mkpath(scribe_dir)
        mkpath(gp_dir)
        save_static_visualizations(
            scribe_visualization;
            output_dir=scribe_dir,
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
            scribe_visualization;
            output_dir=scribe_dir,
            metrics=[:posterior_summary],
            fps=settings.animation_fps,
        )
        save_comparison_animation(
            result.gp.records,
            grid,
            result.gp.truth,
            joinpath(gp_dir, "posterior_exploration.gif"),
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
    let scribe_final=last(result.scribe.records).summary,
        gp_final=last(result.gp.records).summary
        println(
            "$(scenario_title(result.scribe.scenario)) " *
            "VulcanJ model comparison complete.",
        )
        println("  samples: $(last(result.scribe.records).n_samples)")
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
