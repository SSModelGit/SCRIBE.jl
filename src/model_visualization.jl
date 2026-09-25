using Match: @match
using Plots
using Statistics: mean, std

export SCRIBEModelState, SCRIBEObservation
export SCRIBEVisualizationGrid, SCRIBEVisualizationHistory
export scribe_model_history
export plot_posterior_mean_map, plot_posterior_uncertainty_map
export plot_posterior_against_ground_truth, plot_absolute_error_map
export plot_metric_history, plot_predictions_against_ground_truth
export plot_posterior_summary
export animate_posterior_mean_map, animate_posterior_uncertainty_map
export animate_posterior_against_ground_truth
export animate_absolute_error_map, animate_metric_history
export animate_predictions_against_ground_truth
export animate_posterior_summary
export save_static_visualizations, save_animated_visualizations

"""A SCRIBE model together with the information state defining its posterior."""
struct SCRIBEModelState
    smodel
    information
    R
end

"""A realized measurement and the location used to condition the model."""
struct SCRIBEObservation
    X
    z
    R
end

"""A regular two-dimensional grid used to evaluate a SCRIBE posterior."""
struct SCRIBEVisualizationGrid
    x
    y
    locations
end

function SCRIBEVisualizationGrid(x, y)
    let locations=reduce(
            vcat,
            ([xᵢ yⱼ] for yⱼ in y for xᵢ in x),
        )
        SCRIBEVisualizationGrid(x, y, locations)
    end
end

function SCRIBEVisualizationGrid(
    state::SCRIBEModelState;
    n_points=51,
)
    if state.smodel isa EOFClimateModel
        locations = state.smodel.params.locations
        size(locations, 2) >= 2 ||
            throw(ArgumentError(
                "One-dimensional EOF models require an explicit " *
                "SCRIBEVisualizationGrid.",
            ))
        x = collect(range(
            minimum(locations[:, 1]),
            maximum(locations[:, 1]);
            length=n_points,
        ))
        y = collect(range(
            minimum(locations[:, 2]),
            maximum(locations[:, 2]);
            length=n_points,
        ))
        return SCRIBEVisualizationGrid(x, y)
    end

    let centers=state.smodel.params.p[:μ],
        x=collect(range(
            minimum(centers[:, 1]),
            maximum(centers[:, 1]);
            length=n_points,
        )),
        y=collect(range(
            minimum(centers[:, 2]),
            maximum(centers[:, 2]);
            length=n_points,
        ))
        SCRIBEVisualizationGrid(x, y)
    end
end

"""Evaluated posterior states and optional sampling-path metadata."""
struct SCRIBEVisualizationHistory
    states
    observations
    grid
    ground_truth
    sampling_locations
    show_sampling_path
    frames
end

Base.length(history::SCRIBEVisualizationHistory) = length(history.frames)
Base.firstindex(history::SCRIBEVisualizationHistory) =
    firstindex(history.frames)
Base.lastindex(history::SCRIBEVisualizationHistory) =
    lastindex(history.frames)
Base.getindex(history::SCRIBEVisualizationHistory, index) =
    history.frames[index]

visualization_truth(::Nothing, _) = nothing
visualization_truth(ground_truth::Function, X) = ground_truth(X)
visualization_truth(ground_truth, _) = ground_truth

visualization_noise_variance(R::Number) = R
visualization_noise_variance(R::AbstractVector) = mean(R)
visualization_noise_variance(R::UniformScaling) = R.λ
visualization_noise_variance(R::AbstractMatrix) = mean(diag(R))

function posterior_statistics(prediction, variance, ground_truth, R)
    let uncertainty=sqrt.(max.(variance, 0.0)),
        noise_variance=visualization_noise_variance(R),
        uncertainty_statistics=Dict(
            :uncertainty => uncertainty,
            :mean_uncertainty => mean(uncertainty),
            :maximum_uncertainty => maximum(uncertainty),
            :integrated_uncertainty => mean(variance),
        )
        isnothing(ground_truth) ?
            merge(
                Dict(
                    :prediction => prediction,
                    :variance => variance,
                    :absolute_error => nothing,
                    :rmse => missing,
                    :normalized_rmse => missing,
                    :mae => missing,
                    :predictive_log_likelihood => missing,
                    :interval_coverage => missing,
                ),
                uncertainty_statistics,
            ) :
            let error=prediction - ground_truth,
                rmse=sqrt(mean(abs2, error)),
                predictive_variance=variance .+ noise_variance
                merge(
                    Dict(
                        :prediction => prediction,
                        :variance => variance,
                        :absolute_error => abs.(error),
                        :rmse => rmse,
                        :normalized_rmse => rmse / std(ground_truth),
                        :mae => mean(abs, error),
                        :predictive_log_likelihood => mean(
                            -0.5 .* (
                                log.(2π .* predictive_variance) .+
                                abs2.(error) ./ predictive_variance
                            ),
                        ),
                        :interval_coverage => mean(
                            abs.(error) .≤ 1.96 .* uncertainty,
                        ),
                    ),
                    uncertainty_statistics,
                )
            end
    end
end

function visualization_summary(state, grid, ground_truth)
    let moments=posterior_model_moments(
            state.smodel,
            state.information,
            grid.locations,
        ),
        variance=max.(diag(moments[:Σ]), 0.0),
        statistics=posterior_statistics(
            moments[:μ],
            variance,
            ground_truth,
            state.R,
        ),
        coefficient_mean=posterior_coefficient_moments(
            state.information,
        )[:μ]
        merge(
            statistics,
            Dict(
                :coefficient_magnitude => norm(coefficient_mean) /
                    sqrt(length(coefficient_mean)),
                :mean_entropy => evaluate_information_metric(
                    state.information;
                    metric=:differential_entropy,
                ) / length(coefficient_mean),
            ),
        )
    end
end

function visualization_frame(
    state,
    grid,
    ground_truth;
    n_samples,
    observation=missing,
    expected_information=0.0,
    realized_information=0.0,
    variance_reduction=0.0,
)
    Dict(
        :n_samples => n_samples,
        :observation => observation,
        :expected_information => expected_information,
        :realized_information => realized_information,
        :integrated_variance_reduction => variance_reduction,
        :summary => visualization_summary(state, grid, ground_truth),
    )
end

function realized_observation(
    observation::SCRIBEObservation,
    _,
    _,
)
    observation
end

function realized_observation(
    observation::LGSFObserverState,
    _,
    _,
)
    SCRIBEObservation(
        observation.X,
        observation.z,
        observation.v[:R],
    )
end

function realized_observation(
    observation::EOFObserverState,
    _,
    _,
)
    SCRIBEObservation(
        observation.X,
        observation.z,
        observation.v[:R],
    )
end

function realized_observation(observation, sampling_location, R)
    SCRIBEObservation(sampling_location, observation, R)
end

function sampling_location(sampling_locations, index)
    isnothing(sampling_locations) ? nothing : sampling_locations[index]
end

function condition_visualization_state(
    prior,
    observation,
    grid,
    ground_truth,
    n_samples,
)
    let expected_information=mutual_information(
            prior.smodel,
            prior.information,
            observation.X,
            observation.R,
        ),
        variance_reduction=integrated_variance_reduction(
            prior.smodel,
            prior.information,
            observation.X,
            grid.locations,
            observation.R,
        ),
        information=condition_on_measurement(
            prior.smodel,
            prior.information,
            observation.X,
            observation.z,
            observation.R,
        ),
        posterior=SCRIBEModelState(
            prior.smodel,
            information,
            observation.R,
        ),
        realized_information=D_KL(information, prior.information),
        frame=visualization_frame(
            posterior,
            grid,
            ground_truth;
            n_samples,
            observation=observation.z,
            expected_information,
            realized_information,
            variance_reduction,
        )
        Dict(:state => posterior, :frame => frame)
    end
end

"""Build visualization history from observations and an initial model.

`SCRIBEObservation`, `LGSFObserverState`, and `EOFObserverState` entries carry
their own filtering locations. Raw measurement values use the corresponding entry in
`sampling_locations`. Sampling locations are only drawn as a path when they
are explicitly supplied.
"""
function scribe_model_history(
    observations,
    initial_model::SCRIBEModelState;
    sampling_locations=nothing,
    grid=SCRIBEVisualizationGrid(initial_model),
    ground_truth=nothing,
    show_sampling_path=true,
)
    let truth=visualization_truth(ground_truth, grid.locations),
        states=SCRIBEModelState[initial_model],
        frames=Any[
            visualization_frame(
                initial_model,
                grid,
                truth;
                n_samples=0,
            ),
        ],
        prior=initial_model
        foreach(enumerate(observations)) do sample
            index, observation=sample
            realized=realized_observation(
                observation,
                sampling_location(sampling_locations, index),
                prior.R,
            )
            update=condition_visualization_state(
                prior,
                realized,
                grid,
                truth,
                index,
            )
            prior=update[:state]
            push!(states, update[:state])
            push!(frames, update[:frame])
        end
        SCRIBEVisualizationHistory(
            states,
            observations,
            grid,
            truth,
            sampling_locations,
            show_sampling_path && !isnothing(sampling_locations),
            frames,
        )
    end
end

"""Build visualization history by sampling an observation function."""
function scribe_model_history(
    observe::Function,
    sampling_locations,
    initial_model::SCRIBEModelState;
    grid=SCRIBEVisualizationGrid(initial_model),
    ground_truth=nothing,
    show_sampling_path=true,
)
    let observations=map(
            X -> SCRIBEObservation(X, observe(X), initial_model.R),
            sampling_locations,
        )
        scribe_model_history(
            observations,
            initial_model;
            sampling_locations,
            grid,
            ground_truth,
            show_sampling_path,
        )
    end
end

function model_state_frame(
    state,
    previous_state,
    sampling_location,
    grid,
    ground_truth,
    n_samples,
)
    let realized_information=isnothing(previous_state) ?
            0.0 :
            D_KL(state.information, previous_state.information),
        expected_information=isnothing(sampling_location) ?
            0.0 :
            mutual_information(
                previous_state.smodel,
                previous_state.information,
                sampling_location,
                previous_state.R,
            ),
        variance_reduction=isnothing(sampling_location) ?
            0.0 :
            integrated_variance_reduction(
                previous_state.smodel,
                previous_state.information,
                sampling_location,
                grid.locations,
                previous_state.R,
            )
        visualization_frame(
            state,
            grid,
            ground_truth;
            n_samples,
            expected_information,
            realized_information,
            variance_reduction,
        )
    end
end

"""Build visualization history from a sequence containing the initial state
followed by each updated SCRIBE model state. When sampling locations are
provided, they also define each update's expected mutual information and
integrated variance reduction.
"""
function scribe_model_history(
    model_states::AbstractVector{<:SCRIBEModelState};
    sampling_locations=nothing,
    grid=SCRIBEVisualizationGrid(first(model_states)),
    ground_truth=nothing,
    show_sampling_path=true,
)
    let truth=visualization_truth(ground_truth, grid.locations),
        frames=map(enumerate(model_states)) do state_entry
            index, state=state_entry
            previous_state=index == 1 ? nothing : model_states[index - 1]
            location=index == 1 || isnothing(sampling_locations) ?
                nothing :
                sampling_locations[index - 1]
            model_state_frame(
                state,
                previous_state,
                location,
                grid,
                truth,
                index - 1,
            )
        end
        SCRIBEVisualizationHistory(
            model_states,
            nothing,
            grid,
            truth,
            sampling_locations,
            show_sampling_path && !isnothing(sampling_locations),
            frames,
        )
    end
end

visualization_history(
    history::SCRIBEVisualizationHistory;
    parameters...,
) = history

function visualization_history(
    observations,
    initial_model::SCRIBEModelState;
    parameters...,
)
    scribe_model_history(
        observations,
        initial_model;
        parameters...,
    )
end

function visualization_history(
    observe::Function,
    sampling_locations,
    initial_model::SCRIBEModelState;
    parameters...,
)
    scribe_model_history(
        observe,
        sampling_locations,
        initial_model;
        parameters...,
    )
end

function visualization_history(
    model_states::AbstractVector{<:SCRIBEModelState};
    parameters...,
)
    scribe_model_history(model_states; parameters...)
end

surface_matrix(grid, values) =
    reshape(values, length(grid.x), length(grid.y))'

function sampled_path(history, frame_index)
    if !history.show_sampling_path || frame_index == 1
        zeros(0, 2)
    else
        let n_locations=min(
                frame_index - 1,
                length(history.sampling_locations),
            ),
            locations=history.sampling_locations[1:n_locations]
            reduce(
                vcat,
                (reshape(location, 1, :) for location in locations),
            )
        end
    end
end

function overlay_sampling_path!(plot_object, path)
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

function posterior_surface_panel(
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
        margin=0.5 * Plots.mm,
        left_margin=2.5 * Plots.mm,
        bottom_margin=2.5 * Plots.mm,
        right_margin=0.5 * Plots.mm,
        top_margin=0.5 * Plots.mm,
    )
end

function visualization_limits(history)
    let summaries=getindex.(history.frames, :summary),
        predictions=reduce(vcat, getindex.(summaries, :prediction)),
        reference=isnothing(history.ground_truth) ?
            predictions :
            history.ground_truth,
        field_extent=maximum(abs, [reference; predictions]),
        uncertainty_extent=maximum(
            getindex.(summaries, :maximum_uncertainty),
        ),
        error_extent=isnothing(history.ground_truth) ?
            1.0 :
            maximum(
                maximum(summary[:absolute_error]) for summary in summaries
            )
        Dict(
            :field => (-field_extent, field_extent),
            :uncertainty => (0.0, uncertainty_extent),
            :error => (0.0, error_extent),
        )
    end
end

function posterior_mean_title(frame)
    ismissing(frame[:summary][:rmse]) ?
        "Posterior mean — $(frame[:n_samples]) samples" :
        "Posterior mean — $(frame[:n_samples]) samples\n" *
            "RMSE $(round(frame[:summary][:rmse]; digits=3))"
end

function plot_posterior_mean_map(
    history::SCRIBEVisualizationHistory;
    frame_index=lastindex(history),
    limits=visualization_limits(history),
)
    let frame=history[frame_index],
        plot_object=posterior_surface_panel(
            history.grid,
            frame[:summary][:prediction],
            posterior_mean_title(frame);
            color=:balance,
            color_limits=limits[:field],
        )
        overlay_sampling_path!(
            plot_object,
            sampled_path(history, frame_index),
        )
        plot!(plot_object; size=(480, 390))
    end
end

function plot_posterior_mean_map(
    input...;
    frame_index=nothing,
    parameters...,
)
    let history=visualization_history(input...; parameters...),
        index=isnothing(frame_index) ? lastindex(history) : frame_index
        plot_posterior_mean_map(history; frame_index=index)
    end
end

function plot_posterior_uncertainty_map(
    history::SCRIBEVisualizationHistory;
    frame_index=lastindex(history),
    limits=visualization_limits(history),
)
    let frame=history[frame_index],
        plot_object=posterior_surface_panel(
            history.grid,
            frame[:summary][:uncertainty],
            "Posterior uncertainty — $(frame[:n_samples]) samples\n" *
                "mean σ $(round(frame[:summary][:mean_uncertainty]; digits=3))";
            color=:viridis,
            color_limits=limits[:uncertainty],
        )
        overlay_sampling_path!(
            plot_object,
            sampled_path(history, frame_index),
        )
        plot!(plot_object; size=(480, 390))
    end
end

function plot_posterior_uncertainty_map(
    input...;
    frame_index=nothing,
    parameters...,
)
    let history=visualization_history(input...; parameters...),
        index=isnothing(frame_index) ? lastindex(history) : frame_index
        plot_posterior_uncertainty_map(history; frame_index=index)
    end
end

function plot_absolute_error_map(
    history::SCRIBEVisualizationHistory;
    frame_index=lastindex(history),
    limits=visualization_limits(history),
)
    let frame=history[frame_index],
        plot_object=posterior_surface_panel(
            history.grid,
            frame[:summary][:absolute_error],
            "Absolute error — $(frame[:n_samples]) samples\n" *
                "MAE $(round(frame[:summary][:mae]; digits=3))";
            color=:thermal,
            color_limits=limits[:error],
        )
        overlay_sampling_path!(
            plot_object,
            sampled_path(history, frame_index),
        )
        plot!(plot_object; size=(480, 390))
    end
end

function plot_absolute_error_map(
    input...;
    frame_index=nothing,
    parameters...,
)
    let history=visualization_history(input...; parameters...),
        index=isnothing(frame_index) ? lastindex(history) : frame_index
        plot_absolute_error_map(history; frame_index=index)
    end
end

function plot_posterior_against_ground_truth(
    history::SCRIBEVisualizationHistory;
    frame_index=lastindex(history),
    limits=visualization_limits(history),
)
    let truth_plot=posterior_surface_panel(
            history.grid,
            history.ground_truth,
            "Ground truth";
            color=:balance,
            color_limits=limits[:field],
        ),
        posterior_plot=plot_posterior_mean_map(
            history;
            frame_index,
            limits,
        )
        plot(
            truth_plot,
            posterior_plot;
            layout=(2, 1),
            size=(480, 720),
            margin=0.5 * Plots.mm,
        )
    end
end

function plot_posterior_against_ground_truth(
    input...;
    frame_index=nothing,
    parameters...,
)
    let history=visualization_history(input...; parameters...),
        index=isnothing(frame_index) ? lastindex(history) : frame_index
        plot_posterior_against_ground_truth(
            history;
            frame_index=index,
        )
    end
end

function plot_posterior_summary(
    history::SCRIBEVisualizationHistory;
    frame_index=lastindex(history),
    limits=visualization_limits(history),
)
    let frame=history[frame_index],
        posterior_plot=plot_posterior_mean_map(
            history;
            frame_index,
            limits,
        ),
        uncertainty_plot=plot_posterior_uncertainty_map(
            history;
            frame_index,
            limits,
        )
        if isnothing(history.ground_truth)
            plot(
                posterior_plot,
                uncertainty_plot;
                layout=(1, 2),
                size=(900, 390),
                margin=0.5 * Plots.mm,
            )
        else
            let truth_plot=posterior_surface_panel(
                    history.grid,
                    history.ground_truth,
                    "Ground truth";
                    color=:balance,
                    color_limits=limits[:field],
                ),
                error_plot=plot_absolute_error_map(
                    history;
                    frame_index,
                    limits,
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
                    plot_title="Samples: $(frame[:n_samples])   " *
                        "Expected IVR: $(round(
                            frame[:integrated_variance_reduction];
                            digits=3,
                        ))",
                )
            end
        end
    end
end

function plot_posterior_summary(
    input...;
    frame_index=nothing,
    parameters...,
)
    let history=visualization_history(input...; parameters...),
        index=isnothing(frame_index) ? lastindex(history) : frame_index
        plot_posterior_summary(history; frame_index=index)
    end
end

function plot_metric_history(
    history::SCRIBEVisualizationHistory;
    frame_index=lastindex(history),
)
    let frames=history.frames[1:frame_index],
        samples=getindex.(frames, :n_samples),
        summaries=getindex.(frames, :summary),
        mean_uncertainty=getindex.(summaries, :mean_uncertainty),
        maximum_uncertainty=getindex.(summaries, :maximum_uncertainty),
        mean_entropy=getindex.(summaries, :mean_entropy),
        entropy_reduction=first(mean_entropy) .- mean_entropy,
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
            getindex.(frames, :expected_information);
            label="expected MI",
            marker=:circle,
            ylabel="information (nats)",
            xlabel="samples",
        )
        plot!(
            information_plot,
            samples,
            getindex.(frames, :realized_information);
            label="realized KL",
            marker=:circle,
        )

        reward_plot=plot(
            samples,
            getindex.(frames, :integrated_variance_reduction);
            label="integrated variance reduction",
            marker=:circle,
            ylabel="mean variance reduction",
            xlabel="samples",
        )

        model_plot=plot(
            samples,
            getindex.(summaries, :coefficient_magnitude);
            label="RMS posterior coefficient",
            marker=:circle,
            ylabel="model state",
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

        if isnothing(history.ground_truth)
            plot(
                uncertainty_plot,
                information_plot,
                reward_plot,
                model_plot;
                layout=(2, 2),
                size=(900, 680),
                margin=1.5 * Plots.mm,
            )
        else
            let error_plot=plot(
                    samples,
                    getindex.(summaries, :rmse);
                    label="RMSE",
                    marker=:circle,
                    ylabel="field error",
                    xlabel="samples",
                )
                plot!(
                    error_plot,
                    samples,
                    getindex.(summaries, :mae);
                    label="MAE",
                    marker=:circle,
                )
                plot(
                    error_plot,
                    uncertainty_plot,
                    information_plot,
                    model_plot;
                    layout=(2, 2),
                    size=(900, 680),
                    margin=1.5 * Plots.mm,
                )
            end
        end
    end
end

function plot_metric_history(
    input...;
    frame_index=nothing,
    parameters...,
)
    let history=visualization_history(input...; parameters...),
        index=isnothing(frame_index) ? lastindex(history) : frame_index
        plot_metric_history(history; frame_index=index)
    end
end

function comparison_frames(history)
    unique([1, cld(length(history), 2), length(history)])
end

function plot_predictions_against_ground_truth(
    history::SCRIBEVisualizationHistory;
    frame_indices=comparison_frames(history),
)
    let truth=history.ground_truth,
        plot_object=plot(
            truth,
            truth;
            color=:black,
            linestyle=:dash,
            label="identity",
            xlabel="ground truth",
            ylabel="posterior prediction",
            aspect_ratio=:equal,
        )
        foreach(frame_indices) do frame_index
            frame=history[frame_index]
            scatter!(
                plot_object,
                truth,
                frame[:summary][:prediction];
                markersize=2,
                markerstrokewidth=0,
                alpha=0.45,
                label="$(frame[:n_samples]) samples",
            )
        end
        plot!(
            plot_object;
            size=(520, 470),
            margin=1 * Plots.mm,
            left_margin=3 * Plots.mm,
            bottom_margin=3 * Plots.mm,
        )
    end
end

function plot_predictions_against_ground_truth(
    input...;
    frame_indices=nothing,
    parameters...,
)
    let history=visualization_history(input...; parameters...),
        indices=isnothing(frame_indices) ?
            comparison_frames(history) :
            frame_indices
        plot_predictions_against_ground_truth(
            history;
            frame_indices=indices,
        )
    end
end

function posterior_animation(history, plot_frame)
    let animation=Animation()
        foreach(eachindex(history.frames)) do frame_index
            Plots.frame(animation, plot_frame(frame_index))
        end
        animation
    end
end

function animate_posterior_mean_map(
    history::SCRIBEVisualizationHistory,
)
    let limits=visualization_limits(history)
        posterior_animation(
            history,
            frame_index -> plot_posterior_mean_map(
                history;
                frame_index,
                limits,
            ),
        )
    end
end

function animate_posterior_mean_map(input...; parameters...)
    animate_posterior_mean_map(
        visualization_history(input...; parameters...),
    )
end

function animate_posterior_uncertainty_map(
    history::SCRIBEVisualizationHistory,
)
    let limits=visualization_limits(history)
        posterior_animation(
            history,
            frame_index -> plot_posterior_uncertainty_map(
                history;
                frame_index,
                limits,
            ),
        )
    end
end

function animate_posterior_uncertainty_map(input...; parameters...)
    animate_posterior_uncertainty_map(
        visualization_history(input...; parameters...),
    )
end

function animate_absolute_error_map(
    history::SCRIBEVisualizationHistory,
)
    let limits=visualization_limits(history)
        posterior_animation(
            history,
            frame_index -> plot_absolute_error_map(
                history;
                frame_index,
                limits,
            ),
        )
    end
end

function animate_absolute_error_map(input...; parameters...)
    animate_absolute_error_map(
        visualization_history(input...; parameters...),
    )
end

function animate_posterior_against_ground_truth(
    history::SCRIBEVisualizationHistory,
)
    let limits=visualization_limits(history)
        posterior_animation(
            history,
            frame_index -> plot_posterior_against_ground_truth(
                history;
                frame_index,
                limits,
            ),
        )
    end
end

function animate_posterior_against_ground_truth(
    input...;
    parameters...,
)
    animate_posterior_against_ground_truth(
        visualization_history(input...; parameters...),
    )
end

function animate_posterior_summary(
    history::SCRIBEVisualizationHistory,
)
    let limits=visualization_limits(history)
        posterior_animation(
            history,
            frame_index -> plot_posterior_summary(
                history;
                frame_index,
                limits,
            ),
        )
    end
end

function animate_posterior_summary(input...; parameters...)
    animate_posterior_summary(
        visualization_history(input...; parameters...),
    )
end

function animate_metric_history(
    history::SCRIBEVisualizationHistory,
)
    posterior_animation(
        history,
        frame_index -> plot_metric_history(
            history;
            frame_index,
        ),
    )
end

function animate_metric_history(input...; parameters...)
    animate_metric_history(
        visualization_history(input...; parameters...),
    )
end

function animate_predictions_against_ground_truth(
    history::SCRIBEVisualizationHistory,
)
    posterior_animation(
        history,
        frame_index -> plot_predictions_against_ground_truth(
            history;
            frame_indices=[frame_index],
        ),
    )
end

function animate_predictions_against_ground_truth(
    input...;
    parameters...,
)
    animate_predictions_against_ground_truth(
        visualization_history(input...; parameters...),
    )
end

function static_visualization(history, metric)
    @match metric begin
        :posterior_mean => Dict(
            :plot => plot_posterior_mean_map(history),
            :filename => "posterior_mean.png",
        )
        :posterior_uncertainty => Dict(
            :plot => plot_posterior_uncertainty_map(history),
            :filename => "posterior_uncertainty.png",
        )
        :posterior_against_ground_truth => Dict(
            :plot => plot_posterior_against_ground_truth(history),
            :filename => "posterior_against_ground_truth.png",
        )
        :absolute_error => Dict(
            :plot => plot_absolute_error_map(history),
            :filename => "absolute_error.png",
        )
        :metric_history => Dict(
            :plot => plot_metric_history(history),
            :filename => "metric_history.png",
        )
        :predictions_against_ground_truth => Dict(
            :plot => plot_predictions_against_ground_truth(history),
            :filename => "predictions_against_ground_truth.png",
        )
        :posterior_summary => Dict(
            :plot => plot_posterior_summary(history),
            :filename => "posterior_summary.png",
        )
    end
end

function animated_visualization(history, metric)
    @match metric begin
        :posterior_mean => Dict(
            :animation => animate_posterior_mean_map(history),
            :filename => "posterior_mean.gif",
        )
        :posterior_uncertainty => Dict(
            :animation => animate_posterior_uncertainty_map(history),
            :filename => "posterior_uncertainty.gif",
        )
        :posterior_against_ground_truth => Dict(
            :animation => animate_posterior_against_ground_truth(history),
            :filename => "posterior_against_ground_truth.gif",
        )
        :absolute_error => Dict(
            :animation => animate_absolute_error_map(history),
            :filename => "absolute_error.gif",
        )
        :metric_history => Dict(
            :animation => animate_metric_history(history),
            :filename => "metric_history.gif",
        )
        :predictions_against_ground_truth => Dict(
            :animation => animate_predictions_against_ground_truth(history),
            :filename => "predictions_against_ground_truth.gif",
        )
        :posterior_summary => Dict(
            :animation => animate_posterior_summary(history),
            :filename => "posterior_summary.gif",
        )
    end
end

const DEFAULT_STATIC_VISUALIZATIONS = (
    :posterior_mean,
    :posterior_uncertainty,
    :metric_history,
    :posterior_summary,
)

const DEFAULT_ANIMATED_VISUALIZATIONS = (
    :posterior_mean,
    :posterior_uncertainty,
    :posterior_summary,
)

function save_static_visualizations(
    history::SCRIBEVisualizationHistory;
    output_dir,
    metrics=DEFAULT_STATIC_VISUALIZATIONS,
)
    mkpath(output_dir)
    map(metrics) do metric
        let visualization=static_visualization(history, metric),
            output_path=joinpath(
                output_dir,
                visualization[:filename],
            )
            savefig(visualization[:plot], output_path)
            output_path
        end
    end
end

function save_static_visualizations(
    input...;
    output_dir,
    metrics=DEFAULT_STATIC_VISUALIZATIONS,
    parameters...,
)
    save_static_visualizations(
        visualization_history(input...; parameters...);
        output_dir,
        metrics,
    )
end

function save_animated_visualizations(
    history::SCRIBEVisualizationHistory;
    output_dir,
    metrics=DEFAULT_ANIMATED_VISUALIZATIONS,
    fps=4,
)
    mkpath(output_dir)
    map(metrics) do metric
        let visualization=animated_visualization(history, metric),
            output_path=joinpath(
                output_dir,
                visualization[:filename],
            )
            gif(visualization[:animation], output_path; fps)
            output_path
        end
    end
end

function save_animated_visualizations(
    input...;
    output_dir,
    metrics=DEFAULT_ANIMATED_VISUALIZATIONS,
    fps=4,
    parameters...,
)
    save_animated_visualizations(
        visualization_history(input...; parameters...);
        output_dir,
        metrics,
        fps,
    )
end
