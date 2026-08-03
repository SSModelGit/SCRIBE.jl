using LinearAlgebra: Diagonal, Symmetric, inv, norm
using Plots
using Random
using SCRIBE
using Statistics: mean

include("_roms_data.jl")

function field_grid(values, roms)
    grid = fill(NaN, roms.grid_shape...)
    grid[roms.wet_mask] = values
    permutedims(grid)
end

function field_panel(values, roms, title; color, clims)
    heatmap(
        field_grid(values, roms);
        title,
        color,
        clims,
        colorbar=false,
        aspect_ratio=:equal,
        axis=false,
        titlefontsize=8,
    )
end

function paired_snapshot_grid(
    truths,
    results,
    snapshot_ids,
    roms;
    result_name,
    result_color,
    result_limits,
    figure_title,
)
    truth_limit = maximum(maximum(abs, field) for field in truths)
    panels = []
    for (truth, result, snapshot) in zip(
        truths,
        results,
        snapshot_ids,
    )
        push!(panels, field_panel(
            truth,
            roms,
            "Snapshot $snapshot: ground truth";
            color=:balance,
            clims=(-truth_limit, truth_limit),
        ))
        push!(panels, field_panel(
            result,
            roms,
            "Snapshot $snapshot: $result_name";
            color=result_color,
            clims=result_limits,
        ))
    end
    plot(
        panels...;
        layout=(4, 4),
        size=(1600, 1200),
        plot_title=figure_title,
    )
end

function posterior_field_variance(params; observation_variance=1e-4)
    decomposition = params.decomposition
    E = decomposition.modes
    residual = decomposition.residual_variance
    R⁻¹ = Diagonal(1.0 ./ (observation_variance .+ residual))
    P = inv(Symmetric(inv(Symmetric(params.P₀)) + E' * R⁻¹ * E))
    vec(sum((E * P) .* E; dims=2)) + residual
end

function save_training_plots(model, training_data, roms, component, output_dir)
    decomposition = model.params.decomposition
    reconstruction = decomposition.mean .+
        decomposition.modes * decomposition.coefficients
    snapshot_ids = round.(Int, range(
        1,
        size(training_data, 2);
        length=8,
    ))
    truths = [training_data[:, snapshot] for snapshot in snapshot_ids]
    reconstructions = [reconstruction[:, snapshot] for snapshot in snapshot_ids]
    posterior_variance = posterior_field_variance(model.params)
    variances = [copy(posterior_variance) for _ in snapshot_ids]
    relative_errors = [
        100 .* abs.(truth - estimate) ./ max.(abs.(truth), eps(Float64))
        for (truth, estimate) in zip(truths, reconstructions)
    ]
    prefix = joinpath(output_dir, "rams_head_$(component)")
    truth_limit = maximum(maximum(abs, x) for x in truths)
    variance_limit = maximum(posterior_variance)

    reconstruction_plot = paired_snapshot_grid(
        truths,
        reconstructions,
        snapshot_ids,
        roms;
        result_name="EOF reconstruction",
        result_color=:balance,
        result_limits=(-truth_limit, truth_limit),
        figure_title="Ground truth and best-fit EOF reconstruction " *
            "(shared scale ±$(round(truth_limit; digits=2)))",
    )
    savefig(reconstruction_plot, "$(prefix)_training_reconstruction.png")

    covariance_plot = paired_snapshot_grid(
        truths,
        variances,
        snapshot_ids,
        roms;
        result_name="posterior covariance diagonal",
        result_color=:magma,
        result_limits=(0, variance_limit),
        figure_title="Ground truth and EOF posterior field variance " *
            "(scale 0–$(round(variance_limit; sigdigits=3)))",
    )
    savefig(covariance_plot, "$(prefix)_posterior_covariance.png")

    error_plot = paired_snapshot_grid(
        truths,
        relative_errors,
        snapshot_ids,
        roms;
        result_name="percent relative error",
        result_color=:magma,
        result_limits=(0, 100),
        figure_title="Ground truth and percent relative error (>100% saturated)",
    )
    savefig(error_plot, "$(prefix)_relative_error.png")
end

"""Learn and save a fixed EOF model space from ROMS snapshots."""
function construct_roms_eof_model(;
    component=:u,
    temporal_stride=3,
    training_fraction=0.8,
    variance_fraction=0.995,
    max_rank=160,
    process_variance=1e-4,
    artifact=roms_eof_artifact(component),
)
    archive_data = read_roms_velocity(component)
    roms = prepare_roms_velocity(
        archive_data;
        temporal_stride,
    )
    training_snapshots = floor(Int, training_fraction * size(roms.data, 2))
    training_data = roms.data[:, 1:training_snapshots]
    model = initialize_eof_climate_model(
        training_data;
        locations=roms.locations,
        process_covariance=process_variance,
        variance_fraction,
        max_rank,
        algorithm=:randomized,
        oversample=16,
        power_iterations=2,
        rng=MersenneTwister(12),
        interpolation=:nearest,
        metadata=Dict(
            "source" => basename(RAMS_HEAD_ARCHIVE),
            "component" => String(component),
            "temporal_stride" => temporal_stride,
            "training_snapshots" => training_snapshots,
            "process_variance" => process_variance,
            "grid_shape" => collect(roms.grid_shape),
            "wet_mask" => Int8.(roms.wet_mask),
        ),
    )
    decomposition = model.params.decomposition
    reconstruction = decomposition.mean .+
        decomposition.modes * decomposition.coefficients
    training_rmse = sqrt(mean(abs2, reconstruction - training_data))
    relative_error = norm(reconstruction - training_data) /
        norm(training_data .- decomposition.mean)
    model.params.metadata["training_rmse"] = training_rmse
    model.params.metadata["relative_reconstruction_error"] = relative_error

    mkpath(dirname(artifact))
    save_eof_model(artifact, model)
    save_training_plots(
        model,
        training_data,
        roms,
        component,
        dirname(artifact),
    )
    println("Saved $(model.params.nᵩ) EOFs learned from ",
            "$training_snapshots ROMS snapshots to $artifact")
    println("Retained ", round(
        100 * model.params.decomposition.explained_variance;
        digits=2,
    ), "% of anomaly variance at a $temporal_stride-hour interval.")
    println("Coefficient process: identity random walk with Q=",
            process_variance, "I per update.")
    println("Training reconstruction: RMSE=",
            round(training_rmse; digits=5), ", relative anomaly error=",
            round(relative_error; digits=4))
    model
end

abspath(PROGRAM_FILE) == (@__FILE__) && construct_roms_eof_model()
