using SCRIBE.ROMSTools

const RAM_HEAD_ARCHIVE = normpath(joinpath(
    @__DIR__, "..", "bigdata", "rams_head_model_output",
    "stjohn_hourly_5m_velocity_ramhead_v2.mat",
))
using LinearAlgebra
using Random
using Serialization

const WORKSHOP_EOF_RANK = 10
const WORKSHOP_EOF_DIRECTORY = joinpath(@__DIR__, "res", "model")
const RAM_HEAD_EOF_MODEL = joinpath(WORKSHOP_EOF_DIRECTORY, "rams_head_u_eof.mat")

function workshop_roms()
    path = joinpath(WORKSHOP_EOF_DIRECTORY, "roms.jls")
    isfile(path) && return open(deserialize, path)
    roms = let archive=read_roms_velocity(RAM_HEAD_ARCHIVE, :u)
        prepare_roms_velocity(archive; temporal_stride=3)
    end
    GC.gc()
    mkpath(WORKSHOP_EOF_DIRECTORY)
    open(io -> serialize(io, roms), path, "w")
    roms
end

function workshop_eof_parameters()
    if !isfile(RAM_HEAD_EOF_MODEL)
        BLAS.set_num_threads(1)
        roms = workshop_roms()
        n_training = floor(Int, 0.8size(roms[:data], 2))
        model = initialize_eof_climate_model(
            roms[:data][:, 1:n_training];
            locations=roms[:locations], rank=WORKSHOP_EOF_RANK,
            algorithm=:randomized, oversample=16, power_iterations=2,
            rng=MersenneTwister(12), process_covariance=0.0,
            interpolation=:nearest,
            metadata=Dict("temporal_stride" => 3, "training_snapshots" => n_training,
                "grid_shape" => collect(roms[:grid_shape]),
                "wet_mask" => Int8.(roms[:wet_mask]), "workshop_revision" => 2),
        )
        save_eof_model(RAM_HEAD_EOF_MODEL, model)
    end
    params = load_eof_model_parameters(RAM_HEAD_EOF_MODEL)
    params
end

function ram_head_truth_snapshot()
    selection = joinpath(@__DIR__, "res", "eof_snapshot_comparison", "selected_snapshot.txt")
    isfile(selection) ? parse(Int, strip(read(selection, String))) : 5534
end

metadata_value(x) = x isa AbstractArray ? only(x) : x

function ram_head_sampling_rows(roms, n_samples)
    wet_rows = zeros(Int, prod(roms[:grid_shape]))
    wet_rows[roms[:wet_mask]] = axes(roms[:locations], 1)
    grid_rows = reshape(wet_rows, roms[:grid_shape]...)
    path = reduce(vcat, [
        filter(x -> !iszero(x), column % 2 == 1 ? grid_rows[:, column] :
            reverse(grid_rows[:, column]))
        for column in axes(grid_rows, 2)
    ])
    path[round.(Int, range(1, length(path); length=n_samples))]
end

function ram_head_sampling_path(roms, rows)
    wet_cells = findall(reshape(roms[:wet_mask], roms[:grid_shape]...))
    cells = wet_cells[rows]
    hcat(getindex.(cells, 1), getindex.(cells, 2))
end

"""Assimilate one held-out Ram Head current field in a retained EOF basis."""
function run_ram_head_eof_experiment(;
    prior_snapshot=3030,
    truth_snapshot=ram_head_truth_snapshot(),
    n_samples=300,
    sensor_variance=1e-4,
)
    params = workshop_eof_parameters()
    roms = workshop_roms()
    run_ram_head_eof_experiment(params, roms; prior_snapshot, truth_snapshot,
        n_samples, sensor_variance)
end

function run_ram_head_eof_experiment(params, roms;
    prior_snapshot=3030, truth_snapshot=ram_head_truth_snapshot(), n_samples=300,
    sensor_variance=1e-4,
)
    model = eof_model_at_coefficients(
        params,
        params.decomposition.coefficients[:, prior_snapshot],
    )
    truth = roms[:data][:, truth_snapshot]
    rows = ram_head_sampling_rows(roms, n_samples)
    information = SCRIBE.init_agent_info(model.params)

    foreach(rows) do row
        X = permutedims(roms[:locations][row, :])
        R = eof_effective_measurement_covariance(
            model,
            X,
            sensor_variance,
        )
        information = condition_on_measurement(
            model,
            information,
            X,
            [truth[row]],
            R,
        )
    end
    coefficients = posterior_coefficient_moments(information)[:μ]
    posterior = reconstruct_eof_field(model; coefficients)
    Dict(
        :truth_snapshot => truth_snapshot,
        :truth => truth,
        :posterior => posterior,
        :sampling_path => ram_head_sampling_path(roms, rows),
        :grid_shape => roms[:grid_shape],
        :wet_mask => roms[:wet_mask],
        :n_samples => n_samples,
        :rank => model.params.nᵩ,
        :rmse => sqrt(mean(abs2, posterior - truth)),
    )
end

function ram_head_sampling_path!(panel, path)
    plot!(
        panel,
        path[:, 1],
        path[:, 2];
        color=:dodgerblue,
        linewidth=0.8,
        linestyle=:solid,
        alpha=0.38,
        label=false,
    )
    scatter!(
        panel,
        [path[1, 1]],
        [path[1, 2]];
        color=:dodgerblue,
        marker=:circle,
        markersize=4,
        markerstrokecolor=:black,
        label=false,
    )
    scatter!(
        panel,
        [path[end, 1]],
        [path[end, 2]];
        color=:yellow,
        marker=:diamond,
        markersize=5,
        markerstrokecolor=:black,
        label=false,
    )
    panel
end

function ram_head_panel(
    values,
    result,
    title,
    color_limit;
    show_sampling_path=false,
)
    panel = plot_roms_field(
        values,
        Dict(
            :grid_shape => result[:grid_shape],
            :wet_mask => result[:wet_mask],
        );
        title,
        color=:balance,
        clims=(-color_limit, color_limit),
    )
    plot!(
        panel;
        colorbar=false,
        titlefontsize=12,
        margin=1 * Plots.mm,
    )
    show_sampling_path && ram_head_sampling_path!(
        panel,
        result[:sampling_path],
    )
    panel
end
