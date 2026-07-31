using MAT
using Plots
using Random
using SCRIBE

"""
Configuration and cache for the RAMS Head ROMS velocity archive shipped in
`bigdata/`. This is an example of the user-owned loading policy expected by
`eof_model_data_loader`; it is intentionally not part of SCRIBE's core API.
"""
mutable struct RamsHeadVelocitySource
    path::String
    component::Symbol
    temporal_stride::Int
    maximum_snapshots::Union{Nothing, Int}
    cache::Any
end

function RamsHeadVelocitySource(
    path;
    component=:u,
    temporal_stride=24,
    maximum_snapshots=nothing,
)
    component in (:u, :v) ||
        throw(ArgumentError("component must be :u or :v."))
    temporal_stride >= 1 ||
        throw(ArgumentError("temporal_stride must be positive."))
    RamsHeadVelocitySource(
        String(path),
        component,
        Int(temporal_stride),
        isnothing(maximum_snapshots) ?
            nothing :
            Int(maximum_snapshots),
        nothing,
    )
end

function prepare_rams_head_source!(source::RamsHeadVelocitySource)
    !isnothing(source.cache) && return source.cache

    source.cache = matopen(source.path) do file
        raw = read(file, String(source.component))
        latitude = vec(read(file, "lat"))
        longitude = vec(read(file, "lon"))
        ocean_time = vec(read(file, "ocean_time"))
        snapshots = reshape(raw, :, size(raw, 3))

        # The archive contains 349 land rows that are NaN at every time.
        wet = vec(all(isfinite, snapshots; dims=2))
        selected = collect(1:source.temporal_stride:size(snapshots, 2))
        if !isnothing(source.maximum_snapshots)
            resize!(selected, min(length(selected), source.maximum_snapshots))
        end

        (
            data=Matrix{Float64}(snapshots[wet, selected]),
            locations=hcat(longitude[wet], latitude[wet]),
            times=Float64.(ocean_time[selected]),
            grid_shape=size(raw)[1:2],
            wet=wet,
        )
    end
end

import SCRIBE: eof_model_data_loader

function eof_model_data_loader(source::RamsHeadVelocitySource; kwargs...)
    prepare_rams_head_source!(source).data
end

rams_head_locations(source::RamsHeadVelocitySource) =
    prepare_rams_head_source!(source).locations

rams_head_times(source::RamsHeadVelocitySource) =
    prepare_rams_head_source!(source).times

function build_rams_head_eof_model(;
    component=:u,
    temporal_stride=24,
    maximum_snapshots=720,
    rank=12,
)
    source = RamsHeadVelocitySource(
        joinpath(
            @__DIR__,
            "..",
            "bigdata",
            "rams_head_model_output",
            "stjohn_hourly_5m_velocity_ramhead_v2.mat",
        );
        component,
        temporal_stride,
        maximum_snapshots,
    )
    prepared = prepare_rams_head_source!(source)
    model = initialize_eof_climate_model(
        source;
        locations=prepared.locations,
        rank,
        algorithm=:randomized,
        oversample=8,
        power_iterations=1,
        rng=MersenneTwister(12),
        interpolation=:inverse_distance,
        interpolation_neighbors=4,
        dynamics_ridge=1e-8,
        max_spectral_radius=0.999,
        metadata=Dict(
            "source" => basename(source.path),
            "component" => String(component),
            "temporal_stride_hours" => temporal_stride,
            "first_matlab_datenum" => first(prepared.times),
            "last_matlab_datenum" => last(prepared.times),
            "grid_shape" => collect(prepared.grid_shape),
            "wet_mask" => Int8.(prepared.wet),
        ),
    )
    source, model
end

function main()
    source, model = build_rams_head_eof_model()
    output_directory = joinpath(@__DIR__, "res", "eof_roms")
    mkpath(output_directory)

    artifact = joinpath(output_directory, "rams_head_u_eof.mat")
    save_eof_model(artifact, model)
    savefig(
        plot_eof_spectrum(model),
        joinpath(output_directory, "eof_spectrum.png"),
    )
    savefig(
        plot_eof_mode(model, 1),
        joinpath(output_directory, "eof_1.png"),
    )
    savefig(
        plot_eof_coefficients(model; modes=1:min(4, model.params.nᵩ)),
        joinpath(output_directory, "eof_coefficients.png"),
    )

    println("Learned $(model.params.nᵩ) EOFs from ",
            model.params.decomposition.n_samples, " snapshots.")
    println("Retained weighted variance: ",
            round(100 * model.params.decomposition.explained_variance;
                  digits=2), "%")
    println("Wet ROMS locations: ", size(rams_head_locations(source), 1))
    println("Saved learned artifact to: ", artifact)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
