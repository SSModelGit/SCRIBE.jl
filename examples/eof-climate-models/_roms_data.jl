using MAT: matopen

const RAMS_HEAD_ARCHIVE = normpath(joinpath(
    @__DIR__,
    "..",
    "..",
    "bigdata",
    "rams_head_model_output",
    "stjohn_hourly_5m_velocity_ramhead_v2.mat",
))

eof_result_dir(run) = normpath(joinpath(
    @__DIR__,
    "..",
    "res",
    "eof-climate-models",
    String(run),
))

roms_eof_artifact(component=:u) = joinpath(
    eof_result_dir(:offline),
    "rams_head_$(component)_eof.mat",
)

"""Read the ROMS arrays needed by the example from the MATLAB archive."""
function read_roms_velocity(
    component=:u;
    path=RAMS_HEAD_ARCHIVE,
)
    matopen(path) do file
        (
            values=read(file, String(component)),
            longitude=read(file, "lon"),
            latitude=read(file, "lat"),
            times=Float64.(vec(read(file, "ocean_time"))),
        )
    end
end

"""Turn raw ROMS arrays into chronologically sampled wet-cell snapshots."""
function prepare_roms_velocity(
    archive;
    temporal_stride=24,
    sample_offset=0,
    n_snapshots=nothing,
)
    snapshots = reshape(archive.values, :, size(archive.values, 3))
    wet = vec(all(isfinite, snapshots; dims=2))
    sampled = collect(1:temporal_stride:size(snapshots, 2))
    selected = isnothing(n_snapshots) ?
        sampled[(sample_offset + 1):end] :
        sampled[(sample_offset + 1):(sample_offset + n_snapshots)]

    (
        data=Matrix{Float64}(snapshots[wet, selected]),
        locations=hcat(
            vec(archive.longitude)[wet],
            vec(archive.latitude)[wet],
        ),
        times=archive.times[selected],
        grid_shape=size(archive.values)[1:2],
        wet_mask=wet,
    )
end

"""Serve one prepared snapshot as a location-indexed data sensor."""
function roms_snapshot_sensor(snapshot, locations)
    row_at = Dict(
        Tuple(locations[row, :]) => row
        for row in axes(locations, 1)
    )
    (_, X) -> snapshot[[row_at[Tuple(x)] for x in eachrow(X)]]
end

metadata_scalar(value) = value isa AbstractArray ? only(value) : value
