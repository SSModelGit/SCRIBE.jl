using SCRIBE.ROMSTools

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

"""Serve one prepared snapshot as a location-indexed data sensor."""
function roms_snapshot_sensor(snapshot, locations)
    row_at = Dict(
        Tuple(locations[row, :]) => row
        for row in axes(locations, 1)
    )
    (_, X) -> snapshot[[row_at[Tuple(x)] for x in eachrow(X)]]
end

metadata_scalar(value) = value isa AbstractArray ? only(value) : value
