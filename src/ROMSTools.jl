module ROMSTools

using LinearAlgebra
using MAT: matopen
using Plots
using Random
using Statistics

using ..SCRIBE: EOFClimateModelParameters, calibrate_eof_uncertainty
using ..SCRIBE: fit_eof_decomposition, initialize_SCRIBEModel_from_parameters

export read_roms_velocity, read_roms_flow_directions, prepare_roms_velocity
export prepare_roms_component, prepare_roms_curl, prepare_roms_curl_shape
export fit_roms_eof, field_grid, wet_grid_locations
export plot_roms_field, plot_roms_curl

function read_roms_velocity(path, component::Symbol=:u)
    matopen(path) do file
        Dict(
            :values => read(file, String(component)),
            :longitude => read(file, "lon"),
            :latitude => read(file, "lat"),
            :times => Float64.(vec(read(file, "ocean_time"))),
        )
    end
end

function prepare_roms_velocity(
    archive;
    temporal_stride=24,
    sample_offset=0,
    n_snapshots=nothing,
)
    snapshots = reshape(archive[:values], :, size(archive[:values], 3))
    wet = vec(all(isfinite, snapshots; dims=2))
    sampled = collect(1:temporal_stride:size(snapshots, 2))
    selected = isnothing(n_snapshots) ?
        sampled[sample_offset + 1:end] :
        sampled[sample_offset + 1:sample_offset + n_snapshots]
    Dict(
        :data => Matrix{Float64}(snapshots[wet, selected]),
        :locations => hcat(
            vec(archive[:longitude])[wet],
            vec(archive[:latitude])[wet],
        ),
        :times => archive[:times][selected],
        :grid_shape => size(archive[:values])[1:2],
        :wet_mask => wet,
    )
end

function planar_coordinates(longitude, latitude)
    longitude_origin = mean(filter(isfinite, vec(longitude)))
    latitude_origin = mean(filter(isfinite, vec(latitude)))
    radius = 6_371_000.0
    x = radius .* cosd(latitude_origin) .* deg2rad.(longitude .- longitude_origin)
    y = radius .* deg2rad.(latitude .- latitude_origin)
    x, y
end

function navigation_locations(longitude, latitude)
    x, y = planar_coordinates(longitude, latitude)
    finite = isfinite.(x) .& isfinite.(y)
    x_min, x_max = extrema(x[finite])
    y_min, y_max = extrema(y[finite])
    scale = max(x_max - x_min, y_max - y_min, eps(Float64))
    hcat((vec(x) .- x_min) ./ scale, (vec(y) .- y_min) ./ scale)
end

function decimate_roms(roms, stride)
    stride <= 1 && return roms
    wet_indices = findall(roms[:wet_mask])
    selected = wet_indices[1:stride:end]
    mask = falses(length(roms[:wet_mask]))
    mask[selected] .= true
    rows = 1:stride:size(roms[:data], 1)
    Dict(
        :data => Matrix{Float64}(roms[:data][rows, :]),
        :locations => Matrix{Float64}(roms[:locations][rows, :]),
        :times => roms[:times],
        :grid_shape => roms[:grid_shape],
        :wet_mask => mask,
    )
end


function prepare_roms_component(
    path,
    component;
    temporal_stride,
    spatial_stride=1,
)
    archive = read_roms_velocity(path, component)
    navigation = navigation_locations(archive[:longitude], archive[:latitude])
    roms = prepare_roms_velocity(archive; temporal_stride)
    positioned = merge(roms, Dict(
        :locations => Matrix{Float64}(navigation[roms[:wet_mask], :]),
    ))
    decimate_roms(positioned, spatial_stride)
end

"""
Compute vertical vorticity `∂v/∂x - ∂u/∂y` from collocated horizontal
velocity components on the supplied planar x y grid. `u` and `v` must
follow the eastward and northward grid directions, respectively.
"""
function planar_velocity_curl(u, v, x, y)
    nᵢ, nⱼ, nₜ = size(u)
    curl = fill(NaN, nᵢ, nⱼ, nₜ)
    for t in 1:nₜ, j in 1:nⱼ, i in 1:nᵢ
        i₀, i₁ = max(i - 1, 1), min(i + 1, nᵢ)
        j₀, j₁ = max(j - 1, 1), min(j + 1, nⱼ)
        stencil = (
            u[i₀, j, t], u[i₁, j, t], u[i, j₀, t], u[i, j₁, t],
            v[i₀, j, t], v[i₁, j, t], v[i, j₀, t], v[i, j₁, t],
            x[i₀, j], x[i₁, j], x[i, j₀], x[i, j₁],
            y[i₀, j], y[i₁, j], y[i, j₀], y[i, j₁],
        )
        if !all(isfinite, stencil); continue; end

        Δi = max(i₁ - i₀, 1)
        Δj = max(j₁ - j₀, 1)

        dx_di = (x[i₁, j] - x[i₀, j]) / Δi
        dy_di = (y[i₁, j] - y[i₀, j]) / Δi
        dx_dj = (x[i, j₁] - x[i, j₀]) / Δj
        dy_dj = (y[i, j₁] - y[i, j₀]) / Δj

        determinant = dx_di * dy_dj - dy_di * dx_dj
        if !(abs(determinant) > eps(Float64)); continue; end

        du_di = (u[i₁, j, t] - u[i₀, j, t]) / Δi
        du_dj = (u[i, j₁, t] - u[i, j₀, t]) / Δj
        dv_di = (v[i₁, j, t] - v[i₀, j, t]) / Δi
        dv_dj = (v[i, j₁, t] - v[i, j₀, t]) / Δj

        ∂v_∂x = (dv_di * dy_dj - dy_di * dv_dj) / determinant
        ∂u_∂y = (dx_di * du_dj - du_di * dx_dj) / determinant
        curl[i, j, t] = ∂v_∂x - ∂u_∂y
    end
    return curl
end

function velocity_curl(
    u::AbstractArray{<:Real, 3},
    v::AbstractArray{<:Real, 3},
    longitude::AbstractMatrix,
    latitude::AbstractMatrix
)
    x, y = planar_coordinates(longitude, latitude)
    planar_velocity_curl(u, v, x, y)
end

function velocity_curl(
    u::AbstractArray,
    v::AbstractArray,
    x::AbstractVector,
    y::AbstractVector
)
    x_grid = repeat(reshape(x, :, 1), 1, length(y))
    y_grid = repeat(reshape(y, 1, :), length(x), 1)

    curl = planar_velocity_curl(
        reshape(u, size(u)..., 1), reshape(v, size(v)..., 1), 
        x_grid, y_grid)

    return dropdims(curl, dims=3)
end

function prepare_roms_curl(
    path;
    temporal_stride,
    spatial_stride=1,
)
    u = read_roms_velocity(path, :u)
    v = read_roms_velocity(path, :v)
    sampled_times = collect(1:temporal_stride:size(u[:values], 3))
    curl = velocity_curl(
        view(u[:values], :, :, sampled_times),
        view(v[:values], :, :, sampled_times),
        u[:longitude],
        u[:latitude],
    )
    snapshots = reshape(curl, :, size(curl, 3))
    wet = vec(all(isfinite, snapshots; dims=2))
    wet_indices = findall(wet)[1:spatial_stride:end]
    selected = falses(length(wet))
    selected[wet_indices] .= true
    navigation = navigation_locations(u[:longitude], u[:latitude])
    Dict(
        :data => Matrix{Float64}(snapshots[selected, :]),
        :locations => Matrix{Float64}(navigation[selected, :]),
        :times => Float64.(u[:times][sampled_times]),
        :source_indices => sampled_times,
        :grid_shape => size(curl)[1:2],
        :wet_mask => selected,
    )
end

function prepare_roms_curl_shape(
    path;
    temporal_stride,
    spatial_stride=1,
)
    roms = prepare_roms_curl(
        path;
        temporal_stride,
        spatial_stride,
    )
    magnitude = abs.(roms[:data])
    scale = vec(mean(magnitude; dims=1))
    merge(roms, Dict(
        :data => magnitude ./ reshape(scale, 1, :),
        :curl_scale => scale,
    ))
end

function read_roms_flow_directions(path, roms, snapshots)
    source_indices = roms[:source_indices][snapshots]
    wet = roms[:wet_mask]
    u = matopen(path) do file
        values = read(file, "u")[:, :, source_indices]
        Matrix{Float32}(reshape(values, :, length(source_indices))[wet, :])
    end
    v = matopen(path) do file
        values = read(file, "v")[:, :, source_indices]
        Matrix{Float32}(reshape(values, :, length(source_indices))[wet, :])
    end
    speed = hypot.(u, v)
    moving = speed .> eps(Float32)
    u[moving] ./= speed[moving]
    v[moving] ./= speed[moving]
    u[.!moving] .= 0
    v[.!moving] .= 0
    Dict(
        snapshot => hcat(view(u, :, column), view(v, :, column))
        for (column, snapshot) in enumerate(snapshots)
    )
end

function fit_roms_eof(
    roms;
    training_fraction,
    rank,
    oversample,
    power_iterations,
    rng=MersenneTwister(12),
)
    n_training = floor(Int, training_fraction * size(roms[:data], 2))
    training = @view roms[:data][:, 1:n_training]
    decomposition_rank = min(rank, size(training)...)
    decomposition = fit_eof_decomposition(
        training;
        rank=decomposition_rank,
        algorithm=:randomized,
        oversample=min(oversample, decomposition_rank),
        power_iterations,
        rng,
    )
    held_out_count = size(roms[:data], 2) - n_training
    calibration_count = floor(Int, held_out_count / 2)
    calibration_end = n_training + calibration_count
    held_out = @view roms[:data][:, n_training + 1:calibration_end]
    held_out_coefficients = decomposition.modes' * (
        decomposition.weights .* (held_out .- decomposition.mean)
    )
    uncertainty = calibrate_eof_uncertainty(
        decomposition;
        histories=[held_out_coefficients],
    )
    params = EOFClimateModelParameters(
        decomposition;
        locations=roms[:locations],
        process_covariance=uncertainty.Q,
        prior_covariance=uncertainty.P₀,
        interpolation=:nearest,
        metadata=Dict(
            "training_snapshots" => n_training,
            "coordinate_system" => "isotropic local metric / domain scale",
            "grid_shape" => collect(roms[:grid_shape]),
            "wet_mask" => Int8.(roms[:wet_mask]),
            "uncertainty_calibration" => "held-out EOF coefficient history",
            "uncertainty_calibration_snapshots" => size(held_out, 2),
        ),
    )
    Dict(
        :model => initialize_SCRIBEModel_from_parameters(params),
        :n_training => n_training,
        :calibration_end => calibration_end,
        :validation_start => calibration_end + 1,
        :field_scale => max(std(training), sqrt(eps(Float64))),
    )
end

function field_grid(values, roms)
    grid = fill(NaN, roms[:grid_shape]...)
    grid[roms[:wet_mask]] = values
    permutedims(grid)
end

function field_limit(values)
    finite = filter(isfinite, vec(values))
    isempty(finite) ? 1.0 : max(maximum(abs, finite), eps(Float64))
end

function plot_roms_field(
    values,
    roms;
    title="EOF posterior mean",
    color=:balance,
    clims=(-field_limit(values), field_limit(values)),
)
    grid = field_grid(values, roms)
    heatmap(
        grid;
        color,
        clims,
        aspect_ratio=:equal,
        axis=false,
        xlims=(0.5, size(grid, 2) + 0.5),
        ylims=(0.5, size(grid, 1) + 0.5),
        title,
    )
end

function plot_roms_curl(
    values,
    flow_directions,
    roms;
    arrow_stride=7,
    title="curl",
    limit=field_limit(values),
    colorbar=true,
    magnitude=false,
    display_scale=1e3,
    colorbar_title=magnitude ? "|curl| (10⁻³ s⁻¹)" : "curl (10⁻³ s⁻¹)",
)
    vorticity = display_scale .* field_grid(
        magnitude ? abs.(values) : values,
        roms,
    )
    flow_u = field_grid(view(flow_directions, :, 1), roms)
    flow_v = field_grid(view(flow_directions, :, 2), roms)
    panel = heatmap(
        vorticity;
        color=magnitude ? :thermal : :balance,
        clims=magnitude ?
            (0.0, display_scale * limit) :
            (-display_scale * limit, display_scale * limit),
        colorbar,
        colorbar_title,
        background_color_inside=:gray25,
        aspect_ratio=:equal,
        axis=false,
        xlims=(0.5, size(vorticity, 2) + 0.5),
        ylims=(0.5, size(vorticity, 1) + 0.5),
        title,
    )
    rows = 2:arrow_stride:size(vorticity, 1)-1
    columns = 2:arrow_stride:size(vorticity, 2)-1
    x = Float64[]
    y = Float64[]
    for row in rows, column in columns
        direction_u = flow_u[row, column]
        direction_v = flow_v[row, column]
        all(isfinite, (direction_u, direction_v)) || continue
        direction_norm = hypot(direction_u, direction_v)
        direction_norm > eps(Float64) || continue
        direction_x = direction_u / direction_norm
        direction_y = direction_v / direction_norm
        normal_x = -direction_y
        normal_y = direction_x
        arrow_length = 3.0
        head_length = 0.8
        head_width = 0.45
        start_x = column - arrow_length * direction_x / 2
        start_y = row - arrow_length * direction_y / 2
        tip_x = column + arrow_length * direction_x / 2
        tip_y = row + arrow_length * direction_y / 2
        left_x = tip_x - head_length * direction_x + head_width * normal_x
        left_y = tip_y - head_length * direction_y + head_width * normal_y
        right_x = tip_x - head_length * direction_x - head_width * normal_x
        right_y = tip_y - head_length * direction_y - head_width * normal_y
        append!(x, (start_x, tip_x, NaN, left_x, tip_x, right_x, NaN))
        append!(y, (start_y, tip_y, NaN, left_y, tip_y, right_y, NaN))
    end
    plot!(
        panel,
        x,
        y;
        color=:white,
        linewidth=2.8,
        alpha=0.9,
        label=false,
    )
    plot!(
        panel,
        x,
        y;
        color=:gray10,
        linewidth=1.1,
        label=false,
    )
    panel
end

function wet_grid_locations(roms, rows)
    wet_cells = findall(reshape(roms[:wet_mask], roms[:grid_shape]...))
    hcat(getindex.(wet_cells[rows], 1), getindex.(wet_cells[rows], 2))
end

end
