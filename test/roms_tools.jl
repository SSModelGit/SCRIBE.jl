using Plots
using SCRIBE.ROMSTools

@testset "ROMS curl and field rendering" begin
    longitude = [
        -64.7 + 0.0005(i - 1)
        for i in 1:7, j in 1:5
    ]
    latitude = [
        18.3 + 0.0005(j - 1)
        for i in 1:7, j in 1:5
    ]
    x, y = SCRIBE.ROMSTools.planar_coordinates(longitude, latitude)
    Ω = 0.001
    rotation = SCRIBE.ROMSTools.velocity_curl(
        reshape(-Ω .* y, 7, 5, 1),
        reshape(Ω .* x, 7, 5, 1),
        longitude,
        latitude,
    )
    irrotational = SCRIBE.ROMSTools.velocity_curl(
        reshape(x, 7, 5, 1),
        reshape(y, 7, 5, 1),
        longitude,
        latitude,
    )
    @test all(isapprox.(rotation, 2Ω; atol=1e-12))
    @test all(isapprox.(irrotational, 0.0; atol=1e-12))

    shape = (94, 44)
    roms = Dict(
        :grid_shape => shape,
        :wet_mask => trues(prod(shape)),
    )
    curl = vec([
        sin(i / 2) * cos(j / 2)
        for i in 1:shape[1], j in 1:shape[2]
    ])
    directions = repeat([1.0 0.0], prod(shape), 1)
    panel = plot_roms_curl(curl, directions, roms; arrow_stride=7)
    arrow_x = panel.series_list[2][:x]
    arrow_y = panel.series_list[2][:y]
    lengths = [
        hypot(arrow_x[index + 1] - arrow_x[index],
              arrow_y[index + 1] - arrow_y[index])
        for index in 1:7:length(arrow_x)
    ]

    @test xlims(panel) == (0.5, 94.5)
    @test ylims(panel) == (0.5, 44.5)
    @test all(isapprox.(lengths, 3.0; atol=1e-12))
end
