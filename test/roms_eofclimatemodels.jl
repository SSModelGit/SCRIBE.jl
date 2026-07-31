include(joinpath(@__DIR__, "..", "examples", "roms_eof_climate_model.jl"))

@testset "RAMS Head ROMS EOF integration" begin
    source, model = build_rams_head_eof_model(
        maximum_snapshots=96,
        rank=8,
    )
    @test size(model.params.locations, 1) == 3787
    @test model.params.decomposition.n_samples == 96
    @test model.params.nᵩ == 8
    @test model.params.decomposition.explained_variance > 0.9
    @test model.params.metadata["component"] == "u"
    @test length(model.params.metadata["wet_mask"]) == 94 * 44

    mktempdir() do directory
        path = joinpath(directory, "rams_head_eof.mat")
        save_eof_model(path, model)
        loaded = load_eof_climate_model(path)
        @test loaded.params.decomposition.modes ≈
              model.params.decomposition.modes
        @test all(isfinite, predict_SCRIBEModel(
            loaded,
            rams_head_locations(source)[1:4, :],
        ))
    end
end
