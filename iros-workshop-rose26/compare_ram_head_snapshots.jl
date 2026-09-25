using SCRIBE, Plots, Statistics

include("ram_head_eof_experiment.jl")

"""Compare held-out maps with identical priors, sampling paths, and EOFs."""
function compare_ram_head_snapshots()
    params = load_eof_model_parameters(RAM_HEAD_EOF_MODEL)
    stride = Int(metadata_value(params.metadata["temporal_stride"]))
    roms = let archive=read_roms_velocity(RAM_HEAD_ARCHIVE, :u)
        prepare_roms_velocity(archive; temporal_stride=stride)
    end
    GC.gc()
    first_validation = params.decomposition.n_samples + 1
    candidates = unique(vcat(4923, round.(Int, range(
        first_validation, size(roms[:data], 2); length=32))))
    output = joinpath(@__DIR__, "res", "eof_snapshot_comparison")
    mkpath(output)
    results = map(candidates) do truth_snapshot
        result = run_ram_head_eof_experiment(params, roms; truth_snapshot)
        truth, posterior = result[:truth], result[:posterior]
        limit = maximum(abs, truth)
        result[:balance] = min(mean(truth .> 0.2limit), mean(truth .< -0.2limit))
        result[:correlation] = cor(truth, posterior)
        result[:relative_rmse] = result[:rmse] / std(truth)
        println("Snapshot $truth_snapshot: RMSE=$(result[:rmse]), " *
            "correlation=$(result[:correlation]), balance=$(result[:balance])")
        flush(stdout)
        result
    end
    open(joinpath(output, "metrics.csv"), "w") do io
        println(io, "snapshot,rmse,relative_rmse,correlation,bipolar_fraction")
        foreach(results) do r
            println(io, join((r[:truth_snapshot], r[:rmse], r[:relative_rmse],
                r[:correlation], r[:balance]), ','))
        end
    end
    ordered = sort(results; by=r -> r[:relative_rmse] + 2max(0, 0.15-r[:balance]))
    for page in 1:cld(length(ordered), 8)
        subset = ordered[(8page-7):min(8page, length(ordered))]
        panels = reduce(vcat, map(subset) do r
            limit = maximum(abs, r[:truth])
            [ram_head_panel(r[:truth], r, "$(r[:truth_snapshot]): truth", limit),
             ram_head_panel(r[:posterior], r,
                "RMSE=$(round(r[:rmse]; digits=3)), r=$(round(r[:correlation]; digits=2))", limit)]
        end)
        savefig(plot(panels...; layout=(length(subset), 2),
            size=(1100, 220length(subset))), joinpath(output, "candidates_$page.png"))
    end
    foreach(ordered[1:8]) do r
        limit = maximum(abs, r[:truth])
        panels = (ram_head_panel(r[:truth], r, "Ram Head ROMS Ground Truth", limit),
            ram_head_panel(r[:posterior], r, "EOF Posterior after 300 Samples", limit))
        savefig(plot(panels...; layout=(1, 2), size=(1080, 300), margin=0Plots.mm),
            joinpath(output, "snapshot_$(r[:truth_snapshot]).png"))
    end
    println("Comparison saved to $output")
end

abspath(PROGRAM_FILE) == (@__FILE__) && compare_ram_head_snapshots()
