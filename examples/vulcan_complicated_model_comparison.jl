include("vulcan_model_comparison.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    comparison_main(
        ComplicatedComparison(),
        isempty(ARGS) ? :full : Symbol(first(ARGS)),
    )
end
