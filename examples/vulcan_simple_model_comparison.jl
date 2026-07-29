include("vulcan_model_comparison.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    comparison_main(
        SimpleComparison(),
        isempty(ARGS) ? :full : Symbol(first(ARGS)),
    )
end
