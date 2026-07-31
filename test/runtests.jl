using Test

using SCRIBE

include("eofclimatemodels.jl")

if lowercase(get(ENV, "SCRIBE_RUN_BIGDATA_TESTS", "false")) == "true"
    include("roms_eofclimatemodels.jl")
end
