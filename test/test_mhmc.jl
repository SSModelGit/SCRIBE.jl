# Compatibility entry point for the SCRIBE MHMC experiment driver.
#
# The implementation and its plotting environment live under `test/experiments`
# so this file remains parallel to `test_scriptor.jl`.
include(joinpath(@__DIR__, "experiments", "test_mhmc.jl"))
