# Compatibility entry point for the SCRIBE experiment driver.
#
# The implementation and its plotting environment live under `test/experiments`
# so this file is not part of routine package tests.
include(joinpath(@__DIR__, "experiments", "test_scriptor.jl"))
