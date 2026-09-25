#!/usr/bin/env julia

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = ("--precompile" in ARGS) ? "1" : "0"

using Pkg

Pkg.activate(@__DIR__)

local_packages = Pkg.PackageSpec[
    Pkg.PackageSpec(path=joinpath(@__DIR__, "..")),
    Pkg.PackageSpec(path=joinpath(@__DIR__, "..", "..", "VulcanJ")),
]

Pkg.develop(local_packages)
Pkg.resolve()
Pkg.instantiate()
