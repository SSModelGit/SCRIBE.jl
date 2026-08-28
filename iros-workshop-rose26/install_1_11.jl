#!/usr/bin/env julia

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = ("--precompile" in ARGS) ? "1" : "0"

using Pkg

Pkg.activate(@__DIR__)

# Julia 1.11 resolves the local paths declared in Project.toml's [sources]
# section.
Pkg.resolve()
Pkg.instantiate()
