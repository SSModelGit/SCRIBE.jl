# SCRIBE

SCRIBE (Signal-aware Collaboration for Reliable Information-based Estimation)
is a distributed environment-modeling framework for robotic exploration under
limited and intermittent communication. It combines fixed-dimensional spatial
models with information filtering, Covariance Intersection, and asynchronous
consensus so that robots can learn locally and safely fuse their models when
communication becomes available.

## Workshop paper

The method, its distributed update, and the current experimental results are
presented in the [SCRIBE workshop paper](iros-workshop-rose26/scribe_workshop_paper.pdf).
The [LaTeX source](iros-workshop-rose26/scribe_workshop_paper.tex) and experiment
scripts are available in the same directory.

## Examples

Runnable demonstrations are collected in the [examples](examples/) directory.
They include Gaussian scalar-field and EOF climate models, model prediction and
visualization, and information-based exploration with VulcanJ.

## Installation

The example and workshop environments expect `VulcanJ` to be checked out next
to this repository. From the SCRIBE root, install the example environment with
the script matching your Julia version:

- Julia 1.10: [`examples/install_1_10.jl`](examples/install_1_10.jl)

  ```console
  julia examples/install_1_10.jl
  ```

- Julia 1.11 or later: [`examples/install_1_11.jl`](examples/install_1_11.jl)

  ```console
  julia examples/install_1_11.jl
  ```

Append `--precompile` to either command to precompile the environment during
installation. To reproduce the workshop experiments, invoke the corresponding
workshop script instead:

```console
# Julia 1.10
julia iros-workshop-rose26/install_1_10.jl

# Julia 1.11 or later
julia iros-workshop-rose26/install_1_11.jl
```

The workshop installers are available here for
[Julia 1.10](iros-workshop-rose26/install_1_10.jl) and
[Julia 1.11 or later](iros-workshop-rose26/install_1_11.jl).
