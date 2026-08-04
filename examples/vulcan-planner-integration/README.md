# SCRIBE–VulcanJ planner integration

`scribe_exploration.jl` demonstrates the smallest complete division of
responsibilities between the packages:

- the MDP defines physical motion, the mission horizon, and the analytic field
  used to simulate measurements;
- SCRIBE supplies the LGSF coefficient posterior, field predictions,
  observation conditioning, and integrated variance-reduction calculation;
- VulcanJ runs information-based MCTS using those model hooks.

The example uses a 6×6 dictionary of overlapping Gaussian scalar fields and a
spatially correlated coefficient prior. The truth contains domain-scale
variation and two off-grid local features, so it is not constructed from the
learner's basis dictionary.

The planner integration implements:

```julia
VulcanJ.initial_environment_model(...)
VulcanJ.expected_information_gain(...)
VulcanJ.conditional_observation_distribution(...)
VulcanJ.condition_environment_model(...)
```

The reward returned by this example is mean integrated field-variance
reduction over a fixed evaluation grid. It is not Shannon mutual information;
use SCRIBE's `mutual_information` function in the reward dispatch when that is
the desired acquisition objective.

From the SCRIBE repository root, run:

```sh
julia --project=examples \
    examples/vulcan-planner-integration/scribe_exploration.jl
```

For a bounded compilation and rendering check:

```sh
julia --project=examples \
    examples/vulcan-planner-integration/scribe_exploration.jl smoke
```

Results are stored in
`examples/res/vulcan-planner-integration/full/` or
`examples/res/vulcan-planner-integration/smoke/`. The example saves posterior
maps, uncertainty and error diagnostics, metric histories, and focused
posterior-learning animations through SCRIBE's visualization API.
