# SCRIBE examples

## Information-based exploration with VulcanJ

`vulcan_scribe_exploration.jl` uses a SCRIBE Gaussian scalar field as the
environment model supplied to VulcanJ's information-based MCTS planner. The
integration consists of four dispatched methods:

- initialize the planning model;
- evaluate the mutual information of a candidate sample;
- return its posterior predictive measurement distribution;
- condition the model on a simulated or realized measurement.

VulcanJ remains responsible for planning. SCRIBE remains responsible for all
model prediction, conditioning, and information calculations. The script adds
the local VulcanJ and SCRIBE package environments to Julia's load path, with
VulcanJ first so its resolved planning dependencies remain intact. Set
`VULCANJ_ROOT` when VulcanJ is stored somewhere other than
`/home/shashank/cbase/secondary/jbase/VulcanJ`.

The complete profile uses an eight-connected 33×33 navigation grid and a
48-sample mission. Its 36 overlapping scalar-field bases have a spatially
correlated coefficient prior. The synthetic truth is deliberately outside
this native parameterization: it uses a rotated and translated 5×5 dictionary
of narrower GSFs, with asymmetric local and domain-scale structure. Its wells
therefore do not coincide with the learner's regular 6×6 basis centers.
During MCTS, predictive measurements retain their sampled values for model
conditioning but compare by sampling-state history in tree keys. This is exact
for SCRIBE's linear-Gaussian covariance planning and prevents continuous
measurements from fragmenting the deeper search tree. The planning reward is
mean integrated field-variance reduction over a fixed evaluation grid;
expected mutual information and realized KL are retained as separate
posterior diagnostics.

Run the complete experiment with:

```sh
julia examples/vulcan_scribe_exploration.jl
```

A short compilation and plotting check is also available:

```sh
julia examples/vulcan_scribe_exploration.jl smoke
```

Both profiles write a compact four-panel diagnostic animation, a focused
ground-truth-versus-posterior animation, standalone posterior-mean and
posterior-uncertainty animations, a final surface comparison, information and
error histories, and predicted-versus-ground-truth plots. Results are separated under
`examples/res/vulcan_scribe/<profile>/`, so a smoke render cannot overwrite
the full experiment.

## SCRIBE and Gaussian-process comparisons

The comparison facilities run VulcanJ using either SCRIBE or a standard
squared-exponential Gaussian process. The motion model, measurements, fixed
rollout budget, and integrated variance-reduction reward are held fixed so that
the posterior backend is the experimental variable. The two prior
field-uncertainty amplitudes are matched; neither model is fitted to the
ground-truth function. The GP branch uses GaussianProcesses.jl's standard
`GPE`, SE kernel, observation-noise model, posterior prediction, and
conditioning machinery.

The introductory comparison uses a smooth two-lobe field and a single-scale
SCRIBE basis:

```sh
julia examples/vulcan_simple_model_comparison.jl
```

The difficult comparison combines domain-wide variation, local anomalies, and
a curved narrow feature. Its SCRIBE model uses local and broad GSFs together
with a scale-aware prior:

```sh
julia examples/vulcan_complicated_model_comparison.jl
```

Its SCRIBE backend uses two overlapping GSF scales. A dense 9×9 dictionary of
narrow fields carries local variation, while a sparse 4×4 dictionary of broad
fields carries the domain-scale structure. This allocates spatial coverage
according to scale instead of repeating one center grid for every width.
Coefficients are spatially correlated within each scale but independent
between scales, avoiding an artificial assumption that broad and local
features are interchangeable. The resulting prior is normalized at the field
level so its mean pointwise variance matches the GP's prior variance. The
complicated field uses a unit prior field deviation for both backends; the smoother
introductory field retains the broader prior used by the original example.

The full comparisons use 12 fixed MCTS rollouts per sample with a depth-four
lookahead. This keeps the backend comparison repeatable while bounding the
matrix-factorization work performed by the multiscale SCRIBE model.

Append `smoke` to either command for its bounded profile. Closed-loop results
are separated by scenario and profile under
`examples/res/vulcan_model_comparison/planner/`, so a smoke run cannot
overwrite a complete experiment.

## Same-data model comparison

`vulcan_model_only_comparison.jl` removes the planning trajectory as a
confounding variable. It constructs one deterministic farthest-point sampling
sequence and one seeded noisy measurement vector, then conditions SCRIBE and
the GP on precisely the same `(location, measurement)` pairs.

```sh
julia examples/vulcan_model_only_comparison.jl
julia examples/vulcan_model_only_comparison.jl smoke
```

The runner evaluates both the simple and complicated truths. It saves the
shared data in `shared_samples.csv` alongside the comparison figures under
`examples/res/vulcan_model_comparison/model_only/<scenario>/<profile>/`.

## Multirun MCTS ablation

`vulcan_mcts_ablation.jl` evaluates three backends on the complicated field:

- multiscale SCRIBE with dense local and sparse broad GSF dictionaries;
- local-only SCRIBE, which removes the broad GSF dictionary;
- the single-scale squared-exponential GP.

The complete profile aggregates five matched MCTS seeds. Runs execute
sequentially and retain only metric histories, rather than posterior
animations or search trees. The smoke profile uses two seeds.

```sh
julia examples/vulcan_mcts_ablation.jl
julia examples/vulcan_mcts_ablation.jl smoke
```

The aggregate figure reports the mean trajectory with a one-standard-deviation
ribbon. Per-seed final values are saved to `final_metrics.csv` under
`examples/res/vulcan_model_comparison/ablation/complicated/<profile>/`.

All comparison experiments report:

- RMSE and RMSE normalized by the ground-truth field deviation;
- mean predictive log likelihood using latent posterior variance plus the
  measurement-noise variance;
- empirical coverage of the latent truth by the posterior 95% interval;
- integrated posterior uncertainty, evaluated as mean field variance;
- MAE for model-only runs or planning reward for closed-loop runs.
