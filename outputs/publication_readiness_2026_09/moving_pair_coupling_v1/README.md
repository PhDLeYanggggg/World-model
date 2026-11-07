# Does Analytic Pair Coupling Improve Forecast Substitution?

I kept the forecaster and candidate trajectories fixed and tested whether
accounting for two substitutions together improves over the same learned
marginal costs. The squared relative-vector error has a mixed term that
depends only on the two forecast changes, not on an unseen future label.
This is different from discouraging predicted agents from approaching each
other, which did not improve labelled pair geometry in the earlier study.

**The registered component screen fails.** Coupling changes the chosen agents
in 423 of 1,613 queries, with the same count in every query and 3,225 total
substitutions per arm. It does not improve observed relative-position error.

| Readout | Additive | Coupled | Paired gain, exploratory source interval |
|---|---:|---:|---:|
| Normalized squared relative-vector error | 0.241868 | 0.242040 | -0.000172 [-0.001445, +0.000907] |
| Normalized relative-vector error | 0.268478 | 0.268577 | -0.000099 [-0.000642, +0.000393] |
| Known agent ADE gain difference | reference | comparison | +0.0184 pp [-0.1028, +0.1471] |
| Agent paired row-support completion lower | reference | comparison | -0.2392 pp [-0.5647, +0.0339] |

Positive differences favor coupling. Seven of fourteen sources improve on the
squared pair outcome and six on its unsquared counterpart. Neither interval
supports the prospectively specified positive result. The ADE completion
lower bounds wholly unknown row contributions, not full-grid error on partial
tracks. Agent and pair outcomes are not interchangeable.

## Why This Does Not Justify A More Complex Solver

A separate **post-hoc, label-aware diagnostic** uses the same candidates,
edge graph, requested queries and intervention counts. It asks what could
happen if the marginal pair costs were known for the observed common steps.
It does not produce a deployable action file.

| Same-count choice | Normalized squared relative-vector error |
|---|---:|
| Learned marginal, additive | 0.241868 |
| Learned marginal, analytic coupling | 0.242040 |
| Label-aware marginal, additive | 0.213141 |
| Label-aware marginal, analytic coupling | 0.212507 |

At the labelled optimum, the additional coupling gain is 0.000634, or
0.30% relative to the label-aware additive control. The learned-to-oracle
coupled gap is 0.029532, about 46.6 times that incremental gain. The gap
includes information unavailable at inference and is not a learnable-gain
estimate. The comparison identifies a better next priority than tuning the
pair solver: improve estimation of baseline-relative marginal costs. It does
not rule out pair benefits under other candidate banks or intervention counts.

## What Was Run

- Fresh_run: 14 deterministic source-excluded ridge fits, exact same-count
  allocation over all 125,540 rows, outcome readout and a separate complete
  replay. All coefficient and choice arrays and scalar results match exactly.
  The separate oracle diagnostic also reruns to the same aggregate receipt.
- Cached_verified: six original EqMotion seed-17 forecast banks and A donor
  arrays. The frozen actor and the fitted marginal head exclude the source
  being scored. These are previously exposed development data; source exclusion
  within the design is not new independent confirmation.
- Pair support: 3,074,068 observed of 3,787,200 requested pair steps; 1,612 of
  1,613 queries have observed pair support. The unsupported query remains in
  the roster, not a zero-error pair. Agent rows include 122,716 known, of
  which 31,751 are partial, and 2,824 wholly unknown. No imputed future labels.
- The main run (PID 6136) exits 0 in 5.864 s, peak 627,867,648 B; the separate
  replay (PID 6167) exits 0 in 5.954 s, peak 542,310,400 B. Native arm64,
  four configured BLAS threads, no loader workers. No CREATE job is needed.
  Fourteen focused tests pass, including held-source target perturbation,
  observed-mask algebra and exhaustive small-graph allocation checks.

No independent outcome readout, new actor training, new thresholds or changed
evaluation gate. This fixed-count component experiment does not enforce or
certify the original recording-level 2% contract. It neither upgrades a model
nor replaces the failed historical results. One forecaster seed and local image
coordinates, not verified seconds, meters, physical safety, true 3D, a foundation
model or submission readiness. Stage5C execution and SMC remain off.

## Reproduction

Use the existing local A donor and original forecast bank artifacts with native
arm64 `.venv-pytorch`. Completed design files are atomic; `run` reuses verified
completed designs. `replay` refits rather than trusting saved coefficients.
Only aggregate metrics and metadata are committed.

```sh
.venv-pytorch/bin/python scripts/run_m3w_pair_coupling.py run
.venv-pytorch/bin/python scripts/run_m3w_pair_coupling.py replay
.venv-pytorch/bin/python scripts/diagnose_m3w_pair_coupling.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_pair_coupling.py tests/test_m3w_native_bound_estimands.py
```

[Registered protocol](protocol.md) | [All source metrics](analysis.json) |
[Oracle diagnostic](oracle_diagnostic.json)
