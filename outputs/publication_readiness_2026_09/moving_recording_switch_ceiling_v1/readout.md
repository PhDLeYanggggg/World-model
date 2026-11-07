# Why Fixed-Count Selection Fails Under The Original Risk Contract

I optimized an offline, future-label oracle on the same CV/EqMotion paths,
34 exposed development recordings, 14 sources and three actual forecast seeds.
This does not train or change a model. It isolates the imposed switch count
from feasibility of the two trajectories already available for each row.

| Binary candidate bank | Feasible / 102 repeated views | Feasible with positive conservative utility | Infeasible | Unresolved |
|---|---:|---:|---:|---:|
| Free count, analytic optimum | 91 | 91 | 11 | 0 |
| Original per-query fixed count, exact integer optimum | 90 | 90 | 12 | 0 |

All 11 free-count failures have observed positive easy references. None has a
distinct EqMotion action with an easy harm ratio within the original 2%.
The affected recordings have only 1 to 7 known easy rows. Since any weighted
selected easy ratio at most 2% needs at least one such action, this is a
candidate-bank limitation, not a solver timeout. Fixed counts additionally
rule out recording 53 for seed 17, where a free-count witness exists.
Do not average over the 90 or 91 feasible views as though the full roster passed.

The solver returned 90 optimal and 12 infeasible views with no timeouts or
postcheck failures. Every feasible witness is checked by the original
whole-record scorer. The weighted gain objective is distinct from the
unweighted original selected-risk ratios. There are no new certificates,
changed tolerances, imputed positions or saved label-aware row actions.

The same binary oracle replays exactly in a separate invocation. PIDs 13965
and 14081 exit 0 in 59.33 s and 60.74 s. Reuse existing caches and per-record
scalar checkpoints; no training, independent confirmation or deployment.

```sh
.venv-pytorch/bin/python scripts/run_m3w_recording_switch_ceiling.py run
.venv-pytorch/bin/python scripts/run_m3w_recording_switch_ceiling.py replay
```

This is a finite-bank counterfactual calculation, not a deferral learning
guarantee or an application of [conformal risk control](https://arxiv.org/abs/2208.02814).
No new data roles, metric or seconds claims, Stage5C or SMC.

[All views](analysis.json) | [Complementary candidate bank](../moving_recording_bank_ceiling_v1/readout.md)
