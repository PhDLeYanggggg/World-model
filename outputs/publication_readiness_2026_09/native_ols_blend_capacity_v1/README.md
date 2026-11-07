# Does OLS4-Centered Interpolation Expand Forecast Capacity?

The [preceding audit](../native_dual_reference_capacity_v1/README.md)
establishes the ceiling for the native forecast bank at the original
exact query counts. Here the original forecasts remain available and
we add only OLS4-centered blends, with fractions fixed before scoring.
The first expansion adds 12 causal blends, three toward each of four
non-OLS4 motion forecasts. The second adds nine further blends, three
toward each of the learned forecasts. CV remains the fallback and
the denominator for the unchanged 2% recording rule.

The proposed trajectories use only frozen past-input forecasts.
Finding the best combination is a separate hindsight diagnostic:
the solver has labels and must meet both conservative accuracy
comparisons, the original recording constraints and the original
genuine switch count for every query. Matching counts is not
matching the path-displacement magnitude. A verified solution from
a smaller menu carries to the larger menu; a negative bound does not.

The comparisons use 34 exposed development recordings across
14 source groups and three forecaster seeds, not 102 independent
samples. The main paired interval resamples 14 source groups 3,000
times after averaging seeds. A separate descriptive analysis of
negative views compares optimal margins only where both nested solves
were proved optimal. It is conditional on continued failure, not an
overall policy-effect estimate.

## Results

| Nested forecast bank | Max options per row | Verified both | Negative maximin bound | Risk/count infeasible | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 8 | 51 | 45 | 6 | 0 |
| Add causal blends | 20 | 51 | 45 | 6 | 0 |
| Add causal and learned blends | 29 | 51 | 45 | 6 | 0 |

At the same 9,675 genuine switches, neither expansion adds a positive
view, recording, or source group. The paired source-bootstrap
verified-rate differences are both 0 pp [0, 0] on this development
roster; that degenerate interval does not establish population
equivalence. Positive witnesses from the native bank were rescored
and retained; the new solves test capacity on the remaining views,
not whether utility improves on already passing views.

Among the 45 risk-feasible views that remained negative, both nested
optima are proved in every comparison. Causal interpolation improves
the maximin objective by a paired median of +0.120 pp (largest
+0.493 pp), and learned interpolation adds +0.030 pp (largest
+1.074 pp). After all blends, the strongest still-negative optimum
is -1.471 pp. These effects are percentage points of conservative
gain relative to observed CV error mass. No view comes within one
point of the gate.

This rules out the tested three-fraction expansion as a way to
increase dual-reference capacity at this fixed count. It does not
rule out every interpolation or every budget, and it does not
establish what happens under all unknown-label completions.
A new selector over this expanded menu is not supported by the
registered capacity endpoint.

[Scalar results](analysis.json) | [Negative-margin analysis](negative_margin_analysis.json) |
[Score receipt](score_execution.json) | [Replay receipt](replay_execution.json).
Six focused tests passed; a separate 60-case synthetic comparison
gave the same optimal objectives with and without same-person
dominance pruning.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/audit_m3w_ols_blend_capacity.py score
.venv-pytorch/bin/python scripts/audit_m3w_ols_blend_capacity.py replay
.venv-pytorch/bin/python scripts/report_m3w_ols_blend_margins.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_ols_blend_capacity.py \
  tests/test_m3w_dual_reference_capacity.py tests/test_m3w_exact_count_joint_capacity.py
```

[Protocol](protocol.md) and [registration](registration.json) precede
scoring. Per-recording checkpoints preserve solver status and verified
witnesses privately. Replay rechecks the scores of positive witnesses
and recomputes summaries; it does not rerun optimization. The unit is
image pixels at eight observed and twelve forecast annotation steps,
not meters or seconds. No independent data, new training, policy
promotion, Stage5C execution, or SMC.
