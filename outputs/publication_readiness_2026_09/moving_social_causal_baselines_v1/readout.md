# Past-Only Social Motion Controls

## Material Passport

Fresh_run: causal baseline forecasts and metric computation on the registered
MovingDroneCrowd++ population. Cached_verified: the population and three-seed
EqMotion forecast bank, followed by a separate exact scalar replay. No new
neural training, policy fitting, calibration or independent confirmation.

## Question And Selection Rule

Does a simple shared-motion extrapolator explain the observed EqMotion gain?
The definition was committed before this extension's held scores. Six original
controls are augmented with nine median-acceleration controls using the same
nearest eight complete-past neighbors as the forecasters. A separate richer
inventory adds nine controls using all-current partial-history context. These
inventories contain 15 and 24 candidates respectively; the richer comparison
is not input matched to the neural forecaster.

Acceleration is derived from past common velocities. The fixed coefficient
grid is alpha={0.25,0.5,1} and decay={0.8,0.9,1}. Each producer fold chooses a
candidate using only its fitting sources and the original mean-source
CV-relative available-label ADE objective. No held score chooses coefficients.
These are causal image-plane extrapolators, not estimated camera pose or
physical acceleration. Future positions never enter prediction inputs.

## Results

| Producer fitting fold | Selected candidate in both inventories |
|---|---|
| 0 | nearest-eight acceleration, alpha=0.5, decay=0.8 |
| 1 | nearest-eight acceleration, alpha=0.5, decay=0.9 |
| 2 | original history OLS-4 velocity |

The richer inventory selects no all-current candidate. Both inventories thus
produce identical held metrics; these are not two independent discoveries.

| Supported-group secondary contrast | Point (pp) | Grouped 95% CI |
|---|---:|---:|
| EqMotion minus selected control, known labels | +18.3259 | [+9.5118,+26.5624] |
| Paired missing-outcome completion lower | -1.6116 | [-13.6627,+8.9601] |
| Paired missing-outcome completion upper | +38.2633 | [+25.8781,+49.7471] |

EqMotion's mean groupwise known gain over causal CV remains +15.7428%.
The fitting-selected social control is -2.5830% against CV on held groups.
Expanding a candidate inventory did not make the selected control stronger
on held data. It would be misleading to report only the favorable neural
comparison and omit this failure to beat CV.

The interval uses 3,000 source-group resamples after producer-fold and actual
neural-seed averaging. There are 23 supported source groups and 138 repeated
producer/seed/group views per inventory, not 138 independent sites. The
deterministic controls do not acquire independent training seeds merely by
being compared with three neural seeds.

## Interpretation And Limits

This simple P-selected social kinematic family does not explain away the
observed-label EqMotion advantage. It also does not establish improvement
under the missing-outcome lower contrast. Constant-acceleration and turn-rate
own-agent families are not included in this extension, so this is not an
exhaustive claim to beat the globally strongest possible causal baseline.

All 195,286 past-eligible rows are retained. The original 24-group primary is
not estimable because one declared group has no requested 8/12 window. The
23-group results are exposed-source development secondary evidence. Partial
available-label ADE is not complete 12-step truth, and endpoint diagnostics
are not full-population FDE. Overlapping windows are not independent trials.

Original cost floors, allocation policies, data roles and the full 2% risk
contract are unchanged. No independent calibration or confirmation was opened.
This result does not promote deployment or prove a novel architecture,
interaction mechanism, safe world model, true 3D or foundation model. Image
pixels and native annotation steps are not verified metres or seconds.
Stage5C execution and SMC remain disabled.

## Actual Execution

Evaluation: PID 14602, observed exit 0, 24.333172 seconds.
Separate replay: PID 14663, observed exit 0, 24.597323 seconds; every scalar
matches exactly. Three synthetic causal-motion, future-field and fitting-role
tests pass. The full numerical record is in `analysis.json`; execution times
are measured, not projected. Private source arrays and forecasts are not
redistributed.

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_social_causal_baselines.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_social_causal_baselines.py verify
```

Completed receipts are immutable; existing verified outputs are read or
verified rather than silently overwritten to manufacture a fresh experiment.
