# Past-Defined Relative-Motion Diagnostic

## Evidence Passport

Fresh_run: evaluation PID6615 exits0 in32.209474s; independent scalar replay
PID6685 exits0 in31.631269s. Cached_verified: all inputs are the separately
replayed18-model forecast bank, with row/schema/checkpoint/source-role hashes.
This diagnostic trains no model and changes no role, primary or risk gate.
Definition was registered before the parent held readout, not chosen from it.

All195,286past-eligible target rows remain. The graph contains962,093past-defined
agent-pair-query edges across2,789queries, using current nearest8neighbors and
agent-ID tie breaking. Edges are not independent observations or future-defined
interaction labels. All23supported groups retain both excluded producers and
three actual training seeds; the original24-group primary stays not estimable.

## Results

Equal-source-group means, after producer/seed averaging:

| Model | Available pair ADE (pixel) | Available endpoint pair FDE (pixel) |
|---|---:|---:|
| Causal CV | 33.7626 | 88.6805 |
| Fitting-selected floor | 33.0270 | 86.4332 |
| PCT | 38.7339 | 109.5076 |
| EqMotion | 31.6745 | 82.1903 |

Paired EqMotion-minus-PCT gain, scaled by fixed known CV pair-error mass:

| Estimand | Point (pp) | 95% group-bootstrap interval (pp) |
|---|---:|---:|
| Known-label contrast | +19.7990 | [+11.8632, +28.1398] |
| Completion-lower contrast | -24.8693 | [-46.6212, -5.8276] |
| Completion-upper contrast | +64.4672 | [+43.2877, +90.3817] |

These use3,000source-group resamples, not IID edge/window resampling. The known
contrast supports better observed relative-motion prediction than PCT, but the
completion lower bound does not support a robust positive effect. A negative
lower bound is not proof that the unknown true effect is negative. Partial pair
ADE uses only common valid future steps and is not complete12-step truth;
endpoint FDE is diagnostic on supported endpoints, not all requested endpoints.
Unknown pairs retain the fixed path-disagreement bounds, not imputed positions.

## Interpretation And Limitations

This proxy measures relative displacement-vector error. An arbitrary shared
translation of all predicted trajectories is invisible to it, even when every
absolute prediction is wrong; that blind spot has an explicit regression test.
It is not a collision check, metric safety, or the published multimodal JADE/
JFDE metric. It cannot replace absolute ADE, the original2%operational/easy
requirements, or the unestimable full-roster primary. No contribution of a new
interaction architecture is isolated by these two unequal-capacity families.

This compares whole forecast families, not the geometry of partial neural/floor
mixtures chosen by a later joint controller. No mixed-policy consistency claim
follows. Source components are not verified IID geographic sites. Calibration
and independent confirmation remain closed. Pixels/native annotation steps are
not verified metres/seconds. No deployment or submission-readiness promotion.
Stage5C execution and SMC stay off.

## Reproduction

After the parent complete fits, full first refits and all18prediction replays:

```sh
.venv-pytorch/bin/python -u scripts/run_m3w_moving_relational_diagnostic.py evaluate
.venv-pytorch/bin/python -u scripts/run_m3w_moving_relational_diagnostic.py verify
```

The original registration, source code, scalar analysis and execution receipts
are versioned. Large prediction/trajectory arrays remain private. Repeating the
same metric computation must reproduce every saved scalar, not open new roles.
