# Joint Path Accuracy Under Missing Pair Outcomes

**Finding:** Joint selection improves on independent selection for
observed relative motion, but does not establish a gain over the
matched additive control. The apparent benefit also depends on which
motion baseline and which outcome are compared.

## Protocol

All policies use the same 34 exposed recordings, 14 source groups,
1,613 queries and three frozen forecaster seeds (102 recording views).
The four full-amplitude policies each switch 9,675 agents at matching
query counts; the quarter-amplitude family each switches 6,952. Count
matching is within, not between, these families. CV and OLS4 are
reference forecasts, not count-matched policy variants.

The pair graph uses the four nearest agents at the last observed
frame, made undirected. Both agents must be labeled at a step to score
relative vector error. Across repeated views, 9,222,204 of 11,361,600
requested pair-steps are observed (81.2%); at the last step, 647,949
of 946,800 pairs are observed (68.4%). Positive paired differences
favor the policy named first below. All errors and differences are in
image pixels.

## Paper Table: Pair Accuracy And Completion Bounds

| Comparison | Observed Pair ADE Gain [95% CI] | Observed Pair FDE Gain [95% CI] | Full Pair-Grid ADE Bounds |
|---|---:|---:|---:|
| Joint vs independent | +0.231 [+0.139, +0.345] | +0.659 [+0.373, +1.069] | [-0.944, +1.250] |
| Joint vs shared-family scene | +0.256 [+0.110, +0.448] | +0.604 [+0.259, +1.132] | [-1.235, +1.570] |
| Joint vs additive (primary) | +0.013 [-0.009, +0.045] | -0.014 [-0.297, +0.268] | [-0.158, +0.175] |
| Joint vs CV | -0.438 [-1.067, +0.046] | -1.651 [-3.646, -0.157] | [-2.167, +1.822] |
| Joint vs OLS4 | -3.776 [-4.951, -2.536] | -9.501 [-12.947, -6.597] | [-15.636, +9.888] |
| Quarter vs CV | +0.014 [-0.002, +0.033] | +0.022 [-0.019, +0.065] | [-0.005, +0.032] |

Bounds are not confidence intervals: for each missing pair-step, the
triangle inequality bounds the unknown difference in errors by the norm
of the difference of the two predicted relative vectors. Sum observed
differences and the signed bound contributions over all requested pair
steps, then average queries and seeds within sources. The source
bootstrap resamples whole sources 3,000 times. The primary lower bound
is -0.158 pixels with source interval [-0.337, -0.037]. The upper bound
is +0.175, so this does not prove a negative complete-grid effect.

OLS4 has observed pair ADE 28.842 pixels vs 32.618 for joint, but
available scene ADE 48.012 vs 45.423. Relative pair motion and
absolute scene error can favor different predictors. The joint
policy's source-averaged p95 query pair error is 51.396 versus
49.805 for CV. The worst recording view is 80.804 versus 79.686
for CV. The calibrated sparse risk and support arms change pair
geometry in only 18 and 19 of 4,839 query views; neither has a lower
bound with a source interval strictly above zero versus quarter.
The [analysis](analysis.json) includes every arm, the contraction
diagnostic, and all terminal and tail metrics.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_joint_path_audit.py score
.venv-pytorch/bin/python scripts/run_m3w_joint_path_audit.py replay
```

Registration `b11c56e6` precedes the readout. Both original observed
pair contrasts replay; fresh score and replay reproduce the same
artifact in about 11 seconds each. Fourteen scoped tests passed.
No new training, external inference, risk-rule change or deployment.
Pair distances in images are not physical collision or calibrated
world-state measurements. This is not independent confirmation.
