# Post-Hoc Training Score and Target Diagnosis

Defined after the first seed results were observed. No training, threshold search,
held scoring or deployment is performed by this analysis. All six fixed endpoints
are retained. Costs below are normalized signed gain, not percentage gains.

| Branch | Mean target | Median target | Huber location | Mean score | RMSE / constant mean RMSE | Outside unit Huber radius | Requested rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expected_cost_seed17 | +0.0018389 | -0.0027525 | +0.0018038 | -1.0479632 | 1.0505893 / 0.0434014 | 4 | 0.000% |
| cost_supervised_seed17 | +0.0017791 | -0.0028001 | +0.0017483 | +0.0019456 | 0.0394543 / 0.0427015 | 3 | 73.811% |
| expected_cost_seed29 | +0.0023533 | -0.0033275 | +0.0023139 | -1.0437756 | 1.0470834 / 0.0477312 | 5 | 0.000% |
| cost_supervised_seed29 | +0.0023077 | -0.0033524 | +0.0022758 | +0.0019145 | 0.0428965 / 0.0471736 | 4 | 53.215% |
| expected_cost_seed43 | +0.0016813 | -0.0026271 | +0.0016409 | -1.0622768 | 1.0647797 / 0.0441810 | 3 | 0.000% |
| cost_supervised_seed43 | +0.0014440 | -0.0024295 | +0.0013997 | +0.0014093 | 0.0383056 / 0.0424454 | 3 | 56.889% |

## What This Can and Cannot Establish

For a fixed candidate, expected-action cost has score derivative
`-sigmoid(z)*(1-sigmoid(z))*gain`. The Smooth-L1 term has derivative
`clip(z-gain, -1, 1)`. Its constant optimum is a Huber location, not
necessarily the arithmetic mean gain that determines aggregate expected cost.
The comparison above checks whether this distinction is material for the
observed training targets; it does not prove the conditional optimum of a neural head.

The expected-cost branch has no gain-regression supervision, so its signed-cost
RMSE is descriptive only, not a failed promised calibration objective.
The cost-supervised branch jointly changes its candidate and representation;
in-sample score fit does not establish independent calibration or generalization.

Gradient averages concern a hypothetical common score shift with the candidate
held fixed. They are not parameter gradients, a convergence certificate or an
isolated explanation of clipping. Slice targets are future-label diagnostics
and never inference features. No future/test data informs a policy change.

Past-normalized/pixel raw-frame source task only. No Stage5C execution, SMC,
metric/seconds, true-3D, foundation or submission-ready claim.
