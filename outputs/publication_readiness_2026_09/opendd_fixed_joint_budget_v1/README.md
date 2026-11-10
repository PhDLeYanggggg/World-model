# OpenDD: match both count and movement before judging the support score

The earlier fixed-proposal comparison selected the same number of recordings
but did not match total movement from CV across recordings. I matched both
the unweighted and observed-query-weighted sums of all 12 forecast-step
displacements, within 1% of the original mask in each of the 126 directions.
Each of the 252 support-score masks has a checked optimal solution under
these constraints. The original two forecasts for a selected recording
never change. The unconstrained comparison reproduces the earlier results.

| Supported merit versus original merit | Count only | Count and movement matched |
|---|---:|---:|
| Selected recording views per arm | 2,196 | 2,196 |
| Support arm movement / original movement (location mean) | 89.93% | 99.60% |
| Query-weighted movement / original movement | 88.33% | 100.03% |
| Support arm dual passes (original: 710) | 752 | 734 |
| Observed ADE gain (pp) | +0.000272 [-0.000361, +0.001169] | +0.000591 [+0.000076, +0.001347] |
| Conservative ADE gain (pp) | -0.001029 [-0.001652, -0.000437] | -0.000388 [-0.000798, -0.000072] |
| Conservative relative-motion gain (native units) | -0.01066 [-0.01690, -0.00565] | -0.00758 [-0.01325, -0.00387] |

Intervals use 3,000 paired location resamples after averaging repeated
directions and seeds within the seven development locations. None of the
2,196 recording views is treated as a new independent location. The
movement-matched dual-pass contrast is +1.61 pp [-1.09, +4.05]. It is not
reliable evidence of improved admission.

The count-only support arm already spends **less** movement on average.
Matching movement narrows the conservative loss but does not reverse it.
On matched masks, the paired missing-label debit is 0.000979 pp
[0.000286, 0.002095], enough to reverse the positive observed-label point.
Total movement does not determine which recordings or steps carry that
movement, or whether those steps have labels. The sign of the full-grid
effect is not established by either the observed gain or the lower bound.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_fixed_joint_budget.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_fixed_joint_budget.py score
```

`fresh_run`: mask optimization and paired sensitivity analysis.
`cached_verified`: frozen scores, proposals, data and the prior readout.
`not_run`: training, CREATE or new external confirmation. Masks were
published before scoring (`fdff79289`). No thresholds, 2% checks or
deployment decisions changed. [Protocol](protocol.md) |
[Mask freeze](mask_freeze.json) | [Full readout](readout.json).
