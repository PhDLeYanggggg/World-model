# Ranking Supervision Did Not Repair the Neural Controller

## Material Passport

- Result: completed and computationally verified; research improvement not established.
- Source: fresh36 Torch risk heads,72,000 updates and216 evaluated views.
- Reused: hash-verified trajectory forecasts,original hurdle controls,utility heads,
  training-only preprocessing and complete source-exclusion lineage.
- Not run: new dynamics fitting,independent calibration/final confirmation,remote training.
- Decision: no deployment promotion; not submission ready; goal remains unfinished.

## What Changed

Only the risk objective changed. The original occurrence/severity loss receives
a within-locality,margin-weighted pairwise ordering auxiliary. Every head retains
the same22,979 parameters,initialization,sampled rows,2,000 steps and optimizer.
All forecasts and the2% predicted-risk limit remain frozen. The new weight is1,
registered before training,not chosen from held-out outcomes. This is an adaptation
of established pairwise logistic learning,not a new architecture claim.

All36 heads finished,with4,471,012 valid pair draws and zero unknown-label draws.
These are repeated training draws,not independent observations. Summed fitting
time is149.734s,excluding loading,inference,verification and reporting. The
100-step pilot was resumed rather than discarded. Full traces are available in
[training metrics](training_metrics.json) and [loss curves](training_loss.svg).

## Main Comparison

The table counts positive and negative conditional95% intervals among nine
fold-seed comparisons. The remaining intervals include zero. All-ADE ordering
components use percentage points of CV-normalized improvement and keep both
count anchors; they do not select a favorable intervention rate.

| Candidate / risk event | Full-policy change: positive / negative CIs | Ordering at control counts | Ordering at ranked counts |
|---|---:|---:|---:|
| Neural / all | 5 / 0 | 1 / 5 | 1 / 3 |
| Neural / easy | 2 / 5 | 3 / 4 | 3 / 2 |
| Damping / all | 7 / 1 | 0 / 4 | 0 / 4 |
| Damping / easy | 6 / 1 | 5 / 3 | 1 / 2 |

Neural/all improves full-policy ADE in7/9 point comparisons,with changes from
-0.1871 to+0.9454 pp. But its ordering at control counts changes by-0.4522 to
+0.4259 pp and most intervals do not support improvement. In folds0/1,coverage
components are positive for all three seeds;fold2 differs. The same-count
comparison prevents an intervention-count change being mislabeled a general
ordering improvement.

## Strong Baseline and Preservation

| Control | Observed preservation across all/easy event views | Positive-easy check | Zero-CV harm |
|---|---:|---|---|
| Frozen neural hurdle | 6/18 | All18 within2% | 12 views harm zero-CV cases |
| Ranking-augmented neural | 6/18 | All18 within2% | 12 views harm zero-CV cases |
| Frozen damping hurdle | 18/18 | All18 within2% | No harm observed |
| Ranking-augmented damping | 15/18 | Three easy-event views fail | No harm observed |

The six neural views without zero-CV harm contain no zero-CV examples; they are
not evidence that the model can protect that event. New damping/easy failures
occur in all three seeds of fold1,with worst degradation4.9464%. The corresponding
accuracy increases are not deployable gains.

Against equally protected damping,the new neural policies lose all18 all-ADE
point comparisons;17 conditional intervals are strictly negative. Relative
all-ADE differences range from-4.6057% to-0.2753%. All18 hard-subset intervals
favor damping,with point differences from-5.8747% to-0.0989%. Some easy-subset
comparisons favor neural,but neither overall nor hard-case superiority follows.
See [direct comparison](comparison_summary.json) and [all views](results.md).

## Failure Explanation and Next Test

Observed evidence identifies a training-support problem: neural/easy has167,291
valid rank-pair draws across18,000 updates,versus2,355,302 for neural/all.
Logged minibatches have0-20 versus83-182 valid pairs. Forming pairs before
filtering undefined event labels leaves little easy-event ordering supervision.
This is a plausible contributor,not proof of the entire failure mechanism.

The realized-risk target also differs from a ratio of conditional expected
moments. Lower training loss therefore need not produce better deployment risk
ordering. Do not retune the loss weight or decision threshold on these results.
The next one-factor test should pair supported event rows within each fitting
minibatch/locality,while leaving the original minibatch,moment loss,model and
budget intact. Separately,source-supported causal abstention is still needed
for unsupported zero-reference behavior. [Failure analysis](failure_analysis.md)
records both observations and unresolved explanations.

## Verification and Evidence Boundary

All36 checkpoints replay on the first4,096 excluded-index rows each,and all36
sampler sequences match their controls. Complete216-view evaluation reproduces;
36 original controls match the prior study exactly. Separate scalar sorting,
coordinate arithmetic and bootstrap sums verify216 decisions,864 metric
reductions and108 decompositions. There are234 passing tests across37 scoped
files,not the full legacy suite. [Completion receipt](completion_checks.json),
[verification](verification.json),[execution notes](execution_notes.md) and
[gates](gates.md) distinguish these checks from scientific success.

Each fit uses four fitting/eight complete-chain-excluded localities,all drawn
from twelve opened European Squares development sources. Three seeds and3,000
paired locality resamples are conditional,dependent and unadjusted for multiple
comparisons. The data are released detector-track image pixels,obs8/pred12
rawstride12,not historical t50,seconds,metric,human gold,physical safety,true3D
or foundation evidence. Historical Stage37 is not recertified. Reserved roles,
deployment,Stage5C and SMC remain unchanged. [English addendum](paper_addendum.md).
