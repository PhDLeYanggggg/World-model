# Cross-Moment Targets Help Some Controllers, Not Neural Superiority

## Result Passport

- Fresh: 72 real Torch risk heads, 144,000 updates, three seeds and 432 views.
- Cached and hash-verified: forecasts, utility heads, preceding supported-pair controls and source-excluded producer chains.
- Not run: new trajectory fitting, independent calibration/confirmation or CREATE training.
- Decision: no model promotion, no threshold change, no safe neural superiority; not submission ready.

This experiment tests two explanations separately. Batch mode replaces observed
H/(B+H) difference weights with cross-moment weights abs(H_i B_j-H_j B_i),
retaining per-batch normalization. Fitting mode changes only that denominator
to a fixed training-only scale. Both modes start from the original matched
initialization, not an extra-trained control. Both complete decision banks were
frozen before either new readout. Registration commit: `3fe8c81a`.

## Controlled Results

Each cell is the number of positive / negative conditional 95% intervals among
nine fold-seed comparisons. Intervals containing zero are omitted from both
counts. The metric is all-ADE gain difference, in percentage points of the same
CV-normalized scale. Event identifies the risk-head target, not the evaluation
subset. These are dependent, unadjusted development comparisons.

| Comparison | Candidate / event | Full policy | Ordering at control counts | Ordering at treatment counts |
|---|---|---:|---:|---:|
| Cross weights vs share weights | Neural / all | 4 / 4 | 2 / 3 | 4 / 3 |
| Cross weights vs share weights | Neural / easy | 0 / 4 | 0 / 2 | 0 / 3 |
| Cross weights vs share weights | Damping / all | 9 / 0 | 9 / 0 | 9 / 0 |
| Cross weights vs share weights | Damping / easy | 0 / 9 | 0 / 5 | 0 / 5 |
| Fixed vs batch normalizer | Neural / all | 2 / 0 | 1 / 2 | 2 / 2 |
| Fixed vs batch normalizer | Neural / easy | 5 / 1 | 4 / 2 | 5 / 2 |
| Fixed vs batch normalizer | Damping / all | 2 / 2 | 1 / 2 | 0 / 2 |
| Fixed vs batch normalizer | Damping / easy | 4 / 2 | 6 / 1 | 5 / 1 |

The clearest positive finding is the batch-normalized damping/all controller.
Its full-policy advantage is +0.5105 to +1.1595 pp, with positive intervals in
all nine comparisons. Both matched-count orderings also improve in all nine.
This is not merely making more interventions. It is a scoped controller result
for a simple-motion candidate, not evidence of neural trajectory superiority.

The same target change is not uniformly beneficial. Neural/all is mixed,
neural/easy has no positive full-policy interval, and damping/easy worsens in
every full-policy point and interval (-1.5025 to -0.0530 pp). The fixed normalizer
partly improves easy-event ordering, but does not repair every split or seed.
Neural/easy fixed-normalizer full-policy differences remain small: -0.0488 to
+0.0667 pp. Both sequential controls, rather than a selected winner, remain in
[batch results](batch/results.md) and [fitting results](fitting/results.md).

## Neural Versus Equally Protected Damping

| Mode | Neural all-ADE gain range (%) | Positive / negative all CIs | Positive / negative hard CIs |
|---|---|---:|---:|
| Cross weights, batch scale | -3.0628 to -0.1497 | 0 / 15 | 0 / 15 |
| Cross weights, fixed fitting scale | -3.0001 to -0.2069 | 0 / 13 | 0 / 15 |

Each row covers 18 comparisons. All 18 all-ADE points and all 18 hard-ADE points
are negative in each mode. These direct relative gains are not differences of
two CV-relative percentages. The tested repair has not made the neural-candidate
pipeline stronger than its equally protected simple-motion competitor.

Both modes pass the observed positive-easy 2% check in all 18 neural views.
Worst positive-easy degradation is 0.1738% for batch and 0.1722% for fixed scale.
However, each mode harms zero-reference rows in 12 views; the other six have no
zero-reference examples. Neural observed preservation is therefore 6/18, not
general safety. Damping preserves all 18 observed views in each mode, with worst
positive-easy degradation 1.8602% and no observed zero-reference added harm.
This does not authorize promotion from opened-development results.

Fresh label-support verification finds only four zero-CV rows, all from one
locality, each with two valid future labels and no endpoint. Repeated views are
not additional independent safety cases. The failure remains recorded under
the unchanged metric, but those rows cannot validate full-horizon protection.

## Training Mechanism

Each mode uses 36 heads, 22,979 parameters/head and 2,000 updates/head. Fitting
seconds summed over heads are 152.134 and 158.343, excluding preparation and
evaluation. Every original sampling sequence is retained; there are no unknown
label training draws. Both modes expose exactly 5,049,993 supported pair draws,
the same count as the preceding supported-pair experiment. This tests weighting
and normalization, not pair availability or extra data.

Fixed fitting total loss decreases in 30/36 batch heads and 33/36 fixed-scale
heads; ranking loss decreases in 25/36 and 28/36. A fixed diagnostic batch is not
validation evidence. Weight concentration remains substantial: one audited
neural/easy pair can carry 87.83% of batch weight; mean weight-effective pair
counts range 4.90--11.08 for neural/easy and 3.65--4.60 for damping/easy.
Those are weight concentration diagnostics, not independent sample sizes.
See both modes' fitting diagnostics, weight summaries and all training curves.

## Interpretation and Next Action

The conditional-risk target mismatch was worth testing: a real ordering effect
appears for damping/all. It is not a sufficient explanation of the whole neural
failure. A more defensible target and less random normalization do not establish
useful neural dynamics, solve source shift, or create zero-event support.

Before another loss variant, freeze these forecasts and separate attainable
incremental neural benefit over protected damping from controller error. Report
the available headroom and missed/harmful interventions by locality and future
label completeness, retaining all opened-development comparisons. Use any oracle
only as an offline upper-bound diagnostic, never an inference feature or claimed
model. This will determine whether the next controlled fit should change the
trajectory candidate or its risk/support model. Do not loosen the 2% limit,
drop difficult localities, or promote a favorable seed from this readout.

## Verification and Boundaries

All 72 checkpoint replays, 432 views and 72 preceding controls reproduce.
Separate scalar sorting and coordinate arithmetic check 432 decisions, 1,728
reductions and 216 decompositions. 264 tests across 41 scoped files pass; the
full legacy suite is not run. The [completion receipt](completion_checks.json)
binds the summaries, tests, process completion and pre-readout ordering.

All twelve European Squares localities are opened development; each fit uses
four fitting/eight complete-chain-excluded localities. Three seeds and 3,000
paired locality-bootstrap draws are conditional, dependent and unadjusted.
Detector-track image pixels, obs8/pred12 rawstride12; not historical t50,
seconds, metric, human gold, physical safety, true 3D or foundation evidence.
Historical Stage37 is not recertified. Reserved roles and deployment unchanged;
Stage5C and SMC off. Research goal remains unfinished.
