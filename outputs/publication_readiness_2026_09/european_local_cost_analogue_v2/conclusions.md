# Local Cost Alignment Repairs Much of the Overshoot, but Still Fails

## Material Passport

**fresh_run:** 72 closed-form local risk fits and fixed-query evaluations.
**cached_verified:** ancestor neural weights, original queries, selected donor
population, normalization, neighbor identities and two numerical controls.
**not_run:** new neural training, policy execution, held/independent evaluation
and deployment promotion. This is not a new trajectory or world-dynamics model.

The first full invocation failed on a missing diagnostic-writer alias before
saving a group. That failure, the reproducing regression and the numerical-rule-
preserving v2 repair are retained. Corrected run PID82222 exited 0 in 135.378112 s,
peak RSS 6,634,569,728 B, with zero neural optimizer updates. All 72 views retain
all 62,902 query occurrences: 61,335 known and 1,567 unknown. These counts include
repeated contexts/seeds and overlapping windows, not independent examples.

## Controlled Change

The same selected, other-recording reference bank and 32 nearest histories are
used. Fit the bounded local fraction against source-normalized signed-risk error
instead of averaging raw harm/envelope fractions. Equal-source weighting and
frozen reference-prediction offsets are specified before fitting. Query labels
are used only for scoring. No threshold, k, feature, source or data-role search.

The weighted-harm control shares the weights but omits the reference offset.
The global control removes neighbor matching; the shuffled control permutes
paired corrected fractions within donor recordings. Original and uniform-neighbor
predictions, scores, neighbor hashes and normalization reproduce exactly.

## Primary Results

MSE is source-normalized signed-risk prediction error, averaged equally over
72 views. These percentages are not trajectory ADE/FDE improvements.

| Estimator | All risk MSE | Easy risk MSE | Predicted easy harm | EH>H occurrences |
|---|---:|---:|---:|---:|
| Original head | 0.079458745 | 0.000309373 | 1.375607 | 0 |
| Uniform selected nearest | 0.101728186 | 0.002365719 | 370.305280 | 0 |
| Weighted harm | 0.090005651 | 0.000446276 | 55.366321 | 0 |
| Local signed cost, PRIMARY | 0.089637254 | 0.000442400 | 55.772637 | 9,019 |
| Global signed mean | 0.078049697 | 0.000309446 | 17.752598 | 0 |
| Shuffled signed cost | 2.102822411 | 0.005573301 | 594.829128 | 2,015 |

Observed easy-harm mass is 95.560642 for every arm. Lower aggregate prediction
is not automatically better calibration: the primary now underpredicts this mass.
The primary reduces easy MSE **81.30%** and all MSE **11.89%** relative to uniform
selected neighbors, but remains **1.43x original** on easy MSE and **12.81% worse**
on all MSE. No context/head-seed pair beats both controls: **0/18**. All 12 source
means, all six context means and all three head-seed means remain worse than
original on easy MSE. The registered magnitude screen fails, three of five
conditions passing. Do not substitute the global control after readout; it is
slightly worse than original on easy MSE and also underpredicts harm.

## What the Controls Identify

Weighting without the reference offset achieves 99.80% of the primary's easy-MSE
reduction against uniform neighbors, an algebraic decomposition of this fixed
contrast, not a universal causal attribution. The reference offset adds only
0.87% easy-MSE improvement over weighted harm. It also breaks the necessary
ordering EH<=H in 9,019 repeated predictions. This follows from fitting the two
reference-corrected targets separately; the original harm labels and common
positive weighted averages do not have that problem. The violations are kept,
not projected away after seeing them. The shuffled arm also has 856 B+H>E
violations. No moment or physical-validity gate is claimed for the primary.

The primary's smaller MSE does not mean it identifies actual harm well:

| Known-outcome diagnostic | Uniform neighbors | Primary local cost |
|---|---:|---:|
| Easy-harm SSE on positive events | 40.662720 | 41.956992 |
| Easy-harm SSE on non-events | 36.437111 | 2.342901 |
| Predicted harm on positive events | 7.230221 | 1.265996 |
| Predicted harm on non-events | 363.075059 | 54.506641 |
| False-safe signed-score occurrences | 1,699 | 2,444 |

Most error reduction is on realized non-events, while positive-event harm SSE
worsens. Nonzero probabilistic risk on a realized non-event is not a classification
false positive. The event-conditioned figures are descriptive, not inference
features or independent calibration certificates. False-safe counts do not
certify the 2% selected positive-harm/reference budget.

## Support and Failure Slices

Formal k32 support stays complete, but weights can collapse effective reference
support: minimum 1.00013 effective donors and maximum single weight 0.999935.
The median of the 72 view medians is 7.78267, not a pooled query median or an
independent sample-size estimate. Weighted, overlapping references remain
dependent. This reveals leverage risk, not proof that it caused all query errors.

Locality020 improves from easy MSE 0.017224676 to 0.001258581, still far above
original 0.000009287. Locality082 worsens versus uniform neighbors. All are
retained. Neighbor matching is worse than the global signed mean on the primary
MSE, despite being much better than shuffled signed labels. The shuffle does
not prove that the current distance representation is adequate for risk learning.

## Decision

Do not deploy, reroute queries or read independent data. Loss alignment explains
much of the analogue overshoot, but a reliable conditional risk learner is still
missing. The reference-offset benefit is small and structurally inconsistent.
The next repair must examine coupled harm consistency and effective local
support together; merely adding a positivity link, rescaling total harm,
changing thresholds or celebrating lower donor loss would repeat known failures.
Any new repair needs a separate frozen protocol; it has not run here.

Protocol remains obs8/pred12 rawstride12, pixel/local detector-silver, and 2%
positive selected harm/reference. No metric, seconds, human-gold, physical-safety,
true3D, foundation or submission-readiness claim. Exposed upstream models and
preprocessing prevent whole-pipeline OOF claims. No confirmatory CI or p-value
from repeated fitting views. Stage5C/SMC remain off; the long-term goal is open.

See [slices](slice_diagnostics.md), [gate](evidence_gate.md),
[failure taxonomy](failure_analysis.md) and [reproduction](reproducibility.md).
