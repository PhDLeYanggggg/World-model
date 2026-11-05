# Cost-Directed Partitions: Better Regression, Insufficient Decision Repair

## What Was Actually Tested

The previous directional-peer feature branch failed its follow-up screen. This
experiment does not expand that branch. It holds the original120causal inputs,
capacity limits, original two-channel risk means and zero decision boundary
fixed, changing the partition-fitting criterion. Sixteen new forests compare
unweighted sign and cost-weighted sign partitions with eight verified existing
MSE partitions. All24cells are freshly refitted for reproducibility.

This is an intervention-risk model experiment, not neural trajectory training,
a new architecture contribution or an independent evaluation.

## Primary Result

| Context | Risk-MSE reduction | Joint-cost reduction | New/control accepted | New/control harmful admissions |
|---|---:|---:|---:|---:|
| 0 | +10.7439% | -1.8295% | 58 / 1 | 6 / 0 |
| 1 | +1.0417% | +0.0410% | 43 / 10 | 9 / 1 |
| 2 | +35.8044% | +0.2843% | 40 / 0 | 3 / 0 |

The registered cost-partition screen fails **0/3**. The required1%joint-cost
reduction is absent, and all three contexts increase absolute false-safe cost.
The large regression improvement in context2 is real under the specified
source-macro diagnostic, but it is not35.8% trajectory or deployment gain.
Source-macro harmful costs are0to9.6490851e-5,4.2984456e-7to2.8190795e-6,
and0to3.3834201e-8. Relative changes from a zero denominator remain NA.

The sign-only secondary improves MSE11.4521%,3.3376%,31.3327% but joint cost
changes-1.2560%,+0.0090%,+0.0165%. It is neither a meaningful decision repair
nor an eligible replacement for the registered primary.

## Why Majority-Safe Labels Do Not Imply Low Expected Harm

| Context | Weighted jointly nonpositive labels | All-channel positive/negative mass | Easy-channel positive/negative mass |
|---|---:|---:|---:|
| 0 | 90.15% | 0.711 | 6.389 |
| 1 | 72.24% | 5.680 | 56.493 |
| 2 | 63.04% | 5.126 | 52.210 |

These are original normalized signed-target masses on known own-B labels,
using the original source/query measure. They are not full-policy risk ratios.
Even with a majority of jointly nonpositive labels, a relatively smaller set
of positive outcomes has much larger magnitude. The easy-channel mean is
positive in all three contexts. A classifier that merely predicts the common
safe class cannot solve the original expected-cost problem. The causal model
must distinguish the high-cost cases, not just count the frequent ones.

This explains why an indiscriminate mean or majority classifier is unsuitable.
It does not prove irreducible future uncertainty, quantify a population tail,
or show that every rejected candidate should be accepted.

## A Hurdle Rewrite Alone Cannot Repair A Fixed Partition

For a fixed leaf and unchanged weights, the identity is:

`E[y] = P(y > 0) E[y | y > 0] - P(y < 0) E[-y | y < 0]`.

Across all24cells, reconstructed factorized means differ by at most1.11e-16;
original-forest mean reconstruction differs by at most3.89e-16. Both have zero
joint-sign changes. No output or deployment rule was replaced by this audit.
Separately trained occurrence/severity models can differ, but the earlier
108-head joint-event study already tested that route and did not establish a
safe policy. Repeating the algebra would not be a new model repair.

## Fitting Gain Does Not Reliably Transfer Across Sources

When048is held out, cost partitions increase fitting admissions52to710 and
reduce fitting decision cost, but held cost worsens. When073is held out,
fitting admissions rise157to481 while held admissions remain zero. Other
folds remain almost entirely rejecting even on their own fitting complement.
All source folds, including44known rows in065, are retained in
[source_folds.md](source_folds.md). This is heterogeneous source behavior,
not a proven single overfitting mechanism or a justification to exclude a source.

## What Changes Next

Do not run a depth, threshold or objective-weight sweep based on these held
source outcomes. Do not promote the best context or switch to the secondary
criterion. The partition change substantially improved continuous prediction
without obtaining useful, safe decisions. More risk-head complexity alone has
not earned a larger neural training run.

The next diagnostic should trace the frozen forecast bank's high-cost cases:
which producer/action and source creates the positive easy-excess mass, whether
that mass is concentrated in a few identifiable motion/annotation patterns,
and whether the existing causal observations distinguish them. Compare with
prior SOURCE trajectory-cost and Moving forecast studies before proposing a
new predictor loss. This separates a poor candidate-bank tail from a poor risk
estimator; neither is currently established as the sole cause. It may justify
changing the forecast itself rather than repeatedly retuning its selector.

## Unchanged Limits

Fresh fits/diagnostics are not independent confirmation. Known proposals48,061;
unknown proposals1,274; own-B rows62,466; eight source folds, three exposed
contexts, one producer seed. Overlapping windows are dependent. No new CI or
multi-seed claim is made. The31focused passing tests establish implementation
behavior, not scientific success. The preflight test failures are retained in
[preflight_amendment.md](preflight_amendment.md).

No additional nested-C/H or reserved readout. Original2%all/easy positive-harm,
signed easy-net and reference/support requirements remain. Empty controls and
missing support stay NA, without identity certification. Earlier failures and
original24-group primary NA persist. A better risk predictor cannot create
missing future labels/reference events. Moving20,DUT,DroneCrowd,H remain closed.
No deployment or submission-ready claim. Native obs8/pred12 silver image
coordinates, not metric/seconds/true3D/foundation. Stage5C/SMC off.
