# Interpretation And Failure Analysis

## Outcome

Nine real fits, the full repeat and all initial/final metric replays completed.
The separated representation improves both training decision-cost channels in
two contexts, but not in the first. The prespecified all-context fitting screen
fails. Additional held predictions are **not_run**, not zero gain, safe fallback
or a passed safety gate. Original2% rules and support requirements are unchanged.
No identity certificate, deployment change, new H or independent test is added.

## What Improved

For separate64, mean fitting MSE falls20.42%,28.68%,10.07% in critic0/1/2.
Easy decision-cost falls20.54%,75.25%,84.96%. All-decision cost falls30.71%
and20.18% in critic1/2, but increases4.13% in critic0. These are own-fitting
diagnostics, not trajectory ADE/FDE or easy deployment degradation percentages.

The decision branch can fit informative signs. In critic0 its stand-alone
all-cost falls80.08%, from0.00470207 to0.00093668. This does not authorize using
it alone: its risk control has not been validated, and earlier signed-cost-only
training failed held evaluation. No posthoc single-head deployment is selected.

## Why The Main Model Still Fails

Critic0's combined all-cost rises from0.00470207 to0.00489618. The decomposition:

| Error component | Initial | Final | Change |
|---|---:|---:|---:|
| False-safe cost | 0.000801034 | 0.000193686 | -75.82% |
| False-rejection cost | 0.003901035 | 0.004702493 | +20.54% |
| False-safe count | 127 | 51 | Fewer harmful accepts |
| False-rejection count | 1918 | 2256 | More useful proposals rejected |

The combined rule cannot recover useful proposals rejected by either head.
The risk branch alone has all-cost0.00503770; adding the decision branch reduces
that somewhat, but leaves it above initialization. SOURCE048's total all-cost
rises18.98%; SOURCE067 falls1.88%. The048 false-rejection increase is the largest
contributor. This localizes a fitting failure, not a proven cause of unseen-scene
errors. Counts are correlated row/action/channel observations, not independent
examples or a significance test.

The current objective averages risk regression and a separate decision loss;
it does not directly optimize their composed acceptance rule. The evidence
supports investigating this mismatch. It does not prove that removing the
risk head or relaxing its threshold would be safe. Conservative missing-label
supervision remains intact and has not been shown to be invalid.

## Capacity And Source Heterogeneity

Compared with the near-capacity-matched shared128 control, separate64 raises
all-decision cost3.57% and12.60% in critic0/1, while lowering it1.26% in critic2.
Easy cost is lower16.70%,8.75%,1.77% respectively. Thus separation is not uniformly
better than increasing shared capacity. Comparing only with shared64 would
overstate the architectural evidence. Wide and separated counts differ by256
parameters; initialization/gradient sharing also differ.

Even an aggregate fitting pass can conceal a weak source. In critic1, the
combined easy-cost increases in053 and065 while the much larger076 improvement
dominates the aggregate. In critic2, SOURCE059's all-cost is slightly worse
despite the aggregate improvement. This is why the original per-view safety
requirements cannot be replaced with a mean improvement.

## Next Model Change, Not Yet Run

The next focused hypothesis is to train the *composed* intervention decision,
with a cost-aware loss that exposes false rejections as well as false accepts,
while retaining the original unknown-label targets and all/easy risk heads.
Use the same own-B roles and a frozen-budget contrast against this checkpoint
family before considering another exposed-development readout. Do not optimize
thresholds against held outcomes or choose the stand-alone branch posthoc.
This is a proposed model-loss experiment, not a change to the2% contract.

Independent reference support remains a separate data limitation. Better model
weights cannot create missing easy reference events. The original24-group
primary stays NA and prior failed trials stay failed. No claim of submission
readiness, new world dynamics, metric/seconds, true3D or foundation evidence.

## Evidence

[All fixed fits and comparisons](README.md), [numeric summary](result_summary.json),
[source/head attribution](fitting_error_attribution.json),
[exact attribution replay](fitting_error_attribution_replay.json),
[original gates](gates.md), [related-method positioning](method_positioning.md).

All nine head/source decompositions replay exactly. The three own-B observed-mask
bindings match their frozen origins. No new held targets were used for this
posthoc fitting diagnosis. Stage5C and SMC remain disabled.
