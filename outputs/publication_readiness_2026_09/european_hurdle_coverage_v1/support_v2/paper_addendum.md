# Separating Risk Ordering from Intervention Coverage

## Research Question

Occurrence-severity supervision can improve observed easy-case preservation
without improving the ordering of candidate interventions. We evaluate this
distinction for frozen neural and causal-damping forecasts, comparing matched
product-moment regression and occurrence-severity risk heads. Neither the
two-part model nor an additive decomposition alone is claimed as a new method.

## Design

We retain three source-role rotations and three training seeds. Within each
rotation, four localities fit the complete model chain and eight excluded
localities provide development readout. All twelve sources have previously been
opened; this is not an independent final test. The history and forecast lengths
are 8 and 12 detector-track positions at a raw annotation stride of 12, in image
pixels. The study does not supply seconds or metric calibration.

For each candidate and all/easy risk-event target, we freeze utility scores and
both risk heads. We rank the common causal pool by estimated harm/reference
mass, using immutable row IDs for ties. Eligibility and per-locality intervention
counts are fixed before label-support filtering. We retain original product and
hurdle policies on their complete legal pools, their common-pool restrictions,
and each ranking at the other policy's common-pool count. An explicitly
registered pre-readout amendment preserves unequal support in two groups.

Let P and H denote summed ADE under the full product and hurdle policies, Pc and Hc
their common-pool anchors, Hp hurdle ranking at product counts, and Ph product
ranking at hurdle counts. Within each locality/subset, C is the same CV error
sum. Percentage-point components satisfy:

```text
100(P-H)/C = support_difference + 100(Pc-Hc)/C
100(Pc-Hc)/C = 100(Pc-Hp)/C + 100(Hp-Hc)/C
             = 100(Pc-Ph)/C + 100(Ph-Hc)/C
```

The two paths separate ordering and coverage along different anchors. They are
exact accounting identities, not a unique causal mediation decomposition.
Unknown labels remain unknown, undefined locality denominators remain undefined,
and matched-count policies are offline controls that may exceed predicted risk.
All 216 views and both anchors are retained. Paired locality-bootstrap intervals
use 3,000 draws conditional on fitted models; overlapping views are dependent
and no multiplicity correction or independent confirmation is claimed.

## Results

For neural all-event heads, full-policy all-ADE improves in eight of nine point
comparisons, with seven positive conditional intervals. At product intervention
counts, however, only three ranking intervals favor hurdle and four favor
product-MSE; at hurdle counts, one favors each. Coverage components favor hurdle
in all nine comparisons along both paths. For damping/all, every fixed-count
ranking point contrast favors product-MSE, while coverage components are positive.

The easy-event arm shows the complementary tradeoff. All nine neural full-policy
all-ADE contrasts are negative. Original hurdle policies keep worst
positive-easy degradation to 0.6746%, compared with 17.2546% for product-MSE.
Forcing hurdle ordering to product counts increases the worst value to 13.7083%
and fails the 2% limit in six views. Product ordering at hurdle counts instead
keeps the worst value to 1.2078%. It still harms zero-CV rows in five views;
neither the count control nor average easy preservation proves zero-error safety.

Unequal-support contributions to all-ADE are +0.004401 and +0.000796 percentage
points in the two affected damping groups. Their small aggregate magnitude does
not remove the need to preserve support provenance. The matched high-count
neural/easy policies violate the predicted-risk rule in all nine views and are
not deployment candidates.

## Implication and Limits

The evidence motivates separating ordering quality, coverage calibration and
support-aware abstention. It does not show stable neural dynamics superiority
over the equally protected damping control. No new forecaster is trained here,
no favorable fold or anchor selects a model, and reserved roles stay closed.
Computational reproducibility is supported by all-view replay, an independent
arithmetic implementation and 227 scoped tests. Independent-scene confirmation,
stable safe neural advantage and paper-level generalization remain incomplete.
Stage5C and SMC remain disabled; no true-3D, foundation or physical-safety claim.
