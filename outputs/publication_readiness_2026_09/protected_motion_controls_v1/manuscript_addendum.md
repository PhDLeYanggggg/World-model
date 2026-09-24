# Protected Motion Formulas as Forecasting Controls

This addendum reports a completed, preregistered source-development comparison,
not an independent external result. It supplements the earlier evidence draft
without rewriting its frozen findings.

## Method

We apply the same bounded benefit/harm cost interface and conservative switching
rule to six deterministic causal motion alternatives and a full Transformer
forecast. The alternatives are constant position, three damping rates, causal
acceleration and turn rate; constant velocity remains the fallback. Four
leave-one-site outer folds and three seeds are used on the established 8-to-12
native-annotation-step SDD population. Forecasting producers used for Transformer
cost supervision exclude both the outer site and each training row's source site.
The formula controls require no learned forecasting producer.

We train 72 neural formula-cost heads and 84 forests, and verify/reuse twelve
previously fitted full-Transformer neural cost heads. Complete-label supervision,
site-balanced sample draws, the 356-dimensional causal interface, and the strict
predicted harm <=0.1 benefit rule are matched. Tree weights reproduce the neural
sampling counts; this matches empirical data weighting, not optimizer capacity or
wall-clock budget. No threshold or model selection uses outer outcomes. All
decisions are frozen before readout. Future-label availability does not determine
inference membership.

## Findings

The full Transformer with a neural cost head improves equal-site relative ADE by
2.437%, with 1.067% worst-site/seed easy degradation. Identically protected damping
improves mean ADE by 3.635%--3.745%, but incurs 2.494%--3.737% easy degradation.
The greater damping utility persists at matched strict intervention counts, where
its easy degradation remains above the ceiling. Thus the comparison is a
utility/degradation tradeoff, not dominance by either forecasting family.

With forest cost heads, damping005 yields 1.414% gain and Transformer 1.302%; the
paired Transformer-minus-damping interval is [-0.739, +0.674] percentage points.
Conversely, holding the full Transformer fixed, the neural cost head exceeds the
forest by 1.135 pp [0.575, 1.695]. These results support useful learned routing in
the examined development setting, but do not isolate an indispensable learned
forecasting contribution. Routing improvements and forecasting improvements are
different claims.

## Interpretation Limits

All intervals are nominal, unadjusted four-site development bootstraps, not
confirmation. No physical safety, metric/seconds, image/goal contribution, joint
benefit or foundation-world-model claim follows. EqMotion is not included in
this comparison. The subsequent inventory correction establishes that its
pair-excluded producers already exist; a matched full-forecast extension remains
to be evaluated. No retraining of those existing producers is implied.
Existing external readouts are not reused for tuning. The complete tables,
negative controls, matched-count safety analysis, loss records and replay
evidence are linked from [conclusions.md](conclusions.md).
