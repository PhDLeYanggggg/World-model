# Moving Zero-CV Cases: Support Gap, Not Numerical Motion

**The fixed-partition zero-event guard has a relevant-label support gap. No model
or deployment is promoted. Exact representational impossibility is not proven.**

## New Evidence

The complete registered diagnostic covers 36 frozen source fits, three actions,
three seeds and two feature banks. All 175756 cached histories were checked
against their native annotation coordinates; their exact last-step moving flags
agree. All seven moving zero-CV cases were traced to raw annotations. They are
only **three scoped tracks in two recordings**, not seven independent situations.

| Hypothesis | Finding | Conclusion |
|---|---|---|
| Tiny numerical motion falsely makes stopped cases eligible | Last observed displacement is 6, 7 or 16.5 annotation pixels | Refuted for these seven cases |
| Cached zero-CV labels are a conversion error | Raw future coordinates match CV exactly at all twelve requested sample points | Refuted on the fixed grid |
| The complete past already has exact CV motion | Past CV backcast residuals are 3--10 pixels | An exact-history CV veto does not identify these cases |
| Relevant positive labels are nearby in source feature space | None in the nearest 512 moving/effective source rows, in all 126 case/arm/action/seed comparisons | Local event support is missing on the tested representations |
| Exact feature aliases prove prediction is impossible | No exact source-feature match for the seven cases | No impossibility proof; near contexts are not identical contexts |
| Ordinary feature-density rejection alone would clearly identify the cases | In history space, 71.875--93.75% of same-view metadata controls are at least as far from their nearest source neighbor | These histories are not isolated simply by nearest-source distance |

The [complete tables](tables.md), [machine-readable analysis](analysis.json) and
[figure](support_diagnostic.svg) retain every case and all controls. No decision
threshold is selected from these comparisons.

## What the Neighborhoods Show

The source bank is the actual complete, positive-draw, D>0, moving fitting
population, not all nominal source rows. It contains only 2--7 zero-event windows
from 1--3 scoped tracks per view. The frozen training normalization is used;
query-site statistics are never fitted.

In history-only features, the nearest zero-event source row ranks 1330--12720,
or 2.86--13.64% of its source bank. In full cutoff-relative risk features it ranks
21025--42774, or 22.55--87.68%. The first 32, 128 and 512 neighbors contain no
zero events. This is not a claim that those neighborhood sizes define deployment
support: they are fixed descriptive probes.

There are many ordinary moving neighbors. Their candidate benefit/harm labels
are mixed. Full-risk EqMotion neighborhoods have positive mean relative gain
for every case, whereas the actual seven outcomes have zero-CV reference error.
Transformer neighborhoods can have positive or negative mean gain. A conditional
mean can favor intervention without representing a rare harmful outcome well.
The tested explicit atom still misses it. This does not isolate data scarcity
from partition design, nor prove that another causal representation cannot help.

Controls are 32 metadata-hash-selected moving windows per held site, reused
across seeds/actions. Their 384 records per action/feature arm are not 384
independent sites, and the 126 case comparisons are not 126 independent events.
Distances, label proportions and control ranks are not calibrated probabilities,
p-values, confidence bounds or formal out-of-distribution tests.

## Annotation and Time-Grid Findings

All seven cases have exactly zero raw-coordinate CV error on the approved
obs8/pred12 stride12 prediction grid. But CV ADE on the intervening 144 raw
annotation frames is **0.5, 0.625 or 0.8125 pixels**. A grid-exact reference does
not establish a continuous-time or physically deterministic trajectory.
The dense-grid computation is a provenance diagnostic only; it does not replace
the main metric or relax the zero-reference guard.

All twelve future sample points in every case are flagged generated. Six of the
seven supplied histories contain generated rows bracketed by a following
non-generated control after the prediction query. This is evidence of an offline
annotation-provenance concern, not a newly introduced future-feature input.
Control bracketing is not a recorded execution trace of the annotation algorithm.
It does not by itself prove which control was used, but **strict online sensor-
as-of causality is not established**. The benchmark remains prediction relative
to supplied annotations; its code-level input cutoff must not be confused with
the temporal provenance of the source annotations. Generated/silver labels are
not human-gold sensor observations.

These findings do not justify removing seven difficult cases, ignoring their
harm, tuning to them, relabeling exposed sites as confirmation or changing the
approved risk tolerance. The previous failed guard remains failed.

## Research Decision

Do not run another threshold sweep or interpret global event Brier as rare-event
coverage. Do not use future control points as model features. Keep deployment
unchanged and stop treating the exact-zero atom readout as a repaired safety head.

The next highest-value work is **admissible independent trajectory support and
annotation-time provenance**, not a larger forest on the same three event tracks.
Before a further risk-model fit, require a declared source support population and
an independently admissible calibration population. A future support-aware rule
must represent uncertainty about absent event labels, not equate an empty leaf
with zero risk; it must retain fallback and matched-intervention controls. Whether
such a rule can preserve utility under the existing strict criterion remains open.

Independent source admission is not solved by this diagnostic. DroneCrowd remains
closed confirmation, IMPTC remains quarantined, and no external forecast outcomes
were opened. Historical contaminated stages remain exploratory. Stage5C/SMC are
off. No metric, seconds, true3D, foundation, calibrated safety or submission-ready
claim is made. The full research goal is still incomplete.

See [execution evidence](execution_notes.md), [operation/recovery](operation_zh.md)
and the [manuscript addendum](manuscript_addendum.md).
