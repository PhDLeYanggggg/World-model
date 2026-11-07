# Native Query Activation Under Matched Intervention

Keep the 34 exposed recordings, 14 source groups, three forecaster seeds,
native 8/12 annotation-step task, frozen bank and training-derived cuts.
Use the original CV risk reference, OLS4 accuracy comparator, 2% recording
rule and undefined-reference handling. DroneCrowd remains sealed.

For each held source, take the next three cyclically in the fixed sorted
source roster for calibration and the other ten for fitting. No held or
calibration labels train the support model, the additive gate or the
feasibility ranker. For training slates, also omit each training donor
from its additive proposer. Use the preceding fixed classifier design.

Per current query, both new policies use exactly the same proposed set:
the highest feasibility-score option among those allowed by the original
predicted gate. Compare when to use it by normalized predicted gain versus
the predicted probability of the joint feasible event. The support-only
control is the prior algorithm refit with the same ten fitting sources;
include the previous frozen support policy as a separate context control.

The primary comparison is the paired completion-lower utility of
probability versus gain at identical per-recording counts, also matched
by the number of agents switched per query. These two offline arms rank
the full recording's scores and use the refit support policy's realized
count; they are diagnostics, not current-query deployment policies.

Separately, fit a threshold for each score using only the three donor
sources. Target the refit support rule's donor intervention frequency,
not the held count. At ties at the donor boundary abstain conservatively.
These current-query rules do not use later queries or held labels. Their
held intervention counts can differ; calibration targets coverage, not
a guarantee of recording risk. Do not treat the offline exact match as
a causal result or claim that the donor threshold calibrates the 2% rule.

Report local feasibility, both matched and causal switch counts, the
original recording risk, paired lower utility versus CV/OLS4 and pair
geometry. Use 3,000 paired source draws after seed averaging.
