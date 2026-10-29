# Protected Harm Learning Did Not Repair Selected Risk

## What Completed

Fresh_run54 matched continuations,108,000 updates,144 complete fit/held readouts.
All three old controls replay exactly. All isolated benefit/reference outputs
remain bit-exact. A float32 backward defect interrupted the precursor after37
finals; the repaired experiment restarted every arm. No failed-precursor fit is
included in these results. Numerical repair succeeded; the research gate failed.

## Measured Failure Taxonomy

| Explanation | Evidence | What follows |
|---|---|---|
| Runtime/hang prevented a fair test | Repaired54 finals completed in526.97s, readout118.26s, both exit0 | Not the cause of the final negative result |
| Risk training altered protected benefit/reference | B/R/Re exactly match the anchor on every evaluated row | Ruled out for the isolated arm, not for joint risk |
| Less intervention automatically improves safety | Isolated selects24,204 held occurrences versus36,085 joint, but known easy violations47 versus45/72 | Refusal alone does not repair selected risk |
| The architecture only needs longer training | Utility continuation lowers fit MSE0.489310->0.396041 but raises held MSE1.695168->1.967093 | More updates alone did not generalize in this fixed comparison |
| Isolation improves utility at equal budget | Isolated minus joint held full lower utility-1.627860pp, CI[-2.463811,-0.853076]; same-count-0.276264pp[-0.490781,-0.076560] | Both registered contrasts are negative |
| Missing labels alone explain failure |47/72 known easy violations;63 upper-risk violations and4 undefined ratios | Known-label failures remain before missing-outcome uncertainty |
| Problem is only held-recording shift | Isolated already violates known easy risk21/72 fit views versus6 joint | Training decision quality is also inadequate |
| Model never learns harm volume | On the same anchor-selected held cohort, median predicted/observed easy harm rises0.004896->1.530911 | Aggregate estimates change substantially, but this is not calibrated selection |
| Better bulk harm estimates imply a safe selected tail | Isolated own-selected median easy-harm ratio is0.013320, despite the fixed-cohort ratio1.530911 | Selection-conditional underprediction remains |

The last two ratios summarize different cohorts with68 and63 defined views,
respectively. They are descriptive, not a paired causal effect or ECE. The
registered same-anchor cohort comparison is controlled; the own-versus-fixed
comparison diagnoses selection behavior but cannot establish its sole cause.

## Why This Does Not Prove Gradient Conflict

Freezing the base removes gradient updates to B/R/Re, but it also reduces
trainable capacity from24,615 to12,258 and constrains harm through the fixed
benefit envelope. Its negative result says this particular isolation is
insufficient. It does not show that every shared representation is wrong or
that gradient conflict, data shortage or architecture is uniquely responsible.
The bounded odds range may restrict extreme-anchor repair; saturation has not
been causally isolated in this experiment.

Both utility arms underestimate easy harm badly on their selected sets. Risk
arms raise harm on a larger fixed cohort while continuing to select a tail
where harm is underestimated. The isolated arm also preserves the anchor's
reference-mass errors. Joint-risk held signed MSE4.663019 versus isolated5.047722
has a difference CI crossing zero; do not claim a reliable overall MSE difference
between those two arms. Neither achieves safe policy support across all views.

## Support and Scope

Only3/72 isolated held views have full completion support; the required count
is72. Net easy upper-bound preservation fails42/72, even though this is fewer
than joint risk's54. Three isolated views make no intervention; four have
undefined easy risk. Empty/undefined cases are not successful safety evidence.
All registered controls, sources and seeds remain in the report.

There are74 fit and31 held source/recording pairs across12 localities. Row counts
repeat across contexts/seeds. Bootstrap3,000 is locality-grouped but nominal on
already exposed TRAIN data. No outer development or independent roles were read.
No deployment, trajectory producer, Stage5C or SMC changed.

## Shortest Next Test

Do not restart another generic hurdle experiment: occurrence/severity and
coverage-matched hurdles already have negative/limited records in
[the prior hurdle analysis](../european_hurdle_risk_v1/failure_analysis.md).
Before another fit, distinguish selected-score ordering error, frozen-reference
error and causal-support gaps using fixed policies and inner-fit-defined bins.
Retain matched intervention counts, unknown labels and the2% criterion. Any
policy-conditional training repair must state how selecting-model exposure is
excluded; another marginal probability head is not a new mechanism test.

No direct evidence yet supports changing a threshold, discarding a locality,
opening the independent roles, scaling the world model, or upgrading deployment.
