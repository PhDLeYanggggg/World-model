# Cross-Objective Review: Protection at a Large Accuracy Cost

## Material Passport

Completed on 2026-09-22. Fresh fixed-policy evaluation using cached, hash-verified
forecasts and cost heads; no new training. Registration commit `27443938` preceded
readout. The additional veto diagnosis is explicitly post-readout, not a new
registered hypothesis test. No model selection, threshold search or deployment.

The task remains eight observed and twelve predicted annotation steps at stride
12, in SDD annotation pixels. Four development-exposed physical sites, 33
recordings, 175,756 past-eligible queries and seeds 17/29/43. These are not
independent confirmation data. Seeds average errors, not prediction trajectories.
Raw-frame t+50 belongs to a different supplementary protocol and is not evaluated
here. No seconds, metric, true-3D or foundation-model claim. Stage5C/SMC remain off.

## Fixed Comparison

The existing frozen-region head nominates switches from causal constant velocity
(CV) to one fixed neural forecast. The unchanged nomination rule requires positive
predicted net benefit, harm at most 0.1 times benefit, nonzero forecast disagreement
and a past-stop veto. The new review retains a nomination only if its harm and
both existing native/fraction-head harm estimates are at most 0.1 times the
nominator's benefit. All choices are frozen before reading outcome labels.

The primary comparator ranks original nominations by their own predicted net
benefit, selecting exactly the reviewed count in every site/seed. It is an offline
batch diagnostic, not an online deployment rule. Equal counts isolate decision
composition, but do not imply equal realized risk. Both reviewers share fitting
data and upstream models; they are not independent calibrators.

## Results

All percentages below are equal-site improvements over CV; easy degradation has
the opposite sign, so a negative value means improvement. Interventions are
query/seed counts, not independent examples.

| Policy | ADE gain [95% site CI] | FDE gain | Hard gain | Easy degradation | Worst site/seed easy degradation | Interventions |
|---|---:|---:|---:|---:|---:|---:|
| Full nomination | 4.09764 [2.62300, 5.99397] | 4.41091 | 4.25956 | -0.57490 | 3.85432 | 29,668 |
| Cross-objective review | 0.83413 [0.49806, 1.37543] | 0.88967 | 0.56793 | -0.73023 | 0.48136 | 8,627 |
| Same-count nomination ranking | 3.04656 [1.76281, 5.17035] | 3.28898 | 3.99750 | 1.63570 | 4.29167 | 8,627 |

The primary reviewed-minus-matched gain is **-2.21244 percentage points**, with
3,000 paired physical-site bootstrap CI **[-4.51520, -0.76060]**. Every site favors
matched ranking in ADE. Against full nomination the difference is -3.26351 points,
CI [-5.41829, -2.03930]. Review retains only 29.08% of nominations and intervenes
on 1.636% of all query/seed opportunities.

The review passes the observed every-site/every-seed easy ceiling, positive CV
gain and exact-zero-CV checks, but **fails the registered primary joint gate**.
The same-count comparator is not a deployable winner either: its worst easy
degradation is 4.29%. The result is an accuracy/protection tradeoff, not evidence
that review is uniformly useless or that ordinary ranking is safe.

## Scene and Tail Results

Native errors are annotation pixels. P95/P99 summarize the per-query errors
averaged over seeds, not an ensemble forecast or a physical-safety measure.

| Site | Reviewed gain | Matched gain | CV ADE | Reviewed ADE | Reviewed P95 | Matched P95 | Reviewed P99 | Matched P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coupa | 1.65428% | 2.13913% | 11.90821 | 11.71121 | 45.22648 | 44.97510 | 77.44038 | 76.01255 |
| deathCircle | 0.53887% | 6.18075% | 24.81068 | 24.67698 | 73.83408 | 73.19048 | 185.53200 | 172.89373 |
| gates | 0.45725% | 1.59243% | 19.76152 | 19.67116 | 63.05786 | 62.66099 | 118.10106 | 111.26586 |
| hyang | 0.68611% | 2.27395% | 15.92336 | 15.81411 | 48.66319 | 48.27001 | 88.25304 | 86.38508 |

The largest lost gain is deathCircle, not a uniform reduction across scenes.
Reviewed ADE gains by seed are 1.21380%, 0.50164% and 0.78695%. All three are
positive; they do not turn four explored physical sites into twelve independent
sites. The bootstrap describes this development comparison, not unseen-domain
coverage or repeated-study certainty.

## Why the Review Loses Useful Interventions

The new [read-only veto audit](veto_diagnosis.json) partitions the frozen decisions,
uses complete future labels only for diagnostic outcomes, and checks conservation
of counts and net benefit. It does not create another policy.

| Region | All decisions | Complete outcomes | Beneficial | Harmful | Completely unknown |
|---|---:|---:|---:|---:|---:|
| Original nominations | 29,668 | 25,474 | 21,437 | 4,037 | 423 |
| Review accepted | 8,627 | 7,599 | 6,718 | 881 | 101 |
| Review rejected | 21,041 | 17,875 | 14,719 | 3,156 | 322 |
| Matched ranking | 8,627 | 6,975 | 5,858 | 1,117 | 230 |

1. **The veto discards substantial benefit.** Of rejected decisions with complete
   outcomes, 82.34% would improve ADE. Rejected decisions have positive mean net
   benefit in all twelve site/seed views. This is retrospective opportunity, not
   a rule that may use future labels to recover those gains. The same-count result
   independently shows that lost accuracy is not explained by count alone.
2. **Marginal conservatism hides conditional optimism.** Maximum predicted harm
   exceeds realized mean harm in every original nomination region and every
   rejected region. Yet it underestimates mean harm in 10/12 accepted regions.
   Selecting low predicted risk changes the relevant population. A maximum of
   uncalibrated scores is not an upper confidence bound.
3. **Risk discrimination is weak and uneven.** On complete original nominations,
   mean within-view harmful-event AUROC is 0.5745 for the nominator ratio, 0.5113
   for native harm/nomination benefit, 0.6364 for fraction harm/nomination benefit
   and 0.5822 for their maximum. The maximum ranges from 0.4909 to 0.6922. These
   are descriptive event-ranking diagnostics, not calibrated probabilities,
   uncertainty intervals or a proof that event classification solves cost risk.
4. **The reviewers reject different decisions.** There are 9,727 native-only,
   4,478 fraction-only and 6,836 joint vetoes. The better observed fraction-head
   AUROC does not authorize dropping the other head after this readout. No
   reviewer-subset search or retrospective deployment promotion was performed.
5. **Easy protection is empirical, not individual protection.** Accepted complete
   decisions still include 881 ADE-harming query/seed instances. The aggregate
   and scene/seed easy checks can pass while particular agents are harmed.

These facts weaken the hypothesis that cross-objective veto alone repairs
conditional risk estimation. They do not establish a unique cause: cost-scale
sensitivity, ranking quality, upstream forecast behavior and source shift remain
possible contributors. More conservative thresholds alone cannot establish
independent calibration or a useful accuracy/risk tradeoff.

## Missing Outcomes and Claim Limits

The primary ADE supports 172,957 of 175,756 indexed queries. The reviewed policy
selects 1,028 incomplete query/seed futures, including 101 with no outcome at all.
Missing labels are never counted as safe. On complete futures alone, reviewed
gain is 0.87585%, versus matched 3.20981%; the direction remains negative, but
complete-case analysis is not a correction for selection bias.

Triangle-inequality full-grid absolute-gain bounds remain in `analysis.json`.
For gates the reviewed lower bounds are negative in all seeds: -0.00715,
-0.02037 and -0.00770 annotation pixels. Thus even observed positive gains do
not establish improvement for every incomplete-outcome site population.

Current annotations support the offline-annotated observation contract. A
past-indexed loader does not prove sensor-time causality when source interpolation
may depend on later control points. No new data roles were assigned; old closed
roles remain closed. Calibration/confirmation on independent scenes is not_run.

## Evidence and Next Decisions

The findings support retaining a negative mechanism control in the research
record. They do not establish joint-agent benefit, multimodal representation lift,
cross-dataset success, independent safety, or CVPR submission readiness. The
preceding hash-pinned manuscript remains a separate evidence snapshot; this
follow-up does not silently change its original comparisons.

Priorities are now:

1. Resolve the pending independent-scene acquisition/data-role decision. A source
   split that excludes a scene only from the cost head, while its target producer
   saw that scene, is not independent risk calibration. No nested-role reassignment
   or download-warning bypass has been performed.
2. Before another model run, require a repair to address conditional harm ranking
   and calibration, not just high average predicted risk. Freeze its hypothesis
   and matched-risk/count controls in advance; do not choose a critic from these
   held-development outcomes and relabel the result as confirmation.
3. Test the eventual repaired policy on genuinely independent support, retaining
   per-scene easy, missing outcomes, tails and useful intervention rate. Extra
   compute or another architecture alone cannot supply that evidence.

No new policy is deployed. Existing historical Stage26/Stage37 deployments are
not re-certified under this different protocol. The research goal remains unmet.

## Verification and Statistical Scope

All 36 cost-head checkpoint endpoints replay exactly (1,581,804 score rows).
Separate formulas check 36 choices, twelve count matches, 288 scene reductions,
paired intervals and incomplete-outcome bounds. The veto audit independently
recounts its 252 region/subset records against the frozen archives and matches
the previous 48 conditional summaries. Shared preprocessing/model functions
remain: this is same-agent arithmetic verification, not independent replication.
The unchanged 38 scoped experiment tests passed previously; ten new audit tests
pass. The full legacy suite was not rerun.

Statistical review covered all eleven checks: no all-site reversal of the primary
accuracy comparison; no individual guarantee inferred from site averages;
selection/Berkson and collider risks remain in nominated/complete-case subsets;
event counts accompany AUROC; matched controls limit regression-to-mean claims;
missing futures remain visible; post-readout comparisons are descriptive, with
no significance fishing or reviewer selection; historical protocol choices remain
exposed; neither population association nor diagnostic ordering is presented as
a causal explanation or reverse-causal input. Confidence level: caution, owing
to only four explored sites and no independent confirmation.

See [execution and reproduction](execution_notes.md), [primary analysis](analysis.json),
[checkpoint replay](replay.json), [separate formulas](independent_verification.json)
and [fixed registration](registration.md).
