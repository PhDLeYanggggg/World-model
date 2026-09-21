# Forecast-Disagreement Cost Heads: Better Tradeoffs, Not Certified Safety

Date: 2026-09-21. Registration commit: `3b6bb0a7`, before real fitting or
readout. Analysis SHA256:
`19bc91b35db63de0ce626f92f78094a05dccacf79d4426c119ea50a63d6858d5`.

## Evidence Status

- `fresh_run`: 36 real Torch cost-head fits, 108,000 optimizer updates,
  27,648,000 sampled training instances, the complete fixed source readout,
  checkpoint replay and separate arithmetic verification.
- `cached_verified`: fixed neural forecasts, two-site-excluded training producers,
  causal geometry, source membership, prior risk heads and diagnostic cutoffs.
- `not_run`: new forecaster training, independent calibration or confirmation,
  closed-role readout, deployment, Stage5C execution or SMC.

The four physical sites, coupa/deathCircle/gates/hyang, have already informed
research design. Excluding a site from fitting repairs producer lineage but
does not restore its status as an independent test. The protocol observes eight
sampled annotation steps and predicts twelve, with SDD stride12. Errors are in
annotation pixels, not verified metres or seconds. These are not raw-frame t50
results. No true-3D or foundation-model claim is supported.

## Design and Main Result

Let D be the average distance between the two frozen predicted trajectories over
all twelve requested steps. D is available from past-conditioned predictions;
future labels and their validity are not inputs. The realized full-grid benefit
plus harm is at most D, by the ordinary triangle inequality. The new head bounds
its two continuous predicted costs accordingly. This is a consistency condition,
not a novel theorem, calibrated probability, or protection guarantee.

Three arms have the same 356 causal features, 64-wide hidden layer, initial raw
parameters, complete-only training supervision, scene-uniform draws and 3,000
updates per fit. `direct_native` and `bounded_native` use the same native-cost
MSE. `bounded_fraction` divides the cost residual by D, deliberately changing
example weighting. All final checkpoints are evaluated, with no threshold,
checkpoint, seed or model selection. Unknown training labels are never zeros;
zero unknown-label instances enter training.

The primary registered accuracy contrast is positive:

**bounded-native strict minus direct-native strict: +1.46876 percentage points
of ADE gain, paired physical-scene bootstrap interval [1.00675, 1.93077].**

However, that primary arm fails exact-zero protection: seven repeated query/seed
instances suffer added absolute error where CV is exactly correct. Seed17 also
has 2.10895% positive-easy degradation, despite the three-seed average of 1.59233%.
The primary combined accuracy-and-protection hypothesis therefore does not pass.

## All Fixed Arms

Percent gains are relative to CV, averaged equally across physical sites after
averaging seed errors. Positive easy degradation is worse; negative is better.
Switches and zero-CV harms count repeated query/seed instances, not independent
people, tracks, or scenes. All arms use the same indexed cohort.

| Head and policy | ADE gain % | FDE gain % | Hard gain % | Positive-easy degradation % | Switches | Complete zero-CV harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CV floor | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 |
| Uncontrolled neural | 7.63308 | 8.64511 | 10.65751 | 21.70968 | 527268 | 9000 |
| Legacy stop-MSE-strict | 1.29190 | 1.45715 | 1.73956 | 0.60943 | 14576 | 0 |
| Direct-native net-stop | 5.53031 | 6.17091 | 7.92876 | 8.03903 | 144154 | 15 |
| Direct-native strict-stop | 1.68949 | 1.84930 | 2.39845 | 0.82815 | 13702 | 5 |
| Direct-native matched-count | 2.39575 | 2.62395 | 3.60408 | 1.13817 | 14576 | 0 |
| Bounded-native net-stop | 8.05972 | 8.91390 | 10.20068 | 12.46617 | 304182 | 21 |
| Bounded-native strict-stop | 3.15825 | 3.45206 | 3.98028 | 1.59233 | 41816 | 7 |
| Bounded-native matched-count | 2.44236 | 2.66625 | 3.70718 | 1.26597 | 14576 | 0 |
| Bounded-fraction net-stop | 8.37915 | 9.25745 | 10.25133 | 10.70075 | 286649 | 21 |
| Bounded-fraction strict-stop | 2.43683 | 2.69448 | 2.29181 | -0.55804 | 35198 | 0 |
| Bounded-fraction matched-count | 2.34567 | 2.57687 | 3.52702 | 0.91614 | 14576 | 0 |

These twelve arms were fixed before the outcome readout. This table is not a
post-hoc leaderboard used to pick a deployed model. Matched-count arms use
source/seed-wide ranking at legacy intervention counts; they are offline
allocation diagnostics, not causal streaming deployment rules.

## What the Controls Establish

At the same 14,576 total interventions, bounded-native improves on direct-native
by only **0.04661 pp [0.02486, 0.06426]**, positive on all four explored sites.
Its much larger strict-threshold contrast therefore cannot be interpreted as an
equally large gain in ranking quality. Selection coverage/composition and score
parameterization change. The count-matched control is an important limit on the
claim, not an alternative headline to omit.

The fraction objective is a predeclared secondary arm. With strict-stop it gives
2.43683% ADE gain, conditional scene interval [1.67924, 3.11136], zero observed
complete zero-CV harms, and an average positive-easy improvement of 0.55804%.
Each seed stays below 2% easy degradation; the worst site/seed easy degradation
is 1.06690%. Its hard gain is only 2.29181%, not the older >=10% ambition.

| Explored site | Fraction strict ADE gain % | Positive-easy degradation % |
| --- | ---: | ---: |
| coupa | 2.86381 | -2.27833 |
| deathCircle | 3.35892 | 0.83157 |
| gates | 1.28439 | -0.37507 |
| hyang | 2.24019 | -0.41034 |

The same fraction arm is **0.72142 pp worse** than bounded-native at the strict
rule, interval [-1.39262, -0.05022]. At matched count it is 0.09669 pp worse,
interval [-0.23548, 0.02806]. Thus a favorable protection/coverage tradeoff is
plausible, but superior ranking has not been shown. The difference from the
legacy arm is descriptive: the latter also differs in supervision support.

All twelve bounded-native fits reduce complete-outcome cost MSE relative to the
matched direct head. This is not calibrated harm estimation: in all twelve
fraction strict-policy slices, mean predicted harm is still below observed harm.
Neither lower MSE nor the bound justifies treating the scores as safety risks.

## Missing Outcomes and Safety Limits

175,756 unique target queries remain indexed. ADE has 172,957 supported rows;
2,799 are unknown. FDE has 144,010 supported endpoints. Complete future labels
exist for 143,918 queries; 11,566 of these have exactly zero CV ADE. The protected
rule is absolute added ADE at exact zero, without an epsilon or pixel allowance.

| Strict arm | Selected unknown ADE instances | Selected incomplete futures |
| --- | ---: | ---: |
| Legacy | 283 | 3064 |
| Direct-native | 377 | 3440 |
| Bounded-native | 743 | 8118 |
| Bounded-fraction | 387 | 4960 |

The zero observed complete harms in fraction-strict do not identify safety on
its 4,960 incomplete selected query/seed outcomes. Missingness is not assumed
random. Full-grid absolute gain can instead be bounded using known forecast
disagreement at unobserved steps. Fraction-strict seed-mean intervals are:

| Site | Lower absolute ADE gain | Upper absolute ADE gain |
| --- | ---: | ---: |
| coupa | 0.24797 | 0.41790 |
| deathCircle | 0.43994 | 1.14072 |
| gates | 0.04266 | 0.42205 |
| hyang | 0.27431 | 0.41832 |

These are deterministic annotation-pixel identification intervals, **not**
confidence intervals, percentages, physical guarantees, or protected-subgroup
bounds. They allow arbitrary missing Euclidean targets without a smoothness
assumption. The gates/seed43 lower bound is -0.00724: even this supplementary
whole-population average bound is not positive for every individual fit.

## Failure Taxonomy and Research Choice

1. **Average accuracy is not exact-zero protection.** The primary bounded-native
   arm improves ADE but harms perfectly CV-predictable paths and violates the
   positive-easy limit in one seed. Do not deploy it or relax the rule.
2. **Cost fit is not decision calibration.** Every bounded-native MSE comparison
   improves, yet conditional harm remains underestimated. Better regression
   alone is insufficient as the contribution.
3. **Large policy gains are not large ranking gains.** The +1.46876 pp strict
   contrast shrinks to +0.04661 pp under the registered count control.
4. **Weighting changes the tradeoff.** Fraction training protects observed easy
   outcomes better but loses accuracy against bounded-native and does not beat
   it at matched coverage. This is a secondary development signal, not a new
   primary success after seeing outcomes.
5. **Statistical support remains limited.** Three seeds and 3,000 scene-bootstrap
   draws do not create more than four explored physical sites. No independent
   safety calibration or confirmation is available under the approved roles.

I prioritize **baseline-relative reliable intervention** as the current paper
route, not a generic JEPA/Transformer combination or a larger world-model claim.
This is a practical research judgment, not an estimated acceptance probability.
The new fraction arm is worth freezing for a future approved independent check;
it is not promoted to deployment or selected using closed test labels. The next
method question is whether a train-only decision-calibration/support mechanism
can make conditional harm estimates reliable without discarding useful coverage.
It needs a new fixed experiment identity before changing the registered head.

The completed joint-policy null remains a negative control. Do not resweep pair
weights on these outcomes. The elementary distance bound and expected-cost
routing are not claimed as new. Regression deferral already covers fixed
predictors and learned routing ([Mao et al., ICML 2024](https://proceedings.mlr.press/v235/mao24d.html));
reduced coverage need not preserve subgroup performance
([Shah et al., ICML 2022](https://proceedings.mlr.press/v162/shah22a.html)).
Both original proceedings abstracts were checked again on 2026-09-21. This is a
focused prior-art boundary, not a complete novelty review or theorem transfer.

Publication readiness still requires a substantial distinct contribution,
independent scenes with approved calibration/confirmation roles, comparable
strong methods, and reproducible external evidence. More windows from the same
explored scenes cannot fill that gap. The project remains not submission-ready;
no acceptance promise, physical-safety claim, Stage5C or SMC.

See [analysis.json](analysis.json), [replay](verification_with_replay.json),
[separate verification](independent_verification.json), and
[execution notes](execution_notes.md).
