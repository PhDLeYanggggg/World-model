# Fixed Neural Forecasts Do Not Establish A Useful Coupling Gain

Completed 2026-09-20. This is a new decision/evaluation diagnostic over verified
existing models, not new model training, independent confirmation or deployment.

## What Was Run

All 24 fixed combinations were retained: Transformer and EqMotion-K1, three
training seeds, ridge/neural cost heads and conservative/moderate policies.
Each uses the same 970 scene queries and 37,775 agent queries from two explored
UCY recordings of one physical site. ADE has 28,324 complete-path labels; FDE
has 28,335 endpoint labels. Repetition across models does not multiply the
independent population. The approved task remains eight observed and twelve
predicted annotation steps, not seconds. The old primary metric is unchanged.

Every combination compares risk-only independent selection (I), geometry-aware
independent selection (U), and full joint selection (J). U keeps the additive
geometry terms and deletes only the binary-product coupling. U and J match I's
past-agent intervention count with identical issued forecasts, scores, support,
original harm constraint and numerical solver. They do not match realized risk.
The five original controls are also preserved in the complete results.

## Main Result

**The non-additive interaction mechanism has no stable demonstrated forecasting
benefit in this experiment. No new model or policy is promoted.**

| Same-predictor contrast, J minus U | ADE difference | Interpretation |
| --- | ---: | --- |
| Transformer, seed29, ridge/conservative | -0.00000879113 | Tiny favorable difference; one ADE-labeled changed agent |
| EqMotion, seed29, ridge/conservative | -0.00001522417 | Tiny favorable difference; 19 changed agents improve, 11 worsen |
| EqMotion, seed29, ridge/moderate | +0.00000601727 | Tiny adverse difference; 6 changed agents improve, 8 worsen |
| Remaining 21 fixed combinations | 0 | No ADE difference; no changed joint/unary identities |

Differences use the original past-normalized ADE. They are not percentages or
physical distances. Transformer changes two agent identities in one query, only
one of which has a complete ADE label. EqMotion changes 42 identities across 20
queries in the conservative comparison and 20 across nine queries in the moderate
comparison. These subsets overlap across policies and are not independent cases.

Transformer's tiny J-over-U benefit restores the risk-only result: J and I have
identical ADE/FDE there. EqMotion's coupling effects appear only for seed29's
ridge heads and reverse direction between the fixed policies. All neural-cost
head J-versus-U contrasts are zero. This does not show a reproducible learned
interaction contribution.

| Full-joint result across all 12 fixed combinations per model | Transformer | EqMotion |
| --- | ---: | ---: |
| ADE improvement over CV, range (%) | -0.24244 to +0.00546 | -1.84649 to -0.00681 |
| Relative easy degradation, range (%) | 31.31 to 2620.50 | 124.37 to 21821.61 |
| Combinations satisfying easy degradation <=2% | 0/12 | 0/12 |

All 72 new model/control cells fail the same easy-preservation criterion. The
large relative easy percentages have a small denominator: easy CV ADE is
0.00323052 in the original normalized units. Absolute errors, positive harm,
native per-recording errors and hard-slice results are retained in the full
tables; the ratios are neither hidden nor interpreted as physical danger.
Three Transformer configurations have tiny positive overall gains, but that
does not make them safe or establish a coupling advantage.

## Why This Matters

1. **Limited decision effect:** nonzero product terms do not necessarily change
   the optimum. Most fixed policies choose exactly the same identities with or
   without the product term. More solver complexity is not demonstrated value.
2. **Proxy and accuracy differ:** lower forecast-proximity cost can coexist with
   worse trajectory error. The EqMotion moderate contrast is a direct example.
3. **Learned risk is not calibrated safety:** the predicted-harm constraints do
   not protect the observed easy slice. Earlier engineering feasibility checks
   never established a realized-risk guarantee.
4. **Numerical control is necessary but insufficient:** the new risk-only solver
   differs from the old solver on two EqMotion seed29/moderate agent identities.
   All three new controls use the same version, and original prediction errors
   remain exact. The numerical repair does not rescue the scientific hypothesis.
5. **Evidence remains developmental:** one physical site cannot establish
   out-of-scene stability. Three fitting seeds are not three independent sites.
6. **Metric conditioning remains unresolved:** the separate source-population
   audit identified extreme weighting by almost-static histories. This experiment
   retains that primary rather than turning a retrospective metric change into
   a positive result. The metric decision still requires confirmation.

## Verification And Runtime

All 906,600 repeated decision rows exactly reproduce their original identity,
scale, label coverage and floor/candidate forecast errors. Independently checked:
5,439,600 selected-error entries, 216 ADE summaries, 144 paired ADE/FDE contrasts
and 288 native recording reductions. These are verification counts, not sample
sizes. All 23,280 repeated scene decisions match the reference count; zero
unmatched queries. Original legacy-arm replays have zero changed rows. The new
versus old risk-only solver difference above is reported separately.

Both family runs completed on their original devices, with 108 verified batches
each. Transformer main continuation took 479.97 seconds plus a 5.74-second pilot;
EqMotion took 1462.51 seconds plus a 27.60-second pilot. Pilots were included in
the full populations. Neither run was reduced because it was slow. Runtime and
verification notes distinguish sandbox MPS initialization, float32 display
roundoff, and actual solver certificates. The 107 targeted tests pass; unrelated
historical report-writing integrations were not rerun.

Completed resumes for both families exactly reproduce every saved aggregate,
issue zero new forecasts, and retain the completed output identities. The
independent aggregate checker also preserves the original solver certificate
tolerance; its separate float32 display discrepancy reaches at most 7.52043e-8
across both families. This does not alter any saved forecast or decision.

## Evidence Status And Next Step

| Requirement | Status |
| --- | --- |
| Fixed predictors, original rows and frozen rules | Verified |
| Same-solver geometry-aware independent control | Completed |
| All seeds/heads/policies reported | Completed, no selection |
| Stable positive coupling effect | Not established |
| Easy preservation | Failed |
| Independent scene-level uncertainty | not_run: only one explored physical site |
| Independent risk calibration and confirmation | not_run |
| New deployment or submission readiness | No |

The justified next step is to settle and prospectively register the primary
metric decision, retaining all old negatives, then test predictor information
and gain/harm calibration within the permitted training folds. Do not sweep
pair weights on these recordings, relabel these data as confirmation, or keep
joint optimization as the main contribution without new evidence. Geometry
and coupling controls remain useful falsification tools even though this model
family does not earn a positive claim from them.

Stage5C and SMC remain disabled. No true-3D, foundation, metric, seconds-level,
physical-safety or submission-ready claim follows.

- [Every fixed result](complete_results.md)
- [Machine-readable results and independent checks](analysis.json)
- [All new controls in CSV](all_controls.csv)
- [Comparison figure](comparison.svg)
- [Execution, provenance and limitations](execution_notes.md)
- [Design frozen before new scoring](../frozen_interaction_decision.md)
