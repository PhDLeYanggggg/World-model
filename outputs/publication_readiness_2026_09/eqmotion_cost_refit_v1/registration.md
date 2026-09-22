# Predictor-Specific EqMotion Cost Refit

Registration before cost-head training or its new held-source readout.
Prerequisite: completed, hash-verified pair-excluded EqMotion cost caches and
their separate arithmetic audit. No fit can start merely because the upstream
training process finished. All four SDD sites are research-design exposed.

## Fixed Hypothesis

Does training the existing cost-head family on clean EqMotion forecasts improve
intervention over transferring Transformer-trained heads unchanged? This tests
predictor-specific fitting and preprocessing together, not a novel architecture
or an isolated proof that feature shift caused the previous failure.

## Fixed Design

- Twelve outer site/seed views, seeds17/29/43 and the same four source sites.
- Each fitting row uses a producer excluding both its site and the outer site.
  Physically omit outer rows from fitting archives even when a pair cache contains
  them. Keep inference predictions and supervision in separate archives.
- Thirty-six new heads: direct-native, bounded-native and bounded-fraction,
  with the same356features,64hidden units,3,000updates,batch256 and optimizer as
  the previous bounded-cost study. Complete training supervision only. Same
  fitting draws across loss arms; no tuning or seed selection.
- Fit preprocessing on supported fitting rows only. Use identical train-only
  CV/easy/hard cutoffs to the frozen transfer reference; assert their equality.
- Fixed policies: past-stop veto, net gain, harm<=0.1benefit, and offline
  matched-count ranking. All matched-count comparisons use the frozen
  fraction-strict intervention count, not each new arm's preferred count.
- Frozen controls: CV, uncontrolled EqMotion, past-stop EqMotion and all three
  transferred cost-head families. The forecast itself remains unchanged.
- Primary contrast: refitted fraction-strict minus frozen fraction-strict.
  The previous failure motivates this development test and remains reported.
  Do not select a different primary arm after seeing its outcomes.

## Joint Criterion and Reporting

Require each seed's positive ADE gain over CV, zero observed harms among
complete exact-CV outcomes, each seed's positive-easy degradation<=2%, and
a positive lower endpoint for the3,000physical-site-bootstrap paired primary
contrast. Report all arms/seeds/sites, ADE/FDE, hard/easy/tails, intervention,
unknown/incomplete outcomes, conditional harm calibration and full-grid partial
identification bounds. Matched counts separate ranking from abstention.
Confidence intervals on four explored sites are developmental, not independent
confirmation. Even a pass cannot certify missing outcomes or authorize deployment.

Freeze all model scores and policy choices before new held-source outcome
reductions. Replay every cost head and verify arithmetic separately. Save
checkpoints and heartbeats with explicit resume,CPU4/inter-op1/workers0.
Private arrays/checkpoints stay outside Git. The earlier predictors are reused,
not counted as newly fitted heads or independent examples.

No new data roles, closed final tests, threshold search, risk-tolerance change,
independent calibration or deployment. Eight observed/twelve predicted sampled
annotation steps,SDDstride12,annotation pixels; no verified metric/seconds,
true3D or foundation claim. Stage5C and SMC remain off.
