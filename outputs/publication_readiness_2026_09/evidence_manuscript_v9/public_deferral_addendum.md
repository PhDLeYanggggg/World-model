# New Development Control: Published Regression Deferral

This evidence addendum is not yet incorporated into the compiled main PDF.
It preserves the main draft's not-submission-ready status. It does not change
the original primary or use a new confirmation set.

## Method And Reproducibility

We adapted the fixed-predictor, two-stage logistic surrogate of Mao, Mohri and
Zhong, *Regression with Multi-Expert Deferral*, ICML2024,PMLR235:34738-34759,
Section4 Equation3 ([official paper](https://proceedings.mlr.press/v235/mao24d.html)).
Fifty-four source-isolated routers received2000updates each. All predictors,
causal feature fields, original inner-C roles and query-sampling budgets were
fixed. All learned preprocessing used only the fitting source part. Complete
optimizer/sampler states and inference hashes were replayed; the original point
control also reproduced all108blocks and828source-policy readouts exactly.

The public router differs from the point control in output parameterization,
objective, decision rule and treatment of wholly unknown outcome labels.
Consequently it is a task-adapted public baseline, not a pure loss ablation or a
reproduction of the original paper's experiments/theorems. It does not establish
an original M3W method contribution.

## Results

|Development comparison|Known effect pp [95% source CI]|Direct completion lower pp [95% source CI]|
|---|---:|---:|
|Public router minus point proposer|+5.249[+1.307,+9.551]|-2.410[-6.139,+1.249]|
|Public router with shared veto minus point with shared veto|+0.817[-1.047,+2.931]|-0.780[-2.679,+0.594]|

Unguarded public routing improves source-macro available-label ADE versus the
physical floor by12.295%[6.937,17.878], with a floor-relative completion lower
of4.209%[0.165,8.290]. The latter does not prove the direct contrast against the
point control. The guarded policy supports only38/696repeated recording views;
483have known easy-risk violations. Its required nonempty matched comparison
is not estimable because8required source views have empty controls. Its
full-strategy secondary matched known effect is negative,
-0.807pp[-1.747,-0.245]. No original risk/support requirement is relaxed.

## Interpretation

Direct predictive-cost routing can improve average observed-label utility
without solving selected positive-harm estimation. The existing risk critic
does not reliably convert that utility into safe decisions. Proposal-conditioned
risk learning is a next hypothesis, not a completed result. Missing reference
support remains a distinct limitation that architecture changes cannot erase.

This study uses MovingDroneCrowd++ exposed SOURCE and native observation8/
prediction12 annotation order. Its image-pixel errors are not pooled with
European stride-specific or historical SDD raw-frame results. Intervals use
3000source resamples after context/seed/orientation averaging, not independent
windows. No metric, calibrated-time, physical-safety, true3D, foundation,
independent-confirmation or deployment claim is supported.

Full evidence: `../moving_public_deferral_v1/readout.md` and
`../moving_public_deferral_v1/failure_analysis.md`.
