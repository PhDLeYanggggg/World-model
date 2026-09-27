# Development Experiment: Explicit Causal Risk Descriptors

## Method Contrast

We ask whether explicit descriptors can repair source-held risk estimation
without changing the forecaster, protected fallback, utility estimator or
risk tolerance. Six causal quantities describe motion magnitude relative to
past extent, path nonlinearity, turn, neighbor-history availability and
forecast/floor disagreement. A zero-initialized 6-by-64 branch is added before
the existing risk head's GELU. All shared initialization and training draws
match a signed-budget-excess control. The added branch increases capacity by
384 parameters; this is not a capacity-matched semantic ablation.

There are 108 matched source-role groups derived from twelve opened
development localities and three forecast seeds. Each group excludes the two
held localities from forecaster, floor/utility, preprocessing and risk-head
fitting. A risk head receives 2,000 updates, with fitting-only descriptor
normalization. Decisions are frozen before reading held outcomes.

## Results

The descriptor policy improves ADE over the protected floor by 0.2594%
(95% exploratory locality-bootstrap interval 0.1632 to 0.3687), versus 0.1848%
for the control. Its intervention rate is also higher: 8.0235% versus 6.7818%.
When the control is restricted to the same number of interventions in each
current query, the descriptor policy loses 0.0139% ADE (interval -0.0298 to
-0.0022). This disconfirms a ranking-quality explanation for the unmatched
mean advantage.

All/easy risk-budget excess MSE is the fitting objective, not a certificate.
The all-sample held-score MSE decreases slightly, yet 82 of 216 dependent
held views violate the fixed positive-harm budget. Ten other views have
undefined selected-risk ratios. The registered fixed-roster harm-reduction
primary is therefore undefined and cannot pass. Easy net error is preserved,
which does not resolve positive-harm failures.

## Supported Claim

In this development experiment, improving average score fit and increasing
coverage did not produce better same-budget selection or reliable conditional
harm control. This motivates distinguishing prediction quality, intervention
coverage and allocation quality in the main research design.

## Unsupported Claims and Remaining Evidence

This study does not establish scene-joint superiority, calibrated risk,
independent external generalization, physical safety or a deployable upgrade.
It introduces no new trajectory dynamics, pixel reconstruction, generative
rollout or modality encoder. Strong public method comparisons and genuinely
independent calibration/confirmation are still missing. No submission-ready
or acceptance claim follows from this negative development experiment.

The protocol is image-local detector silver, obs8/pred12 rawstride12, not
metric or seconds-level prediction. The experiment is appropriate for a
transparent ablation/limitations section, not the paper's main positive claim.
