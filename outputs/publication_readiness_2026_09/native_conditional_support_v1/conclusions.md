# Training-Only Conditional Support and Geometric Identity

## Material Passport

2026-09-21; `fresh_run` in-fit head replay and training-only support arithmetic;
`cached_verified` nested forecasts, gain/protected-risk heads. No new held-source
row was scored, no model was fitted, and no threshold was searched. The source
scenes remain design-exposed. These in-fit predictions are not OOF risk
calibration, even though upstream forecasters have valid nested exclusions.

All twelve views verify the full-horizon triangle inequality: benefit and harm
cannot exceed D, the mean distance between candidate and CV rollouts. On complete
queries where CV_ADE is exactly zero, protected harm equals D exactly, up to
1.066e-14 floating reduction error. D is computed on the requested twelve-step
grid using only the two past-conditioned forecasts and the observed scale;
no future-validity mask enters D. Labels/membership are supervision only.

Therefore E[H*I(CV_ADE=0)|X] = D(X)*P(CV_ADE=0|X), provided X includes the frozen
candidate and CV predictions and the event concerns the complete requested
grid. This is an elementary identity, not a new theorem or a calibrated risk
guarantee. It is a valid special case of joint-risk factorization because D is
known, unlike multiplying an easy probability by an arbitrary mean future harm.

## Observations

- Each view has 7,012-10,001 complete zero-CV rows versus 1,999-2,677 positive
  protected-harm labels. Candidate-agreement rows can supervise baseline
  predictability even though their intervention harm is structurally zero.
- Moving/non-stop zero-CV rows number only 2-7 per view, from 1-3 scoped tracks.
  These are overlapping rows, not independent new events. Such scarce support
  is a serious limit, not evidence that a new head will solve transfer.
- Even on their own fit inputs, nine of twelve protected heads admit 3-9
  harmful zero-CV rows through the fixed positive-gain/0.01 rule. The other
  three admit none. Domain shift alone cannot explain those in-fit failures.
- Multiplying the existing protected-event probability by D gives lower in-fit
  protected-cost MSE in all twelve views: 0.0633-0.2988 versus 0.1499-0.5710 for
  direct magnitude regression (ranges are across views, not paired endpoints).
  This arithmetic check uses the old event, so does not validate a new fitted
  baseline-predictability model or a held policy.

The identity/support result motivates a fixed factorial fit: baseline-zero
classification with/without geometric cost loss, and with/without explicit
past-only constant-velocity consistency features. Keep the rollout and gain
heads frozen, compare every arm, and do not promote on fit MSE or global AUROC.
The follow-up must test actual selected harm, unknown-label coverage and the
simple causal-stop control. No data roles or strict zero-reference protection
are changed. No Stage5C/SMC, metric/seconds/foundation claim or new deployment.
