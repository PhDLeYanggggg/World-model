# Centering the Risk Score Is Not a Safe Policy Repair

## Result

Both frozen head families fail the registered development screen. Fitting-only
score centering reduces some positive harm, but loses more useful interventions.
This is not a deployment upgrade or new neural-dynamics result.

| Head family | Centered versus same-count raw ADE gain % | 95% locality bootstrap interval | Fixed-denominator harm reduction pp |
|---|---:|---:|---:|
| Pointwise | -0.003354 | [-0.008823, -0.000168] | +0.000995 |
| Subset aggregate | -0.005225 | [-0.011821, -0.000433] | +0.001010 |

The aggregate contrast is negative in7localities and tied in5; the pointwise
contrast is negative in6and tied in6. No locality has positive mean ADE gain
in either same-count comparison. These are nominal development intervals, not
independent-confirmation claims or multiplicity-adjusted inference.

## What Changed, and What Did Not

The previous probe fitted two nonnegative offsets per head on the two fitting
sources. This trial applied those fixed offsets without retraining or threshold
search. It retained both existing head families and all108role/seed groups.
Forecasts, utility predictions, fallback, data roles and risk tolerance stayed
fixed. The same-count raw-risk control uses exactly the same count per query.

Registration350d73f0 preceded causal action generation. Action-freeze237a4dd9
preceded the new held-development readout. The twelve opened source-training
development localities remain the only data used. Independent selection,
calibration and confirmation sources remain closed.

## Failure Taxonomy

1. **Coverage collapse.** Aggregate intervention fell from7.7985%to0.6717%;
   pointwise fell from7.9341%to0.6827%. The centered policies entirely abstain
   in200/216and203/216dependent views, respectively. Empty selected-risk
   denominators remain undefined, not zero-risk certificates.
2. **Allocation still loses benefit at matched counts.** Aggregate centering
   removes0.006691ppbenefit while adding only0.000485ppbenefit. It removes
   0.001105ppharm while adding0.000096ppharm. The net change is-0.005197pp
   against the fixed floor-error denominator. This identifies lost benefit,
   not merely fewer actions, as part of the negative accuracy result.
3. **Residual safety failures.** Even after widespread abstention,6aggregate
   and5pointwise views violate the original2%selected-harm screen. Counts of
   violations alone are misleading when nearly all other views are undefined.
4. **Loss improvement did not transfer.** The prior analytic fit improved its
   exact weighted training objective by about0.10%median. That does not imply
   conditional held-source risk calibration or useful forecast allocation.
5. **Not a solver failure.** Both centered and count-matched raw joint policies
   report zero solver-fallback queries. The added statistical constraint, not
   optimizer failure, accounts for the changed feasible sets.
6. **Replay bookkeeping defect.** The first replay stopped after confirming the
   first group's arrays: tuple/list identity containers compared unequal after
   JSON serialization. A separate adapter canonicalizes identities only. The
   registered code and original outputs remain unchanged; full replay and
   independent verification status are recorded in the verification receipt.

Easy cases still pass the error-preservation screen; no zero-CV harms were
observed. That limited success does not rescue the undefined/violating risk
screen. Neither the floor-relative ADE gain nor the harm diagnostic substitutes
for the original primary risk requirement.

## Narrower Research Conclusion

A global offset can fix a weighted mean residual on fitting sources without
fixing conditional ranking or selected-set risk on new source-role views.
This experiment rejects this particular global-centering repair. It does not
prove that every conditional risk model or every joint allocation method fails.

The next justified question is where score magnitude and the per-query error
budget cease to be comparable. Before a new training run, inspect fitting-only
admission loss by the all/easy axes and by the protected-error exposure, using
source-excluded internal folds. If a conditional or exposure-scaled repair is
tested, register it against the same-count raw policy, not only against the
nearly abstaining centered policy. Do not choose new offsets, margins or a
winning family on these held-development outcomes.

## Evidence and Limits

- fresh_run: new policy actions, held-development scoring and paired bootstrap.
- cached_verified: frozen predictors, risk networks, fitted offsets and splits.
- not_run: independent confirmation, independent calibration, new dynamics
  training, cold raw rebuild and the side-effectful full legacy test suite.
- Obs8/pred12, raw-frame stride12, image-local detector-silver. This is not
  metric, seconds-level, human-gold, physical-safety, true3D or foundation evidence.
- Historical Stage35/37/43/44 results retain their exposure/lineage limitations.
- No deployment change. Stage5C and SMC remain disabled. The research and
  submission goals remain incomplete; this is a verified negative mechanism
  experiment only once its verification receipt passes.

Full tables, locality/seed breakdown and the Chinese reproduction guide are
in this directory. Raw arrays, row-level details and checkpoints stay private.
