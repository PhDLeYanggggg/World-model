# Newly Admitted Harm Under Joint Training

## Material Passport

Fresh own-B fitting attribution and exact replay, 5 November 2026. This is a
diagnosis of existing neural models, not new training or a held test result.
Parent weights/data are cached_verified. No new V/H outcomes are accessed.

## Finding

| Context | All observed harmful proposals | Newly admitted harmful proposals | Their median target | All harmful median target | New harmful weighted cost | Recovered useful weighted cost |
|---|---:|---:|---:|---:|---:|---:|
| critic0 | 464 | 4 | 0.005242 | 0.122769 | 0.00000404 | 0.00105909 |
| critic1 | 1,467 | 17 | 0.026056 | 0.058217 | 0.00006375 | 0.00080371 |
| critic2 | 14,363 | 141 | 0.072066 | 0.275317 | 0.00049922 | 0.00901202 |

Critic2 repairs one previous harmful admission but admits 141 new harmful
ones, explaining the increase from 22 to 162. SOURCE059 accounts for 101 new
harmful admissions, SOURCE060 for 29, and SOURCE073 for 11. Some errors remain
large: the new-harm 95th percentile is 0.386435 and maximum is 1.278855.
They are not all negligible boundary errors.

The positive target mass is already greater than the safe-benefit mass.
Negative/positive mass ratios are 0.610821, 0.139117 and 0.147125. Thus simple
benefit-mass dominance is not supported. However, the absolute-target-weighted
sign loss gives relatively weak gradients to lower-magnitude harmful cases.
The new-error median is below the overall harmful median in every context.
This motivates a training-weight allocation test, not a proven unique cause.

Costs sum the two normalized target channels, matching the parent attribution.
Training uses their mean, a factor of two that does not alter signs or mass
ratios. Counts are fitting rows, not independent samples. Observed target-safe
means a label sign, not a deployment certificate. The existing original safety
failure and original 24-group primary NA remain unchanged.

## Verification And Next Test

Diagnosis PID8483 completed in 33.58 s; replay PID8997 in 33.12 s, both exit 0.
All three contexts replay exactly. Four focused attribution tests pass.
The initial synthetic expectation used channel means instead of the existing
sum convention; the fixture was corrected before real-data diagnosis. No
scientific result or evaluation rule was changed to address that test failure.

Next fixed comparison: original joint loss, uniform positive-weight increase,
and a positive-only penalty floor at the own-fitting weighted median. The
last two have the same total positive coefficient mass, isolating allocation
from a uniform increase. Risk MSE, unknown labels, inference and the original
2% rules remain unchanged. See [protocol](../moving_harm_floor_v1/protocol.md).

Native obs8/pred12 image coordinates only. No metric, seconds, true 3D,
foundation or human-gold claim. Stage5C and SMC remain disabled.

[Numbers](analysis.json) | [Exact replay](verification.json)
