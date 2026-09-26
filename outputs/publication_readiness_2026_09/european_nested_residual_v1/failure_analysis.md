# Why The Nested Residual Repair Failed

## Observed Failure

OOF context repair has no positive full-input easy-harm MSE interval against
the original estimator across six assignments. Its point estimates range
from -5.0019% to +0.1976%. Motion-only points are all negative, from -22.0430%
to -3.2028%. The registered primary and tail/coverage guards fail. This is not
just failure to beat a weak comparator; the original and both matched cyclic
controls are retained. No model is promoted.

## Direction Versus Magnitude

Post-freeze, descriptive accounting uses the exact identity
`MSE_change = 2 mean(original_error * shift) + mean(shift^2)`.
It does not estimate or deploy a held-optimal shrinkage coefficient.

| Inputs / residual source | Improved | Non-helpful aggregate direction | Helpful direction, excessive magnitude |
|---|---:|---:|---:|
| Full / OOF | 39 | 18 | 15 |
| Full / matched next | 50 | 7 | 15 |
| Full / matched previous | 53 | 10 | 9 |
| Motion-only / OOF | 35 | 29 | 8 |
| Motion-only / matched next | 40 | 18 | 14 |
| Motion-only / matched previous | 39 | 21 | 12 |

Each row contains 72 dependent locality/seed views, not 72 independent tests.
No view is exactly unchanged. A majority of locally improved views can coexist
with unfavorable assignment-level relative errors: the improvements and harms
have different magnitudes. Selecting the 39 improved full-input OOF views
would be outcome-guided reporting and is not done.

The decomposition identifies two failure patterns, not causal root causes.
It does not justify saying that global shrinkage alone fixes the problem.
The applied shifts include clipping; a new coefficient would require its own
fitting-only specification, not a coefficient chosen from these held errors.

## Target And Distribution Transport

The OOF producer uses two fitting localities; the original outer estimator
uses three. Cyclic controls match the two-locality size, update budget and
seed, but not composition. The OOF easy cut disagrees with the outer-fitting
event on a median 2.5245% of fitting-bank rows, up to 15.0120%; cyclic medians
are 0.9878% and 1.2659%. This establishes target-event drift, not that it alone
causes the MSE failure. The minimum five motion-only easy-harm rows are also
weak support, even though numerical fitting succeeds.

The fixed seven-feature additive repair does not reliably transport residual
direction or magnitude. Lower fitting loss (median 0.651791 to 0.433402) does
not demonstrate held calibration. No broad claim of undertraining, a defective
Torch runtime, universal OOF failure or missing model capacity is supported.

## Next Falsifiable Step

Keep all current predictions fixed and account for event-cut changes separately
from changes in the producer's residual scale/composition. In particular,
measure the label-side contribution `H * (E_inner - E_outer)` as a diagnostic,
not an inference feature or a new target used to train on held outcomes.
Only then register a fitting-only event/scale transport repair if the measured
effect supports it. Preserve the original strong control and every locality.
Do not launch another threshold sweep or an unsupported larger architecture.

All results are source-development risk-head evidence in detector pixels,
eight observed/twelve predicted annotation steps. They are not trajectory
gains, independent calibration, physical safety, metric/seconds, human-gold,
true3D or foundation success. Stage5C and SMC remain off.
