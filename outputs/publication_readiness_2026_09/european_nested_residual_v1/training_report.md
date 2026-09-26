# Inner Risk-Head Training

Result source: fresh_run. PID 54473 exited normally. The committed inner
prediction freeze is 9259a611, before residual fitting or new outer readout.

| Item | Observed value |
|---|---:|
| Completed native-Torch risk heads | 432 |
| Optimizer updates | 864,000 |
| Updates per head | 2,000 |
| Sum of recorded fit time, seconds | 787.943793 |
| Median initial fixed-batch normalized cost loss | 0.651791 |
| Median final fixed-batch normalized cost loss | 0.433402 |
| Unknown-label rows sampled | 0 |
| Computational threads / interop threads / workers | 4 / 1 / 0 |

Fit time excludes the surrounding source checks, data construction, saved
prediction inference and later reproduction checks. It is not total wall time.
The pilot's 100 steps are part of the total, not a discarded tuning run.
Loss is measured on each head's fixed fitting-batch diagnostic. Its decrease
is not evidence of held-locality cost accuracy or trajectory improvement.

All fitting subsets are retained, including the motion-only subset with only
five easy-harm rows. Numerical support is not statistical power. Checkpoints
contain optimizer and random states for exact resume and remain private.
The original outer cost estimators and trajectory forecasters are unchanged.
No independent model selection, risk calibration, confirmation, policy
deployment, Stage5C or SMC was performed.

Three implementation records remain visible: support-report Boolean
serialization was fixed before training; the reused fitter's in-sample
provenance flag was corrected before residual fitting; receipt-path loading
was repaired after readout. None changes the scientific specification.
The second change has explicit numerical-equivalence tests; all preserve
the completed checkpoints and predictions.

Scope remains eight observed/twelve predicted annotation steps, image pixels,
source-development evidence. No metric/seconds, human-gold, physical-safety,
true-3D or foundation claim follows from this training receipt.
