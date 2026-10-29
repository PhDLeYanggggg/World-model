# Full-Range Bounded Harm: One Matched Repair

## Material Passport

Posthoc TRAIN-development hypothesis, registered before new training/readout.
The preceding144-view selected-tail diagnosis motivates this change. Previously
opened recordings remain exploratory, never independent final evidence.

## Single Change

Replace bounded +/-5 log-odds shifts with bounded additive probability shifts.
For frozen anchor harm h, benefit b, envelope e and easy fraction q:

cap=max(e-b,0); h'=h+clip(cap*tanh(z_h),-h,cap-h).
q'=q+clip(tanh(z_q),-q,1-q); easy harm=h'*q'.

The implementation uses increments for exact initial identity and float64
intermediates for numerical stability. Outputs/weights stay float32. This
retains0<=easy harm<=harm<=envelope-benefit and unchanged benefit/reference.
It can escape near-zero anchor odds. Both range and gradient geometry change;
they are not separately causally identified. Frozen benefit can still constrain
the feasible harm envelope. This is not an unbounded residual or forecaster.

18 fits: six unchanged source contexts x three seeds17/29/43. Same original
utility anchors, initial model parameters, protected base, trainable12,258 and
total25,407 parameters, source/query sampling, fresh optimizer, lr0.001,
weight decay0.0001, clip5,2,000 additional updates and fixed risk dual/loss.
Compare each final against its archived isolated-odds control; require exact
initial weights, input hashes, source preprocessing, sampler state/draw hash,
steps and parameter counts. No refit of the six completed controls is needed.

## Evaluation and Decision

Freeze all18 finals before reading any new held predictions. Replay all six
parent controls exactly and evaluate full-range in the same144 fit/held views.
Use full policies plus per-query matched-count paired lower utility, missing
outcome completion bounds, fixed2% easy/all risk and easy preservation.
Bootstrap3,000 by locality after averaging repeated views. Nominal, not
independent/simultaneous inference. Seeds are cost-head seeds, not new producers.

Advance only if lower95% limits for full and matched utility are positive
against ALL six controls AND all72 held views have full completion support.
Empty/undefined policies do not pass. MSE alone cannot advance. Keep negative
sources, unknown outcomes, easy violations and all comparators. No test-based
thresholds, extra optimization trials or favorable checkpoints.

## Runtime and Scope

Native arm64 CPU4/workers0, real100-step resume/direct pilot,18 full fits.
Atomic checkpoints/receipts, heartbeat,12h limit, explicit --resume. CREATE only
read-only existing TRAIN input streaming; no scheduler submission is needed.
No outer development, selection, calibration or confirmation opened. Private
checkpoints/caches remain out of Git. No deployment changes in this experiment.
Observation8/prediction12 at raw stride12, detector tracks, pixel/local units.
No metric/seconds/physical-safety/gold/true3D/foundation claim. No Stage5C/SMC.
