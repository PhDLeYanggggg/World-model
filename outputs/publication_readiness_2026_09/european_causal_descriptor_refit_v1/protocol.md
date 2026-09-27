# Matched Causal-Descriptor Risk-Head Refit

## Material Passport

Planned fresh108risk heads against108cached_verified signed-excess controls.
Frozen forecast banks, protected floor, utility, all/easy risk budgets, source
roles, target definitions and source-balanced training draws. All12already-opened
European training localities; no independent selection/calibration/confirmation.
This is hypothesis generation followed by another development experiment.

## Single Changed Factor

Add six explicit continuous causal descriptors: mean/last observed displacement
over inherited past extent; path nonlinearity; mean turn over adjacent nonzero
steps; valid neighbor-history occupancy; predicted neural/floor disagreement
over extent. They use current/past observations and frozen predicted rollouts.
All six are retained; no held-selected subset or motion-threshold rule.
No reference-error, future completeness, future goal or target latent input.

Descriptors enter a zero-initialized6x64linear branch added before the existing
hidden GELU. Shared380x64+64x4weights and initial outputs/RNG match the control.
This adds384parameters (24,644to25,028); it is a small feature/capacity change,
not a parameter-count-matched proof that the semantic descriptors are unique.
Descriptor mean/std use fitting-known rows with the existing equal-source
weights; original preprocessing is unchanged. The geometric extent clamp is
inherited, not a metric scale or a guarantee of unit invariance.

The model learns the SAME signed all/easy excess loss: positive harm minus2%
of protected-floor error. Its four nonnegative output components are score
bases, not identified expected costs or calibrated probabilities. AdamW,
learning rate.0003, decay.0001, width64, batch256, gradient clip5 and2,000updates
are unchanged. The existing positive-utility/moving/99%support guards remain.
Pilot100updates are included, then resumed. Checkpoints every500updates, CPU4,
interop1, workers0, no resource probing. No held checkpoint or threshold choice.

## Fixed Evaluation

Freeze all held predictions/actions before reading held outcomes. Compare the
floor, ridge, moment-MSE, signed-excess control, new descriptor model and a
count-matched signed-excess control. For the matched control, use the new count
within each current locality/recording/frame, rank eligible rows by the old
max(all/easy) signed score and break ties by rowID. Include unknown-label rows
in decisions, not just scored rows. Never borrow later-frame intervention slots.

Primary: paired reduction in selected positive-harm ratio against this matched
control. Require positive lower95%locality CI. Also require positive lowerCI
for equal-count ADE advantage and floor ADE gain, nonzero locality coverage,
all216dependent views with defined selected risk<=2%, easy degradation<=2%,
and zero harm on zero-CV-error rows. Empty risk denominators remain undefined
and fail the complete-roster gate. Lower coverage or better training loss alone
does not count as a successful method. Equal-count control is a ranking
diagnostic, not a certified2%policy. Keep per-source/seed and complete/partial
label results; unknown actions remain explicitly unevaluable.

Average dependent producer/fit/seed views within each of12fixed localities;
bootstrap3,000times, seed101531. Preserve the parent's mean-of-view-ratios
estimator, not the later diagnostic pooled estimator. These development CIs
are not confirmation or simultaneous safety guarantees.

## Resources and Integrity

Use lossless compressed checkpoints and decision masks. Freeze prediction
hashes before readout and reproduce scores from checkpoints instead of storing
another score bank. Avoid duplicating large source caches. Verify compressed
checkpoint roundtrip and interrupted/resumed
fitting. Pilot estimates full space/time demand before all108fits. Preserve
10GiBfree disk and save/stop on real resource failure rather than mislabel a
partial study. No unapproved deletion, authentication change, duplicate HPC
job or login-node computation. Each completed group is resumable and bound to
the parent seal, source roles and fitting data hashes.

No training or gate outcome is claimed at registration. Image-local detector
silver, obs8/pred12 rawstride12. No metric/seconds/human-gold/physical-safety,
true3D/foundation claim. No Stage5C execution or SMC.
