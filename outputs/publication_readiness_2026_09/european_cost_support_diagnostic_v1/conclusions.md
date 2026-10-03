# Frozen Cost-Head Diagnosis: Training Improves, Recording-Held Error Does Not

## Result and Evidence Role

All 72 frozen heads have been replayed from hash-verified original and cost
weights. Predictions, actions and policy readouts are unchanged. Diagnostic
calculations are `fresh_run`; model weights and their training provenance are
`cached_verified`. New fitting, transfer and independent confirmation are
`not_run` because this is a frozen development diagnosis, not a new model trial.

The cost experiment's advance failure is unchanged. This work identifies a
training-to-validation failure with concentrated positive-link extrapolation;
it does not establish a successful repair or certify either policy.

## What Has Been Ruled Out

Every one of the 72 heads improves mean per-tree TRAIN MSE, raw ensemble TRAIN
MSE and final projected ensemble TRAIN MSE. Mean raw ensemble change is
-0.140572, nominal 95% locality CI [-0.170978, -0.110682]; projected TRAIN change
is -0.124240 [-0.150777, -0.099481]. The per-tree loss change plus fixed penalty
reconstructs the saved optimizer loss to a maximum residual of 1.23e-15.

Therefore this run is not explained by incomplete optimization or by a reversal
between the fitted tree objective and the ensemble's TRAIN error. Per-tree and
projected ensemble objectives remain different, but their difference is not
evidence that training failed to fit these data. It also does not rule out all
benefits of a future ensemble-aware objective.

In recording-held validation, 63/72 mean tree errors, 55/72 raw ensemble errors
and 43/72 projected ensemble errors increase. These are repeated heads over 12
exposed localities, not 72 independent validation domains. Eight of 12 locality
means worsen after projection.

## Where the Error Appears

| Quantity, cost minus original | Locality mean | Nominal 95% CI |
|---|---:|---|
| Validation raw ensemble score MSE | +12171.294559 | [4.463675, 29999.991208] |
| Validation projected score MSE | +0.104153 | [0.018611, 0.220039] |
| Easy-harm moment contribution | +0.085562 | [0.011610, 0.196884] |
| Outside majority TRAIN leaf-quality boxes | +0.104975 | [0.024601, 0.215570] |
| Inside majority TRAIN leaf-quality boxes | -0.000822 | [-0.008066, 0.007373] |
| Low TRAIN effective easy-harm event support | +0.106977 | [0.022800, 0.222236] |
| Easy-harm multiplicative ratio above four | +0.087624 | [0.013320, 0.196671] |

These are normalized decision-score errors, not trajectory ADE/FDE changes.
The large raw value is a cost-estimator tail, not an executed trajectory or a
measured physical accident. Feasibility projection suppresses most of it and
must not be removed on the basis of this study. Raw error is concentrated in
localities 067 and 110, with an additional contribution from 007.

Easy harm accounts for 82.15% of the net final MSE increase. The support and
rate strata overlap; they cannot be added as independent causes. The TRAIN
quality boxes are axis-aligned descriptive bounds, not an uncertainty model.
Their outside/inside association motivates a controlled extrapolation test but
does not establish causality. Both all-zero-TRAIN-harm strata contribute zero
to the *change* in MSE; this does not show the original predictions there are
accurate. Sparse nonzero easy-harm support remains implicated.

## Prediction Error Is Not Selected-Policy Harm

Almost all excess final MSE is outside the selected cohort: its global-weighted
change is +0.104155, compared with -0.000002209 in the cost-selected cohort.
Reducing a global tail is therefore not sufficient evidence of safer selection.

The four known-label risk failures in 112/124 retain the original actions. Each
selects two rows. In 124 the known easy-harm row is outside every leaf-quality
box, despite effective TRAIN easy-harm support of about 6.56. In 112, known
easy-harm cohort means are outside 68.36%-71.88% of leaf-quality boxes; mean
zero-TRAIN-EH fractions are 0%-0.78%, not universal absence of harm events.
These are tiny, repeated failure cohorts, not four independent demonstrations.

The three upper-risk failures in 067 require completion for 9-10 unknown
outcomes per head. Known easy-harm rows there have much lower mean outside-box
fractions, about 7.69%-9.56%. A repair of feature extrapolation cannot be assumed
to repair missing-outcome support. Unknown outcomes remain unknown and remain
in the policy evaluation; they are not removed or treated as harmless.

## Next Controlled Repair

The next minimal test should isolate the extrapolation mechanism: a
TRAIN-defined leaf-local feature extension that leaves every known TRAIN
prediction unchanged, compared with the frozen unrestricted positive link.
Freeze its definition before reading its validation results. Keep tree routing,
forecasts, costs, risk budgets, unknown-completion rules and selection logic
unchanged; verify TRAIN invariance and report full and same-count comparisons.

This is not another threshold search, global rescaling or a demonstrated fix.
It must pass the source-development comparison before any transfer advance.
Known selected-risk failures and unknown-outcome support remain separate
requirements even if global MSE improves. A bound on features is not a risk
guarantee. No independent selection, calibration or confirmation roles are opened.

## Runtime and Reproducibility

Registration commit: `5ce4f685`. Native arm64, four compute threads, zero workers.
Pilot: 48.27 seconds, peak RSS 6.472 GB. Full diagnosis: 478.13 seconds, peak RSS
9.219 GB. All 152,728,232 remote checkpoint bytes rehashed. No new local numerical
cache, no new model weights, no Slurm submission, no interrupted active training.
There are 6,624 independent policy scalar checks in the runner and 13,390
independent scalar/bootstrap checks in the final verifier. Twenty-five focused
tests pass; this is not a claim that the entire legacy repository test suite ran.

The 3,000-resample bootstrap uses 12 locality blocks and fixed seed 20261003.
It is nominal, post-hoc development evidence and not search-adjusted. The parent
experiment used a different fixed bootstrap seed, hence its interval differs
slightly while its per-head scores and aggregate point estimate are identical.

See [summary](summary.json), [independent verification](verification.json),
[per-locality and failure tables](findings.md), [protocol](protocol.md), and
[Chinese reproduction guide](reproduction_zh.md).

Detector-silver, image-local obs8/pred12 at rawstride12 only. No metric, seconds,
physical-safety, true 3D, foundation or submission-ready claim. Stage5C and SMC
remain off. The research goal remains active; deployment has not changed.
