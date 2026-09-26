# Direct Costs With Auxiliary Easy Membership

## Material Passport

Parent 9987ba4b completes frozen factor attribution: neither realized-factor
substitution consistently repairs conditional cost. No new policy or
independent access is justified. This registered source-only experiment
compares a direct nested-cost model with the same model plus an auxiliary
easy-membership task. It is a standard multi-task control, not a novelty claim.
New fitting/readout are not_run at registration; inherited source features,
forecasts and original costs are cached_verified. Historical Stage35/37
deployment claims remain exploratory after their lineage audit.

## Fixed Intervention

Retain the original 383->64->4 nested moments:
D=softplus(a), H=envelope*sigmoid(b), D_E=D*sigmoid(c), H_E=H*sigmoid(d).
Their outputs are direct moment estimates; sigmoid(d) is a harm fraction,
not a membership probability. Add an independent linear membership readout
from the shared 64-wide hidden layer. Both arms have 24,901 parameters.

- cost_only: original mean four RMS-normalized squared errors plus zero BCE.
- membership_aux: the same cost objective plus one BCE(E) term.

E=1(0<CV_error<=fitting-only positive-error25th percentile). Future-derived E
is training/evaluation supervision only. Unknown labels are excluded, zero
CV error stays outside E. Do not stack fitted probabilities as features or
multiply membership probability into inferred costs. Auxiliary coefficients
are fixed at 0/1 before fitting, with no search. Original model initialization,
training RMS, locality-balanced draws, AdamW/clip and budgets are matched.
The cost-only auxiliary branch is capacity-matched but receives no useful
gradient. Original cached model remains a stronger independent implementation
check; report numerical prediction differences, do not assume equality.

## Budget And Roles

Six source assignments x three seeds(17/29/43) x two forecast pairs(full and
motion_only) x four leave-locality-out folds x two arms =288 new fits and
576,000 updates. Each fit uses the same 2,000 updates/batch256/width64/lr.0003.
Both arms train all four original costs; downstream comparison preserves the
original D/D_E reference predictions exactly and replaces only H/H_E.
The previous conditional-cost system is a negative diagnostic comparator;
beating it alone does not pass. Training labels and preprocessing exclude
each held locality and original producer exclusion is inherited and verified.

Register and commit before fitting. A100-update real control pilot is included
in the final budget and resumes. Freeze all causal predictions/checkpoint
hashes and push before new held readout. This is previously exposed source
development, not independent model selection, calibration or confirmation.
No thresholds, candidate goals, forecasting endpoint or risk tolerance change.

## Fixed Analysis

All known and positive-disagreement costs, all source assignments retained.
Compare auxiliary vs cost-only, original, previous conditional, and control
vs original. Primary requires all six full-input positive-disagreement
easy-harm MSE improvement intervals positive versus BOTH cost-only and
original. Tail top10 harm-capture, coverage absolute-log-error and all-harm
MSE guards allow no negative or non-estimable interval versus either strong
control. Secondary membership Brier/AUROC/log-loss do not rescue cost failure.
No policy readout or deployment even if this component passes.

Average three seeds within locality; 3,000 bootstrap resamples of four
localities per source assignment(seed47131). Repeated source roles/windows
are dependent, CIs exploratory and not multiplicity-adjusted. Full/motion
disagreement populations differ and cannot alone prove modality contribution.
Report cost/BCE losses separately; total objectives are not comparable.

## Runtime And Boundaries

Native arm64 Torch CPU4/interop1/workers0, checkpoints and heartbeat every
200updates, exact resume tested, one process and10GiB disk reserve.
Read-only CREATE status was checked during the parent diagnostic; no new
remote job needed for this measured small-head budget, remote M3W path remains
unverified. Preserve all existing jobs and weights. Weights/cache stay private.
Eight observed/twelve predicted annotation steps; image pixels, detector-
derived labels. No metric/seconds, human-gold, physical safety, true3D,
foundation or submission-ready claim. Stage5C/SMC remain off. Routine audit
is delegated; formal submission remains user-confirmed.
