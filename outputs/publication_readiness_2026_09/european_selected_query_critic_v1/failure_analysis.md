# Why selected-set risk learning did not repair the policy

## What the completed comparison says

The 36 full Torch fits and 72-group policy readout are complete. A separate
complete replay (PID51271,1696.211s) exactly reproduces all72 groups,144 original
full-policy controls and the canonical summary. Exact replay is reproducibility,
not a new independent sample.
All results are exposed SOURCE fitting/resubstitution. They are not independent
generalization, calibration or successful new trajectory dynamics.

The prespecified selected-set joint versus row-local joint contrast is negative:
known ADE -0.610407pp, conditional locality CI[-0.880023,-0.349336]; full paired
wholly-unknown completion-lower -4.872588pp, CI[-11.205362,-1.301837]. All 12
locality known contrasts are negative, as are all three head-seed means.
The separately reoptimized common-count control is also negative: -0.440822pp
known CI[-0.714955,-0.193132] and -2.201911pp lower CI[-3.941785,-0.879762].
Thus the full contrast is not explained solely by different intervention counts.
This does not identify a unique causal reason that pooling fails.

## Distinct failure mechanisms and evidence limits

1. **The query objective is weak in its original units.** All 36 fixed-monitor
   total losses fall, but the median original query-score term worsens. Actual
   same-batch query/point gradient medians are 0.000809 and 0.000708. The separately
   registered scale diagnostic measures old/actual query RMS medians of 48.274
   and 17.588 across channels. This supports testing a unit repair; it is not
   proof that increasing this loss repairs policy utility or safety.
2. **Training selection and final selection differ.** The query loss learns
   risk on the ORIGINAL joint-selected subset. The new independent rule and
   joint allocator can change both actions and selected rows. Training on that
   original mixture need not calibrate every newly optimized mixture. This is
   an implementation-level mismatch, not a quantified causal attribution.
3. **Lower marginal MSE is not lower selected-policy risk.** All-known-legal
   risk MSE falls from 0.151659 to 0.107182 row-local and 0.108521 selected-set.
   Yet selected-set joint has 0/444 supported recording views, 362 known all-risk
   violations and 393 known easy-risk violations. MSE is neither ECE nor the
   selected harm/reference criterion.
4. **The new critics open a much larger action region.** Row/set joint select
   440,480/432,796 reused row views, versus 265,671 for the original joint rule.
   This can improve known fitting ADE while worsening missing-outcome exposure.
   Selected unknown row views are 8,471/8,010, versus 4,098 originally. These are
   descriptive changes, not evidence that expansion alone causes every failure.
5. **Protected outputs restrict finite-label interpolation.** Frozen benefit
   and reference moments constrain what harm corrections can represent. The
   separate triangular projection exposes label targets outside that region.
   These finite realizations do not prove that conditional expected moments
   are misspecified. We keep the constraints and report the limitation.
6. **Pooling adds no new observations.** The selected-context arm uses the same
   past-derived features and original actions, with mean-pooled learned hidden
   vectors. It is an inductive-bias change, not newly observed scene geometry,
   pair dynamics, goal evidence or a proof of better multi-agent representation.
7. **Predicted feasibility does not imply observed finite-record compliance.**
   The 2% all/easy predicted budgets still compose algebraically across queries.
   Learned moment bias, final selection and unknown support can invalidate the
   observed ratio. Empty easy reference is undefined, not a zero-risk pass.

## What the next experiment can isolate

The separately registered query-unit comparison changes only the selected-query
loss denominator. Its architecture, source/query draws, pointwise objective,
weights, forecaster and decoder remain unchanged, with cached-verified old
critics and new paired critics. Better query fitting is a manipulation check;
full retained utility and the original complete-recording risk are the tests.
No threshold is tuned on these policy outcomes. Independent roles remain shut.

## Statistical interpretation

Locality intervals are conditional exploratory summaries after averaging reused
context/head-seed views. They do not adjust for the broader adaptive research
search, and three critic seeds are not three forecast producers. Mixed locality
means are not individual causal effects. The full roster and unknown rows are
kept; support is not repaired by dropping hard recordings or empty comparisons.
No independent confirmation, verified physical safety, metric/time conversion,
human-gold, true3D, foundation, deployment, Stage5C execution or SMC claim follows.
