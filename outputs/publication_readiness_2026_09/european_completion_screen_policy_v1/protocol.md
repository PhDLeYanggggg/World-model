# Finite-Completion Source Screen: Frozen Policy Contrast

## Hypothesis

The preceding diagnostic recovered source-validation support for nine candidate
views previously rejected only for missing selected outcomes. Test whether
replacing that unknown-outcome veto with the conservative finite-completion
screen improves actual intervention decisions. This is a source-level support
rule change, not new neural training or new forecast generation.

Use the same 72 frozen fits, MSE/final/initial candidates, past-only actions,
support domains and predicted-risk thresholds. The only eligibility change is
the separately verified completion screen at the existing 2% all/easy selected
positive-harm budgets, positive net-gain lower mass and <=2% easy degradation
upper bound. Do not discard zero-floor easy harm or switch to full-grid ADE.
Rank eligible candidates by the unchanged known-source-validation gain, then
fewer actions, then MSE/final/initial. This ranking statistic is NOT a lower
bound on relative gain over unknown completions. If none passes, use the floor.

No per-row future-availability mask enters inference. Source-level validation
model choice uses outcomes, as did the parent; it is not independent calibration.
The diagnostic results and earlier transfer readouts have been exposed, so the
follow-up is development-only. It cannot restore independent confirmation.

## Freeze and Evaluation

1. Register code, config and protocol before deriving source choices.
2. Commit all 72 choices, then reconstruct causal predictions and actions for
   the same 216 source-to-other-fitting-locality views.
3. Verify the parent's action hashes exactly, freeze all new action hashes and
   commit before computing new transfer metrics.
4. Compare parent policy, completion-screen policy, their per-query same-count
   selections and the unchanged floor. No post-readout selection or thresholds.

Primary: native ADE improvement of completion-screen policy over parent policy,
with coverage differences explicit. Secondary: same-query matched intervention
count ADE improvement. Also report all/easy/hard gain against floor, FDE,
selected all/easy positive-harm ratios, worst easy ADE degradation, unknown
interventions and finite-completion support on each evaluated view. The latter
is an offline diagnostic, never a future-aware deployed gate.

Use three already trained seeds17/29/43 and a 3,000-resample paired bootstrap
after aggregating dependent views within each of12 localities. Intervals are
nominal exploratory intervals, not repeated-look corrected or independent
confirmation. Distinguish numerical gains, source-support repairs, risk passes
and evidence of learned neural contribution. Step-zero priors are not learned
improvement. Unknown/zero-reference views do not count as safe passes.

No independent selection, calibration or confirmation role is opened. No new
training, row cache, checkpoint or GPU job. CPU4/inter-op1/workers0; planned
aggregate output <=16MiB. Forecasts and inputs are cached_verified; new choices,
actions and readout fresh_run. Scientific units remain image-local detector
silver, obs8/pred12 with stride12 raw frames. No metric/seconds, true3D,
foundation, human-gold or physical-safety claim. No Stage5C or SMC execution.
