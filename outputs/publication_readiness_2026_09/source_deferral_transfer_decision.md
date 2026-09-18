# Fixed Source-Site Readout of Frozen Cost Deferral

## Material Passport

Prospective registration before any new held-source forecast from these six
models. The question follows the completed training-only experiment: does its
small relative-cost routing benefit persist outside its fitting rows?

Six final neural checkpoints and three matched dense controls are
`cached_verified`. New inference for all six is `fresh_run`; the three saved
dense predictions are replayed exactly and their statistics recomputed.
No new training, hyperparameter search or deployment is authorized here.

Bookstore was excluded from these model fits, but its labels and old-model
results were already explored in the project. This is explicitly an exploratory
source-site diagnostic, not a fresh independent test or a way to repair prior
test-selection lineage. Main development, calibration and confirmation stay
closed. The complete approved main task and its primary estimand are unchanged.

## Frozen Design

- Use all six step-10,000 endpoints: expected-cost and cost-supervised variants,
  seeds 17/29/43. No seed, milestone or variant is selected after scoring.
- Retain the exact score > 0 action from training. Zero/negative scores return
  the stationary baseline exactly. No score calibration, alpha search or
  retrospective easy/hard filter is allowed.
- Evaluate each proposal and its hard action, alongside all three matched dense
  mask-only cosine controls and stationary CV. Dense control models have the
  same sampling streams, data and update budget, but 33 fewer gate parameters.
- Use all 6,944 complete registered bookstore queries: seven recordings,
  181 recording-scoped agents, one physical site. The training complement remains
  15,430 rows across four other source sites. No cohort or horizon change.
- Hard labels use the original training-only 90th percentile of baseline ADE.
  Zero-error and moving-label slices are evaluation-only, never policy inputs.

The fixed primary diagnostic is cost-supervised hard action versus stationary
CV and versus its matched dense control. Report every other arm and all fixed
contrasts regardless of direction. A relative gain over a losing neural model
does not establish a positive gain over CV. All-baseline output is not success.

## Readout and Uncertainty

Report ADE/FDE, native annotation-pixel ADE, hard/moving gain, absolute zero-target
harm, requested and actual intervention, binary future-oracle regret, 95th/99th
error quantiles, per-recording results and the worst recording. The oracle is
evaluation-only. Average errors over the three seeds before population statistics;
do not average predictions into an unregistered ensemble.

Use the established 2,000 paired recording resamples, seed 38113. Windows from
one recording are kept together and each arm shares identical resamples.
Report intervals for gain over CV and differences against dense control and
the arm's own proposal, for all, hard and moving slices. These intervals are
conditional on seven recordings from one previously explored physical site;
recordings may remain dependent through shared site conditions. They are not
scene-generalization confidence intervals or confirmatory significance tests.

Also report equal-recording and equal-scoped-agent sensitivity. These are
additional estimands, not replacements chosen because they give a nicer result.
Keep the default window-weighted source diagnostic distinct from the approved
equal-physical-site main estimand. Relative easy degradation is undefined for
zero-error CV rows, so absolute harm does not silently pass the 2% main gate.

## Engineering Checks and Placement

Check the entire saved training artifact hash chain, final model identity,
train/held recording and scoped-agent disjointness, row alignment and bounded
finite output. The new runner cannot fit or select a model. Verify all six
prediction/score snapshots and three dense predictions through exact replay.
Repeated evaluation must leave ancestors, checkpoints and registered summaries
unchanged; new prediction arrays remain local only.

Local native arm64 CPU4/inter-op1/workers0 is sufficient: the prior real training
completed in 37 minutes with about 2.7 GiB RSS, and current free disk is 65 GiB.
This readout requires nine short forward passes and aggregate statistics, not
a new training matrix. Record PID/heartbeat and keep private prediction files
atomically. Interrupted runs verify existing arrays rather than inventing scores.
CREATE has only a historical authentication/project-path blocker on record;
live remote assets and jobs remain unknown. No new remote job is needed here.

## Decision Boundary

A positive source result would still require independent protocol-compliant
confirmation, calibrated risk and joint-intervention evidence. A negative
result is retained, with no new threshold search on bookstore. If needed, the
separate next experiment would isolate frozen-candidate gain learning from
joint proposal/rejector optimization using training data only; it is not run
under this registration.

Offline supplied annotations may be interpolated/generated, not strict sensor
as-of observations or human intention gold. The source task remains 8-to-12 at
stride12, +144 raw frames, annotation pixels/past-normalized coordinates. No
meters, seconds, true-3D, foundation or submission-ready claim. No Stage5C
execution or SMC. No changes to deployment.
