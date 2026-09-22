# Matched Prefix-Risk Supervision: Frozen Development Experiment

2026-09-22. Registration before real fitting and readout. Previous scalar loss
training is complete, but protection failed. No new result is claimed here.

## Hypothesis and Unchanged Protocol

Full-trajectory benefit/harm supervision misses harm on shorter observed prefixes.
Learning all twelve prefix costs may retain useful interventions while better
protecting easy cases. The preceding support diagnosis motivated this experiment;
it does not prove the repair works. This is not a new independent test.

Keep the approved 8-observed/12-predicted annotation-step task (raw stride 12),
annotation pixels, native supported ADE primary, physical-site equal weighting,
three seeds 17/29/43, same four development-exposed sites and clean nested OOF
forecasters. Fitting row R and outer site O remain excluded from its producer.
No data role, scene, risk tolerance, input contract, primary metric or future
support population changes. Incomplete outcomes stay in supported evaluation and
full-grid bounds; unknown is not safe. Offline annotation, not sensor-time input.

## Two Capacity-Matched Arms

Each of twelve views fits two 356-input, 128-hidden, 24-output networks. Both have
identical seed initialization (zero output weights/bias), complete-label fitting
rows, source-balanced draws, fixed 4x fit-region emphasis, train-only preprocessing,
AdamW lr .001/weight decay .0001, batch 256, 12,000 steps, clip 5. Checkpoint every
500, heartbeat every 100, CPU4/interop1/workers0. This is 24 new risk heads, not
24 newly trained world models. Forecasts and latent features are reused, verified.

- terminal_repeat: twelve copies of the full-12 benefit/harm pair and disagreement.
- prefix: benefit/harm and causal forecast disagreement at each prefix k=1,...,12.

Both minimize the same distance-weighted compositional log loss, averaged across
twelve tasks. Outputs use softplus benefit/harm fractions plus unit slack, scaled
by causal disagreement. Equal output capacity separates supervision from capacity.
Task-dependent distances/gradients differ intentionally; scalar reference bitwise
equivalence is not assumed. No partial-label training is added in this first test.

## Fixed Policies Before Label Readout

Every policy retains the existing past-stop veto and positive terminal disagreement.
Terminal strict selection requires predicted benefit > harm and harm <= .1 benefit.

1. control_terminal: last output from terminal_repeat, terminal strict rule.
2. profile_terminal: last output from prefix, terminal strict rule.
3. profile_guard: profile_terminal AND harm_k <= .1 benefit_k for all twelve k.
4. control_matched: rank control terminal net benefit at exactly profile_guard count.
5. profile_matched: rank profile terminal net benefit at exactly profile_guard count.

Matched counts are per held source/seed; tie break by stable row id. Matching uses
causal decisions only, not outcomes. Zero-disagreement prefixes have zero cost and
do not veto; positive terminal gain remains required. Future validity, prefix length,
labels and outcome-defined easy/hard groups never enter any inference rule.
Gapped future availability is a limitation, not repaired by a contiguous prefix rule.

## Primary Comparison and Decision

Primary: profile_guard minus control_terminal in equal-site ADE gain over causal CV.
Use the existing 3,000 physical-site paired bootstrap after averaging seed errors,
not seed forecasts. Require lower paired CI > 0, positive CV gain for every seed,
easy degradation <=2% for every scene/seed and seed aggregate, and no harm to
complete exact-zero-CV outcomes. All conditions are conjunctive.

Also report profile_guard vs the verified scalar log reference and both equal-count
controls. To claim better discrimination rather than abstention, require positive
paired CI vs control_matched as well. No secondary winner promotion if primary fails.
Report ADE/FDE, hard/easy, per-scene/seed, tails, switch counts, missing labels,
full-grid bounds, prefix cost error/underprediction on complete held labels and
supported prefix-only diagnostics. Prefix diagnostics cannot replace the primary.

All four sites have informed development. Three seeds and site bootstrap do not
create independent confirmation. No calibrated probability, conformal guarantee,
physical safety, external generalization, seconds/metric or novelty claim follows.
Independent calibration/confirmation and joint-scene contribution remain gaps.
Stage5C and SMC remain off. Deployment remains false, even if development gates pass.

## Execution and Reproducibility

Use native arm64 `.venv-pytorch/bin/python scripts/run_m3w_prefix_cost.py`.
First `--audit-only`, then `--view coupa_seed17 --arm terminal_repeat --stop-at 100`
(use the actual registered view key if named differently), then `--resume` for
the full matrix, `--evaluate`, `--verify`, and `scripts/verify_m3w_prefix_cost.py`.
Exact-resume tests and input/label isolation tests precede real fitting. Source,
config, code, reference and checkpoint hashes are bound. Decisions are frozen before
new held-label readout. Reports and code are public; large/private arrays and weights
are not committed. Fit time is reported separately from validation and preparation.
