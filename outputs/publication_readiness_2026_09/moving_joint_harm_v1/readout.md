# Matched Neural Joint-Harm Study

**fresh_run:** 108 real Torch heads, 216,000 saved primary updates, complete nested-C evaluation and separate exact inference replay. **cached_verified:** parent forecasters, original roles, admission map, own-source preprocessing and frozen proposers. **not_run:** new H readout, independent confirmation, deployment and submission.

The complete primary is not estimable, not passed. Original full-policy risk/support requirements fail.

## Model Change

Both arms have 28,489 parameters, identical initial weights, own-part preprocessing, query draws, optimizer settings and update budgets. The event arm adds exclusive-event cross entropy plus positive-event conditional-magnitude loss to the same five-moment and net-gain objective. This identifies the combined auxiliary treatment, not each component separately. The product-moment control has the same factorized architecture; its unsupervised probability factors are not a calibrated probability baseline. Fitting-only event priors are also reported. Neither arm is a new trajectory forecaster or a generative world model.

Each critic and all learned preprocessing exclude its held validation sources. A separately fitted frozen proposer determines the action; the critic may retain or reject it, not choose another amplitude. All 108 new heads froze before the new validation readout.

## Registered Primary

|Arm vs own query-count/amplitude-matched control|Known pp [95% CI]|Direct lower pp [95% CI]|Unavailable /276|
|---|---:|---:|---:|
|product_moment|not estimable|not estimable|26|
|event_supervised|not estimable|not estimable|24|

The event-arm direct completion-lower contrast is primary. Matching fixes each query's intervention count and amplitude histogram, not displacement magnitudes. Empty required source views remain NA; no switch is forced. Original 24-group primary remains not_estimable_not_passed. The 23 available source units are exposed development sources, not recovered independent test data.

## Secondary Results

|Full-strategy secondary contrast|Known pp [95% CI]|Direct lower pp [95% CI]|
|---|---:|---:|
|product_moment minus own matched control|-0.9308 [-1.7381,-0.2764]|-1.3922 [-2.5353,-0.5176]|
|event_supervised minus own matched control|-1.0312 [-1.8864,-0.3692]|-1.5431 [-2.8105,-0.6015]|
|Event veto minus product veto|+0.2754 [-0.0617,+0.6649]|+0.0989 [-0.2435,+0.4462]|

Event-versus-product full policies have different intervention rates and are not a matched-ranking contrast. A positive secondary does not replace a missing primary. Intervals use 3,000 source-group resamples after context/seed/orientation averaging. The 276 repeated source views and 696 record views are not independent sample sizes.

## Original Full-Policy Safety

|Policy|Supported /696|Known all-risk violations|Known easy-risk violations|Undefined selected easy risk|Switch % [95% CI]|
|---|---:|---:|---:|---:|---:|
|frozen_proposer|36|494|532|100|+56.6891 [+46.3596,+65.9227]|
|product_moment_veto|41|386|405|220|+18.0149 [+12.1018,+24.4172]|
|event_supervised_veto|47|382|399|213|+18.6854 [+12.9584,+24.9254]|
|product_moment_matched|39|386|404|219|+18.0149 [+12.1018,+24.4172]|
|event_supervised_matched|40|372|401|206|+18.6854 [+12.9584,+24.9254]|

Original 2% risk, easy-net and support requirements are unchanged. Identity certification was not added. Counts overlap; zero-reference and no-intervention views do not automatically pass.

## Same-Row Mechanism Checks

|Same-row diagnostic (positive means error reduction)|All legal known actions [95% CI]|Fixed proposer choices [95% CI]|
|---|---:|---:|
|EH_Brier_gain_over_fit_prior|-0.0911 [-0.1628,-0.0401]|not estimable|
|EH_Brier_reduction|+0.0688 [+0.0413,+0.0987]|not estimable|
|EH_fraction_MSE_reduction|+0.0519 [+0.0323,+0.0746]|not estimable|
|H_Brier_gain_over_fit_prior|-0.0827 [-0.1302,-0.0492]|not estimable|
|H_Brier_reduction|+0.0745 [+0.0464,+0.1054]|not estimable|
|H_fraction_MSE_reduction|+0.0509 [+0.0316,+0.0721]|not estimable|
|R_scaled_MSE_reduction|+0.0966 [+0.0143,+0.2103]|not estimable|

MSE reductions are product minus event; Brier gain over fit prior uses the event model and its own fitting-source prior. These are query-balanced available-label diagnostics, not calibration guarantees or proof of decision utility. Unknown futures are never negative occurrence labels; their original conservative loss remains. Partial labels keep the original available-label ADE meaning, not imputed full trajectories.

## Runtime And Reproduction

- pilot: PID31368, 74.15s, peak RSS 7.187GB.
- train: PID31956, 7954.35s, peak RSS 9.770GB.
- evaluate: PID43447, 299.39s, peak RSS 5.557GB.
- verify: PID43995, 297.89s, peak RSS 5.595GB.

Six real resumed/direct pilot states and the first full-budget refit agree exactly. All 54 paired initializations, sampling streams and preprocessing states match. The initial training attempt exited after a completed model/refit because an aggregate prior ndarray was not JSON serializable. A narrow wrapper fixed receipt serialization and resumed unchanged checkpoints. The terminal train receipt is the resumed attempt, not total wall span; see runtime_fix.md.

Use `.venv-pytorch/bin/python scripts/run_m3w_moving_joint_harm_runtime.py PHASE` for pilot/train/evaluate/verify. Completed receipts are immutable; do not rerun completed phases in place. Use `--resume` only for an interrupted unchanged phase. Native arm64 CPU4/interop1/workers0; no MPS claim.

Native observation8/prediction12 annotation-order image pixels; not raw-frame t50, verified seconds, metric, physical-safety, true3D, foundation or human-gold evidence. Stage5C and SMC remain off.
