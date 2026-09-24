# European Source Studies: Execution and Reproduction

## Recorded Execution

| Phase | Execution | Status |
|---|---|---|
| Fixed forecast matrix | PID17317; 18 fits, 72,000 updates; final readout2026-09-24T20:47:49Z | fresh_run, exit0 |
| Nested cost heads and controls | PID20927; 9 ridge +9 neural fits; 18,000 neural updates; readout20:50:48Z | fresh_run, exit0 |
| Full forecast metric reproduction | PID20906; final20:48:41Z | cached_verified, exit0 |
| Full cost/control metric reproduction | PID21213; final20:53:15Z | cached_verified, exit0 |
| Forecast checkpoint replay | 18 models,128 excluded rows per model, exact | fresh inference on cached_verified inputs, exit0 |
| Cost-head checkpoint replay | 18 heads,4,096 excluded rows per head, exact | fresh inference on cached_verified inputs, exit0 |
| Same-count decision audit | all1,152queries per seed/head checked; three contrasts | fresh arithmetic on cached_verified decisions, exit0 |
| Relevant regression suite | 127passed,2.10seconds | engineering validation, not model evidence |

Native arm64 PyTorch2.12.0, CPU4/inter-op1, workers0. No CUDA/MPS inference is
claimed. No CREATE submission or resource probing. Fixed budgets are complete;
no quick/medium relabelling, early stopping or test-selected checkpoint.

The forecast receipts sum to1,915.68fitting seconds, excluding cache creation,
inference and verification. UTC logs contain a substantial elapsed-clock gap
within complement1_seed29; its recorded fitting duration is77.03seconds.
The same process subsequently completes. The cause of the gap was not verified;
do not claim either OpenMP deadlock or continuous accelerator utilization.
Recorded timing is not a hardware benchmark or measured CPU/GPU device time.

## Frozen Scientific Identity

Registrations/configurations, source receipts, fold assignments, code identities,
checkpoints and prediction arrays are retained. Completed identity-bound code
is not edited to improve old results. The cost runner's receipt-path fix predates
its first launch/identity; a regression test checks both valid and changed bytes.
It changes no model choice, cost target, risk limit or row selection.

Inputs come only from the twelve admitted source-training localities. Future
targets and validity masks are used only for training loss and masked evaluation.
Unknown labels remain in inference populations. Raw-source admission and complete
raw replay are older verified steps, not falsely described as repeated by these
new training runs. Reserved source roles and DroneCrowd remain closed.

## Reproduce Without New Training

Run from the repository root with its native arm64 environment:

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_source_forecast.py --verify
.venv-pytorch/bin/python scripts/run_m3w_european_source_intervention.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_european_source_models.py --forecast
.venv-pytorch/bin/python scripts/verify_m3w_european_source_models.py --intervention
.venv-pytorch/bin/python scripts/audit_m3w_european_joint_contrasts.py
.venv-pytorch/bin/python scripts/report_m3w_european_source_studies.py --forecast --intervention
.venv-pytorch/bin/python scripts/plot_m3w_european_source_studies.py
```

The first two replay all metrics from verified caches. The next two perform
sampled new checkpoint inference, not full retraining. The contrast audit checks
actual decisions rather than trusting matching flags. Hash/identity disagreement
must stop replay; do not overwrite a receipt to make it pass.

To recover an interrupted fixed matrix, use the corresponding runner's
`--train --resume --evaluate` path only when no other process holds its lock.
Completed models are verified, not silently refitted. A new scientific experiment
requires a new versioned registration and directory. No new data role is opened
by these commands.

## Loss Interpretation

Forecast loss is masked native ADE divided by a fitting-site reference scale,
with the registered sampling correction. It is not validation loss. A legacy
trace key called `mean_past_normalized_ADE` actually contains unweighted native
minibatch ADE in this path; its name is not a normalization claim. Neural cost
loss is continuous positive benefit/harm regression with4xunderestimation harm
weight. Ridge has no gradient-loss curve. All logged curves are retained; noisy
loss does not establish convergence or generalization.

## Publication and Storage Boundaries

Only code, configurations, reports, aggregate metrics and generated scientific
vector plots are intended for Git. Source archives, feature/history/latent
caches, checkpoints, arrays and media remain private. The unrelated staged
worktree entries are excluded from this commit. Formal submission, independent
calibration, confirmation, Stage5C and SMC were not executed.
