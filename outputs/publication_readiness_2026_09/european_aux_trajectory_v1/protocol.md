# Registered Auxiliary Training-Trajectory Instrumentation

## Material Passport

Previous goal turn: progress, a verified negative projection-intervention
experiment at 3e81d8c735f36acfe8145d93a32c97643d420273. Current new question:
when does the cap-event auxiliary cost deficit emerge during training?
No model, checkpoint, threshold or policy selection is part of this study.

## Fixed Design

Reconstruct all 432 original strong-base training trajectories: 144 views,
three seeds, full/motion inputs, cost-only/true-cap/shuffled-cap arms. Retain
the original 383 features, GELU64 architecture, original initialization,
four-cost objective, auxiliary coefficient1, all-known-row support,
locality-balanced sampler, AdamW lr.0003, weight decay.0001 and clip5.
Run all 2000 updates per head, 864000 updates total. This is fresh instrumented
reconstruction of already trained models, not a new model variant or
independent retraining replication.

Save model/optimizer snapshots at steps200,600,1000,1400,2000. Do not select
the best-looking time point. Exact final model, AdamW state, sampler/torch RNG,
sample counts, preprocessing, targets and numerical trace must equal the
original checkpoint. Runtime and new provenance IDs are allowed to differ.
Compare every prescribed numerical trace point exactly. The original first
pilot additionally logged step100; retain and disclose that old-only trace
point rather than falsely treating a logging difference as a parameter change.
If equality fails, stop interpretation and diagnose; do not accept approximate
results as evidence about the original trajectory.

At each of2160 snapshots, evaluate all supported fitting rows in chunks, by
locality and by fixed fitting severity strata. Record four-cost normalized
MSE, easy-harm normalized MSE/SSE and true-event BCE. Severity bins are zero,
positive[0,50],(50,90],(90,99],(99,100] using positive easy-harm target quantiles
from original equal-locality fitting weights. Unknown rows are never treated
as zero costs. Ties can make bins empty: report unsupported, never invent
support. Verify that bin counts and SSE sum to the positive stratum.

At the same snapshots record task gradients on the prior diagnostic's eight
fixed fitting-only, site-balanced 256-row batches. Record actual-training and
true-event BCE gradients against cost4 and positive-envelope easy harm on
shared parameters. No virtual projection or new optimizer intervention here.
The cost-only model's auxiliary gradient can be undefined; it is not agreement.

## Analysis

All five times, all three arms, both feature families and all six assignments
remain visible. Compare true vs cost-only and true vs shuffled fitting loss
trajectories. Average seed and excluded-locality context within fitting
locality; report3000 paired resamples of the four localities. These are
descriptive fitting intervals from overlapping contexts/models, not
independent validation or generalization. Empty strata remain unsupported.

Report the first recorded time with a negative interval, whether it persists,
and where severity-bin errors accumulate. These are fixed-grid observations,
not discovery of the exact onset or proof of causation. Plot gradient conflict
over time alongside loss differences; association does not prove interference.
Do not select a checkpoint from these fitting results. A subsequent repair
must be separately registered and validation-selected, retaining strong controls.

Earlier membership-gradient and severity-transport negative findings are
reused, not relabeled fresh. This new observation is time evolution on the
strong cap-event task. Do not repeat generic gradient-projection or severity
weight grids without a specific new prediction supported by this trajectory.

## Data Roles and Claims

Only exposed source fitting data are used. Nested row-locality-excluded
targets remain loss/diagnostic labels, never input features. No future endpoint,
central official velocity, test-endpoint goals or test normalization is added.
Shared source containers are not filesystem-blinded; target arrays are indexed
by fitting IDs. Independent selection, reserved calibration and confirmation
remain unopened. No fresh outer outcome readout or deployment change.

Obs8/pred12 native annotation steps, detector pixels. No metric, seconds,
physical safety, human-gold, true3D or foundation claim. Stage5C and SMC off.

Local native arm64 CPU4/interop1/workers0. Start with one real head through
step200 for runtime estimation, then resume full fixed budgets. Preserve10GiB
free disk. Atomic checkpoints every200 steps, per-snapshot receipts, heartbeat
and an exclusive file lock. No automatic downgrade for slowness. Keep original
checkpoints and unrelated Git changes; submit no raw data or checkpoint files.
