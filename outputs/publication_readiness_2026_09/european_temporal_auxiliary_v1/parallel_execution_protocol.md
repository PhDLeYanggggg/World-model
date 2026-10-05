# Four Disjoint Execution Shards

## Decision and Scope

The verified real TRAIN pilot (job 37798513, completed 0:0) estimates 156435.58
seconds for the original serial run. Its `local_time_feasible: false` receipt is
preserved. This additive execution amendment does not waive that gate: four
disjoint jobs have an estimated 39108.90 seconds each, below the unchanged
43200-second per-job limit. These are estimates, not measured full runtimes.

The original 72 source/seed identities are assigned in frozen manifest order
as `heads[shard::4]`. Each shard retains all three arms for 18 identities,
giving 54 fits per shard and all 216 fits overall. Each receives four CPU cores,
16 GiB memory and 12 hours; at most four run concurrently. A separate one-core,
2 GiB, 20-minute join verifies completion. Queue limits remain enforced by Slurm.

## Scientific Invariants

The original optimizer, model, targets, loss weights, initialization, random
sampling, TRAIN preprocessing and 2000-update budget are unchanged. The three
100-update pilot checkpoints resume exactly. No validation or independent-role
arrays are transferred or scored. No checkpoint, threshold or model selection
is added. Partial runs are not results and do not open the seven-arm readout.
These are auxiliary neural cost heads around frozen forecasters, not new world
dynamics or JEPA/Transformer forecaster training.

Each shard has a private process lock, heartbeat and event log. A shared process
lock excludes the original sequential runner. All checkpoint saves take a common
exclusive write lock and recheck the original global 256 MiB checkpoint cap,
2 MiB atomic-write headroom and 10 GiB personal-storage reserve. Interruptions
retain the last atomic model/optimizer/RNG checkpoint; resume changes no budget.

## Submission and Completion

Code and this registration must be committed before transport. Both the four-job
array and its after-success join are submitted held. Their identities and hashes
are recorded before release. An uncertain submission/release is inspected, not
automatically resubmitted. No other project, job, environment or data is modified.

The join requires all four actual task IDs to be COMPLETED with exit 0:0, all
four frozen assignments, all 216 unique 2000-step metadata records and checksum-
verified checkpoints. It publishes the original-compatible training freeze plus
array accounting and shard provenance. The existing reader must still verify
the completed join job, full manifest, original code, every fit and checkpoint.
Scientific evaluation remains a separate operation after the complete freeze.

## Evidence Boundaries

Execution equivalence tests are synthetic engineering checks. The pilot verified
real TRAIN optimization and exact resume but did not demonstrate primary-loss or
downstream improvement. Results remain exposed source-development evidence;
independent calibration and confirmation remain closed. No deployment promotion,
metric/seconds/true-3D/foundation claim, Stage5C execution or SMC is authorized.
