# Real TRAIN Pilot: Numerical Checks Pass, Single-Job Time Gate Fails

Result source: **fresh_run**. CREATE job **37798513** completed **0:0** in
3 minutes 31 seconds. Its original pilot receipt, numerical code hashes and four
checkpoint hashes/sizes were freshly verified. This is one TRAIN source and
head seed 17, not the planned full three-seed comparison.

## What Ran

- No-auxiliary, row-mean auxiliary and temporal auxiliary: 100 updates each.
- A separate 100-update uninterrupted temporal replay: exactly equal to the
  interrupted/resumed temporal state, except elapsed time.
- All three arms used matching query draws and initial primary architecture.
- All losses/gradients were finite. Four checkpoints total 904,595 bytes.
- Peak Python RSS: 1,404,203,008 bytes (about 1.31 GiB).
- Validation predictions, independent roles and deployment were not touched.

These are small real-data optimizer checks, not a new forecaster, a completed
216-fit experiment, independent confirmation or evidence of scientific lift.

## Loss Evidence

The fixed TRAIN monitor is distinct from changing sampled minibatch loss.
The first logged monitor below is after step 1, not the unobserved step-0 value.

| Arm | Primary monitor at step 1 | Primary monitor at step 100 | Auxiliary at step 1 | Auxiliary at step 100 |
|---|---:|---:|---:|---:|
| No auxiliary | 0.98992604 | 0.99686575 | 2.58474541 | 2.58474541 |
| Row mean | 0.98992604 | 0.99309129 | 1.77379322 | 1.24703503 |
| Temporal | 0.98992604 | 0.99350804 | 2.58167934 | 1.86898172 |

Auxiliary training loss declines, but the primary monitor does not improve over
its first logged value in this short pilot. That is not hidden or interpreted as
downstream success. The frozen 2,000-update experiment and seven-arm readout are
still required; no model, seed or threshold is chosen from this pilot.

## Resource Gate

Measured fitting/replay phase: 108.6358 seconds. The registered deliberately
conservative extrapolation includes setup/replay overhead at every scaled step
and yields **156,435.6 seconds / 43.45 hours** for one complete sequential job.
Its `local_time_feasible` field is **false**. The original single-job submit gate
must not be bypassed or the pilot receipt edited to claim success.

The next execution design is four disjoint groups of 18 source/seed identities,
54 arm fits each, keeping every one of the original 216 fits and 2,000 updates.
The same conservative estimate divided across four jobs is 10.86 hours per job;
this is a planning estimate, not measured runtime or a reservation. A fresh
17:37 UTC quota read confirmed ordinary CPU eligibility (700 CPU per-user limit,
500 jobs / 1,000 submissions); a four-job, four-CPU-per-job design is within
those stated limits. Other running/pending work remains untouched. Queue delay
and actual per-source cost can still differ.

Before submission, this needs a separately frozen execution amendment, disjoint
work assignments, exact numerical-equivalence checks, per-shard checkpoint and
heartbeat handling, serialized writes under the existing global storage caps,
and an all-216 completion join. Partial shards must never authorize readout.
No parallel full-training job has yet been submitted.

Stage5C and SMC remain off. Current evidence stays exposed-development,
dataset-local/pixel-space with its existing time and label limitations.
