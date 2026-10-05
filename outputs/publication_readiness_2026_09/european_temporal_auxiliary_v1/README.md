# Matched Temporal Auxiliary: Trainer and Readout Ready, Real Fit Not Run

## Material Passport

- Mode: experiment implementation and engineering verification.
- Evidence: native Torch synthetic optimization/resume tests; one real TRAIN
  forward/backward check; storage and bounded remote-access observations.
- Not evidence: new trained performance, safer decisions or independent testing.
- Registration: `f8606491`; original targets, roles, primary loss and risk budget
  unchanged. [Protocol](protocol.md), [run status](run_status.json).

## What Changed

The new trainer has three matched arms: no auxiliary, row-mean auxiliary and
temporal auxiliary. They share the existing five-moment nonlinear cost decoder,
width-32 encoder, initialization, query draws, optimizer and fixed update budget.
The only loss difference is the registered auxiliary term. Auxiliary heads have
the same shape and initial parameters. No model receives future labels or masks
at inference. These are cost heads around fixed trajectory predictors, not new
trajectory-generating Transformer/JEPA models.

Resume fingerprints cover features, primary/auxiliary targets, source recording
IDs, preprocessing, optimizer settings and registration identity. Tests prove
the no-auxiliary arm exactly reproduces original primary training, all three arms
share draws, and interrupted/resumed training equals uninterrupted training.
An interrupt never replaces a valid checkpoint with a potentially half-applied
AdamW step. The last atomic checkpoint is replayed instead.

## What Was Verified

28 focused tests pass in 3.64 seconds: 16 new training/storage checks and 12
existing temporal-target/reporting checks. The full legacy suite was not run.
Optimizer and checkpoint tests use synthetic fixtures; they do not establish
real-data loss convergence or generalization.

The real-data inspection reconstructed TRAIN preprocessing for one source/head,
verified its original checkpoint and producer-chain/recording partition, and
checked that temporal labels reproduce the original primary signed/reference
labels. This first source has 109,454 TRAIN rows, including 106,960 with known
supervision. One query-balanced draw has 141 rows. All three arms perform finite
backward passes with identical initial primary loss 0.6975652575. Different
auxiliary losses reflect different targets, not comparative learning gains.
There were **zero optimizer updates**, no new real checkpoint and no validation
prediction scoring. [Exact inspection receipt](input_check.json).

## Why Formal Training Did Not Start

The real pilot command stopped at the registered storage guard, before loading
the scientific dataset or fitting. At admission, free disk was 9,220,136,960 bytes;
the inherited 10 GiB reserve plus 256 MiB checkpoint allowance and atomic headroom
requires 11,007,950,848 bytes. The shortfall was 1,787,813,888 bytes, about 1.67 GiB.
It is a resource admission failure, not a neural training crash or negative model
result. The reserve was not relaxed and unrelated files were not deleted.

A synthetic 380-feature, width-32, eight-update checkpoint measured 112,713 bytes.
Extrapolating to 216 heads plus one replay gives 24.46 MB, but this is only a
storage estimate; longer traces and real inputs can change it. It does not
override the inherited reserve or justify reporting the full training as run.

CREATE's authorized read-only owner-file query timed out after 20.02 seconds.
Current M3W directory ownership, quota and scheduler state remain unobserved.
The latest `simulation model` chat also reports unstable CREATE access; no
simulation environment, dataset or job was modified. Old queue observations are
not used to infer current M3W availability. No job was submitted or restarted.

## Remaining Work and Decisions

1. Restore checkpoint-safe local space or stable authorized CREATE access. About
   2 GiB additional local free space would cover this observed shortfall; recheck
   immediately before fitting because free space changes.
2. Run the real three-arm pilot and exact interrupted-resume replay. Use its
   measured runtime and memory to place the full fixed-budget 216-fit experiment.
3. Run the now-implemented [fixed readout](readout/protocol.md) only after the
   complete training freeze is committed. It retains all four strong controls,
   full and original-selected error, full/query-matched utility, unknown-label
   completion and the original 2% selected easy-risk rule. Twenty-eight new
   readout tests pass, 56 with the existing trainer/target tests. Its scalar
   verifier separately checks actions, weights, ratios, paired completion and
   locality bootstrap. Synthetic integration is not real model evaluation.
4. Only a controlled utility/risk improvement can support further transfer.
   Auxiliary loss or temporal MSE improvement alone is insufficient.

No deployment or CCF-A readiness promotion. The long-term goal remains active;
the latest scientific result is still the preceding diagnostic, not this code
change. Independent roles stay closed. Image-local detector-silver rawstride12
obs8/pred12 only; no metric, seconds, physical-safety, true-3D or foundation claim.
Stage5C and SMC remain off.

See [Chinese run and recovery instructions](reproduction_zh.md).

The October 5 12:05 UTC read-only CREATE retry timed out after 20.05 seconds;
remote ownership, quota and scheduler state were not observed. The simulation
chat has no new completion message. No remote write, scientific execution or
job operation occurred. Resource recovery is still needed for real fitting.
