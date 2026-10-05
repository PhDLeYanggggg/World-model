# Cache-free CREATE checkpoint readout

This execution extension changes only checkpoint access for the already frozen
seven-arm readout. It neither changes the scientific evaluator nor reads new
validation predictions during implementation or registration.

Collection requires a successful full-training Slurm job, the exact TRAIN input
manifest, original registration, all 216 fixed-final head receipts, all checkpoint
hashes and sizes, and unchanged executable code. Partial/pilot training, duplicates,
unexpected paths, missing arms, changed weights or prior validation scoring fail
closed. Only small exact-byte metadata files are collected. Checkpoints remain in
the owned M3W experiment; no new local model cache or validation array cache.

The complete training freeze and collection inventory must be committed before
scientific arrays or model tensors are loaded for evaluation. The original
readout still reconstructs TRAIN preprocessing/input hashes, source identities,
recording splits and matched arms; replays four strong controls exactly; and
uses the existing scalar verifier and locality bootstrap without modification.

A bounded, owned read-only stream supplies each registered neural checkpoint in
memory. Its hash/length and exact identity are checked before deserialization.
The scoped checkpoint loader is restored on exceptions as well as success.
It is not a new prediction rule, normalization or checkpoint-selection mechanism.
Login nodes perform file hashing/transport only; all scientific inference remains
local. Missing connectivity stops observation, never justifies a duplicate job.

Synthetic serialization/loader parity tests are engineering evidence, not real
readout. Actual real-data collection and evaluation remain `not_run` until the
full training prerequisites are verified. Prior exposed development data stay
development data. Independent roles remain closed; no deployment promotion,
Stage5C, SMC, metric, seconds-level, human-gold, true-3D or foundation claim.
