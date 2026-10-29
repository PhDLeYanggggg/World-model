# Fit-Only Repair Verification

After all18 fixed rational fits freeze, recompute actual harm/easy-harm logit
Jacobians on all72 fit views for rational and preceding full-range heads.
Use identical eligible, anchor-selected, full-range-selected and rational-selected
cohorts across both arms. No held inference, threshold, weight or checkpoint
choice. Whole original TRAIN packet transport may include internal held rows;
discard them before forwarding. No independent roles. Fresh prediction hashes
must match frozen full-range fit readouts and ordinary rational forward.

Report counts, unknowns, exact-zero/near-zero J, and normalized positive harm
underprediction at those derivatives. Do not use tanh/clip flags from the old
mapping to characterize the rational mapping. This check diagnoses whether
the targeted obstruction was removed, not downstream success. Float32 gradient
underflow and boundary anchors may remain. It cannot change the registered
training/readout gate. Labels only in posthoc accounting, never input.

Register code, tests, protocol and training-freeze hash before execution.
Atomic per-view receipts, explicit resume, single writer, heartbeat and12h limit.
Native arm64 CPU4/workers0. Public aggregates only, no private data/model upload.
No metric, seconds, human gold, physical safety, true3D or foundation claims.
Stage5C and SMC disabled. Same exposed TRAIN-internal evidence status throughout.
