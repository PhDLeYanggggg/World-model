# Frozen Harm-Head Jacobian Diagnosis

## Fixed Scope Before Computation

Fresh inference and autograd only on all72 INNER-FIT views: six source contexts,
four sources, three head seeds. Compare the already frozen isolated-odds and
full-range finals on identical inputs. No optimizer, model selection, thresholds,
calibration or deployment changes. No outer development, independent selection,
calibration or confirmation role. Original TRAIN packet files contain both
internal partitions; transport verifies whole-file hashes, but discard the held
partition immediately and do not forward or analyze it in this diagnosis.

Fit-only preprocessing, checkpoint hashes and original prediction identities
must reproduce. Inspect derivatives of all/easy harm with respect to the two
risk logits. The network is row-local; summed-output autograd yields each row's
Jacobian. Preserve original batches4096 to compare prediction hashes exactly.
Use frozen float32 network/output and float64 arithmetic as in training.

Account on four causal cohorts: eligible, anchor-selected, odds-selected and
full-range-selected. These fixed cohorts are shared across arms. Unknown labels
remain in inference and counts. Outcome labels enter only posthoc accounting.
Report exact-zero and near-zero (L1 derivative <=1e-6 times causal cap) rates,
positive harm-underprediction allocated to those rows, hard clip indicators,
zero-cap counts and observed-label excess above the frozen cap. These are not
probabilities, conditional-mean proofs, causal effects, or deployment guarantees.
Repeated rows across sources/seeds/contexts are occurrences, not independent n.

If most fit own-selected easy-harm underprediction lies at exact-zero Jacobian
and hard lower clipping accounts for it, that supports testing one smooth
bounded replacement. It does not prove generalization or make a passing gate.
Otherwise investigate selected-risk support/calibration rather than automatically
adding another mapping. No repair is selected using new held/test outcomes.

## Operational Contract

Commit code/tests/protocol/registration before execution. CPU4, inter-op1,
num_workers0, native arm64 environment; CREATE read-only approved TRAIN streaming,
no scheduler submission. Atomic immutable per-view receipts, single-writer lock,
heartbeat, explicit resume,12h limit. Completed outputs immutable. All72 views
required, no hidden subsample. Public aggregate JSON only; no rows/cache/checkpoints
or raw images/videos in Git. Hash-verified readout receipt replay in check mode.

Obs8/pred12, raw stride12, detector tracks, pixel/dataset-local coordinates.
No seconds, metric, human gold, physical safety, true3D or foundation claim.
Historical Stage35/37/43/44 results remain exploratory. Stage5C and SMC disabled.
