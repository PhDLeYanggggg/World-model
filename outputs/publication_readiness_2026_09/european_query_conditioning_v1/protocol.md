# Query-Head Conditioning: Frozen Fitting-Only Diagnostic

## Material Passport

Mode: execute and validate the next causal diagnosis; prior goal turn made real
progress (36 corrected fits and72-view negative result). No new scientific data
role, risk tolerance, forecasting horizon or deployment authorization is needed.
Existing obs8/pred12 rawstride12, positive selected easy harm/reference2% and
source provenance remain unchanged. All source localities are exposed TRAIN.

## Question

For the newly trained self/peer risk heads, does frequent gradient clipping
actually obstruct fitting the selected-risk problem, or is there still a
near-flat harm decoder on the same selected rows? Earlier general gradient
conflict and rational-map audits already exist. This extends them to the new
equal-parameter branches, not a renamed rerun or a clipping sweep.

## Fixed Work

Reuse hash-bound36 final self/peer heads and original24 source packets. Discard
internal held data before any forward; never open independent roles. Evaluate
72 complete inner-fit context/source/head-seed views, both arms on four fixed
causal cohorts: eligible, anchor-selected, self-selected and peer-selected.
The primary shared cohort is frozen self selection, not each arm's own subset.
Rows with unknown targets remain in context/inference and derivative accounting;
only supervised loss and residual calculation require known labels.

Compute actual local derivatives of all/easy harm with respect to the two raw
risk logits; reuse the earlier zero/near-zero definitions and signed residual
accounting. Verify autograd forward equals ordinary frozen inference and base
outputs remain exact. Report underprediction on the identical cohort, not only
each model's own potentially easier selection.

For each of36 heads, use8 fixed next batches drawn from the copied saved final
sampler states,16 queries per source as in training. Both arms receive exactly
the same queries. Split parameter gradients into primary moment/signed-score
loss, regret, weighted all-risk and weighted easy-risk terms; report norms,
cosines and total clipping factors. All source preprocessing, dual values,
temperature and objective scales remain frozen.

On disposable copies only, take one AdamW step with the exact saved optimizer
history under (a) original clip5, (b) no clipping. Compare same-batch objective,
components and signed-risk MSE on frozen self-selected rows. There are576
disposable steps, no saved trained candidate. Deep-copy optimizer state and
verify original model/checkpoint hashes are unchanged. An unclipped local
improvement does not establish held generalization, longer-run stability or
authorization to remove clipping in deployment.

## Execution and Decision Boundary

Commit protocol/code/tests before real data execution. Native arm64 CPU4,
inter-op1/workers0. Local computation is appropriate for small frozen heads;
CREATE is used only to stream approved TRAIN packets with byte/hash checks.
Per-view/probe receipts are atomic; explicit resume recomputes deterministically
and checks completed receipts.12h timeout is a recovery boundary, not a reason
to substitute a smaller experiment. No checkpoint duplication or large cache.

No new held scores, threshold search, forecast training, submission or deployment.
Report repeated row counts as occurrences; summed source-scaled error masses
are diagnostics, not independent samples or a pooled metric-space benchmark.
If clipping is not a consistent obstruction, do not use its frequency alone
to justify another clipping/learning-rate sweep. A concrete implementation
defect can be fixed with regression evidence; any new training comparison
requires a separately frozen matched protocol before outcome readout.

Detector-silver and local coordinates only. No metric, effective-seconds, physical
safety, human-gold, true3D or foundation claim. Stage5C and SMC remain disabled.
