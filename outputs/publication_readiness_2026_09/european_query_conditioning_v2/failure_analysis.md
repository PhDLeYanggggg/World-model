# Failure Taxonomy After Fixed-Cohort Diagnosis

1. **Numerical verifier defect: established and repaired.** Separate float32
   backward reductions differ most proportionally after cancellation. The old
   elementwise check failed two components; double precision confirms the
   decomposition. v2 uses a precision-aware vector check plus a separate double
   check on every batch. This does not repair scientific prediction quality.
2. **Selected fitting underprediction: established.** The self head predicts
   only1.44% of observed easy-harm mass on its frozen fitting selection. Peer
   context raises this to24.43% on exactly the same cohort, but signed-risk MSE
   gets worse. A pure held-scene-shift explanation is insufficient.
3. **Residual decoder conditioning: established locally.** Easy output/logit
   derivatives are near-zero on75.10%/67.95% of the common self cohort. Most
   positive easy underprediction lies there. Few derivatives are exactly zero;
   absence of hard-clipping dead regions does not ensure useful sensitivity.
4. **Gradient clipping: partial local obstruction, not sufficient explanation.**
   Unclipped disposable steps reduce total loss more in249/288 batches. Easy
   signed-risk MSE improves in82, worsens in74 and ties in132. A larger update
   is not automatically a better conditional risk estimate. Some objective
   changes are much worse without clipping. No unclipped training is certified.
5. **Loss/decision mismatch: supported as a next hypothesis, not proven root
   cause.** The weighted risk objective changes actions through the soft gate;
   it need not improve estimated moments on a fixed selected subset. About
   half the batches have conflicting data/risk directions. Our one-step result
   cannot separate this from model representation and target variability.
6. **Information insufficiency: unresolved.** Mean-pooled same-frame context
   failed, but this does not show that all neighbor representations, images or
   causal features are useless. No new feature intervention was tested here.
7. **Outcome support: unresolved.**1,567 occurrences in the main cohort have
   unknown futures. They stay unknown, not safe or dropped at inference. The
   historical support gate is not restored by these fitting-only calculations.

The previous context result lost matched known benefit and added known harm;
unknown penalties worsened it further. The new fitting diagnosis is consistent
with poor conditional risk estimation, but is not an independent replication.
Do not recertify historical Stage37 scores or reinterpret easy positive-harm
risk as net easy ADE. Do not select favorable head seeds, cohorts or batches.

No new forecaster, external-domain validation, statistical confirmation,
physical-safety guarantee, metric time/scale evidence or world-model success
was obtained. Existing result cards and reserved data roles stay unchanged.
