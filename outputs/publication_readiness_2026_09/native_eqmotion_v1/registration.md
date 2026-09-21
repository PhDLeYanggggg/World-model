# Native-Loss EqMotion K=1 Comparator

Registered 2026-09-21 before any new real fit or score readout. Twelve fits:
four explored SDD source sites times seeds17/29/43. No new source roles. The
original val/test/main/external/bookstore roles stay closed. This is development
evidence, not independent confirmation or a new M3W method claim.

## Question

Does the observed native-coordinate Transformer advantage over CV persist against
a public multi-agent forecast family trained with the same loss and draw budget?
The old EqMotion experiment used another objective and population; it is not a
matched native-loss comparator. No direction of advantage is assumed here.

Use the locally pinned author EqMotion core, commit
`5aec2e0b61c511fa93a24138dd90da59a089084b`, MIT, verified byte by byte.
The existing adapter emits head0 fixed before training, K=1, without a future
label-dependent choice. Skipping unused parallel heads preserves the selected
head arithmetic, as covered by existing tests. This is **not** reproduction of
published minADE20/minFDE20, nor a claim to beat the paper's benchmark.
Author paper: [Xu et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html).
Original proceedings metadata and abstract were checked again on2026-09-21.

## Fixed Training

Use all175,756past-eligible target queries from33recordings of the four approved
source-training sites, excluding the held site from every gradient update and
loss normalizer. Existing unknown/partial label masks are retained; changing
supervision completeness would invalidate matching to the frozen Transformer.
The same complete aligned neighbor eligibility, observed-context conditioning,
eight observed/twelve requested steps and SDD stride12 apply. Coordinates are
annotation pixels with unverified physical scale/time. Future endpoints and
validity masks remain supervision/evaluation only.

Architecture: author hidden64/channels64/layers4, fixed head0. No new bounded
output wrapper is added to EqMotion. The existing Transformer has a motion-bound
wrapper. This compares two fixed prediction systems, **not** an isolated test
of equivariance, equal parameter count, or equal FLOPs. Both get4,000updates,
batch64, identical scene-uniform draws, AdamWlr0.0003/decay0.0001, cosine minimum
ratio0.01 and gradient clip5. Compare per-row draw counts, fitting IDs and loss
factors against the saved Transformer training checkpoint for every fold/seed.

Native ADE training weights, training-CV scene normalizers and importance
correction are exactly the existing native-coordinate objective. No held labels
are used for fitting normalization, early stopping or hyperparameters. Final
fixed-budget endpoints are used. Record checkpoint, optimizer/random state,
draws, finite-loss trace and PID heartbeat. CPU4/inter-op1/workers0. First100
updates of one registered fit provide the real timing smoke; resume the same fit
without changing the budget. Slow progress is not failure.

## Fixed Readout

Complete all twelve endpoints before new held-source inference. Save all
predictions and identities before computing metrics. Compare EqMotion, the
cached-verified native Transformer, CV and constant position on identical rows.
No candidate selection or fallback is trained in this experiment.

Primary: equal-site native ADE gain against CV and paired EqMotion-minus-
Transformer ADE-gain contrast,3,000physical-site bootstrap draws after averaging
seed errors. Keep each seed, site, native ADE/FDE, complete/static/moving/hard,
positive-easy, exact-zero-CV damage, unknown-label coverage and error tails.
Hard is training-CVq75; positive-easy is training positive-CVq25. These masks
are outcome-defined diagnostics, never inference features. Exact-zero CV
protection remains unchanged; no epsilon or pixel tolerance. Unknown outcomes
do not become zero error or safe cases.

A positive result would fill a comparator gap, not establish M3W novelty,
independent safety or publication readiness. Negative results remain visible.
No per-site model selection, threshold search, teacher distillation, new risk
calibration, Stage5C or SMC. Local feasibility is assessed from the real pilot;
CREATE is used only if justified and accessible, not as a substitute for verified
completion. A runtime correction after registration requires a new identity.
