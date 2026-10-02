# Numerical Readout Amendment

All scientific questions, strata, models, actions, risk budget, unknown-outcome
handling, bootstrap and claims remain those of diagnostic_v1. No training or
threshold search. Use all72 frozen heads; independent roles stay closed.

The v1 full replay stopped on head4 at a5.32251621e-9 mismatch against the
registered signed-score error difference. Three completed reports, the pilot,
source code and registration are preserved and hash-bound. This is not evidence
of a model failure. v1 has no completed72-head summary.

Two numerical inconsistencies were reproduced independently: v1 promoted labels
before the score transform although the old reader transformed float32 labels;
v1 multiplied scale and float32 RMS before division although the old reader
divided separately. A synthetic regression reproduces both. v2 retains the
historical label-transform dtype and sequential normalization. Prediction
hashes, action hashes and exact registered readout must still match. Tolerances
are unchanged. Report each numerical effect for each real head.

The moment attribution still uses the linear score Jacobian with float64
coefficients. It decomposes the error relative to the historical transformed
label, and is not a causal ablation. Reference predictions remain fixed.

The original streaming runner is reused through an explicit replacement of
the diagnostic module and output namespace only. The model, inference and
original evaluator are not patched. Unknown labels remain unknown. Run actual
pilot/full replay, independent scalar/bootstrap verification, then decide
whether evidence supports a separately registered change to the learning loss.

Data remain exposed-development detector-silver, image-local obs8/pred12
rawstride12. No independent confirmation, metric/seconds/physical-safety,
true3D, foundation or submission-ready claim. Stage5C and SMC remain off.
