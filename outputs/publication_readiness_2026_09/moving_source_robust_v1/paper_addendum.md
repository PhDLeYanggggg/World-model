# Source-Loss Aggregation: Negative Development Experiment

## Matched Design

We tested whether upweighting the worst training-source objective improves
neural intervention-risk learning. Three frozen relational critics were each
continued under two objectives: the original mean objective and a fixed
0.75 mean +0.25 worst-source mixture. Six models retain identical paired
initializations, causal inputs, normalization, query draws and1,000 additional
optimizer updates. Each contains68,356 parameters. Source names define
training groups only, not inference features or thresholds.

The design was committed before training. The original2% all/easy positive-
harm, signed easy-net and reference/support requirements were not modified.
Unknown outcomes retain their existing bound supervision. No identity-fallback
certificate was introduced. Both real-data resume tests and a full refit are
exact, and all fitted metrics and error attributions replay exactly.

## Findings

Relative to equal-budget continuation, the worst fitting-source objective
decreases2.149%,6.936% and1.980%. However, joint decision cost changes by
+1.465%,+8.233% and-1.160%, respectively. Harmful-admission cost changes by
+5.046%,-31.409% and+22.435%. Lower worst-source surrogate loss therefore does
not provide a consistent decision improvement in this experiment. All five
non-worst sources have higher combined fitting loss under the treatment.

The preregistered internal follow-up screen requires improvement without
worse total or harmful-admission cost in at least two contexts. It passes
zero contexts. Accordingly, additional nested-C and reserved-H readouts
are not run. No held improvement, independent confirmation or safety pass
is inferred from these training results.

## Interpretation

This is a negative test of one source-loss allocation, not evidence against
all robust learning methods. Worst-group optimization itself is established
prior work, not a contribution claimed here. The fixed convex mixture is not
the exact group-DRO algorithm or its guarantee. The methodological target
remains useful scene-level neural intervention under an unchanged risk contract.

Sources are already exposed, windows overlap, and the three contexts share
one producer seed. No sampling confidence interval or population safety claim
is supported by this fitting-only result. Image-coordinate obs8/pred12 is not
metric or seconds-level evidence, nor true3D/foundation-model evidence.

This addendum is separate from the compiled main manuscript. Its numerical
results do not revise that manuscript's PDF or certify submission readiness.
See [complete evidence](README.md), [source losses](source_losses.md) and
[failure analysis](failure_analysis.md). Stage5C and SMC remain disabled.
