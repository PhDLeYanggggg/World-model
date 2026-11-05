# Gradient Conflict Is Present, But This Repair Is Not Supported

## What Was Recomputed

All nine frozen neural heads were evaluated on complete own-B fitting
populations with their saved query-frequency weights. The audit decomposes
seven objective components and compares twelve disposable updates using the
saved AdamW moments, original learning rate, clipping and weight decay.
No updated checkpoint is persisted. No new V/H or reserved outcome is read.

| Rank context | Risk/rank cosine, offset0 | Offset1999 | Protected MSE vs full rank | Joint decision cost vs full rank |
|---|---:|---:|---|---|
| critic0 | -0.927862 | -0.925437 | Better | Worse |
| critic1 | 0.077603 | 0.148454 | Unchanged | Unchanged |
| critic2 | -0.679455 | -0.697222 | Better | Unchanged |

Two of three contexts exhibit conflict at both prespecified pair offsets.
Only one of three protected updates lowers risk MSE without increasing joint
decision cost. The prospective follow-up requires at least two of three for
both conditions, so the projection-training hypothesis is **not supported**.

The largest conflict is not the largest useful repair. In critic0, protected
MSE is0.00974223 versus0.00978255, but joint cost rises from0.00702483 to
0.00721168 and useful rejections rise from2880 to2936. In critic2 the change
in MSE is small:0.612924467 to0.612920098. Neither observation establishes
safe held-scene utility. A derivative at a final checkpoint does not identify
the complete training mechanism, and this negative screen does not rule out
all gradient methods.

## Relation To Existing Methods

Gradient projection is an existing multi-task optimization idea, not a new
M3W contribution. Yu et al. propose modifying conflicting task gradients in
[Gradient Surgery for Multi-Task Learning, NeurIPS2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).
This audit uses a narrower, one-sided projection of only the extra ranking
gradient on risk parameters. It is not a complete PCGrad reproduction or a
theorem about the project's empirical2% safety rule.

## Engineering Evidence

Initial PID22551 completed in46.3178s; exact numerical replay PID22887 in
46.2598s. An initially failing synthetic test changed a label-availability
mask and then incorrectly expected the immutable-data check to accept it.
The check correctly rejected the altered hash. The repaired test separately
checks rejection and the loss-level exclusion of unknown pairs.

The original registration remains intact. `validation_revision.json` records
the test correction and read-only replay wrapper; the scientific evaluator,
protocol, checkpoints and analysis remain unchanged. Revised verification
PID23156 reproduced the same analysis. Completed read-only replay PID23372
reproduced all nine gradient decompositions and twelve updates in47.0975s,
leaving all nine then-existing public files unchanged.

These are engineering and own-fitting results, not an independent test or a
new trained model. The earlier nine-fit negative result still stands. No
identity exemption, threshold change, empty-control pass or retrospective
promotion occurs. Original24-group primary remains NA; independent confirmation
is not_run. Native obs8/pred12 image coordinates are not metric, seconds,
true3D, foundation or human gold. Stage5C and SMC remain disabled.

## Next Decision

Do not launch a projection or generic loss-weight sweep from these results.
The next bounded model-input test checks whether fixed neighbor-slot ordering
causes avoidable sensitivity and whether a fixed permutation average actually
improves fitting error and decision cost together. Sensitivity alone will not
authorize a larger training run; its own prospective criterion must pass.
