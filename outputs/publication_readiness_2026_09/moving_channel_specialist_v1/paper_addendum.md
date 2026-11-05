# Shared Versus Channel-Specific Risk Partitions

We evaluated whether separating all-agent and easy-agent risk regressors repairs
the discrepancy between risk-prediction MSE and useful intervention. On eight
source-excluded fitting folds in three already-exposed producer contexts, two
32-tree regressors were compared with a shared 64-tree regressor. A shared
32-tree control isolated the per-output ensemble-size change. Causal features,
targets, weights, fitting-only preprocessing and admission thresholds were fixed.

The primary source-macro decision-cost reductions were -0.0337%, +0.4615% and
0%, with harmful admissions changing from 0/1/0 to 0/19/0. The fixed internal
follow-up screen failed in all three contexts. MSE changes were inconsistent,
and the largest context continued to reject every known proposal. A separate
boundary audit attributed the preceding representation model's MSE improvement
almost entirely to unchanged rejections. These results do not support output
partition sharing as a sufficient explanation for the admission failure.

All new models and cached controls were reproduced by complete refitting;
this establishes implementation reproducibility, not independent confirmation.
One producer seed, exposed development, dependent windows, partial labels,
different realized node counts and absent full-policy risk evaluation limit
interpretation. Original support and 2% safety conditions remain unchanged.
This negative component result is not a main-method success or a submission
readiness claim. It is a separate addendum, not included in the compiled PDF.
