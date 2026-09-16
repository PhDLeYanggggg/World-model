# Single-Factor Robust-Loss Development Ablation

Registered decision: 2026-09-16, under the user's delegated research authority.
This is adaptive development after observing v1 seed 17/29 failures, not a new
untouched test or a retrospective claim of preregistration before all research.

The fit-only scale audit found that a small fraction of large normalized future
targets dominates squared target energy. The first development seeds rejected
the learned policies. These observations motivate a falsifiable training change,
not evidence that the change already works.

Change only the forecaster objective from coordinate MSE to coordinate
Smooth-L1 with beta=1 in the same past-normalized coordinates. Keep the original
targets, every eligible row, normalization, architecture, optimizer, update
budget, seeds, folds, baseline, risk-head losses, easy definition, policy grid
and evaluation rules unchanged. No target clipping, record deletion, new goals,
future input, revised easy budget or independent confirmation is permitted.

Run all three seeds (17/29/43), each with a full-fit forecaster, three scene-held
forecasters, identical-row OOF ridge/neural cost heads and the frozen development
comparison. Compare candidate error, oracle headroom, selected policy and easy
damage against the completed MSE run. A lower training loss is not evidence of
improvement, and numerical values of MSE and Smooth-L1 loss are not comparable.

Version the protocol and code; retain the MSE checkpoints, losses and reports.
Old code-bound artifacts must be replayed at their recorded implementation,
never accepted by editing their hashes. Source admission and historical exposure
do not change. If this one-factor repair fails, inspect the causal scale floor,
native-unit versus normalized-error sensitivity and a stronger public predictor
before any further policy-threshold search or model expansion.

No deployment upgrade, formal risk guarantee, metric/seconds claim, Stage5C
execution or SMC is authorized by this ablation.
