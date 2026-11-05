# Source-Robust Neural Risk Continuation

## Question

The previous feature probe failed across all three own-B contexts. Rather than
add further proximity features or change a deployment threshold, test whether
the average objective neglects high-risk training sources. This is a model
hypothesis, not an established explanation of the development failure.

## Fixed Contrast

Continue each of the three completed relational neural critics in two arms:

1. `mean_continue`: unchanged original joint objective L.
2. `source_robust`: 0.75 L + 0.25 max_s L_s, where L_s is the mean original
   query loss for training source s represented in the current complete-query
   batch. Risk MSE, decision loss and joint conjunction loss all use their
   existing definitions and coefficients within each source.

The coefficient is fixed once at0.25, not selected on development outcomes.
Both models retain the same68,356 parameters,524 input fields, source roles,
causal normalizers, initial model, query draws, AdamW learning rate1e-4,
weight decay1e-4, clip5 and batch72. No new candidate-neighbor features.
Both reset optimizer and query sampler symmetrically. Source IDs choose
training loss groups, not inference features or per-source thresholds.

Each of six fits runs1,000 additional optimizer updates from its frozen
2,000-update parent, fixed last iterate. Save every100updates and verify
minimum original query coverage. Both risk and decision encoders must update.
Unknown outcomes retain their original bound supervision. Source weighting
deliberately changes their relative contribution by source; they are never
treated as observed futures. No future endpoints, label availability or easy
truth enters inference.

## Checks And Readout

Real pilot: both arms50+50 resume versus100 uninterrupted updates on the largest
context. A full1,000-update first robust fit is independently repeated exactly.
Freeze all six fits before any new exposed nested-C evaluation. Recompute full
own-fitting metrics and per-source losses under actual query visit weights.

Internal follow-up screen, distinct from scientific safety: at least two of
three contexts must reduce worst-source original objective by>=1% relative to
mean continuation, without increasing aggregate known joint sign-cost or
false-safe cost. If it fails, do not read out additional nested-C predictions.
Retain all fits, costs, loss curves and failed results. If it passes, use the
unchanged existing readout and all/easy risk, signed easy-net, reference and
support requirements. Main contrast is robust minus mean continuation, not
robust versus an untrained or lower-budget control.

These are already exposed SOURCE/own-B/nested-C development roles, one producer
seed across three contexts, not three independent seeds or final confirmation.
Moving20, DUT, DroneCrowd and H stay closed. Original24-group primary remains
not estimable; empty support is not a pass. Original2% budgets unchanged.
No identity-fallback certification, threshold search or deployment promotion.
Native obs8/pred12 image coordinates only. Stage5C/SMC remain off.

## Method Positioning

Group robustness is not a new contribution by itself. Sagawa et al., ICLR2020,
define worst-group training objectives and show that low worst-group training
loss need not imply good worst-group generalization. Their regularization
results motivate separating fitting and held development evidence. This small
fixed convex mixture is a task-specific diagnostic, not their exact stochastic
group-DRO algorithm or its guarantee. The current experiment holds regularization
fixed to isolate aggregation; it cannot establish an optimal robustness method.

Primary source read: [paper, v2, sections2-3](https://arxiv.org/pdf/1911.08731).
Our contribution, if supported later, concerns intervention utility/harm and
scene-level decision evidence under the original risk contract, not the mere
presence of a minimax loss. No paper-ready claim follows from fitting success.

## Execution

Local native arm64 PyTorch, CPU4, interop1, workers0; checkpoint and heartbeat.
No CREATE job needed for this scale. Commit this protocol before fitting.
No data, caches or large checkpoints go to GitHub. A runtime problem requires
diagnosis and unchanged resume, not silent sample or objective downgrading.
