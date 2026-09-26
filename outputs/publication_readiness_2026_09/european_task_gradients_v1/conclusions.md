# Task Gradients: Broad Optimizer Interference Is Not Supported

## What Was Run

Registration `b12c9cc9` was pushed before the calculations. All 288 frozen
heads were examined on their original supported fitting batches, with 576
disposable AdamW updates. No checkpoint, policy or threshold was updated.
The main calculation took 156.665 seconds excluding registration, plotting
and subsequent verification. Results are fresh_run; source models and
fitting labels are cached_verified. New held-out outcome readout is not_run.

## Main Finding

The preregistered trigger for a gradient-conflict repair is false.

| Auxiliary head diagnostic, 72 dependent fitting views per pair | Full | Motion only |
|---|---:|---:|
| Negative BCE / total-cost shared gradient cosine | 5 | 2 |
| Median total-cost cosine | +0.1870 | +0.2113 |
| Auxiliary virtual total cost better than cost-only step | 68 | 70 |
| Auxiliary virtual total cost worse | 4 | 2 |
| Auxiliary virtual all-harm component worse | 45 | 46 |
| Auxiliary virtual easy-harm component worse / better / tied | 30 / 42 / 0 | 33 / 37 / 2 |

This does not support a dominant final-batch conflict between classification
and the *total* cost objective. It also does not rescue auxiliary training:
the previous strong held-locality cost gate failed, and the damage components
remain mixed. Local total loss and retained risk outputs are different things.
All four cost components are trained, but only all-harm and easy-harm outputs
are replaced in the paired readout; the original denominators are retained.

The [figure](gradient_optimizer.svg) compares gradient geometry with actual
saved-momentum AdamW counterfactuals. Positive y means the auxiliary step is
worse. Neither axis is a held-out trajectory improvement.

## Controls and Caveats

The cost-only controls have identically zero shared membership gradient;
their cosines are undefined. Three full and four motion-only control views
still have different virtual costs: nonzero membership-head gradients alter
global clipping, even though they do not directly enter the shared encoder.
All seven cases have distinct, active clip factors. This is an optimizer
coupling, not evidence of a shared membership gradient in those controls.

These are dependent fitting views over four localities per assignment, not
288 independent scientific replicates. One fixed batch at step 2,000 cannot
identify early-training interference, a global cause of failure, or held-scene
generalization. Improvements relative to a disposable cost-only step do not
mean the completed auxiliary model beats the original trained model.

## Decision

Do not launch PCGrad or a task-weight sweep on this evidence. Keep the prior
policy unchanged and preserve the failed auxiliary gate. The next controlled
question is whether the auxiliary target is aligned with the *magnitude* of
harm, not merely whether an example belongs to the easy class. Before fitting,
measure fitting-only harm support and lock one severity-aware supervision
comparison with the same original strong comparator. Ordinary harm-only
regression already failed in the conditional-cost study and should not be
reintroduced as an untested idea.

Detector-derived pixels, 8 observed / 12 predicted annotation steps. No metric,
seconds, human-gold, physical-safety, true3D, foundation, new deployment or
submission-ready claim. Stage5C and SMC remain off. The research goal is ongoing.
