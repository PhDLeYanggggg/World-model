# What the Strong-Base Auxiliary Experiment Tests

Let X contain only native causal features, and d(X) be the known disagreement
envelope of the frozen forecast pair. The four supervised costs are the
reference cost D, nonnegative excess cost H, and their positive-CV-easy
components D_E and H_E. The original output parameterization is retained:

    z = GELU(W X + b)
    D_hat = softplus(a(z))
    H_hat = d(X) sigmoid(h(z))
    D_E_hat = D_hat sigmoid(e_D(z))
    H_E_hat = H_hat sigmoid(e_H(z))
    p_cap = sigmoid(c(z))

This enforces nested nonnegative costs, not a calibrated risk guarantee.
The probability head shares z but never multiplies a cost output. In reported
held comparisons D and D_E are copied from the frozen reference estimator;
all four native outputs nevertheless contribute to the training objective.

The auxiliary event is I[H_E > H_inner], where H_inner comes from a teacher
that excluded the row's locality. Labels exist only on known rows with d>0.
The cost loss still uses all known rows, including d=0. Preprocessing, cost
RMS scales and sampling weights are fitting-only. Future outcomes are labels,
not inputs. No already fitted event-classifier scores are stacked into X.

For each minibatch, the objective averages four RMS-normalized squared cost
errors over all sampled rows, then adds the binary log loss averaged over
valid-event rows. The auxiliary term is zero for an event-empty batch.
Its coefficient is0 for the reconstructed original control and1 for both
true-event and within-locality-shuffled labels. All arms use identical
initial weights, all-known locality-balanced samples and2000 updates.
Total objectives across different arms are not comparable loss curves;
the cost and event components must be read separately.

## Why Reconstruction Comes First
The preceding auxiliary design changed several base choices simultaneously.
Its failure did not isolate the effect of the auxiliary task on the original
estimator. Here every no-auxiliary model must reconstruct the corresponding
original parameter state and held cost scores. Numerical tolerance and exact
bitwise agreement are recorded separately. This establishes a strong matched
comparison, not the causal importance of each earlier changed design choice.

## What a Positive Result Would and Would Not Mean
True labels beating both no auxiliary and shuffled labels on held cost MSE
would support a useful task-specific representation effect in this setting.
It would still require tail/coverage guards and all registered assignments,
not a favorable average or a selected seed. Classification-only improvement
cannot establish expected-cost improvement. Mean-cost accuracy also does not
follow from observing individual outcomes above a predicted mean.

Two-locality fitting teachers and three-locality outer teachers can define
different cap events. Rare events, limited locality support, shared-gradient
interference and magnitude shift remain possible failure modes; none is
proven merely by a negative interval. The descriptive fit/held diagnosis
does not select a loss weight, model or threshold after readout.

These are previously exposed source-development data. Three-seed averaging
and locality bootstrap account for some variability but do not manufacture
independent confirmation. There is no new forecasting or scene-joint policy
evaluation here. No deployment decision follows from this experiment alone.
All costs remain in detector pixels over native annotation steps; no physical
safety, metric, seconds-level, true3D or foundation interpretation is made.
