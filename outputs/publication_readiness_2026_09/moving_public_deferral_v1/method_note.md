# Why This Control Does Not Change The Evaluation Contract

## Scientific role

The existing point head estimates benefit, harm and reference costs, then applies
the original risk rules. The public deferral head instead learns which frozen
predictor minimizes expected cost using a cost-sensitive surrogate. Both receive
the same past-only information. For the guarded comparison, both proposals are
then judged by the same independently fitted point critic. This separates two
practical questions: whether cost-sensitive routing helps, and whether the
existing critic can safely admit the proposals it produces. It does not prove
the cause of any performance difference in isolation.

An earlier public deferral adaptation used European source data. This experiment
does not relabel that result as new; it fits new routers on the original Moving
inner-C source partitions with source-excluded validation.

## Equation check

The [original two-stage formulation](https://proceedings.mlr.press/v235/mao24d.html)
treats the base predictor's loss as the cost of action0 and each expert loss as
the corresponding other action's cost. For a fixed row, the coefficient on
the logistic loss for action k is the sum of all other legal action costs.
Minimizing its conditional expectation makes the largest ideal logit correspond
to the smallest expected cost. This is not ordinary oracle-class cross entropy.

The implementation uses log base2 as in the logistic surrogate. Supervision is
detached from gradients, and rows without observed future labels do not acquire
invented zero costs. The sole legal floor on a no-choice row contributes zero
routing loss. The floor is legal for routing but is not counted as an intervention.

## Important limitations fixed before readout

- Available-label ADE is not the missing full-trajectory error.
- The original point learner includes conservative envelope supervision for
  wholly unknown futures. The public surrogate excludes those outcome labels.
  The causal row roster and sampler are matched, not the effective supervised
  loss support. Any difference therefore cannot be attributed to the algebra of
  the loss alone.
- Causal per-row scaling preserves the pointwise conditional cost ordering, but
  changes the weighting of different inputs for a finite model.
- A softmax distribution here is a surrogate score, not the probability that an
  action is best or safe. Matching uses its logit margin only as a ranking.
  With K legal actions and positive total conditional cost S, the unrestricted
  optimum is p_k=(S-E[c_k])/((K-1)S). Consequently p_k is at most1/(K-1).
  With all seven actions legal, even the ideal score is at most1/6; applying a
  generic0.5confidence threshold would reject every ideal decision. This is an
  algebraic property of the surrogate, not an observed performance result.
- Different output parameterizations and decision rules prevent interpreting
  this as an otherwise identical loss-only ablation.
- Query-level context, legal-action masks and source-balanced sampling are task
  adaptations. The paper's bounded-loss guarantees do not automatically apply.
- The public surrogate optimizes predictive cost, not a2%positive-harm ratio.
  A lower average error cannot establish the separate risk requirement.
- Source-excluded validation inside exposed SOURCE is not a new independent test.

The original full-policy risk, easy-net and support checks are retained. Empty
matched controls remain not estimable. No identity fallback certification is
introduced. All previous failures remain recorded. Reserved confirmation data,
Stage5C execution and SMC are not opened by this experiment.
