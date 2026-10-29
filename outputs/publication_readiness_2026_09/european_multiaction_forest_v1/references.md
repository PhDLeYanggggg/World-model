# Conventional Estimator Reference Check

Checked 29 October 2026 against primary sources. This is a strong conventional
control, not a newly proposed tree method or a reproduction of the original
paper's experiments.

- Pierre Geurts, Damien Ernst and Louis Wehenkel. *Extremely randomized trees*.
  Machine Learning 63, 3-42 (2006).
  [Publisher record and abstract](https://link.springer.com/article/10.1007/s10994-006-6226-1).
  Author/title/volume/pages and the randomized feature/cut-point ensemble
  description were checked. The full subscription article was not read here.
- [ExtraTreesRegressor, scikit-learn 1.8.0 documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html).
  This version matches the installed runtime. The initially consulted stable
  documentation is currently 1.9.1; its version is not used as the runtime claim.
  Actual full-data recovery and deterministic tree-state checks establish this
  implementation's behavior, not a documentation statement alone.

The primary forest regresses five continuous, normalized moments using squared
error and query/action weights. Its fixed fitting monitor is a deterministic
subset of up to 4,096 candidate rows, not the full objective. Neural minibatch
loss values and this monitor must not be treated as directly comparable loss
curves. Both methods use the frozen fitting-only target RMS; neither receives
future labels or label-support masks at inference.
