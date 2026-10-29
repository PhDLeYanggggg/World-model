# Reference Coverage and Relative Harm: Technical Note

This is a self-contained method appendix for an engineering ablation, not a
standalone novelty claim or a proof of model safety. Its empirical scope is
already-opened source development. Image coordinates and requested raw-frame
steps are not verified physical units.

## A. Pointwise Output Support

For observed context `x`, let `c_t(x)` be the training-selected causal baseline
at requested step `t`, and `B_t(x) >= 0` the original causal motion budget.
The raw two-dimensional output of the frozen neural architecture is `z_t`.
Its forecast construction is

```
s(z) = z / sqrt(1 + ||z||^2)
p_t = c_t + B_t s(z_t).
```

For positive radius, `s` maps finite raw vectors to the open unit ball. Hence
the construction can only produce points inside the ball centered at `c_t`.
A different causal baseline need not lie in this ball, even when it is the
reference used by the training loss. A finite penalty cannot remove that
geometric exclusion.

Let the two reference trajectories be causal CV and damping .97, both computed
from the same observed history. Define a fixed margin `rho = .05` and

```
B'_t = max(B_t, (1+rho)||r_CV,t-c_t||, (1+rho)||r_D97,t-c_t||).
```

**Pointwise coverage.** If `B'_t > 0`, each nonzero reference displacement is
strictly inside the new output ball. For either reference set
`q_t = (r_t-c_t)/B'_t`; then `||q_t|| <= 1/(1+rho) < 1`. The finite raw output

```
z_t = q_t / sqrt(1-||q_t||^2)
```

produces `c_t+B'_t s(z_t)=r_t`. If `B'_t=0`, both references equal `c_t` and
are represented exactly. For a nonzero reference, the required raw-output
norm is bounded by `1/sqrt((1+rho)^2-1)`, avoiding a boundary-only witness.

The center is also still representable (`z_t=0`). No original pointwise output
is removed because `B'_t >= B_t`; the old open ball is contained in the new
one. These facts concern the output construction for a fixed context and
arbitrary raw vector. They do **not** establish that one finite Transformer
can produce all required vectors across the dataset, that an optimizer finds
them, or that observed context identifies a future trajectory.

The implementation changes no learned parameter or inference field. It uses
neither future errors nor calibration data to set the radius. The selected
baseline remains the initial forecast; this ablation does not change the
initialization anchor to CV.

## B. Why Coverage Does Not Imply Risk Control

For a fixed forecast and reference under a nonnegative trajectory loss, write
the errors `C_i` and `R_i`. Define

```
b_i = max(R_i-C_i, 0)
h_i = max(C_i-R_i, 0)
R_i-C_i = b_i-h_i.
```

Net improvement can coexist with large positive harm because gains on some
rows cancel losses on others in the net sum. The relative positive-harm
constraint instead concerns `sum(h_i)/sum(R_i)` on the registered cohort.
The all/easy and recording-level requirements must therefore remain explicit;
they cannot be replaced by overall ADE or a favorable net easy change.

Even perfect pointwise reference coverage supplies no bound on this ratio.
For example, a model can output a different allowed trajectory when the
reference is correct. If reference error tends to zero while model error
stays positive, relative harm can grow arbitrarily. At a zero denominator,
the ratio is not estimable; inserting an epsilon would change the estimand.
These are algebraic counterexamples, not new observations from real data.

The preceding averaged harm penalty is not an exact recording-level constraint.
Its train-selected reference also differs from the final CV/damping .97 floor.
Reference coverage repairs neither issue by itself. Changing a candidate
forecast additionally invalidates learned candidate-specific benefit/harm
scores. Such scores need retraining and independent calibration before a
deployment claim.

## C. Label-Aware Geometry Diagnostic

For a known target point `y_t`, the infimum error over the independent-step
closed-ball relaxation is

```
ell*_t = max(||y_t-c_t||-B_t, 0).
```

The mean of `ell*_t` over valid future labels is a lower bound on ADE for
outputs within that ball. Unknown labels remain unknown. The ball closure
can include limiting points that require unbounded raw outputs, and this
oracle chooses separately for each target/step. It is not a trained causal
predictor, an inference input, or evidence of generalization.

This relaxation already showed that the remaining 082/51 failure was not
explained by the old hard radius: CV was representable on all78fitting rows.
The current coverage-only intervention leaves that producer unchanged and
therefore cannot be presented as a repair of that specific obstruction.

## D. Empirical Test and Scope of Attribution

The registered ablation retains the data, folds, three seeds, loss weights,
normalizers, sampling and optimization budget. Six radius-unchanged fits must
reproduce their complete control states and forecasts exactly. Three affected
fits retain matched sample/RNG histories. This separates a geometric
construction change from a new architecture, teacher or tuned threshold.

All endpoints precede the fixed readout. Repeated producer/seed errors are
averaged within locality before the3,000-draw paired locality bootstrap.
Twelve source localities, not overlapping windows, are the uncertainty units.
The uncertainty remains conditional on this exposed development cohort.

Known labels and completion/support accounting remain separate. The full
deployment policy must satisfy the original2%requirements on all required
recordings. Equal-count comparisons measure utility only; empty comparison
risk is not_estimable. Neither this algebra nor a source-development gain
establishes independent calibration, physical safety or submission readiness.
