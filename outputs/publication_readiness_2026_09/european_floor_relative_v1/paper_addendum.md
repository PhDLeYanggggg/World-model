# Learning Relative to the Action Actually Replaced

## Question

An intervention controller can reject a neural prediction yet still lose accuracy
because its default action is weaker than another protected causal policy.
The preceding frozen-decision experiment isolated that default-action mismatch.
This experiment asks a different question: does training the controller against
the actual protected default improve its decisions under matched information
and compute? It does not train or claim a new trajectory forecaster.

## Construction

Let C be the constant-velocity rollout, D the protected damping-or-CV rollout,
and N the frozen neural candidate. For reference R in {C,D}, training targets are

    benefit_R = max(ADE(R) - ADE(N), 0)
    harm_R = max(ADE(N) - ADE(R), 0).

The utility head estimates both costs. Within the unchanged CV-defined event E,
the risk head estimates error mass E[1_E ADE(R) | x] and positive harm mass
E[1_E harm_R | x]. The all event contains supported rows; the easy event uses
positive CV error and the original fitting-only CV cutoff. Unknown supervision
is excluded, not assigned zero error. Events are supervision, never inference
inputs. Future positions are used only to construct fitting targets or evaluate
saved decisions.

All four utility/risk reference combinations use the same 380 causal features,
training-only CV cost scale, source-balanced draws, network widths and 2,000
updates. Outputs use a shared rollout-distance envelope spanning C-to-N and
D-to-N. Output priors follow the same target-dependent initializer; they are not
claimed to be numerically identical. Both registered ranking normalizers remain
in the comparison. No mode or seed is selected from the results.

The four fitting localities are split into the same two disjoint halves. Floor
heads trained on one half predict the other half. This supplies out-of-source D
targets. On the eight excluded development localities, D comes from the frozen
four-source policy. Thus every producer excludes the scored locality, but the
two-source to four-source producer-size shift remains a limitation. Source
exclusion does not imply identical target distributions.

At inference, each arm defaults to D. N is allowed only for observed motion,
positive predicted net benefit, positive predicted event mass, and predicted
harm no larger than 2% of predicted event mass. This is a learned pointwise
surrogate, not an independently calibrated safety certificate. A zero-error
reference case is reported separately from positive-easy percentage error.

## Evaluation and Evidence Status

Both complete decision banks are frozen before the new development readout.
Each of 36 mode/fold/seed/event groups reports five policies: the four factorial
target combinations and the preceding frozen rebased policy. ADE is evaluated
on all, easy, hard and complete-label rows. Endpoint FDE, tail errors, worst
locality, unknown-label switches and zero-reference harm remain visible.

Three seeds and 3,000 paired locality-bootstrap resamples per view describe
conditional development uncertainty. Twelve opened localities and overlapping
windows do not become independent confirmation because predictions are
cross-fitted. Multiple related views have no simultaneous-coverage guarantee.
Independent selection, calibration and confirmation roles remain closed.

Numerical results are in `results.md`, `summary_metrics.json`, the full compact
view tables, and the per-fold locality CSV files. Interpretation is in
`conclusions.md` and `failure_analysis.md`; negative arms must not be omitted.
Training loss traces measure optimization, not downstream generalization.

These are EuropeanSquares released detector tracks in image pixels, observing
8 steps and predicting 12 at raw annotation stride12. They are not verified
human-gold labels, meters or seconds, and not a revalidation of historical raw
t+50 scores. Controller gains cannot establish new neural dynamics learning,
physical safety, true 3D, a foundation model, or publication readiness.
No deployment change, latent-generative execution or SMC is authorized by this
development result. Related-work novelty and independent evidence remain
separate requirements for the main paper.
