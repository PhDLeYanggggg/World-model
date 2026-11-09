# Lead-Specific Missingness and Asymmetric Easy Risk

Keep the source rotation, original risk scorer, fixed proposal bank and
query-local decoder. The prior conditional-risk/query-swap policy must
replay exactly. Three fresh heads per fold/seed use 1,000 updates:

1. Conditional risk with the same initialization and risk draws as the
   original conditional head. Multiply the Huber penalty by three when
   observed easy excess is underestimated. The other targets are unchanged.
2. History-only probabilities of an observed annotation at each of the
   12 future steps, trained with binary cross entropy.
3. Candidate-specific observed benefit by lead, trained only on steps
   with known targets. This predicts a conditional gain surrogate.

For each lead combine predicted observed gain and its probability with
the exact missing-step geometry, weighted by one minus that probability.
As an ablation reuse both heads and replace per-lead probabilities with
their per-row average. This keeps the row's expected number of visible
steps, both models, observed benefit estimates and path bank fixed.
Neither readout treats annotation availability as physical observability
or changes which missing outcomes are charged by the evaluator.

Cross the flat and lead-aware readouts with the frozen and asymmetric
risk heads. The primary comparison uses common achievable counts in
each query, starting from the minimum uncapped count and shrinking the
cap deterministically if the policies cannot reach it. The same four
arms are reported at their own counts as coverage diagnostics. Compare
known and conservative utility, OLS4 utility, original recording passes,
worst signed easy change, neighbor relative displacement, and Brier
score by lead. Resample the nine sources after averaging three seeds.

Calibration selects only on analytic paths in separate calibration
sources, over the four arms and the previous six reserve/support settings.
Require all original recording checks and positive conservative
source-macro utility against CV and OLS4. Otherwise use CV. All choices
and visibility estimates freeze before held labels are used to score.
No A outcomes, independent confirmation, deployment change or new
metric/time claim.
