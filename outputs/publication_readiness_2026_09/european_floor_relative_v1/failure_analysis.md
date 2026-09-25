# Failure Taxonomy

## Confirmed in the Matched Experiment

1. Reference-consistent targets are not sufficient. Both-target treatment loses
   to the CV-target control in 33/36 all-ADE views, with 20 negative and 0 positive
   paired intervals. Equal causal inputs, budget and draws rule out simple
   information or update-count advantages between these arms. Output priors
   remain target-dependent by the same initializer rule.
2. More conservative intervention is not necessarily better selection. In the
   all-event fits, CV-target switch ranges are 6.22%-41.19% / 6.64%-40.76%; both-floor
   ranges are 4.33%-32.82% / 4.59%-32.82% for batch/fitting. Reduced intervention can
   discard benefit. A matched-count experiment is still needed before claiming
   ordering improved or worsened independently of coverage.
3. Positive-easy preservation does not cover zero-reference cases. All new
   policies meet the positive-easy 2% check, yet each arm damages zero-CV cases in
   every view that contains such cases. The easy event explicitly excludes zero
   reference error; an average all-event target is not a zero-support certificate.
4. The past-motion guard is not a stop/start guard. All four zero-reference rows
   have zero latest displacement but earlier movement. Their two held-out folds
   have no zero-CV fitting examples. This newly computed post-hoc diagnostic
   suggests a concrete support-aware abstention test, not a post-hoc deployed fix.
5. Averages hide locality failures. Both-floor hard-ADE average gains remain
   positive, but worst hard localities lose 2.8055% / 1.8413%. Tail metrics remain in
   the CSV files; no locality was removed to improve an average.
6. Training completion is not learning success. All 234 heads completed 2,000
   updates; 129/144 fixed risk losses decrease. The matched target hypothesis
   still fails. Utility first/last losses use different batches and are not a
   reliable standalone convergence comparison.

## Not Identified by This Experiment

The floor-target producer sees two fitting localities, while the frozen
evaluation floor sees four. That shift could affect labels and priors, but this
experiment does not isolate it. The earlier neural-producer transport diagnosis
does not automatically establish the effect of the new floor-producer shift.
Nor does this experiment distinguish irreducible future uncertainty, detector
noise, insufficient feature support and model approximation error.

Four zero cases from one locality, each with two labels and no endpoint, do not
identify a full 12-step stationary future. A rule derived from the future zero
label would leak. Any next rule must depend only on observed history, prediction
disagreement and fitting-only support, and be evaluated with its lost-benefit
cost as well as its avoided harm. Unknown-label switches remain unscored, not
assumed correct. Long-term metric or physical safety remains unsupported.

## Next Registered Repair

Keep both normalizer modes and the frozen predictor/floor controls. Isolate a
causal support/stop-start guard from risk-target changes. Compare the guard to
equally selective controls, report reachable and discarded neural benefit, and
keep near-zero and positive-easy strata distinct. Diagnose the selected harm
with fitting-only data before choosing a calibration family. Do not relax 2%,
open confirmation data, delete failed fits or promote a seed from this table.

No new trajectory dynamics contribution, independent confirmation or deployable
neural-policy claim is established. Image-pixel raw annotation steps only;
released automatic labels are not human gold. Deployment is unchanged, and
Stage5C/SMC remain off.
