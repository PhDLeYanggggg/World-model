# Development Failure Localization: Draft Addendum

To distinguish marginal support failure from conditional risk miscalibration,
we froze the forecasting, fallback, utility and risk models and stratified
their nested source-development predictions using fitting-source quartiles of
eight causal descriptors. We separately inspected future-label completeness
and realized reference error as evaluation-only strata. All108paired groups
and twelve localities were retained. We report unadjusted locality-bootstrap
intervals and do not interpret them as independent confirmation.

Observed selected harm persisted inside the existing input-radius guard. Among
complete-label examples, the locality-average pooled harm ratio was2.27%
[1.64%,2.92%]; these examples accounted for61.28% [52.58%,68.18%] of observed
harm. The high rollout-disagreement stratum had5.27% [2.68%,8.07%] selected
harm, versus0.96% [0.74%,1.20%] in the low stratum. Useful oracle actions
remained inside the causal eligibility pool, representing16.27%
[13.24%,19.07%] of reference error. This is an upper-bound diagnostic, not
achieved prediction performance.

The evidence motivates explicitly motion-conditioned risk estimation, while
ruling out an explanation based solely on out-of-support inputs or incomplete
future labels. It does not establish that the descriptors cause improved
selection, that subgroup filtering is safe, or that another learned model will
succeed. Detector noise and source composition remain plausible confounders.
The underlying selection experiment still fails its registered risk and
equal-count ranking criteria. A matched, preregistered feature ablation and
independent-source calibration/confirmation are required before any main
method or deployment claim.

This is development-only image-local detector-silver evidence, obs8/pred12
rawstride12. No metric, seconds, physical-safety, true3D or foundation claim.
No independent scenes were opened; no Stage5C or SMC was executed.
