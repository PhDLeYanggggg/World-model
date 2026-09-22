# Fixed Cross-Objective Review and Equal-Count Control

## Material Passport

Source-only development policy experiment, registered before this policy's
readout. All four sites have already informed design. No new training, data
roles, independent calibration, confirmation, deployment or novel-method claim.

## Motivation and Hypothesis

The frozen-region policy gains4.09764% ADE but fails scene/seed easy protection.
Adaptive emphasis did not beat it or repair that failure. Earlier same-population
diagnosis found native/fraction heads often less optimistic on intermediate-head
choices, although all objectives underestimated harm on their own held choices.
Test whether using a different objective to review nominations rejects costly
decisions better than simply reducing intervention count.

This hypothesis is informed by prior development results. The new policy and
its primary comparison are fixed before evaluating their outcomes; this does
not make the underlying scenes an untouched test set.

## Fixed Policies

One frozen EqMotion forecast per query/seed. The nominator is the completed
fixed-region head, not a newly selected winner. Its original strict policy keeps
positive predicted net gain, nonzero disagreement, the past-stop veto, and
harm<=0.1*benefit. Review with the existing equal-budget native and fraction heads.
Retain a nominated query only if max(nominator harm,native harm,fraction harm)
is at most0.1*nominator benefit. No thresholds or reviewer subsets are searched.
Review uses continuous scores, not an uncertainty interval or calibrated probability.

Compare exactly three policies:

1. The unchanged full nomination policy.
2. The reviewed subset.
3. Nomination-only net-gain ranking at the reviewed count within each site/seed;
   use stable query-ID tie breaking. This is a batch diagnostic, not an online rule.

The reviewed and matched policies must have exactly the same number of choices
in every site/seed. Both remain subsets of the original nominations; empty review
means empty matched control, never forced intervention. No future label or label
validity determines decisions or eligibility. Freeze all decisions before readout.

## Primary Requirement and Readout

Primary: reviewed-minus-matched-nomination equal-site ADE gain has positive lower
3000-paired-site-bootstrap bound. Also require positive lower bound against CV,
each seed positive against CV, aggregate AND each scene/seed positive-easy
degradation<=2%, and no harmed complete exact-zero-CV outcomes. No post-readout
secondary promotion. Report accuracy sacrificed relative to full nomination,
hard/easy, tails, worst site, counts, cost bias, unknown support and missing-outcome
gain bounds. A reduced count by itself is not evidence of better risk discrimination.

Reuse approved8observed/12predicted annotation-step task, stride12, annotation
pixels, same4sites/33recordings/175756queries, seeds17/29/43. Seeds average errors,
not forecast trajectories. Data/teacher ancestry and fitting quantile definitions
remain unchanged. These four explored sites provide developmental intervals only.
Overlapping windows are not independent experimental units. No new risk tolerance.

## Prior Work Boundary

Ensemble uncertainty is established, not a new contribution from combining
heads. The original Deep Ensembles paper studies probabilistic models and proper
scoring rules; its results do not certify our maximum of shared-data cost estimates.
Reading scope: introduction and sections2.1-2.2, not a full-paper read attestation.
[Lakshminarayanan et al., NeurIPS2017](https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf).

Selective Ensembles studies consistency across randomized training runs, whereas
this control reviews costs from different objectives for one fixed forecast.
Consistency is not the same claim as baseline-relative easy-error preservation.
Scope here: author abstract only; HTML fetch failed, no theorem applicability claimed.
[Black et al., author preprint](https://arxiv.org/abs/2111.08230).

The reviewers share fitting data and upstream producers. Do not call them
statistically independent critics, calibration data, or a confidence certificate.
A positive result would motivate a later independently calibrated method, not
establish novelty or authorize deployment by itself.

## Execution

Use nativearm64 .venv-pytorch, CPU4/inter-op1,workers0. No new optimizer updates.
Hash-check existing weights, forecasts and decision archives. Store new decisions
under ignored data and publish only scalar reports/hashes. Replay all36head
endpoints, then separately check choices/counts/metric reductions. Keep runner
lock and heartbeat. Original closed roles remain closed; pending independent
data-role decisions are not bypassed. Stage5C/SMCoff; no metric/seconds/true3D claim.

```text
.venv-pytorch/bin/python scripts/run_m3w_cross_objective_review.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_cross_objective_review.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_cross_objective_review.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_cross_objective_review.py
```
