# Failure Analysis

| Proposed explanation | Evidence in this experiment | Verdict |
|---|---|---|
| Prediction-disagreement support alone caused the loss | History-only rejections have negative net value in all 36 views per target family | Incomplete explanation; observed-history support is also misaligned with utility |
| Same-source overlap is the only bad requirement | Dropping the overlap requirement usually helps relative to joint, but separate marginal support still loses all-ADE in every view | Extra conservatism costs benefit, but is not the whole failure |
| A broader support rule reliably ranks the right agents | Same-current-frame equal-count controls produce mixed positive/negative intervals | No stable allocation advantage established |
| Stopping defect returned | Every factor is a subset of the frozen stop controller; all supported zero-CV cases remain unharmed | No observed regression, with very limited partial-label support |
| Better all-ADE implies every locality is safe | Floor-target stop still loses 2.80551% on its worst hard locality | False; locality failures must remain visible |
| Removing a support axis fixes cross-domain representation | No forecaster or producer was retrained or matched in this experiment | Not tested; cannot infer transport causality |

## The Measured Mechanism

The support rule estimates whether motion/prediction features resemble common fitting-source ranges. The deployment objective is different: does replacing the floor forecast improve expected error enough to justify its risk? A rare motion can be exactly where the neural forecast is useful. The exact removal ledger shows that rejecting such rows often gives up more benefit than the errors it avoids.

This is not just a loss in aggregate caused by one normalization setting. Neither normalization establishes a positive factor-guard all-ADE interval against stop. History-only rejection has negative net value in every view under both target families. At the same time, some both-marginal failures are beneficial to reject, and some positive-easy equal-count comparisons favor support filtering. The rule is not uniformly harmful on every row or slice; it fails as a reliable full-policy repair.

## Unresolved Producer Confound

The cached system uses two-source inner cross-fitted producers for fitting features/targets and four-source producers for held-development forecasts. Both exclude the source being scored, but they need not generate identically distributed disagreement or utility signals. The factorization leaves this mechanism unchanged. It therefore cannot tell whether producer mismatch, missing causal context, weak risk learning or their combination explains the remaining controller limitation.

A valid follow-up must bind each row to a source-excluded producer, compare matched producer regimes at equal training budget, and keep the predicted opportunity set explicit. Changing both forecast quality and the control rule without these anchors would confound improved dynamics with improved selection.

## Data And Claim Limits

Twelve opened EuropeanSquares localities and overlapping indexed histories provide development evidence, not a new independent test. Per-fold intervals resample eight source localities. Complete-label and partial-label results cannot be silently exchanged. Detector track errors, unknown future labels and four short zero-CV examples constrain safety conclusions. No domain-general transport, interaction-conflict reduction, independent calibrated risk, metric timing or physical validity result is established here.

Deployment stays unchanged. No new neural training in this experiment, no Stage5C execution, no SMC, no true-3D/foundation claim.
