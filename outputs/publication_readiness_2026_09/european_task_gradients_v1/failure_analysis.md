# Failure Analysis: Distinguish Optimization from the Target

## Hypotheses Tested

1. **Broad shared-gradient conflict:** not supported at the frozen final fitting
   batch. Only 5/72 full auxiliary views have negative total-cost cosine; 68/72
   virtual auxiliary steps lower total cost more than the cost-only alternative.
2. **All-harm component interference:** locally present. Its cosine is negative
   in 42/72 full views and its virtual component loss is worse in 45/72. This
   does not establish that it caused the prior held-locality failure.
3. **Uniform easy-harm interference:** not supported. Full has 29/72 negative
   easy-harm cosines, with 30 worse and 42 better virtual component losses.
4. **A successful total objective proves useful costs:** false. The auxiliary
   experiment's matched held-locality gate remains failed; some trained output
   components are discarded in favor of the frozen original denominators.
5. **Zero shared gradient implies identical optimizer steps:** false under
   all-parameter clipping. Seven cost-only control views have different clip
   factors after the auxiliary-head gradients are included.

## Target Mismatch Is Still a Hypothesis, Not a New Result

An ordinary binary easy-membership classifier estimates P(E=1 | X). The nested
harm-fraction readout instead represents E[H E | X] / E[H | X] when the
denominator is positive. These are not generally the same quantity: severity
and easy membership can be dependent conditional on causal features.

For fitting supervision, a nonnegative H-weighted Bernoulli loss has that
harm-weighted ratio as its population optimum. This observation motivates a
possible severity-aware auxiliary comparison, not a performance guarantee.
Heavy tails can make its gradients high variance; zero-harm rows provide no
weighted membership supervision. Future H/E can be training targets or
training weights, never inference inputs. No such new fit was run here.

The previous direct harm-only head was already inferior to the original in
five of six full-input cost intervals. Dropping denominator losses is therefore
not an evidence-backed automatic fix. Earlier hurdle/ranking and conditional
severity repairs must remain in the comparator ledger rather than be forgotten.

## What This Diagnostic Cannot Resolve

It does not measure earlier optimizer states, learning on other batches,
irreducible target ambiguity, annotation quality, independent scene transport,
or a deployment safety guarantee. It does not establish that more capacity,
longer training or a particular task coefficient will repair the cost model.
Do not replace a failed primary metric with a favorable gradient count.

## Next Controlled Action

Use fitting-only, recording-aware support checks to decide whether one
harm-weighted auxiliary target is estimable. If supported, register a single
matched supervision change with unchanged source roles, model capacity,
sampling and update budget; retain ordinary auxiliary, cost-only and original
cost readouts as controls. Freeze predictions before source-held readout.
No independent selection, calibration or confirmation access is authorized
by this diagnostic itself. No threshold search or deployment advance.
