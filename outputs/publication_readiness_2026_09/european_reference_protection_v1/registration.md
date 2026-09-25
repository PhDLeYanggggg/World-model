# Reference Protection Versus Shared Continuation

## Material Passport
Registered before new training and source-C outcome readout. Parent study
14aa2255 is cached_verified: all 62 artifacts and 85 implementation bindings,
plus the matched-transport audit source, are checked. No new forecaster is
trained. Six opened selection localities are not evaluated; 12 reserved
calibration and six confirmation localities remain closed.

## Question and Falsifiable Hypothesis
The sampling repair improved full-B easy-harm MSE in 13/18 settings, but harmed
both reference-mass components in all 18 and improved C easy-harm MSE in only
4/18. Exposure alone is insufficient. Does isolating harm learning while
preserving the reference moments improve held-source risk and accuracy over
equal-budget shared continuation?

This tests reference preservation, not a claim that gradient conflict has
already been proved. A shared-hidden-layer tradeoff, optimization budget and
transport error are possible explanations. This experiment cannot distinguish
all causes or establish a novel multi-task architecture.

## Two Warm-Started Arms
Both start from the same preceding uniform mean head after its fixed 2,000
updates, not from the failed sampler head. Both reset AdamW, use its unchanged
B-only feature normalization and RMS/cost scales, and continue the identical
uniform equal-locality RNG stream for 2,000 additional updates. Learning rate
0.0003, weight decay0.0001, clip5, batch256, width64 are unchanged. The same
fixed B diagnostic batch is retained. No early stopping or budget search.

- Continued: update the original four-output shared network.
- Protected: freeze a copy of the original network for D_all and D_easy;
  update another copied network for H_all and H_easy only. The delivered
  reference moments must remain bit-identical for every C row. Harm nesting
  and the causal rollout envelope are unchanged. The four-component loss is
  still recorded; the two reference terms are constants in this arm.

Both arms have 24,836 optimizer-registered parameters. The protected arm has
49,672 total stored parameters due to its frozen copy, and its two unused
reference output rows receive no loss gradient. Thus this is not equal total
model memory or identical effective-gradient parameter count. Protected
inference uses two forwards. These costs are reported, not hidden as a speed
or architecture benefit. Freezing reference moments is not freezing every
forecast, utility score or action outcome into an all-fallback policy.

Same source roles: A produces R/P forecasts, B fits risk, C excludes both
current fitted chains. All six ordered source assignments, seeds17/29/43,
full and motion-only pairs: 72 new fits, 144,000 additional updates. A real
100-update pilot resumes inside the first head's budget. Choosing this fixed
matched budget before readout avoids a C-based continuation or checkpoint
choice. No fitted C normalizer or threshold. B learning curves are diagnostics,
not validation-selected results.

## Readout and Failure Criteria
Freeze and push 576 policy views before C labels are read: six controls per
group (R, raw neural, raw ridge, old mean dual/joint, failed sampling joint)
and five per new arm (all, dual, scene, joint, query-count-matched hash).
Controls are cached_verified; 360 new views are freshly evaluated.

Primary contrasts: protected-joint versus continued-joint complete observed
risk and all ADE. Also compare corresponding all/dual/scene policies, continued
versus old mean joint (extra-budget control), protected versus old mean,
sampling, raw neural/ridge, independent dual and matched-count hash.

Report all/easy/hard/complete ADE, FDE, tails, interventions, all/easy positive
harm, zero-CV harm, unknown-label accounting and same-action moment diagnostics.
The positive-harm budget remains 0.02; net easy must remain within 2% in every
supported locality. Net easy alone does not satisfy complete risk. A pass in
one seed or source assignment is insufficient for promotion.

Three seeds averaged within locality, then 3,000 resamples of the four C
localities per source assignment. Source-role views overlap and C has prior
development exposure: no independent confirmation or multiplicity-adjusted
discovery. Hash controls match counts, not realized risk. Greedy allocation is
not optimal or collision-aware. If protected reference fits remain unchanged
but harm still fails, report the remaining harm/transport problem rather than
declaring the reference guard sufficient.

## Runtime, Prior Work and Limits
Native arm64 CPU4, interop1, workers0; atomic checkpoint/RNG/optimizer resume,
PID/heartbeat and a 10GiB free-disk reserve. Previous fitting costs make local
execution reasonable. The current CREATE queue was checked read-only; no job
was changed or submitted. Remote M3W asset inventory is not_run.

[Yu et al., NeurIPS2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
study detrimental multi-task gradient interference and propose gradient
projection. This study uses no PCGrad and inherits none of its results or
guarantees. Freezing an output branch is a controlled diagnosis, not a novel
method claim. Reading scope: introduction and Sections2-4, not an empirical
reproduction of that paper.

Obs8/pred12 annotation steps at raw stride12, image pixels, detector-derived
labels. No future endpoint, central velocity, test endpoint goals or C/test
normalization in inference. No metric/seconds, human-gold, physical-safety,
true3D, foundation or submission-ready claim. Stage5C and SMC remain off.
