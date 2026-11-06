# Does The Donor Effect Survive Risk-Head Seeds?

I kept the teacher cache, data, model, update budget, source exclusions
and decision rule fixed, and added risk-head seeds29/43 to the verified
seed17 result. This is not a three-forecaster-seed main experiment.

Verdict: **three_risk_seed_donor_information_replicated_not_deployable**.
Passing information contexts by seed: `{'17': 3, '29': 2, '43': 2}`.
No deployment upgrade. The original full-policy safety rules still apply.

## Seed Results

| Risk seed | Context | Cost reduction vs own % | MSE reduction vs own % | Cost reduction vs forest % | False-safe cost nonworse than forest |
|---|---|---:|---:|---:|---|
| 17 | critic0 | +20.710862 | +64.337175 | +4.962925 | False |
| 17 | critic1 | +32.267086 | +16.726303 | -10.859887 | False |
| 17 | critic2 | +10.692731 | +11.997147 | +2.244043 | False |
| 29 | critic0 | +9.283236 | +79.368644 | +7.270917 | False |
| 29 | critic1 | +21.367678 | +30.708264 | -4.972277 | False |
| 29 | critic2 | +1.048867 | +14.383194 | +4.844898 | False |
| 43 | critic0 | +16.010883 | +83.034396 | +9.395852 | False |
| 43 | critic1 | +29.924973 | +16.911014 | +6.171747 | False |
| 43 | critic2 | +2.861279 | -13.618112 | +3.789925 | False |

Positive reductions are better. These are source-macro known-proposal
risk/decision costs, not trajectory ADE/FDE or a full-policy safety ratio.
The cohort remains48,061known proposals and1,274unknown proposals. Unknown
outcomes are outside this known-only information estimate, not certified safe.
Evaluation normalization and zero threshold are identical to the frozen
comparison. No seed or checkpoint is chosen by these results.

## Seed Variation

| Context | Cost reduction vs own, mean +/- sample SD % | Min / max % |
|---|---:|---:|
| critic0 | +15.334994 +/- 5.743716 | +9.283236 / +20.710862 |
| critic1 | +27.853245 +/- 5.737448 | +21.367678 / +32.267086 |
| critic2 | +4.867626 +/- 5.125436 | +1.048867 / +10.692731 |

## Source-Cluster Uncertainty

3000paired source-component replicates;2/3/3components per context.
The same sampled components are used for all arms and risk seeds.
Risk seeds are averaged, not resampled; windows are never sampling units.

| Reference | Seed/context-macro point reduction % | Exploratory95% interval % |
|---|---:|---:|
| own | +12.161989 | [+2.656382, +27.179109] |
| forest | +4.150244 | [-4.749503, +8.121513] |

Here seed averaging means averaging individual-head costs, not risk scores.
These intervals do not apply to the mean/unanimity ensemble diagnostics.
Intervals are conditional on these eight exposed source components and
fixed seed17 teachers. Small-cluster bootstrap and three head seeds do not
establish new-domain coverage, independent confirmation or complete pipeline
seed stability. Negative forest comparisons and source-level tradeoffs remain.

## What Actually Ran

- fresh_run:48new risk fits,48full exact refits and new-seed development readout.
- cached_verified:24seed17heads/results, frozen donor/own-B arrays and teacher lineage.
- Primary new updates196,608; repeats another196,608; separate pilot4,608.
- The pilot matches the complete frozen seed17engine and seed29resume exactly.
- not_run:new outer C/H, independent confirmation, new teacher seeds or deployment.
- All three risk seeds are retained; no best-seed selection or threshold tuning.
- Original2% harm, easy-net and support rules stay fixed; empty controls stay NA.
- Original24-group main NA and old failures remain. No identity certificate.
- Native8/12image-coordinate silver annotations; no verified meters or seconds.
- No true3D/foundation, Stage5C execution or SMC. Not submission-ready.

[Per-source seed results](source_seed_results.csv) | [Loss trace](loss_trace.csv)
| [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)
