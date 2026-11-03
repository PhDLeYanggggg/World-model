# Prior Work And Scope Of The Feature Test

Fresh primary-source reading on3November2026,while full cache replay runs and
before any new cost-head fit or held feature-effect readout. This is positioning,
not a literature-completeness or novelty certificate.

## Contextual Confidence Is Already Established

Cao et al.,CCTR,AAAI2024,pp20949--20957,DOI10.1609/aaai.v38i19.30085,
learn variance calibration using error-based regularization and a contextual
postprocessor. Its observed-motion/social features and scene spatiotemporal
embedding produce per-prediction scaling,with the postprocessor fitted on
validation data. Thus neither conditioning confidence on history/context nor
using a neural representation to predict uncertainty is new by itself.
[Author-hosted paper](https://tanrui.github.io/pub/CCTR-AAAI.pdf).

Read the calibration setup and method sections,not a full independent
reproduction. The publisher endpoint was blocked and the institutional copy
timed out;the author's linked copy was accessible. The current M3Wbank is a
deterministic fixed-head candidate/floor pair,not CCTR's predictive-distribution
interface. A faithful uncertainty-baseline port needs that distinction resolved;
the present feature experiment is not an implementation or rejection of CCTR.

## Joint Metrics And Joint Decisions Are Different

Weng,Hoshino,Ramanan and Kitani,Joint Metrics Matter,ICCV2023,
distinguish per-agent best-of-Kerror from choosing one joint forecast draw
before averaging agent errors. They evaluate joint metrics and collision
proxies and add joint reconstruction terms to existing stochastic forecasters.
[Primary paper](https://arxiv.org/pdf/2305.06292).

Read setup,metric definitions and joint-objective sections1--4;not a new run
of their benchmarks or a reading of all supplementary experiments. The CVF
endpoint returned403;the primary arXivcopy was accessible. Their protocol's
time/unit/radius conventions cannot be imported into our uncalibrated pixels.

From the metric definitions,with one fixed joint prediction and the same
aggregation,the min-over-Kdistinction disappears. This is a mathematical
inference,not a new measured result. Renaming current per-queryADEasJADEwould
not prove interaction modeling. A same-count allocation test asks a different
question,and lower error alone does not establish lower multi-agent conflict.

## What This Experiment Can Establish

The matched primary tests whether a frozen interaction-feature bundle improves
an existing relative-benefit/harm/reference critic under the unchanged missing-
label and per-record risk evaluation. It does not train a new forecaster,prove
JEPA/Transformer contribution,or establish scene-joint optimization. Its raw
neighbor coverage differs from the existing444features;positive results still
need a raw-original-context control before encoding-only attribution.

The actual paper hypothesis still needs demonstrable intervention utility,
matched joint/independent decisions,complete risk support and independent
scenes. Neither these citations nor a positive SOURCE mean alone satisfies it.
