# Membership-Conditional Cost Model Card

## Material Passport

All 288 direct/conditional cost heads completed 576,000 updates (fresh_run).
The membership
classifier, forecasting producers, reference-cost moments and data lineage
are cached_verified. No new policy, trajectory dynamics, calibration or
independent confirmation is included. This is source development. Registration
2c321649 preceded training; prediction freeze a5746d3c preceded held readout.

## Architecture and Supervision

Each new head uses 383 causal features, a width-64 GELU layer and two outputs:
24,706 parameters. Direct predicts bounded all/easy expected harm. Conditional
learns separate harm estimates within E and not-E, including zero-harm easy
examples. E uses the inherited training-only positive-CV-error 25th percentile;
zero CV errors are excluded from E. All future errors are supervision only.

Both receive equal new training budgets and draws. Conditional additionally
uses the frozen 24,641-parameter membership MLP at prediction composition;
total system capacity and past training cost are not matched. Its probability
is not a fitted input or loss argument for the cost head. Training composed
metrics still use in-sample membership probabilities and are fitting-only
diagnostics, not independent performance evidence.

At inference H_easy = p m_E and H_all = p m_E + (1-p) m_notE. Experts are bounded
by causal forecast disagreement, guaranteeing numeric nesting only, not
physical safety. Original predicted reference-cost denominators remain fixed.
The constant-p control retains the same experts and fitting prevalence.

## Use and Limits

No deployable-policy change follows from an expected-cost component result.
Report every role, seed, pair and held locality; use the registered gates,
not the best-looking contrast. Full/motion-only conditional populations differ.
Weights, per-row outputs and raw data remain private. Checkpoints include
optimizer, sampler, preprocessing and RNG state for exact resume.

Detector-derived image pixels and annotation steps only. No metric/seconds,
human-gold, physical-safety, true3D, foundation or submission-ready claim.
Stage5C and SMC remain off.
