# Pooled Analogue Diagnostic Card

## Purpose and Status

A fixed-cohort information probe, not a deployed forecasting or safety model.
Pooled reference support passes and the easy-information screen passes; risk
magnitude fails. No new neural training or policy release. Data-derived tables
are fresh_run; ancestor checkpoints, query hashes and the original input audit
are cached_verified. Independent evaluation and deployment are not_run.

## Inputs and Targets

- Inputs: frozen 380-dimensional causal features from history, neighbors and
  available baseline/candidate rollouts; original eligible selection unchanged.
- The feature includes log(observed scale), so complete coordinate invariance
  is not claimed. No observed future endpoint, central velocity or test goals.
- Donors: four sources in the same predictor/controller context, inner-fit only;
  exclude the query's complete source-qualified recording from normalization,
  neighbors and labels. Other contexts, inner-held and reserved roles are barred.
- Unknown donor features may enter donor statistics, never supervised labels.
  Unknown query outcomes remain in the cohort and support accounting.
- Targets: known donor H/envelope and easy-H/envelope. Predictions multiply the
  query causal envelope; raw native coordinate-unit error is not pooled.
- Fixed controls: original risk head, estimation-only continuation, pooled mean,
  32 nearest donors, same neighbors with paired labels shuffled within recording.
- All five output moments are not retrained: B/R/easy-R are copied. Joint
  envelope failures are counted, never projected away after evaluation.

## Boundaries and Limitations

Original TRAIN exposure includes upstream model, feature preprocessing and
selection. Donor exclusion is not whole-pipeline OOF. Sixty-two thousand query
occurrences are not independent sample size; no confirmatory uncertainty is
attached. Pooling fixes absence of references but can import mismatched label
incidence and magnitude. Donor-rich sources may dominate. Event rank metrics
remain undefined on non-event views, without deletion or zero substitution.

Protocol: obs8/pred12, raw stride12, detector-silver, pixel/local coordinates.
No human gold, seconds, verified meters, physical safety, true3D or foundation
claim. Risk definition remains 2% positive selected harm/reference. No Stage5C,
SMC, submission action or deployment change. Historical contaminated results
are not rehabilitated by this run.
