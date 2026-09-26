# Harm Tail Data Card

## Provenance and Role

The study reuses hash-verified European source-development arrays and frozen
forecasts from the registered lineage. It trains new locality-excluded risk
heads, rather than recollecting data or retraining the trajectory predictors.
The registered observation/prediction lengths are 8/12 annotation steps at
raw stride 12. Coordinates are image pixels; label provenance is detector-derived.
No conversion to seconds or meters is asserted.

For each ordered source assignment, A provides the previously fitted
forecast/causal feature chain, and B provides four controller localities.
Each head trains on three B localities and reads the fourth only after its
predictions have been frozen. Normalization, cost scale, positive-CV-error
25th-percentile cutoff and score bins are learned on the three fitting
localities. Unknown targets are not sampled.

C is excluded from the currently fitted A/B chain but has been historically
opened. It is source-development evidence, not independent confirmation.
Original whole-B models and C scores are cached-verified; B diagnostic scores
and bins are recomputed in this round. Inner-held metrics are fresh_run.

## Independence and Availability

The six previously opened selection localities are not accessed this round.
Twelve reserved calibration and six confirmation localities remain closed.
The primary endpoint and 2% risk tolerance do not change. No test normalization,
future input, central official velocity or test-endpoint goal construction is
introduced. Historical contaminated/test-selected scores remain exploratory.

The four folds use different train-fitted diagnostic easy cuts. Their rows
cannot be pooled as observations of one fixed event. Three-seed means are
computed within locality; uncertainty resamples four localities within each
source assignment 3,000 times. Role/seed views are dependent and are not an
independent-scene sample-size expansion.

Zero-event statistics are unestimable where appropriate. Fewer than 20 positive
or negative events is flagged weak support. AUROC/AP measure harm-event
ranking, while harm coverage measures predicted-to-actual mass; these are
different questions. Simple forecast disagreement is retained as a control.

Private arrays, history/latent caches, checkpoints, images and third-party
assets are not published. Public aggregate evidence and hashed source bindings
support local reproduction when the original authorized data are available.
