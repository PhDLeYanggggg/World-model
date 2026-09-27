# Matched Query-Risk Heads: Model and Data Card

## Intended Use

Development-only comparison of risk-supervision aggregation units. These heads
control intervention between an already frozen causal floor and an already
frozen neural forecast. They are not new trajectory generators, a foundation
model, or a deployment safety certificate.

## Architecture and Training

Each head has 25,028 parameters: the same 380-dimensional causal representation
and six descriptor inputs, a width64 hidden layer, and four nonnegative score
bases. The added descriptors summarize past displacement relative to extent,
path nonlinearity, past turning, neighbor occupancy and disagreement between
the two causal forecast rollouts. Harm-like outputs have a causal envelope.
Because only all/easy signed excess is supervised, separate expected cost
moments are not identified. Nonnegative output bases are not calibrated
probabilities or uncertainty bounds.

The pointwise and query models share initialization, preprocessing, source
roles, every sampled query and row, optimizer settings and 2,000 updates.
There are 108 paired groups and three forecasting seeds. The two losses differ
only in whether individual signed errors are squared before or after averaging
within a retained current-frame query. Equal query weighting is used in both.
The grouping unit is locality/recording/current frame; no temporal future
attention or future-label feature is introduced.

## Data Passport

- Cached_verified pool: 318,969 rows in twelve already-opened European
  source-training development localities.
- In each role assignment: four producer localities, four controller/floor
  localities, two new-head fitting localities and two held development localities.
- Independent selection, calibration and confirmation remain closed.
- Observation8/prediction12 at raw-frame stride12; image-local coordinates.
- Detector-derived silver labels, not independently human-validated gold.
- Unknown future costs are excluded from supervised fitting, but their causal
  rows remain in inference and action counts. Unknown harm cannot be certified.
- Query cohorts are retained evaluation cohorts, not a guarantee that every
  visible agent in a physical scene is present.

All fitted normalizers and labels follow the fitting role. Forecasting contexts
are constructed from sites, history, origin, geometry, recordings and current
frames. Future coordinates and masks enter loss/evaluation only. No central
velocity, test endpoint goals or test normalization statistics are inputs.

## Evaluation and Restrictions

Checkpoints are fixed before action generation; all actions are committed
before held outcome readout. Count-matched ranking is diagnostic. Joint policies
share the nominal 2% predicted all/easy budget but can differ in intervention
rate, so their accuracy contrast is not rate matched.

The registered primary has a disclosed structural zero-coverage defect: ten
parent views force zero actions and undefined selected-risk denominators.
No primary success or deployment promotion can be claimed in this experiment.
Predeclared secondary metrics and negative findings must remain visible.

Bootstrap uses 3,000 locality resamples after averaging dependent views. Twelve
development localities, overlapping queries and repeated seeds do not become
hundreds of independent datasets. No metric, seconds, physical-safety,
true3D, foundation or independent-confirmation claim is supported. Stage5C and
SMC remain disabled; the protected deployment policy is unchanged.

## Storage and Reproduction

Model/optimizer/RNG checkpoints and row-level arrays remain private and ignored
by Git. Public manifests contain hashes, not raw data. Native arm64 CPU4,
interop1 and workers0 are used; the first full paired fit is replayed from
scratch, with all heads subject to causal prediction replay and independent
decision accounting. See `operation_zh.md` for commands and limitations.
