# Fixed-Selection Causal Analogue Probe

## Material Passport

Previous turn:progress,36 real continuations/72,000 updates and72 fitting views
completed and replayed. Average error fell but selected easy-risk repair failed.
This is a new fitting-only information probe, not another loss-weight, clipping,
threshold or output-cap sweep. No new held, outer development or independent
selection/calibration/confirmation role. No trajectory training or deployment.

## Question and Fixed Controls

Does cross-recording similarity in the existing380 causal features contain
conditional harm information within the frozen original-self selection?
Retain all6contexts x3headseeds x4sources =72 views. Original and estimation-only
predictions must replay bit-exactly against the preceding frozen readout.
The original causal selection is fixed before the probe; unknown outcomes
remain in every inference query. No future target can create a query or feature.

For each query recording, reference rows are eligible rows of other recordings
in the SAME fitting source. Feature means/std use all those causal reference
rows, including rows with unknown outcomes. Only known reference outcomes with
positive causal disagreement envelope can supply regression labels. No row
from the query recording enters its normalization, retrieval or labels.

Standardize380 features with reference mean/std (std floor1e-6), clip at8,
Euclidean distance, exactly32 neighbors, no search or tuning. If fewer than32
known donors exist, retain queries as unsupported; never assign zero risk.
Uniform neighbor means predict H/envelope and easy-H/envelope, multiplied by
the query's causal envelope. Compare source-mean fractions and the identical
neighbors with labels permuted WITHIN donor recording (fixed seed20261029).
Permutation preserves recording-wise distributions and donor support. Both
harm labels stay paired. No target-record labels affect its own predictions.

Original B/R/easy-R are copied for signed-risk evaluation, not refitted. Raw
analogue moments obey0<=easy-H<=H<=envelope, but may violate B+H<=envelope with
the frozen benefit head. Count those violations; do not silently project them
or execute a policy. This is an information/estimation probe, not a coherent
deployable replacement. Original and estimation-only are contextual controls;
primary comparisons use matched analogue constraints.

## Measures and Fixed Information Screen

On the full frozen selected cohort report source-normalized all/easy signed-
risk MSE, harm mass, false-safe count, conditional harm AUROC/AP, prevalence,
known/unknown/supported counts, reference and neighbor hashes. No unsupported
query is dropped from the primary: any unsupported row makes that view's full-
cohort MSE undefined. Report supported-only scores as descriptive, never primary.
Unknown labels remain outside supervised MSE, with their coverage explicit.

The descriptive information screen requires nearest32 equal-view easy MSE to
beat both source-mean and within-recording-shuffled controls, and at least12/18
context/head-seed pairs to beat both after averaging their4source views. Absolute
numerical slack1e-12. No seed/source removal or threshold changes. AUROC/AP are
secondary; undefined event support stays undefined. No fitting CI or claim of
independent significance. Even a pass cannot authorize deployment or new held
readout; a later model comparison needs separate registration.

## Important Exposure Limit

The reference-bank calculation excludes the target recording, but the upstream
forecaster, original risk head and fixed selection have their existing training
exposure. This is NOT honest whole-pipeline OOF, validation or independent
generalization. Its role is to compare feature matching with a matched shuffled
control while avoiding trivial same-track/window copying. A negative result
does not prove irreducible uncertainty or rule out every causal representation.

## Execution

Native arm64 CPU4/inter-op1/workers0. One real full-view pilot, then all72 views,
then complete retrieval replay. Chunk256 queries; all donors used, no sample cap
or quick downgrade. Atomic per-view JSON permits explicit interrupted resume.
Heartbeat and PID,12h boundary,10GiB disk reserve. No new arrays/checkpoints/raw
data in Git; private event log only. CREATE read-only original TRAIN packets,
no scheduler submission. Preserve unrelated index and earlier results.

Keep obs8/pred12 rawstride12 and2% positive selected harm/reference criteria.
Detector-silver,pixel/local only. No metric/seconds/human-gold/physical-safety,
true3D/foundation claim. Stage5C and SMC stay off.
