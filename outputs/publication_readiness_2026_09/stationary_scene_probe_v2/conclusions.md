# Static Context Helps Some Start Scores, Not Forecasts

## Material Passport

Mode: executed and replay-verified fit-only experiment. Canonical fit rows and
frozen model settings are bound by `configs/m3w_stationary_scene_probe_v2.json`.
No new development, calibration or confirmation labels were opened. Evidence:
local real-source audit, supervised classifier/regressor fits, model replay and
descriptive whole-scene-held comparisons. Not a neural world-model result,
independent confirmation, deployed policy or submission-ready contribution.

## Experiment Completed

The original and geometry-repaired versions each fitted 36 start classifiers
and 36 multi-output trajectory regressors. The corrected version retains all
365 stationary windows, 31 agents, 45 runs, physical-scene folds, 8/12 native-step
task, labels, three seeds and fixed 0.9 probability gate. Feature arms are pooled
neighbor context, plus static obstacles, plus directional neighbor summaries.
The models are logistic/Ridge and ExtraTrees, not newly trained Transformers.
These small local fits take seconds; they are not presented as long neural runs.
No higher-cost GPU/HPC allocation is warranted for this diagnostic.

The first implementation incorrectly treated two adjacent segments' identical
closest corner as ambiguous direction. The fix distinguishes different closest
surface points from duplicate corners. All 35 falsely undefined Hotel frames are
now defined. Exact original sources, models and results remain locally archived;
the original registration and public metrics are not overwritten. Two tiny
positive guarded results in that first version disappear after the correction.
They must not be cited as gains. See [repair](../stationary_scene_probe/corner_repair.md).

## Inputs That Are Actually Available

Both ETH recordings provide static obstacle XML, maps, H matrices and reference
images. ETH has four line segments; Hotel has four segments and three circles.
The XML is used as an unverified static-scene proxy, without physical/semantic
safety claims. The reference images visibly contain people and have unverified
capture time. They are not model inputs. No video frames, supplied destinations
or supplied group labels enter the experiment. The README's assumed destinations
for all subjects do not establish split-safe goals.

Under the source's H convention, all ETH points and 82.78% of Hotel points fall
within the reference dimensions. This is a numerical availability check, not
pixel alignment validation or meter calibration. The ETH annotation/video-clock
conflict remains open. Neither these images nor the obstacle maps are promoted
to a verified as-of visual modality or human-gold annotation set.

## Corrected Results

| Direction / ExtraTrees | Features | Start AUC | Brier improvement vs training prior | Native ADE | ADE improvement vs CV |
| --- | --- | ---: | ---: | ---: | ---: |
| Hotel -> ETH | pooled | 0.6692 | -0.03195 | 0.25823 | -3.31% |
| Hotel -> ETH | + static map | 0.8048 | +0.00617 | 0.26726 | -6.92% |
| Hotel -> ETH | + directional neighbors | 0.5090 | -0.07899 | 0.26088 | -4.36% |
| ETH -> Hotel | pooled | 0.5003 | -0.00820 | 0.09455 | -346.70% |
| ETH -> Hotel | + static map | 0.5387 | +0.02894 | 0.06626 | -213.07% |
| ETH -> Hotel | + directional neighbors | 0.5023 | +0.01957 | 0.07455 | -252.23% |

Three-seed means. CV native ADE is 0.24997 on ETH and 0.02117 on Hotel; these are
separate dataset-local coordinate results, not pooled metric error. Brier lifts
are absolute probability-score differences, not trajectory percentages. In the
static-map ETH direction Brier is positive in only two of three tree seeds.
In Hotel the Brier improvement accompanies weak rank discrimination, so it must
not be interpreted as a reliable switchability model. All linear arms and all
other settings are retained in [results](results.md).

All 36 corrected unrestricted regressors are worse than CV. The fixed 0.9 gate
produces zero or negative ADE improvement in every corrected setting. For the
static-map trees, ETH always falls back; Hotel gains are -1.86%, 0%, -2.13% across
seeds. This is neither safe positive transfer nor a new deployable predictor.

All easy labels in this subset have exactly zero CV error. Percentage degradation
is therefore undefined, not zero. The report retains absolute harm and intervention
rates, including damage from the small number of high-confidence mistakes. No
additional gate was selected after reading held outcomes.

## Failure Taxonomy

**Start classification is not trajectory identification.** Static distance cues
can help rank or adjust the probability of any future annotation change without
determining direction, duration or displacement. ETH endpoint angular errors for
these regressors average about 120--140 degrees where both endpoints are nonzero.
A probability-only gate cannot correct the direction of an admitted forecast.

**The two sites have different displacement distributions.** Stationary-subset
native CV ADE is 0.24997 in ETH and 0.02117 in Hotel. Cross-site regression can
inject movement into trajectories that remain still. Supplied map extent is not
a verified shared kinetic scale, and larger all-target mean predictions are not
automatically useful. This study does not disentangle annotation precision,
site behavior, horizon-time mismatch and physical coordinate scale.

**Endpoint labels do not summarize every change.** Hotel has 129 windows with a
recorded change, but only 92 nonzero final displacements: 37 return to the starting
coordinate by the final requested step. ETH has 59 changed and 59 nonzero-endpoint
windows. Rounding, annotation interpolation and real return motion remain possible
explanations; none is established here. All-row ADE retains these cases.

**Support and domain imbalance remain substantial.** The positive ETH direction
has only five agents. Repeated windows and seeds cannot create independent sites.
Adding directional neighbor summaries does not recover a useful trajectory model.
No image/body orientation or verified route semantics were available as inputs.

**The geometry bug mattered, but did not explain all failure.** Correcting the
shared-corner ambiguity removes false exclusions and erases the first run's small
positive guarded artifacts. It does not establish useful start trajectories.

**Normalization is not an excuse for these results.** Every row in this exact-
stationary subset has the same numerical-floor past normalizer. Thus its native
and past-normalized percentage gains coincide. Switching units would not turn
the present negative forecasts into positive ones. The broader question of a
future benchmark's primary metric is separate from this failed experiment.

## Decision and Next Work

Do not promote this static-proxy head, launch a larger stationary-start network
on its apparent classification success, or lower thresholds using the held scores.
The frozen forecasting results and historical policy status remain unchanged.
The added scene mechanism has a limited probability signal, not dynamics lift.

Before another full forecast comparison, settle the prospectively specified
main-table error units and the role of normalization. A user decision has been
requested on a separate native-ADE/FDE-primary protocol. It is not approved or
executed here. The present primary protocol and all negative outcomes are retained;
previously exposed sites cannot become independent tests through relabeling.

Independently, useful new support would be verified synchronized scene/body
orientation or route context, continuous labels with known construction, and
more source-eligible independent sites with stationary-to-moving episodes. Those
requirements should be checked against actual assets before building another
encoder. Native ADE/FDE alone would not resolve the demonstrated direction errors,
small support, absent independent confirmation or unsupported joint-intervention
contribution. No number of additional stage reports can replace those experiments.

## Verification and Scope

All 144 saved fitted models across the original and repaired versions reproduce
their saved probabilities/forecasts within 1e-12 (observed max difference
2.22e-16). Fourteen focused tests pass, including geometric equivariance, future
input rejection, shared-corner handling, preserving zero-floor easy harm and
complete seed/setting reporting. The unchanged legacy suite was not rerun.

No p-value, independent-scene CI or new safety guarantee is claimed. This remains
2.5D trajectory research in dataset-local coordinates/native annotation steps,
not verified metric/seconds, true 3D, foundation or successful world dynamics.
Stage5C and SMC remain disabled. Third-party images, maps, source rows, caches and
model checkpoints remain excluded from Git; code and aggregate reports are saved.

The corrected command was also resumed to a separate report directory: all36
model pairs were hash-verified and reused, with zero new fits. All24 registration
bindings match. The resume console's model count describes the completed inventory,
not fresh retraining; the separate report marks each pair cached_verified.
