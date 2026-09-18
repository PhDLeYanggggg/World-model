# SDD Auxiliary Training: What Improved and What Did Not

Result source: fresh training and analysis; source/cache identity and checkpoint
replays cached_verified. This is an exploratory fit-cohort comparison, not an
independent test or deployment certification. The main task remains eight
observed and twelve predicted annotation steps, past-normalized ADE and equal
physical-scene weighting. No closed evaluation role was opened.

## Completed Experiment

The registered matrix contains two source schedules, three input variants,
three seeds and three physical-site folds: 54 real Torch fits. Every fit uses
2,000 first-phase updates and 4,000 main-task updates. The first phase uses
either the same main training fold or the original SDD train-40 source. The
optimizer and main minibatch stream reset at the phase boundary; weights carry
over. Normalization is fitted only on the main training fold and held fixed.
Final checkpoints, not held-score-selected checkpoints, produce the results.

The architecture is the existing small geometry MLP / image CNN forecaster,
not a new JEPA, Transformer or foundation model. All 324,000 updates completed.
Summed fit time is 10,660.60 seconds. No runtime failure or slow-run downgrade
occurred. All 54 prediction replays are exact.

The full auxiliary cache has 229,333 eligible overlapping windows and 254,841
unique past crops from 40 recordings in five scene folders. Every auxiliary fit
draws 128,000 windows with replacement and visits 97,892-98,207 unique rows;
neither a full epoch nor independent coverage of every row is claimed.

## Main Outcome

| Input | No-SDD gain vs CV | SDD gain vs CV | SDD vs matched neural control |
| --- | ---: | ---: | ---: |
| Geometry | -1.34976% | -0.80523% | +0.53728% |
| Geometry + coverage masks | -1.37243% | -0.81626% | +0.54864% |
| Geometry + past RGB | -1.98266% | -1.27335% | +0.69552% |

SDD exposure reduces neural degradation relative to matched neural controls.
It does not establish a gain over CV: all 54 individual held-scene fits remain
negative, and all 54 fail easy preservation. No model is promoted.

The source-effect descriptive 95% scene intervals are [-0.05998%, +5.01533%]
for geometry, [-0.06014%, +4.91578%] for masks, and [+0.38541%, +2.70073%]
for RGB. These resample only three already exposed physical sites after seed
averaging. They cannot supply independent confirmation or a population safety
guarantee, even when an interval excludes zero.

## Failure Taxonomy

### 1. Better Training Fit Does Not Transfer

All 54 fit-cohort gains are positive, ranging from +0.56870% to +2.17476%, while
all held-scene gains are negative. Thus the model and optimizer can improve the
training objective, but that improvement is not a reliable cross-scene forecast.
This is compatible with overfitting, domain mismatch and insufficient observable
state-change cues; the experiment does not uniquely identify one cause.

### 2. Source Benefit Is Scene-Dependent

| SDD input | ETH gain vs CV | Hotel gain vs CV | Grouped Zara gain vs CV |
| --- | ---: | ---: | ---: |
| Geometry | -0.22560% | -0.36373% | -7.22531% |
| Coverage masks | -0.21206% | -0.39398% | -7.32295% |
| Past RGB | -0.84601% | -0.78637% | -6.79648% |

These are seed means, not extra independent experiments. Zara remains the
largest relative degradation, although SDD reduces that damage. A favorable
source contrast must not hide negative absolute performance on every site.

### 3. RGB Does Not Establish a Stable Visual Contribution

RGB versus the matched mask control is -0.60197% without SDD and -0.45339% with
SDD. The latter descriptive interval is [-0.63261%, +0.49055%]. RGB helps relative
to masks on Zara but hurts ETH and Hotel. Readable, aligned past pixels are a
prerequisite, not proof that the current CNN extracts useful future-motion cues.
Low-resolution crops and domain-specific appearance remain hypotheses. No actor
visibility or full semantic-alignment guarantee follows from crop replay.

### 4. Starting Motion Remains Poorly Predicted

The fixed static-to-movement slice has 188 windows across two fit sites. Its
normalized CV ADE is 194.88812; SDD geometry and RGB yield 194.89600 and 194.89577.
The networks do not repair the dominant missing-start/direction problem under
the fixed primary. The large normalized value must not be read as meters.

On 177 static-stays windows, CV has exactly zero error. SDD geometry, masks and
RGB add mean normalized ADE of 0.03164, 0.02373 and 0.03187. Percentage improvement
is undefined for this zero-reference slice. It is not zero and not a reason to
discard the slice. A report-export division by zero was repaired without
changing training, labels, losses or the primary metric.

Moving-stops is a limited positive diagnostic: SDD geometry gains 3.68880%
against CV, while SDD RGB loses 4.04288%. The train-defined hard-q75 slice remains
negative for all six arms. Event slices overlap and are not additive cohorts.

### 5. Easy Preservation Fails in Absolute Terms Too

| SDD input | Easy relative degradation across nine fits | Absolute normalized ADE harm |
| --- | ---: | ---: |
| Geometry | 325.30% to 7,255.27% | +0.03505 to +0.18985 |
| Coverage masks | 320.18% to 7,192.82% | +0.03033 to +0.18821 |
| Past RGB | 536.80% to 5,281.05% | +0.02463 to +0.13819 |

Near-zero easy CV denominators amplify percentages, but absolute harm is also
positive in every fit. This does not justify loosening the 2% gate or opening
closed labels to find a favorable fallback threshold.

## What This Comparison Cannot Isolate

- Source pretraining changes data exposure and initialization jointly. The
  no-SDD arm sees 6,000 main updates; the SDD arm sees 2,000 SDD plus 4,000 main
  updates. We cannot isolate useful transferred representation from reduced
  main-domain overfitting without further matched exposure controls.
- Main-only normalization clips 3.76-4.13% of auxiliary feature entries at the
  registered limits. This measures a distribution mismatch, not its causal effect.
- Source labels are complete/partial/absent for 188,358/37,360/3,615 windows.
  Membership is past-only; masked losses preserve this distinction. No all-empty
  loss batch occurred. Different supervision support remains part of the source
  contrast, not an unreported exclusion.
- SDD stride 12 makes the twelfth forecast point +144 raw frames. This is not
  ETH/UCY elapsed-time alignment, a t+50 result or a seconds-level benchmark.
- Supplied histories contain retrospective annotation interpolation. This is
  offline annotated-history forecasting, not verified strict sensor-as-of input.

## Next Falsifiable Steps

1. Separate source learning from reduced main exposure with a registered
   4,000-main-update-only control and a controlled source-supervision ablation.
   Keep architecture, primary and scene folds fixed; do not select on old test scores.
2. Measure independent start/stop/turn support and observable directional cues
   within the admitted training source before another large trial grid. More
   overlapping windows alone do not resolve rare-event or appearance mismatch.
3. Require a positive candidate forecast plus easy preservation before returning
   to routing. Resolve independent-site calibration/confirmation separately;
   repeated bootstrap over the same three sites cannot create that evidence.

These are follow-up hypotheses, not executed experiments or an approved new
primary/split. The long-term M3W objective remains active and unmet. No metric,
true-3D, foundation, independent-generalization or submission-readiness claim.
Stage5C and SMC remain disabled.

Evidence: [all fit metrics](fit_metrics.csv), [analysis](analysis.json),
[full report](report.json), [replay](replay.json), [reproducibility](reproducibility.md).
