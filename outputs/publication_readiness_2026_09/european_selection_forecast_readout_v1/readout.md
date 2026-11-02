# First Frozen Selection-Locality Forecast Readout

## Scope And Provenance

Result source: **fresh_run** feature conversion, inference and selection evaluation;
**cached_verified** SOURCE-trained checkpoints and SOURCE-only normalizers.
No new training, threshold search, learned preprocessing fit or model selection
from these outcome values was performed in this readout. It is a supporting
forecast/input ablation, not the paper's joint-versus-independent policy result.

The 36 real Torch models cover three actual forecaster training seeds (17,29,43),
three SOURCE producer folds and four detector representations. All predictions
were frozen and fully replayed before the six model-selection localities' labels
were opened. These six localities are **now exposed for model selection**, not an
untouched test. Risk calibration, final confirmation and DroneCrowd remain closed.

## Fixed Population

All 38,102 past-supported targets from 28 recordings remain. There are 37,254
nonempty-label targets, 21,434 complete-12 targets, 15,820 partially labelled
targets, 848 wholly unknown targets and 27,694 available final endpoints.
No future-completeness filter selected the cohort. The task is obs8/pred12 at
raw-frame stride12 in local-image pixels. Detector-derived silver is not human gold.

ADE is averaged over each target's **available future labels**, not complete12 ADE.
Full-roster endpoint FDE is not estimable; available-endpoint FDE is diagnostic.
The registered completion convention bounds wholly unknown target costs using
forecast disagreement; it does not recover every unobserved step of partial
tracks or turn this into a fully observed 12-step benchmark.

## Registered Primary And Secondary

|History minus disabled, percentage points|Point [95% locality bootstrap CI]|
|---|---:|
|Registered paired completion-lower primary|-2.53 [-3.67, -1.49]|
|Known-label secondary|+2.60 [+1.73, +3.81]|
|Paired completion-upper diagnostic|+7.73 [+6.05, +9.96]|

The known secondary is positive in all six localities, but the registered lower
primary is negative in all six. **The primary fails.** Missing-label uncertainty
is not observed hidden harm. Independent known risk violations also prevent safety.
Individual lower-bound gains must not be subtracted to replace the paired lower.

Bootstrap: 3,000 resamples of the six localities after averaging SOURCE folds
and training seeds. Repeated predictions are not independent targets. These are
conditional model-selection development intervals, not final confirmation.

|Representation|Known-label ADE gain vs SOURCE floor, % [CI]|Individual completion-lower gain, % [CI]|Worst known easy net degradation|Risk-supported repeated record views|
|---|---:|---:|---:|---:|
|floor|+0.00 [+0.00, +0.00]|+0.00 [+0.00, +0.00]|0.00%|0/252|
|disabled|+9.07 [+6.98, +10.96]|+2.81 [+1.33, +4.32]|40.81%|0/252|
|current|+8.52 [+6.35, +10.55]|+2.63 [+0.75, +4.49]|59.13%|0/252|
|noise|+8.42 [+6.33, +10.26]|+2.22 [+0.67, +3.81]|38.39%|0/252|
|history|+11.67 [+9.10, +14.13]|+4.70 [+3.44, +5.95]|47.98%|0/252|
|physical_0|-141.44 [-172.59, -107.37]|-159.70 [-192.03, -124.85]|778.18%|0/252|
|physical_1|+2.64 [-1.41, +6.76]|-6.10 [-10.05, -1.89]|36.76%|0/252|
|physical_2|-2.93 [-6.00, -0.22]|-6.38 [-10.01, -3.09]|83.40%|0/252|
|physical_3|+6.25 [+2.91, +9.54]|-0.05 [-3.38, +3.06]|30.88%|0/252|
|physical_4|+4.79 [+0.20, +9.56]|-1.52 [-6.01, +3.22]|35.25%|0/252|
|physical_5|-11.66 [-21.26, -2.08]|-20.21 [-30.08, -10.48]|147.70%|0/252|

Physical0=constant position; physical1=causal CV; physical2=damped .90;
physical3=damped .97; physical4=OLS4; physical5=OLS8. The reference is the
**producer's SOURCE-selected fixed baseline**, not a validation-selected or
per-row oracle baseline. It differs by SOURCE fold. Every repetition retains
the same reference within its matched representation comparison.

The detector branch has 512 parameters within 89,026 total forecaster parameters.
Disabled uses zero detector contribution, so it is not an active-capacity-matched
control. Current and noise activate that same branch. Exploratory known-label
history-minus-current: +3.16 [+1.72, +5.13] pp; history-minus-noise:
+3.25 [+2.33, +4.46] pp. These additional comparisons do not change the
registered primary and are not multiplicity-adjusted new primary claims.

## Localities And Actual Training Seeds

|Selection locality|Known paired effect, pp|Paired completion lower, pp|
|---|---:|---:|
|eu-locality-087|+1.29|-3.33|
|eu-locality-092|+2.67|-0.97|
|eu-locality-093|+2.03|-4.99|
|eu-locality-103|+5.33|-2.16|
|eu-locality-104|+2.61|-1.13|
|eu-locality-125|+1.67|-2.58|

|Actual forecaster training seed|Mean known paired effect, pp|Mean paired lower, pp|
|---|---:|---:|
|17|+3.15|-2.36|
|29|+1.60|-3.12|
|43|+3.06|-2.09|

These are paired representations of actual training seeds, not three stochastic
resamples of a single trained forecaster. Each seed is still repeated across
the same six localities and three SOURCE folds.

## Safety And Decision

History full substitution supports **0/252** repeated recording-risk views:
249 known all-risk and 247 known easy positive-harm violations remain. Worst
known easy net degradation is **47.98%**. The original positive-harm and easy-net
budgets remain 2%; favorable average ADE cannot offset positive harm.

The floor itself has no intervention and zero known harm. Its empty selected
easy denominator is not estimable, not proof of a safe positive learned switch.
The 252 views are 28 recordings repeated for nine producers, not 252 recordings.

**No new deployment. No submission-readiness claim.** There is known-label
forecast-input utility on previously unscored localities, but no supported safe
complete policy, joint-decision contribution or independent confirmation yet.
Current evidence remains 2.5D/pseudo-3D agent-trajectory state modeling, not true3D,
metric/seconds-level prediction or a foundation model. Stage5C and SMC stay off.

## Execution And Replay

Readout PID 83158: 20.824071s, peak RSS
1,504,280,576B. Separate raw-label/metric replay PID 83420:
20.560468s, peak 1,433,534,464B; all28 raw labels and
all54 metric groups reproduce exactly. Full36 model inference had separately
replayed exactly before outcome opening. That inference replay is not repeated
inside the label/metric verification.
