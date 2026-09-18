# Why Equal-Episode Training Failed

## Result

All 24 new held-site fits are negative. Equal-site ADE gain versus stationary CV
is -37.3268% for geometry and -54.9917% for centered appearance. Conditional
four-site intervals are [-52.3613%, -26.0828%] and [-79.7777%, -39.1095%].
Uniform controls lose only 0.0704% and 0.7622%. Centered appearance is worse than
geometry under both samplers; the new contrast is -17.6649 percentage points.
No new model is deployed and no favorable seed or scene is selected.

## A Measured Objective Shift

Every training episode received equal probability, irrespective of its window
count. Group keys use past observations, not future labels. Nevertheless, shorter
episodes are correlated with later annotation changes, so their greater sampling
mass changes the distribution of supervision. The same per-example loss does
**not** imply the same expected training objective under a different sampler.

| Excluded site | Uniform future-change probability in training | Episode-sampled probability | Weighted CV cost / original scale |
| --- | ---: | ---: | ---: |
| coupa | 38.62% | 61.71% | 1.890 |
| deathCircle | 47.49% | 73.54% | 1.840 |
| gates | 46.80% | 71.22% | 1.792 |
| hyang | 44.09% | 66.79% | 1.835 |

These probabilities use future labels for post-hoc diagnosis only. They never
entered the sampler or inference features. Actual saved draw counts and every
trial's probabilities are retained in [diagnosis.json](diagnosis.json).

All twelve geometry models improve the episode-weighted training risk over CV
by 4.52% to 13.29%, yet lose 10.61% to 14.80% on the original unweighted training
distribution. All twelve visual models improve weighted training risk by 9.23%
to 21.71%, yet lose 14.46% to 19.73% unweighted. Thus optimization is demonstrably
active, but its fitted tradeoff is misaligned with the retained evaluation.
This establishes an objective/distribution mismatch, not a unique explanation
of every held-site error or proof that input information is sufficient.

## Harm And Remaining Failure

Static-target absolute harm rises to 0.576017 annotation pixels for geometry and
0.765669 for centered appearance, compared with 0.001644 and 0.020320 under uniform
sampling. Static CV has zero error, so percentage degradation is undefined and
cannot be declared below 2%. Nonzero targets also lose 8.7639% and 14.7767%; the
failure is not confined to static jitter. Centered hard-slice gains are negative
at all four sites. No hard/easy deployment gate passes.

Per-candidate equal-site oracle gains increase to 2.6052% and 3.2816%, while actual
forecasting worsens sharply. Larger future-informed oracle headroom is therefore
not evidence of usable gain prediction, correct direction or learned dynamics.
Do not rescue this result with a threshold selected on these held outcomes.

The source audit also shows a real support limitation: 207 half-box-excursion
windows correspond to 55 annotation episode groups/47 scoped tracks, with only
two relevant tracks at gates. Almost all have moving-neighbor observations.
Neither reweighting nor oversampling creates new independent events. These
counts do not by themselves prove that the task is unlearnable.

## Engineering Checks And Scope

All 240,000 updates completed. Twenty-four checkpoints replay exactly, all
weighted draw streams regenerate, twelve paired-arm streams match, and group
probability masses verify. Six OOF archives recompute. Completed resume adds no
updates and preserves 84 artifacts. Twenty-eight scoped tests pass, including
weighted interrupted/resumed fitting. No full legacy suite rerun is claimed.

The readout is unchanged: all original rows, equal-site normalized ADE, fixed
three seeds and 2,000 conditional site-bootstrap draws. Four explored sites and
shared training folds are not independent confirmation. Offline interpolated
annotations are not sensor-as-of observations or human gold. Dataset-local
annotation pixels/raw frames remain nonmetric and have no verified seconds scale.

## Next Falsifiable Repair

Retain these failures and the uniform controls. Do not quietly adopt episode-
weighted evaluation. If a balanced sampler is used again, separate its exposure
effect from its objective shift: derive the exact uniform-risk importance factor
`1 / (N_train * p_train(row))`, test its expected loss/gradient and weighted resume,
then register a matched training comparison before reading new held scores.
No clipping or normalization of these factors should be introduced silently,
because that would change the expected risk again.

That correction has not been trained here and cannot guarantee predictive lift.
It does not address the limited event support or missing intention cues. A new
data/representation route still needs identifiable past information and enough
independent source events, under the existing role and rights constraints. No
main/outer/external score, new policy, Stage5C execution or SMC is introduced.
