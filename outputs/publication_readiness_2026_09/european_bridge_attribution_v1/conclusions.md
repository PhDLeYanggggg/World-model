# What the Bridge Contributes

## Completed Work

I completed a registered attribution experiment, not another backbone trial.
Registration b6bb367e preceded training; freeze89f359a9 preceded outcome readout.
All36 new motion-only neural cost heads finished2,000 updates, with36 new ridge
fits. The36 full-pair neural and36 full-pair ridge fits were reused with hashes.
All396 policy views were evaluated on38,102 targets at7,087 queries across the
same six already-opened model-selection localities. Three seeds and3,000 paired
locality bootstrap resamples are reported. Calibration/confirmation stayed closed.

## Main Finding

The earlier development gain is reproducible, but I cannot attribute it to
consistently better neural ranking or a strong general neural-dynamics effect.

| Registered question | Three-seed mean result across six source-role assignments | Interpretation |
|---|---|---|
| Full neural vs ridge, same forecasts | -0.145470% to +3.099757% all ADE; 3 positive CIs, 2 negative, 1 overlap | Mixed, not a universal neural-scoring advantage |
| Full neural vs ridge, matched query counts | -0.090299% to +0.071141%; 0 positive CIs, 2 negative, 4 overlap | Better ranking at matched coverage is not demonstrated |
| Motion-only neural vs ridge, matched counts | -0.146654% to -0.008130%; all6 CIs negative | Ridge ranking is better in this motion-only control |
| Full neural vs retrained motion-only neural, all ADE | -0.322571% to +0.560254%; 2 positive CIs, 4 overlap | Neural trajectory access is not a robust overall contribution |
| Same full-vs-motion comparison, hard ADE | -0.152995% to +0.511910%; 0 positive CIs, 1 negative, 5 overlap | Hard-dynamics contribution not established |
| Same full-vs-motion comparison, easy ADE | +0.778339% to +4.293073%; all6 CIs positive | A narrower, development-only easy-subset benefit remains |

The six comparisons share readout localities and source models; they are not
six independent confirmations. Intervals are exploratory, not multiplicity
adjusted. These percentages use each named comparator, not a common denominator.

## What Still Works

The full neural bridge exactly reproduces the prior aligned all-risk decisions.
It improves over the producer-training-selected motion forecast by2.893989%-
11.358978% across18 configurations, with worst easy degradation0.730345% and
18/18 net-easy passes. The retrained motion-only neural bridge also improves
over the corresponding selected motion forecast by2.892281%-10.956458%, with
zero worst-locality easy degradation in all18 views. Ridge alternatives are
competitive: the full ridge system has18/18 easy passes and zero degradation.
The gain from learning when to select motion forecasts remains useful.

## What I Am Not Claiming

Matched counts do not imply shared estimated risk budgets. The full neural
bridge exceeds its realized positive-harm easy ratio in68/108 dependent
locality/views, despite passing net easy preservation. Full ridge still exceeds
that ratio in41/108 views, and motion-only neural in38/108. None is calibrated.

No winner or deployment is promoted. No new trajectory predictor, JEPA or
Transformer was trained here. A benefit from access to existing neural forecasts
on the easy subset is not evidence that this run learned new dynamics. Historical
Stage37/43/44 scores do not acquire clean-test status through this experiment.
This remains image-pixel, native annotation-step development evidence, not
metric/seconds, human gold, physical safety, true3D or foundation success.
Stage5C and SMC remain off. The project is not yet submission-ready.

## Next Decision

Keep neural and ridge scoring in the comparison set. The next useful repair is
reference-relative calibration and support-aware coverage, with the simplest
competitive scorer as a control, not a larger architecture by default. Before
opening the reserved calibration role, freeze a small method family and check
finite-sample risk identifiability and power on already exposed source/development
data. Confirmation stays sealed until that protocol and model are fixed.
