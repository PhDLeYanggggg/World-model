# Reference Protection Did Not Repair Conditional Risk

## Primary Result
I completed the registered matched continuation: 72 real Torch risk-head fits,
144,000 additional updates and 576 frozen policy views. Registration dcc3cf7b
preceded training; decision freeze 81fec73d was pushed before this round's C
outcome readout. No trajectory forecaster, threshold or source role changed.
There are 360 fresh readout views and 216 cached_verified control views.

Protecting reference-cost predictions does not establish an advantage over
equal-budget shared continuation. Full protected-joint versus continued-joint
all-ADE changes range from -0.0543% to +0.2267% across six three-seed source
assignments. All six locality-bootstrap intervals include zero. This is a
failure to demonstrate the proposed repair, not proof of exact equivalence.

| Full forecast pair | Complete observed risk /18 | Net easy passes /18 | Easy-positive-harm violations /72 dependent locality views |
|---|---:|---:|---:|
| Original mean joint | 6 | 18 | 15 |
| Equal-budget continued joint | 4 | 18 | 16 |
| Reference-protected joint | 5 | 18 | 16 |

Full protected-joint also has two all-positive-harm violations. Preserving net
easy error does not control the positive harm suffered by individual cases.
Against the preceding raw neural rule, the protected all-ADE contrast has one
positive interval, two negative intervals and three overlaps. I am not
promoting this model to deployment or to the main paper method.

## What Additional Training Explains
Continued-joint versus the original mean-joint changes all ADE by -0.0082%
to +0.5228%: three positive intervals and three overlaps. Protected-joint
versus the original changes it by +0.0011% to +0.7490%, also only three
positive intervals. A favorable comparison with the old checkpoint therefore
does not isolate the reference guard as its cause. Both new arms have worse
complete-risk pass counts than the original mean head.

The motion-only control is not a rescue. Complete-risk passes are 12/18 for
the original, 9/18 for continued and 6/18 for protected. Protected preserves
net easy in 18/18 settings, with worst degradation 1.8602%; continued passes
17/18, with worst degradation 2.0119%. Both the favorable and unfavorable
parts of this comparison remain in the results.

## Secondary Findings
For independent dual gates, reference protection improves full all ADE by
0.1145%-0.6192% over continued learning, with six positive intervals. But its
complete-risk passes fall from 15/18 to 14/18. This secondary accuracy result
does not overturn the registered joint-policy result or establish safety.

Protected joint allocation improves full all ADE by 0.5318%-2.7373% and hard
ADE by 0.5543%-3.7551% over its independent dual gates, with six positive
intervals each. However, complete-risk passes fall from 14/18 to 5/18. Against
the same-query-count hash control, all-ADE changes +0.0391% to +0.6609%, with
five positive intervals and one overlap. These counts are not matched for
realized harm. The evidence supports a development accuracy effect of
allocation, not a risk-matched superiority or physical-safety certificate.

## Interpretation and Boundaries
Reference predictions were preserved bit-for-bit, so an implementation failure
to freeze them cannot explain away this negative result. Freezing an estimated
denominator does not make it correct, and it does not fix conditional harm
transport. The full-population and fixed-action diagnoses are reported in
[fitting_transport_analysis.md](fitting_transport_analysis.md).

In full B, protected easy-harm MSE improves over matched continuation in
17/18 settings, but only 3/18 improve on C. On the same fixed raw-action mask,
full-B harm coverage decreases from 0.61847 at the original checkpoint to
0.54905 protected; C coverage falls from 0.58180 to 0.52150. This locates a
selected-population estimation problem in addition to transport failure.

C is excluded from current A/B fitting but historically opened for development.
Three seeds are averaged within locality before 3,000 bootstrap draws of four
C localities. Repeated roles and unadjusted multiple comparisons are not
independent confirmatory evidence. Six selection localities were not evaluated;
twelve reserved calibration and six confirmation localities remain closed.

This remains obs8/pred12 annotation-step, raw-stride12, image-pixel research
with detector-derived labels. No metric/seconds, human-gold, physical-safety,
true3D, foundation or submission-ready claim. Stage5C and SMC remain off.
