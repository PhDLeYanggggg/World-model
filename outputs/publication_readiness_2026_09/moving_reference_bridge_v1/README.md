# Learning where to stay between CV and OLS4

I trained a forecast that moves along the line between CV and OLS4
at each future step. Along that line the two distances sum to the
distance between the references; going off it can add disagreement.
To test whether the model knows when to leave the line, I compared
a gate that predicts future label availability from past motion and
current neighbors with one that uses the average availability by
step on the fitting sources. A third model stays on the line.
None sees the realized future mask at inference.

**Fresh run:** 27 matched PyTorch fits and 21,600 updates on arm64,
about 103 seconds, four Torch threads and no worker processes.
Within each of three source rotations, three groups fit the model,
three calibrate it, and three are held. Three seeds give 72 views of
24 exposed development recordings, not 72 independent recordings.
**Cached verified:** source roles, row ordering and both v155 paths.
The three arms share their initialization, training draws and loss.
Checkpoints capture model, optimizer and sampler every 100 updates.

| Forecast | Available ADE / endpoint FDE, px | Lower gain vs OLS4, pp | Original risk passes | Joint passes |
| --- | ---: | ---: | ---: | ---: |
| CV | 94.11 / 246.62 | -55.77 | Undefined edited support | 0/72 |
| OLS4 | 91.60 / 234.45 | 0.00 | 0/72 | 0/72 |
| Earlier OLS4 objective | 89.15 / 232.59 | -5.56 | 0/72 | 0/72 |
| Bridge only | 94.06 / 246.44 | -55.61 | 54/72 | 0/72 |
| Mean-by-step gate | 93.26 / 245.08 | -53.43 | 8/72 | 0/72 |
| Learned availability gate | 93.28 / 244.92 | -53.86 | 9/72 | 1/72 |

The bridge's mean peak displacement from CV is only 0.56 px;
OLS4's is 167.56 px on this roster. Its higher risk pass count
reflects how close it stays to CV, not a gain over the strong
reference. Strict calibration accepts **0/9** options; CV is the
fallback and its undefined edited support is not a pass.

The learned gate's Brier score is **0.1761** versus **0.1835**
for the training-only mean; the paired difference is
**-0.0074 [-0.0377, +0.0189]**. The interval does not establish
better support prediction. At matched recipients and per-row
peak (mean 4.67 px), its direct conservative gain versus the
mean gate is **-1.233 pp [-2.289, -0.464]**. The matched gain on
available labels is **-0.099 pp [-0.337, +0.077]**; the additional
loss in the lower contrast is a debit for missing positions,
not observed evidence of worse trajectories there. The raw
conservative contrast is -2.357 pp [-4.448, -0.874].
Intervals resample nine source groups after averaging the seeds
(3,000 draws). Each comparison has its own missing-position debit,
so its lower bound is not the subtraction of two standalone bounds.

I did not promote the new forecast. The loss encourages proximity
to CV without creating reliable gain against OLS4, and the
matched comparison gives no reason to increase the learned gate's
capacity on these recordings. The next useful experiment should
prioritize source-separated data with stronger future-label support
and check whether an earlier, genuinely stronger forecast can be
selected safely there. I did not fill any missing future labels,
alter the 2% rule, or open Moving A or DroneCrowd confirmation.
Stage5C and SMC were not run. Coordinates are image pixels and
forecast lengths are annotation steps.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Readout](readout.json) |
[Manuscript v157](../evidence_manuscript_v157/README.md)
