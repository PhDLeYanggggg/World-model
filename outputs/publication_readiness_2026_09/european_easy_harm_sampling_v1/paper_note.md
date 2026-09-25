# Candidate Experimental Subsection: Exposure Is Not Calibration

We test whether low exposure to costly easy-case errors explains a conditional
risk predictor's failure. Keeping candidate forecasts, architecture, mean loss,
normalization, initialization and optimization budget fixed, we replace
uniform within-locality training draws with an equal mixture of uniform mass
and supervised easy-harm mass. Each loss is reweighted by its exact p/q ratio.
This preserves the expected objective without implying an unbiased adaptive
optimizer update or lower variance. Importance sampling is established prior
work, not our methodological novelty.

Across three seeds and all six ordered producer/controller assignments, the
sampling intervention does not consistently improve source-held accuracy.
Full joint-policy all-ADE gain over the unchanged uniform mean control ranges
from -0.1691% to +0.4865%, with two positive, two negative and two overlapping
locality-bootstrap intervals. Complete observed risk passes decrease from
6/18 to 4/18, although every setting preserves net easy error within 2%.

The in-training and transport diagnostics diverge. Easy-harm component MSE
improves in 13/18 full-pair training settings but only 4/18 source-C settings;
both reference-mass components worsen in all training settings. On a shared
fixed action mask, the median ratio of predicted to observed easy harm on C
changes from 0.5818 to 0.5783. Thus additional positive exposure alone does
not resolve the conditional underestimation.

Joint allocation nevertheless outperforms individual dual constraints by
0.8446%-2.6287% all ADE, with all six intervals positive. The query-count-matched
control also favors joint allocation, but this comparison does not match
realized harm. We therefore interpret it as an allocation mechanism worth
further study, not a deployment or risk guarantee.

Limitations: four C localities per assignment, overlapping role views,
historical source-development exposure, detector-derived labels, and
unadjusted multiple comparisons. Calibration and confirmation remain closed.
There is no newly trained dynamics model in this experiment. Units are image
pixels and annotation steps, not meters or seconds; no human-gold, physical
safety, true3D or foundation claim. Stage5C and SMC remain off.

Prior work: [Katharopoulos and Fleuret (ICML 2018)](https://proceedings.mlr.press/v80/katharopoulos18a.html).
The present fixed label-mass mixture is not their gradient-bound sampling
algorithm and inherits no speedup guarantee.
