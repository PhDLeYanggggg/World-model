# Locality-Excluded Harm Tail Diagnosis

## Material Passport
Registered before new training or held-locality outcome readout. The preceding
reference-protection study, commit 14c3bae0, is cached_verified: all 60 public
artifacts and 86 source bindings are checked. Its main accuracy repair failed,
and selected easy-harm remained underestimated despite better training MSE.
This study repairs the diagnostic's in-sample limitation; it does not promote
a deployment policy or retrain a trajectory forecaster.

## Question
Can the current causal risk features rank harmful cases in a locality excluded
from risk fitting, or is the main problem inaccurate harm magnitude despite
useful ranking? Separate occurrence ranking, costly-tail retrieval and mean
harm estimation. A good AUROC does not establish calibrated costs or safety.

## Fit Design and Exclusion
For each existing source A/B assignment, seed17/29/43 and full/motion-only
forecast pair, hold one of the four B localities out and train on the other
three. This gives 144 fresh mean-moment heads at 2,000 updates each, 288,000
updates total. Same width64, batch256, lr0.0003, weight decay0.0001 and clip5.
Cold-start the original four-moment head, not the failed protected checkpoint.
The source-A forecast pair and causal input schema remain unchanged.

Every inner fold recomputes normalization, cost scale and the positive-CV
25th-percentile easy cut from its three training localities only. Held labels
and features cannot change this definition; a perturbation test checks it.
Thus inner easy definitions differ between folds. Do not pool them as one
fixed-event calibration estimate or directly compare them with the existing
whole-B-cut C endpoint as an identical target. This diagnostic does not change
the primary evaluation protocol or its 2% risk limit.

The diagnostic causal subset is positive forecast-disagreement envelope. It
uses no B-fitted teacher selection mask. Feature policy bits, where present,
come from source A; held B is excluded from that producer. The original mean
objective is retained; its auxiliary group diagnostic uses only constant and
envelope masks. Future positions supply loss/evaluation labels only.

Train score-bin edges at weighted quantiles 0.5/0.9/0.95/0.99 on supported
training rows with equal-locality weights. Scores are predicted easy-harm
moment, moment/disagreement-envelope, and the causal envelope control. None
is advertised as an event probability. Freeze all heads, scores and bin edges
and push their manifest before held-out readout. No held-out early stopping,
model selection, threshold fit or event redefinition after readout.

## Readout
Report per-locality, all supported rows and the fixed envelope-positive subset:
positive counts/prevalence, AUROC/AP, AP/prevalence, top10%-ranked harm-mass
share, oracle top1% harm-mass concentration, MSE, predicted/actual harm mass,
and every train-defined bin. Tied scores receive proportional mass, not
label-dependent tie breaking. Zero-event populations are not_estimable, not
perfect scores; fewer than 20 positive or negative rows is flagged weak support.

The primary diagnostic contrast is moment versus envelope top10%-ranked harm
capture on held B. Average three seeds within locality, then 3,000 bootstrap
draws of four held B localities per source assignment. Retain all six assignments
and both forecast pairs. Additional fraction-score and AUROC/AP comparisons
are descriptive and unadjusted for multiplicity. These are dependent source-
development views, not new independent scenes or confirmatory trials.

Also diagnose the original full-B mean model on B and C using its original
event definition and fresh B-only bins. C predictions/targets are existing
hash-verified source-development artifacts. This is not fresh C training or
unopened confirmation. New inner models are not used to choose C policies.
Only source A/B/C data are used; the six selection, twelve reserved calibration
and six confirmation localities are not evaluated.

## Interpretation and Next Action
If occurrence ranking looks good but tail retrieval does not beat the envelope,
do not label magnitude calibration the sole missing step. If tail ranking is
consistent but harm mass remains biased, investigate selection-aware magnitude
or uncertainty estimation. Neither branch authorizes another C threshold sweep,
silent error clipping, a relaxed budget or a risk certificate. The result will
determine one subsequent registered intervention; no new policy is deployed here.

## Runtime and Prior Work
Local native-arm64 CPU4, interop1, workers0, checkpoint/optimizer/RNG resume,
PID/heartbeat and 10GiB free-disk reserve. A 100-update real pilot resumes
inside the first head's fixed budget. CREATE was checked read-only; no remote
job is submitted or changed. Remote M3W asset inventory remains not_run.

[Hebert-Johnson et al., ICML2018](https://proceedings.mlr.press/v80/hebert-johnson18a.html)
distinguish aggregate prediction accuracy from calibration across identifiable
subpopulations. This motivates the diagnostic, but we do not implement their
algorithm, prove multicalibration or inherit a domain-shift guarantee. Reading
scope: abstract and introduction, not a reproduction or full theorem audit.

Obs8/pred12 annotation steps, raw stride12, image pixels, detector-derived
labels. No metric/seconds, human gold, physical safety, true3D, foundation or
submission-ready claim. Stage5C and SMC remain off.
