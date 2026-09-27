# Matched Signed Risk-Budget Objective

## Question

The previous144-head source cross-fit learns better average moments than
constants, yet neural accepted-subset positive harm is2.85% in fitting sources
and4.87% in held sources under a nominal2% screen. Test whether aligning the
objective directly with the signed risk-budget excess improves this mismatch.
This is an exploratory training-source objective experiment, not a new official
primary trajectory endpoint or an independent safety certificate.

## One Changed Factor

Keep the exact previous355 causal features, forecaster bank, candidates,
three-source-fit/one-source-held roles, seeds17/29/43, mean/std/CV cost scale,
initial network weights,22914parameters, envelope constraints, site-balanced
sample stream, AdamW learning rate0.0003, clip5,batch256 and2000updates. Train144
new heads. The first100-update pilot resumes inside the budget. Match every
sampler/RNG state and draw count to the sealed two-moment control. Fresh control
inference must match the old cache. Do not retrain or select a new forecaster.

Let y=(CV_ADE, positive(candidate_ADE-CV_ADE)) and p=(p0,p1) be the existing
two-output parameterization. Change only loss:

    previous: mean((p0-y0)^2, (p1-y1)^2)
    new:      ((p1-0.02*p0)-(y1-0.02*y0))^2

Both objectives use the identical fitting-only CV cost normalization. No extra
loss weight, clipping, epoch selection or new target normalization. The new
components are an INTERNAL SCORE BASIS, not separately identified/calibrated
reference/harm moments. Only q=p1-0.02*p0 is supervised. Future labels construct
training targets only, never inference inputs. Unknown labels remain excluded,
zero-reference positive harm retained. Target gradients are detached.

## Readout and Gates

Commit registration before training and freeze all new/control scores before
comparative scoring. The outer readout is not scored; independent selection,
calibration and confirmation remain closed. No new thresholds are chosen.

Primary mechanistic contrast: held-source MSE of the signed excess, new versus
matched two-moment control, for the neural candidate. Positive percent means
lower MSE. Report the damping contrast, all three seeds, fitting comparisons,
constant score skill and native/scaled error. Average dependent role/seed views
within each fixed source locality, then equal-weight twelve localities;3000
locality bootstrap draws, seed71431. Undefined ratios stay undefined.

For both objectives report the exact same fixed all-risk-only diagnostic rule,
q<=0: coverage, positive harm/reference on the screened subset, net ADE gain vs
CV on all rows, positive-easy and hard rows using producer-training cutoffs,
and zero-reference absolute harm. These are NOT the old full utility/easy policy,
not easy degradation inferred from positive-harm ratios, and not scene-joint
deployment results. Benefits do not cancel positive harm in the risk target.

A positive lower primary interval supports the loss hypothesis only. Report
separately whether every held locality has nonzero coverage, observed screened
positive-harm ratio<=2%, easy net degradation<=2%, and zero reference harm.
Passing these exploratory screens still does not certify calibration or permit
deployment. Never call zero-action fallback a learned benefit. A later calibrated
policy needs strictly nested fit/calibration exclusions through the whole
producer chain; ordinary cross-fit outputs cannot silently reuse the outer
source's labels through another head.

## Runtime and Claims

Native arm64 CPU4/interop1, workers0, no resource probing. Checkpoint/heartbeat200
updates and atomic resume; retain10GiB disk reserve. Previous fit measured under
seven local minutes and8.12GB peak RSS, so this small-head experiment remains
local. No new HPC jobs, remote environment changes or simulation-project access.
Cached_verified forecasts plus fresh risk-head training, not fresh raw extraction.

Observed8/predicted12 at raw-frame stride12, detector-derived silver image-local
tracks. Not human gold, metric, seconds, physical safety, true3D or foundation.
No Stage5C execution, SMC, independent-role opening, deployment or submission.
