# Reference-Cost and Harm Cross-Fit Diagnosis

This source-development experiment diagnoses the matched intervention failure;
it does not introduce a deployment policy. The improved frozen forecaster has
raw benefit, but its protected controller loses to equally protected damping.
The previous selected-set diagnostics suggest two distinct errors: inaccurate
positive harm and inflated predicted reference cost. A ratio can appear safe
because its denominator is too large even when harm is overpredicted overall.

## Fixed Design

Reuse the verified forecast bank and the twelve already opened source-training
localities. For each of nine producer/seed banks, each of its two four-locality
controller rosters, and each of two candidates, leave out one controller locality.
Fit an unchanged all-risk head on the other three. Exactly144 fresh Torch heads,
2000updates each, three seeds17/29/43. The producer, inner held locality and
outer readout are excluded from every head's fitting normalizers and labels.
The outer readout is not scored. Independent model selection, risk calibration
and confirmation remain closed. Loading a verified bank is not fresh extraction.

The unchanged355-feature causal schema includes past histories, neighbors, CV
and candidate rollouts. Targets are CV ADE and positive candidate harm relative
to CV; zero-CV observations remain included, unavailable futures stay unknown.
The envelope and features read no future labels or masks. Same64-wide head,
AdamW, learning rate0.0003, batch256, site-balanced sampling, clip5. First-head
100-update pilot resumes inside its2000-update budget. CPU4/interop1, native
arm64, workers0. Keep10GiB disk reserve; no new raw/cache copying or HPC jobs.

Commit registration before training, then hashes of all fitting/held predictions
before comparative scoring. Report both in-sample and inner-held predictions
from the SAME three-source head. Training-defined score edges use weighted
quantiles0.5/0.9/0.95/0.99. No full-four-source utility/easy heads select inner-held
rows: that would leak the held controller locality into this diagnosis.

## Outcomes and Interpretation

Primary diagnostic: reference-cost MSE skill over its fitting-only constant,
on the inner-held locality, and its paired change from the same head's equal-site
fitting skill. Positive skill means smaller MSE. Harm-MSE skill, actual/predicted
reference and harm moments, severe-harm mass ranking, fixed score-bin calibration,
and predicted harm/reference <=2% screening are secondary. The screen uses only
this all-risk head; it is NOT the previous complete deployment policy. Report
accepted harm and reference separately and retain zero-reference absolute harm.
No threshold/model selection from these outcomes; no claimed conformal guarantee.

For each held locality average the dependent producer/seed views, then average
the fixed12localities; locality bootstrap3000, seed71431. Overlapping rows and
repeated fits are not independent samples. Missing ratios remain undefined.
Report seed breakdowns. Train-vs-held gaps are diagnostic, not a causal proof
that representation, optimization or distribution shift alone caused the error.

After diagnosis, a loss/representation/support repair requires a separate fixed
contrast. Do not silently substitute a repaired head into this experiment or
open the independent roles. Technical replay is not empirical model benefit.
Observed8/predicted12, raw-frame stride12, detector-derived silver trajectories,
image-local coordinates; no metric, seconds, human-gold, physical-safety, true3D
or foundation claim. No Stage5C execution, SMC or deployment change.
