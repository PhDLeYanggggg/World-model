# Matching the Tree Objective Does Not Repair Neural Risk Ranking

## Evidence Status

2026-09-22. `fresh_run`: twelve native Torch cost-head fits, fixed decisions and
readout, checkpoint replay and separate arithmetic verification. `cached_verified`:
the original neural and ExtraTrees scores, causal forecasts, nested producer
lineage and preprocessing. `not_run`: new trajectory-forecaster training, external
confirmation, independent calibration, raw-frame t+50 supplement or deployment.

Registration was committed and pushed as `a9292047` before fitting/readout.
The four SDD physical sites are development-exposed. Neither the three seeds nor
3,000 physical-site bootstrap resamples create an independent confirmation set.
The task is 8 observed / 12 predicted annotation steps at raw stride 12, in SDD
annotation pixels. No verified metric scale, seconds-level or online-annotation
causality claim is made. Dataset-provided past annotations are the input contract.

## Question and Controlled Change

The preceding equal-count experiment favored forest relative-risk ranking over
neural ranking. The forest and neural head used different losses. This experiment
tests whether matching their empirical squared-cost objective repairs that gap.

The new neural head uses the same 356 causal inputs, 128 hidden units, 45,954
parameters, initial weights, fitting rows, preprocessing, sampler stream, region
weights, seeds and 12,000-update budget as the old ramp-action neural head. Only
the data-fit loss changes from compositional log loss to distance/region-weighted
two-output fraction squared error. Forest capacity, optimizer and regularization
are still different; this is not a controlled architecture-only comparison.

No held-out threshold, checkpoint, seed or model is selected. The final fixed
budget is evaluated. Unknown future costs are never used as zero-cost labels.

## Fixed Results

Percentages below are relative to causal constant velocity (CV). Seed errors are
averaged before equal-physical-site aggregation. Negative easy degradation means
improvement. Switch counts include repeated query/seed instances, not independent
samples. Ratio policies have exactly the same per-site/seed counts; these counts
do not also match forecast displacement mass.

| Fixed policy | ADE gain | FDE gain | Hard gain | Worst site/seed easy degradation | Switches |
| --- | ---: | ---: | ---: | ---: | ---: |
| New square-loss neural, strict | 3.44707% | 4.97638% | 3.58948% | 1.09155% | 37,030 |
| Old log-loss neural, strict | 3.39756% | 4.90029% | 3.67549% | 0.91149% | 33,793 |
| New square-loss neural, relative risk | 2.59970% | 3.82482% | 2.58474% | 0.10713% | 22,539 |
| Old log-loss neural, relative risk | 2.80697% | 4.12167% | 2.97143% | 0.16376% | 22,539 |
| Forest, relative risk | 3.53029% | 5.16628% | 3.76414% | -0.13424% | 22,539 |
| New square-loss neural, net gain | 3.83871% | 5.31275% | 5.76892% | 2.50887% | 22,539 |
| Old log-loss neural, net gain | 3.82776% | 5.28068% | 5.80729% | 2.55829% | 22,539 |
| Forest, net gain | 4.07418% | 5.61974% | 6.17470% | 2.70229% | 22,539 |

The registered primary difference, new-strict minus old-strict ADE gain, is
**+0.04951 percentage points**, with paired development-site bootstrap interval
**[-0.00527, +0.13864] pp**. The superiority gate fails. All seeds have positive
gain versus CV, aggregate and every site/seed observed positive-easy checks pass,
and complete exact-zero-CV harmed cases fall from one to zero. Passing those
checks does not replace the failed primary conjunction or prove population safety.

At fixed counts, new-relative-risk minus old-relative-risk is **-0.20728 pp**,
interval **[-0.25325, -0.14071]**; all four site contrasts are negative. Compared
with forest relative-risk, the new neural head loses **0.93059 pp**, interval
**[-0.99940, -0.85137]**. Net-gain ranking does not rescue protection: every
net-gain arm exceeds the 2% worst-site/seed easy ceiling. No secondary winner is
substituted for the registered primary result.

## Why This Repair Was Rejected

Separate full-fitting-row objective reconstruction shows that the new loss lowers
its weighted squared objective versus the old neural head in **9/12 views**.
Equal-view means are 0.08215949 (new), 0.08293925 (old) and 0.08158509 (forest).
These are fixed fitting diagnostics, not held-out model-selection scores or
comparable native pixel losses across views. The new neural head beats forest
on this fitting objective in only 3/12 views.

Despite this fitting change, complete-label selected mean harm is underestimated
in **all 12 held views** for both the new strict and new relative-risk policies.
Median realized/predicted harm ratios are **2.75984** for new strict and
**5.75455** for new relative-risk. The old relative-risk head's corresponding
ratio is **3.84045**. The forest on its own relative-risk selection has median
0.60927 and no underpredicting view. Own-selection conservatism is not an
independent risk-calibration certificate.

The specific conclusion is that objective matching alone does not repair
conditional harm estimation or useful risk ordering on these explored scenes.
It does not establish irreducible uncertainty, prove all neural architectures
inferior, or justify another unregistered width/loss/threshold sweep.

## Missing Outcomes and Interaction Limits

New strict selects 5,680 incomplete-future instances, including 606 with unknown
ADE. New relative-risk selects 2,917 incomplete instances, including 329 unknown.
Their deterministic full-grid gain lower bounds remain negative in some gates
site/seed slices. No absent label is counted as safe, and missing-label easy
preservation remains unverified.

The existing context-proximity and smoothness proxies are replayed, not physical
safety measurements or evidence of learned joint dynamics. Lower proximity at
matched counts cannot replace the failed accuracy contrast. Earlier scene-joint
controls also remain negative. A causal-only check of interaction support in the
new protected pool is appropriate before spending on another joint solver/model;
it must not retune pair weights or consult future targets.

## Verification and Decision

All twelve checkpoint scores and 96 fixed choices replay. Separate code checks
1,581,804 fitting-view row instances, 527,268 held score rows and 768 scene
reductions. The score check is exact under the original float32 operation order.
Forty relevant tests pass; the full legacy test suite was not rerun. Separate
arithmetic is by the same implementing agent, not external scientific replication.
See [execution notes](execution_notes.md), [replay receipt](replay.json) and
[separate verification](separate_verification.json).

Analysis SHA256: `fb56409613d2aba1950bb6d7c8f2db64a5bb4430181b4c691e0693bee45dc5f4`.

Keep the ordinary forest as a strong developmental comparator, not a newly
deployed model. Keep earlier primary failures and the pinned manuscript snapshot
unchanged. Independent scene admission/calibration roles, competitive forecaster
evidence, useful joint-agent contribution and external confirmation remain open.
Submission readiness is unmet. Stage5C and SMC remain off; no true-3D, foundation,
human-gold or new deployment claim is made.
