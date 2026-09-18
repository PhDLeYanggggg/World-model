# Past RGB Gives a Fragile Window-Level Gain, Not Robust Transfer

## Evidence Status

Date: 2026-09-18. Status: complete fixed information probe; no model promotion.
`fresh_run`: 30 real Torch fits, 36 prediction cells, 60,000 optimizer updates.
`cached_verified`: approved main/source inputs and the previous geometry-only
study, not newly converted data or newly trained old controls.
`not_run`: a new trajectory policy, independent confirmation, physical-time or
metric validation. Stage5C and SMC remain disabled.

Registration SHA256:
`9edee7d4fe9ef0b97d370ad7804f78ec094102580ef5e7ba12e8e0adf5596af9`.
Report SHA256:
`91914580ba5bad24c66b87878ad0d8258253356c2e62151c779807b3e2f0bc7a`.
The design was pushed at `db7d30f4` before the full experiment. No held-score
checkpoint, threshold or ensemble was selected.

## What Was Tested

Can actual past pixels improve prediction of later annotation-coordinate change
beyond the same geometry and image-coverage information? The matched arms have
43,849 parameters, the same initial weights, sampled rows, optimizer and fixed
2,000-update budget. A shared frame CNN encodes eight past 32 x 32 crops; a small
MLP combines them with 476 past-only geometry features. The control zeros RGB
but retains coverage. There is no trajectory decoder or latent rollout.

Three schedules use main-only, SDD-only or half-main/half-source sampling mass;
seeds are 17, 29 and 43. Source-only fits are shared between the two held-site
directions, hence 30 fits rather than 36 independent models.

Main support is 365 stationary-history queries from 31 local IDs: ETH 81/5 and
Hotel 284/26. Zara has no such queries and is not_run for this probe. Approved
SDD train40 provides 22,374 complete-label stationary queries from 726 local IDs
in 36 videos/five sites. Another 6,460 incomplete stationary queries are unscored,
not mislabeled negatives. Labels mean any later annotation-center change, not
human intent or verified body-motion gold. Completeness can itself limit the
population to which the results apply.

The primary 11,966-query forecast study, eight observed/twelve predicted native
annotation steps, equal-site past-normalized ADE and sealed roles are unchanged.
Source stride12/+144 raw frames is not physically equated to main annotation steps.
Supplied retrospective histories remain an offline-annotation protocol, not
strict sensor-as-of observations.

## Fixed Results

Positive values below are **absolute Brier reductions**, not trajectory gain
percentages. Each value averages losses over three seeds, not their probabilities.

| Training | RGB vs mask, ETH | RGB vs mask, Hotel | RGB vs own train prior, ETH | RGB vs own train prior, Hotel |
| --- | ---: | ---: | ---: | ---: |
| Main only | +0.016172 | -0.002236 | +0.058104 | -0.196108 |
| SDD only | -0.090782 | +0.002252 | -0.258466 | -0.020036 |
| Mixed | +0.039378 | +0.036045 | -0.002996 | -0.000676 |

The registered window-weighted RGB contrast is positive in both sites under
mixed training, in all three seeds. That positive result is retained, not
redefined away. It is nevertheless insufficient for a useful transfer claim:

- Mixed RGB is slightly worse than its own constant training prior in both sites.
  Its ETH/Hotel Brier is 0.277526/0.266622. On Hotel the source-prior constant is
  better still, at 0.247935. No RGB schedule beats its own training prior in both
  held directions.
- With equal local-agent weighting, mixed RGB-minus-mask lift is **-0.019117**
  on ETH and **-0.038780** on Hotel. Their 2,000-resample conditional intervals
  are [-0.133658, +0.083091] and [-0.149999, +0.062546]. Window length and
  heterogeneous per-agent effects materially change the average.
- Removing one local ID at a time leaves mixed Hotel's agent-balanced contrast
  negative in all 26 omissions. ETH's sign changes across omissions. These are
  descriptive sensitivity results, not independent confidence guarantees.
- SDD-only RGB harms ETH in every seed; its conditional agent interval for
  RGB-minus-mask is [-0.135363, -0.072244]. A small Hotel row gain cannot offset
  or conceal this failure.

All probability scores, calibration errors, seed values and intervals are in
[results.md](results.md), [fit_metrics.csv](fit_metrics.csv) and [analysis.json](analysis.json).

## Failure Taxonomy

**Prevalence and weighting effects are measured.** For mixed ETH, the row Brier
gain over mask decomposes into +0.079563 from mean-probability shift and -0.040185
from the varying-prediction term. For mixed Hotel the corresponding terms are
-0.000793 and +0.036838. Thus Hotel does improve on a harmful variable mask
predictor, but this is not enough to beat its own constant prior. Against a
constant prior, the varying-prediction term is negative for both source-only and
mixed RGB in both sites. This is an exact descriptive score decomposition, not
a causal explanation or an inference-time recalibration.

**Main-only overfit is measured.** Training on the 81 ETH rows reaches nearly
zero training Brier but has Hotel Brier around 0.52. Across the schedules, equal
update budgets do not mean equal per-row exposure: seed17 main-only samples
each Hotel/ETH row on average 450.7/1,580.2 times; source-only averages 5.72.
Mixed source averages 2.87 while its main side averages 224.7 or 787.8. Paired
RGB/mask exposure is identical; comparisons between schedules must retain this
difference. These are repeated draws, not independent examples.

**Finite-budget source learning is not an impossibility result.** Source RGB
training Brier is approximately 0.213-0.220, slightly below mask 0.220-0.222;
it is not near-perfect source fitting. This study does not establish convergence,
prove that pixels contain zero information, or identify architecture size as the
root cause. The source training error is not a held-source generalization score.

**Image availability is verified; posture quality is not.** Every requested
history has supported, temporally changing crops. The 36,234 unique source past
crops have median mapped box extent about 10.9 x 12.7 model pixels; 31.5% have
an axis below eight pixels. This may limit fine cues but is not an isolated
resolution ablation or proof of cause. The main cache has no verified box extent
for an equivalent audit. Compression, other agents and alignment can also cause
pixel change. See [pixel audit](pixel_information_audit.md).

**Cross-domain semantics remain unresolved.** Main and source annotation clocks,
camera appearance, label resolution and movement magnitudes are not equated.
The classification objective detects coordinate changes, not a shared verified
behavioral event. Existing training support cannot by itself resolve that gap.

## What Comes Next

The next falsifiable check should separate source-internal generalization from
source-to-main transfer: first audit class/agent support per physical SDD site,
then register matched RGB/mask source-site-held-out fits using only the already
admitted train40 population. Keep main development/calibration/confirmation
closed. This is an exposed-source diagnostic, not a new independent test set.
Do not fit a held-site prior, change the main metric or run another switching
threshold search to rescue these examples.

If RGB has no useful held-source information, investigate target semantics,
visible body support and representation/training budget before another transfer
head. If it generalizes within source but not to main, isolate clock/label and
appearance mismatch. Only a supported probability cue justifies a subsequent
trajectory-direction/utility test. No residual or deployment follows automatically.

State-change recognition alongside a strong motion baseline is already discussed
in prior work; it is not by itself a new method claim. The proposed joint
intervention contribution still needs matched forecasting benefit and independent
evidence. [Uhlemann et al., Sections V-B and V-D](https://arxiv.org/html/2308.05194v3).

## Verification and Limits

The full invocation took 47.11 minutes; summed fitting time was 2,693.54 seconds.
This includes 100 training-only pilot updates plus 59,900 subsequent updates.
All 30 checkpoint probability replays are exact. Fifteen paired-stream checks
pass, and completed resume preserves 91 immutable artifacts and the report hash
with zero new fits/updates. Thirty-four focused tests pass; the full legacy suite
was not rerun. Original aggregate figures were visually inspected.

Five ETH and 26 Hotel IDs, repeated windows, contemporaneous agents and exposed
sites prevent independent scene-level confirmation. Bootstrap intervals are
conditional descriptions, not a new-scene certificate. Multiple fixed contrasts
are reported without selecting a favorable model or claiming multiplicity-adjusted
significance. The primary window estimand has not been replaced by the agent
sensitivity check.

This remains 2.5D/dataset-local or pixel-space research, not true 3D, metric,
seconds-level, foundation or submission-ready evidence. No current deployment
policy changes. [Reproduction](reproducibility.md), [evidence gates](gates.md).
