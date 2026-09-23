# Fixed Source Predictors for External Validation

## Outcome

Six real Torch models completed the pre-registered fitting budget on 2026-09-23:
the existing deterministic Transformer and EqMotion control, each with seeds
17/29/43. There were 24,000 optimizer updates and 1,536,000 sampled training rows
in total. Summed fitting time was 4,473.20 seconds, or 74.55 minutes, on local native
arm64 CPU with four compute threads, one interop thread and no loader workers.
The run was not reduced to a timing pilot or quick experiment. No CREATE job was
needed or submitted.

`fresh_run`: training and initial endpoint checks. `cached_verified`: a second
process reloaded all six checkpoints, verified 234 bound dependencies, matched
per-seed sampling counts, and reproduced all 384 fixed source-input predictions
exactly by hash. Loss aggregation is cached-log analysis, not a second training
run. The 157 scoped regression tests pass. The broad unrelated legacy suite was
not run as part of this change.

This completes a fixed predictor bank, not independent external validation or
scientific admission. Reserved-source inference, calibration and confirmation
are all `not_run`. No model or seed was selected from external results. No new
deployment, cross-dataset gain, method superiority or submission readiness is
claimed. Stage5C and SMC remain off.

## Fitting Population and Budget

All 175,756 eligible source rows from 33 recordings are in the fitting pool:
coupa, deathCircle, gates and hyang. All four sites were already development
exposed. They are not independent confirmation data. Each model used 4,000
updates with batch 64 and the existing native-coordinate objective, site-balanced
sampling and source-fit normalizers. This is a fixed stochastic budget, not a
claim that every eligible row was sampled. The unique row counts were 121,070,
120,874 and 120,922 for seeds 17, 29 and 43, identical across the two model families.

| Predictor | Seeds | Trainable parameters | Total updates | Summed fit seconds |
|---|---|---:|---:|---:|
| Transformer | 17, 29, 43 | 88,514 | 12,000 | 249.16 |
| EqMotion control | 17, 29, 43 | 580,676 | 12,000 | 4,224.04 |

EqMotion checkpoints also retain unused frozen author heads; stored parameter
count must not be reported as the trainable count. Full checkpoint bytes remain
local and Git-ignored. The first 100 EqMotion updates were a real timing pilot
resumed in place, not discarded and rerun. Checkpoints were saved every 200
updates with optimizer, sampler and Torch RNG state; heartbeat every 50 updates.
No runtime hang, nonfinite training loss or failed training process occurred.

The main prediction task remains eight observed/twelve predicted native
annotation steps. The SDD input grid in this population has stride 12 raw frames;
the new external prefix adapter is explicitly stride 1. These are not asserted to
cover equal physical time. Pixel or dataset-local coordinates are not metric,
and offline annotation prefixes do not establish sensor-time label provenance.

## What the Losses Do and Do Not Show

The [full loss table](training_losses.md) and [486 logged batches](training_loss.csv)
are retained. EqMotion's initial loss falls substantially as its new head fits;
the smaller Transformer's logs change less, and seed 17's first-ten versus last-ten
mean does not improve. Neither observation establishes convergence or superiority:
these are noisy source-training minibatches, not validation curves. The endpoint
was fixed before training. No training-loss winner replaces the six-model bank.

The separate past-normalized ADE debug field can be large for near-stationary
histories and is not the native-coordinate training objective. Gradient norms
in the log precede clipping at 5. Reporting these fields as held-out accuracy or
unclipped update size would be incorrect.

## Why This Refit Was Needed

Earlier development comparisons leave one SDD site out per predictor. Choosing
one of those outer-fold models according to external performance would introduce
an avoidable selection decision. This refit uses every approved development site
under a fixed budget, leaving six unambiguous predictors for subsequent work.
It is not a new architecture search or a positive external experiment.

Full-source in-sample errors must not become out-of-fold risk-head targets. The
existing row-site-excluded and pair-excluded producer caches remain necessary for
fitting gain/harm heads and the complete downstream policy. Before external
readout, that policy chain, source-use/exposure checks, source-specific admission
and the one-shot evaluation rules must also be frozen. DUT's two author-described
locations are limited calibration support, not a certificate of 2% risk. DroneCrowd
is reserved as one collection; its 45 exclusion groups are not 45 independent sites.

## Reproduction and Artifacts

Registration was committed before training in `cdff798e`. The authoritative
configuration is `configs/m3w_external_predictor_refit_v1.json`. Model paths and
individual hashes are in [analysis.json](analysis.json); replay evidence is in
[replay.json](replay.json). Training artifacts remain under
`data/stage_cvpr2027_experiments/external_predictor_refit_v1/trials/`.

```sh
.venv-pytorch/bin/python scripts/run_m3w_external_predictor_refit.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_external_predictor_refit.py --resume
.venv-pytorch/bin/python scripts/run_m3w_external_predictor_refit.py --verify
.venv-pytorch/bin/python scripts/summarize_m3w_external_refit.py
```

On an existing completed run, `--resume` validates completed receipts rather than
retraining. `--verify` requires completed endpoints and unchanged source bindings;
it cannot invent missing models. Reproduction needs the verified local source
assets and pinned EqMotion implementation; the light public report is not a
self-contained training-data distribution.

Analysis SHA256:
`494af734ba3ddd9f18369c7485e28252d860b804e7db81776b1bc77713beb0a5`.

The initial CSV export used CRLF line endings, which the repository whitespace
check rejects. The exporter now writes LF; a regression test and parsed-record
comparison confirm all 486 log records are unchanged. No checkpoint, training
analysis or scientific result was modified by this formatting repair.

Next substantive step: complete the source-only gain/harm policy using proper
out-of-fold producers, bind the past-only external adapter and source admission,
then carry out explicitly limited calibration and frozen external confirmation.
The research goal remains active and unmet; routine auditing is delegated.
