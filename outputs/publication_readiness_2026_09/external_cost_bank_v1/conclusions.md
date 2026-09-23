# Fixed Source-Only Gain/Harm Bank: Completed

2026-09-23. The registered six neural cost heads and six matched ExtraTrees
controls are trained and replayed. This completes cost-head fitting for the
external-study predictor bank, not external validation or a new deployment.

## What Ran

- `fresh_run`: six width-64 neural gain/harm heads, 3,000 updates each with batches
  of 256; 18,000 updates and 4,608,000 sampled rows in total. Six forests each
  reach 128 trees, reusing the corresponding neural head's exact per-row counts.
- `cached_verified`: 24 previously fitted whole-site-out trajectory producers,
  all checkpoint training IDs, source-only normalizers, prediction arrays and
  row alignments. No row's forecast producer trained on that row's physical site.
- The population is 175,756 source queries from four development-exposed SDD
  sites. Of these, 143,918 have complete twelve-step supervision; 31,838 do not.
  Inference eligibility remains past-only. Missing future supervision does not
  remove an agent from the inference population or create a synthetic label.
- Native arm64 Torch 2.12.0, CPU four compute threads / one interop thread;
  workers=0. The registered sixteen-tree timing pilot continued in place.
  Summed head fitting time is 209.470 seconds, excluding preparation and checks.
  The earlier six forecasting-model fits took 74.55 minutes; they were not rerun.

The targets are the positive and negative parts of causal-CV minus candidate
native ADE. The causal mean trajectory disagreement bounds their sum. Both heads
fit the corresponding fractions with squared error. Matching sampling weights
does not make neural SGD and tree fitting identical algorithms or capacities.
There is no best-baseline hard-label classification or test-driven threshold fit.

## Verification

All 319 source/code/config bindings verify. A separate process reloads all twelve
endpoints and reproduces 768 fixed source-input cost rows exactly by hash. A second
verification implementation reconstructs Euclidean targets, complete-label
support and all six samplers, and checks both head outputs. All checks pass.
This is independent arithmetic in the same research task, not an independent
researcher's replication or held-out scientific confirmation.

The final focused suite passes 158 tests. It covers the new cost assembly,
matched objectives, zero-disagreement cases, unknown-label exclusion,
checkpoint resume, independent verification, report export, past-only adapter
and admission restrictions. The full historical repository suite was not run.

See [analysis.json](analysis.json), [replay.json](replay.json),
[independent_verification.json](independent_verification.json),
[execution.json](execution.json) and [training_losses.md](training_losses.md).

## What the Losses Do Not Prove

All six final neural minibatch losses are lower than their initial values; the
minibatches differ. This is not convergence or generalization evidence. One
forest's full fitting MSE rises slightly from the sixteen-tree to 128-tree
checkpoint. All twelve registered endpoints are retained, with no loss-based
winner selected. The [complete log export](training_loss.csv) includes 234 rows.

The cost heads see all source sites, even though their forecast targets come
from site-excluded producers. Their fitting errors must not be called OOF policy
evaluation. An outer held-site composite evaluation still needs pair-excluded
producers. A second limitation is the change from three-site OOF producers to
four-site final predictors: the distribution of cost-head inputs can shift.
Complete-case supervision can also bias the learned costs toward retained tracks.
Neither issue is repaired merely by a finite loss or a hash check.

## Unfinished Evidence

| Item | Status |
|---|---|
| Registered source-only cost fitting | `fresh_run`, complete |
| Cached OOF producer validation and endpoint replay | `cached_verified`, pass |
| Full joint controller and support/fallback contract frozen | `not_run`, not yet complete |
| DUT predictive admission and independent calibration | `not_run`, source conditions and limited site support remain |
| DroneCrowd independent confirmation | `not_run`, reservation is not verified site independence |
| Accuracy gain, calibrated safety or new deployment from this bank | Not established |
| Stage5C execution / SMC | Disabled |

Next, bind the complete fixed controller to these predictors and heads, verify
its input-only operation and source-admission rules, then execute only the
permitted calibration/confirmation protocol. Do not select a seed, family or
threshold by reading reserved confirmation outcomes. DUT's two author-described
locations are not 27 independent sites; DroneCrowd's 45 conservative exclusion
groups are not 45 independently verified sites. Small calibration support must
be reported, not inflated using overlapping windows.

The observation/prediction contract remains 8/12 native annotation steps.
SDD source stride12 and the external stride1 interface are not verified equal
physical durations. Costs here are pixel-space. No seconds, metric, physical
safety, true-3D, foundation-model or submission-readiness claim follows.

## Reproduction

From the repository root, with the registered local assets present:

```sh
.venv-pytorch/bin/python scripts/run_m3w_external_cost_bank.py --resume
.venv-pytorch/bin/python scripts/run_m3w_external_cost_bank.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_external_cost_bank.py
.venv-pytorch/bin/python scripts/report_m3w_external_cost_bank.py --verify
```

On the completed run, resume verifies existing endpoints instead of retraining.
Checkpoints, OOF predictions and source arrays remain local and Git-ignored.
The report exporter needs only the lightweight reports; it does not train,
evaluate a model or open a reserved dataset.
