# Nested Keep-Relative Selection

## Result

The registered primary comparison failed. The inner-selected policy passes
48 full-grid checks, compared with 55 for keeping the incumbent allocation
and 52 for the preceding paired selector. It rescues 7 views but destroys
14 previously passing views. Deployment remains unchanged.

This experiment was completed before the user's research pause on
2026-11-10. This README was subsequently assembled for the Claude handoff
from the existing results, without new fitting, scoring, or threshold search.

## What Was Tested

The fixed factorial compares a linear head and a 48/32 GELU MLP, each trained
on either absolute full-grid success (BCE) or keep-relative success difference
(squared error on tanh score differences). All use the same causal features,
training-only normalization, fixed update budget, and seeds 17/29/43.

Nested training excludes the outer location, validation location, donor and,
where applicable, teacher target from the corresponding upstream producers.
The completed matrix contains 105 new rankers, 105 count heads, 252 inner
selectors and 84 outer selectors: 260,400 optimizer updates in total.

The original registration is preserved. The author-approved
[selection amendment](selection_amendment.md) uses net gain over keep across
all validation recordings. No-exchange recordings add zero **difference**,
not a safety pass. Unresolved solver rows retain bounded unknown outcomes.
The original 2% rule and outer evaluation are unchanged.

There are 26/126 empty-exchange inner validation cells. Across repeated inner
validation roles, 3,636 recording views comprise 1,786 exchange slates, 1,848
verified no-exchange views, and 2 unresolved views. They are not 3,636
independent recordings. Selection uses site/seed averaging, not pooled rows.

## Frozen Readout

The roster remains 202 recordings, seven locations and three seeds, giving
606 repeated views. The eligible mask has 101 views and the forecast edit
budget is 780. Reallocating one edit is not another forecast edit budget.

| Policy | Original pass | Full-grid pass | Material full-grid pass | Rescued | Destroyed | Exchanges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent | 60 | 55 | 20 | 0 | 0 | 0 |
| Prior paired selector | 54 | 52 | 20 | 10 | 13 | 82 |
| Linear absolute | 55 | 51 | 17 | 7 | 11 | 84 |
| Linear advantage | 52 | 48 | 16 | 7 | 14 | 88 |
| MLP absolute | 54 | 50 | 17 | 5 | 10 | 87 |
| MLP advantage | 54 | 51 | 19 | 9 | 13 | 79 |
| Inner-selected primary | 52 | 48 | 15 | 7 | 14 | 90 |
| Its own-mask uniform, expectation | 52.9756 | 48.4955 | 15.2335 | 6.0393 | 12.5438 | 90 |
| Hindsight mask oracle, diagnostic | 102 | 100 | 36 | 45 | 0 | N/A |

The selected policy is linear advantage for rdb1/rdb2, linear absolute for
rdb3/rdb4/rdb6/rdb7, and MLP absolute for rdb5. Choices were frozen in commit
`37b871236` before scoring. No model was chosen from this outer table.

| Registered contrast, full-grid percentage points | Equal-location mean | 95% paired-location bootstrap CI |
| --- | ---: | --- |
| Inner-selected minus incumbent | -0.26247 | [-0.78740, 0.00000] |
| Inner-selected minus own-mask uniform | +0.42218 | [-0.11035, +1.40273] |
| Inner-selected minus prior paired | -0.14998 | [-0.44994, 0.00000] |

The 3,000 bootstrap draws resample locations after averaging seeds within
location. Every primary contrast needed a strictly positive lower bound;
none qualifies. Pooled pass totals and equal-location contrasts can have
different signs because recordings per location differ greatly. Neither is
silently substituted for the other. Only seven location clusters are available.

The primary policy passes 16/16/16 views by seed; incumbent passes 19/17/19.
All seven net lost passes are in rdb1 (36 versus 43); zero net differences
elsewhere do not imply no offsetting rescues or destructions. The primary
still has 402 risk-undefined views and 121 known risk violations. It does not
become a safe deployment because the no-exchange gain is zero.

## What the Diagnosis Shows

On repeated training donors, MLP absolute chooses a passing option in 52.69%
of slates; on the 101 held eligible views it does so in 18.81%. Its uncalibrated
sigmoid ECE rises from 4.73% to 43.71%. MLP advantage rescues 696 training
slates without destroying one, but rescues only 9 and destroys 13 held views.
These are in-sample diagnostics, not independent training examples or evidence
of generalization.

Linear advantage predicts a mean selected gain of +0.4762 on held eligible
views while the observed mean is -0.0693. These are bounded pass-difference
scores, **not probabilities**. A relative target does not itself correct the
selection-induced overestimation. Reducing capacity alone also fails: the
best linear arm reaches only 51 full-grid passes.

Training losses are finite. Mean outer-head minibatch losses at update 100
and 400 are 0.3141/0.2643 (linear absolute), 0.0844/0.0796 (linear advantage),
0.0915/0.0172 (MLP absolute), and 0.0214/0.0093 (MLP advantage). BCE and MSE
are different losses and must not be ranked by their numerical magnitudes.
The trace is sampled every 100 updates, not a full-epoch validation curve.

The all-available inner slates and the frozen outer eligibility mask still
represent different selection populations. Sparse sites, population mismatch,
and location shift remain plausible explanations, not separately proven
causes. This result does not justify tuning a new threshold on these outcomes.

## Evidence and Boundaries

- `fresh_run` in the preceding execution: full training, frozen scoring,
  exact-choice replay, and the post-freeze diagnostic.
- `cached_verified`: reused upstream models, forecast bank and frozen controls;
  artifact hashes and training roles were checked by the recorded audits.
- This handoff: reads existing files and checks integrity only. It is not a
  new scientific replication.
- `not_run`: new independent external confirmation, new release, manuscript
  integration, full-repository tests, Stage5C execution, and SMC.

The pre-score and final audits replay 126 inner validation cells and 21
outer choice cells, recompute training-only normalization, and verify nested
producer exclusions. The recorded targeted test run was **27 passed**. It is
not a claim that the entire repository's test suite passes.

These are exposed OpenDD development results on the 8-observed/12-predicted
protocol. They are not the historical SDD raw-frame t50 task, not independent
external confirmation, and not evidence of metric calibration, seconds-level
prediction, true 3D modeling, physical safety, or foundation-model capability.
The recording-wide observed-history aggregation supports retrospective
allocation, not an online streaming claim.

The separate concurrent five-arm draft in `opendd_nested_advantage_v1` remains
untouched. Its initial four-arm registration stopped on a hash mismatch with
zero updates. It must not be mixed with this completed four-arm experiment.

## Assets and Reproduction

Public evidence is in this directory: `protocol.md`, `registration.json`,
`selection_amendment.md/json`, `producer_freeze.json`, `inner_freeze.json`,
`selection.json`, `head_freeze.json`, `pre_score_audit.json`, `readout.json`,
`audit.json`, and `generalization_diagnostic.json`.

Local-only weights, caches and logs are under
`data/stage_cvpr2027_experiments/opendd_advantage_factorial_v1/`.
The current main GitHub repository does not contain those private assets.
The native arm64 runs used CPU4, interop1, workers0, and atomic checkpoints
every 100 updates. All full-matrix calls reached a normal `complete` event.
Peak training process RSS was about 1.59 GiB. No new CREATE job was submitted.

The following commands are reference instructions, not a request to restart
the paused task. Repeating a training phase restores a matching checkpoint;
missing checkpoints cause real training. Do not delete registrations to bypass
identity or hash errors.

```bash
cd /Users/yangyue/Downloads/World
export LC_ALL=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_advantage_factorial.py register
.venv-pytorch/bin/python scripts/run_m3w_opendd_advantage_factorial.py producers
.venv-pytorch/bin/python scripts/run_m3w_opendd_advantage_factorial.py inner
.venv-pytorch/bin/python scripts/run_m3w_opendd_advantage_factorial.py outer
.venv-pytorch/bin/python scripts/audit_m3w_advantage_factorial.py --pre-score
.venv-pytorch/bin/python scripts/run_m3w_opendd_advantage_factorial.py score
.venv-pytorch/bin/python scripts/audit_m3w_advantage_factorial.py
.venv-pytorch/bin/python scripts/diagnose_m3w_advantage_factorial.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_advantage_factorial.py tests/test_m3w_onpolicy_donors.py tests/test_m3w_incumbent_slate.py
```

For the full project, paper, external cohorts, HPC locations, constraints and
unfinished tasks, see [the project handoff](../../../CLAUDE_HANDOFF.md).
