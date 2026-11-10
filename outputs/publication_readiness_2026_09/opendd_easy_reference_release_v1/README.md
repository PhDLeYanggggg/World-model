# OpenDD: finding an easy reference does not find a better edit

I kept the earlier sign nominations fixed and trained a separate head
to predict whether either edited vehicle would contribute a positive
easy reference to the original risk check. I compared the original
sign score, sign times predicted reference availability, and reference
availability alone. A released forecast always uses the same two
edits; only the ordering for calibration changes.

| Rank | Held releases | Undefined easy reference | Original risk passes | Risk and both motion references | Also improves additive incumbent |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sign | 32 | 16 | 12 | 8 | 6 |
| Sign times reference | 58 | 19 | 28 | 15 | 9 |
| Reference only | 2 | 2 | 0 | 0 | 0 |

The new head discriminates the easy reference with a mean directional
AUROC of 0.692 and Brier score of 0.215. The combined score lowers the
fraction of releases with an undefined easy reference from 50.0% to
32.8%, but 19 releases still have no reference and 11 more fail the
original check despite an estimable reference. The absolute number of
strict passes rises with the release count, while the strict pass rate
drops from 18.8% to 15.5%. This is not a fixed-count comparison.

The combined rule changes conservative improvement over CV from
+0.000012 to -0.000017 percentage points. Its paired difference against
sign is **-0.000029 pp** (95% location bootstrap
**[-0.000058, -0.0000025] pp**). The point effect is small, but it goes
in the wrong direction. The 3,000 resamples use seven location means
after averaging repeated calibration roles and training seeds, not
8,814 independent cases. Easier scoring support is not evidence of
lower trajectory error. I kept the existing release policy.

## Reproduction and scope

- `fresh_run`: 63 arm64 PyTorch fits, 12,600 updates, three seeds.
  Four compute threads, no data workers, optimizer and sampler saved
  every 100 steps, and per-cell PID progress.
- `cached_verified`: original donor forecasters, sign nominations
  and additive anchors; original sign thresholds, release counts and
  effect estimates reproduced exactly. The donor anchor differs
  from the frozen additive anchor at calibration and held evaluation.
- `not_run`: no new independent external evaluation, CREATE job,
  Stage5C or SMC. The seven OpenDD locations have already been used
  for development. The original 2% rule and external splits are unchanged.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_easy_reference_release.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_easy_reference_release.py calibrate
.venv-pytorch/bin/python scripts/run_m3w_opendd_easy_reference_release.py score
```

Scores were published at `1144d823d` before calibration, and thresholds
at `710931183` before held scoring. The [protocol](protocol.md) specifies
the unchanged screen. See the [calibration receipt](calibration_freeze.json)
and [held readout](readout.json). The next decision is whether the
locked method has a measurable effect on an independent cohort, not
whether another score increases AUROC on these seven locations.
