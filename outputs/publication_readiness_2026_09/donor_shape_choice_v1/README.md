# Can A Learned Chooser Use Cross-Site Motion Shapes?

I trained a small network to choose between the original trajectory and the
[donor-retrieved shape](../donor_motion_shape_v1/README.md). The two shapes
share the same recipient, query, and peak displacement from constant
velocity (CV). All policies make the same 1,457 nonzero edits. This isolates
shape selection from selecting more recipients or changing the timing.

## Fitting And Validation

I predicted four paired targets: conservative utility versus the original
shape and changes in the all, easy, and signed-easy risk numerators. A
gain-only rule reads the first output. A risk-aware rule subtracts a penalty
for predicted increases in the other three. Both can keep the original
shape at each scheduled edit. Neither can delete the edit.

For each outer site, model fitting and threshold selection exclude that
entire site. Inner predictions also exclude their own validation site.
Every fitted row uses a retrieval library and cut estimates that exclude
its own site. The six donor validation sites have equal weight. Thresholds
maximize mean donor lower utility on legal row-amplitude options; the
risk-aware rule also requires nonpositive donor-macro changes in all three
risk numerators. These are calibration proxies, not recording-level safety
certificates. I did not use inner nominees from outer controller fits whose
training could have seen the held site.

## Outer Results

OpenDD development, 202 recordings across seven sites, three frozen
controller and chooser seeds (606 repeated recording-seed views).
The 95% descriptive intervals resample the seven sites 3,000 times
after seed averaging. Gains are in percentage points of observed CV
error mass; the paired lower bound charges missing-label disagreement.

| Shape chooser | Donor shapes | Original risk passes | Risk and both references pass | Direct lower gain vs prefix, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Original shape | 0 | 112 | 77 | - |
| Causal prefix rule | 764 | 124 | 90 | reference |
| Learned gain only | 1,029 | 133 | 95 | -0.016 [-0.067, +0.023] |
| Learned risk-aware rule | 15 | 112 | 77 | +0.011 [-0.017, +0.060] |
| Hindsight utility only | 769 | 176 | 153 | +0.071 [+0.028, +0.127] |

The risk-aware rule reverted entirely to the original shape in three
outer folds, including the two largest sites. Its apparent gain over
the prefix rule did not survive the interval, and it lost 13 joint
passes. The gain-only rule gained five passes but did not improve
conservative utility. Neither learned rule advances to calibration.

## Why The Fixed Slots Are Limiting

I then optimized over the same two possible shapes at every frozen slot,
using outcomes only for this hindsight diagnostic. I maximized the lesser
of the two conservative gains under the original risk constraints and
rechecked every qualifying solution with the original recording scorer.
Each recording used binary enumeration up to 12 edits, or an integer
program above that size. No case was unresolved.

| Mutually exclusive outcome | Views |
| --- | ---: |
| Could qualify with a hindsight assignment | 157 |
| No easy-reference support among the selected rows | 334 |
| Cannot beat both references even without risk constraints | 84 |
| Can beat both without risk constraints, but not with them | 31 |
| Total | 606 |

Of the 334 undefined views, 172 had no edit and 162 had edits but none
with a positive easy reference. Keeping the same recipients and nonzero
radii makes both groups invariant to shape choice. The 157 qualifying
views are an upper limit for this particular two-shape menu and slot
assignment, not for the dataset or arbitrary trajectories. Even the
utility-only hindsight rule reached 153. This makes recipient support a
more direct next target than searching new shape thresholds. The next
comparison should hold query clocks and counts fixed while testing causal
support-aware nomination against the current nomination policy.

## Relation To Forecast Retrieval

[MANTRA (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Marchetti_MANTRA_Memory_Augmented_Networks_for_Multiple_Trajectory_Prediction_CVPR_2020_paper.html)
already retrieves past/future embeddings from an associative memory for
multimodal forecasting. I do not claim novelty for motion retrieval. Here
retrieval is a controlled candidate generator for a different question:
what changes when shape changes, but the intervention slots and maximum
displacements do not? MANTRA was not fitted as a benchmark in this run.

## Reproduction

- `fresh_run`: 84 native arm64 Torch fits (63 double-excluded and 21
  outer), 25,200 updates, nested readout, and 606-view hindsight analysis.
- `cached_verified`: frozen OpenDD inputs, count and nomination models,
  exact original and prefix scores, and zero-update replay of all 84 fits.
- `not_run`: new DroneCrowd calibration or confirmation. No policy promotion,
  no Stage5C or SMC. Native 8/12 annotation steps and dataset-local
  coordinates, not seconds or verified physical units.

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_donor_shape_choice.py
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_donor_shape_choice.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_donor_shape_ceiling.py
```

Local first fit was about 50 s; replay was about 39 s with four Torch
threads and zero data workers. The original command resumes at a 100-update
checkpoint. All 18 focused tests passed. Source and scalar receipts do not
include data or weights.

[Fit protocol](registration.json) | [Thresholds and results](analysis.json) |
[Hindsight protocol](ceiling_registration.json) | [Hindsight results](ceiling.json)
