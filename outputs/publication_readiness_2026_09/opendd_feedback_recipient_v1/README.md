# Do Earlier Forecast Errors Help Choose Recipients?

The [fixed-slot shape audit](../donor_shape_choice_v1/README.md) found
334 views without selected easy-reference support. The earlier
[same-budget recipient study](../recipient_slot_v1/README.md) showed
that changing recipients could help in hindsight but that its causal
rankers did not. I tested whether earlier, fully elapsed forecast
errors supply the missing information.

## Information At The Decision

I reconstruct the latest available mean errors of CV and OLS4 on
the same pedestrian, using only observations from earlier cohorts
whose entire horizon ends **before the current eight-step window
starts**. The models receive normalized scalar errors, not previous
target positions. No current-window future or across-recording state
enters the decision. This arm uses additional recording history; it
is not a strict eight-observation model.

- **Availability:** has earlier feedback, age, observed fraction;
  the error channels are zero.
- **Aligned:** those same fields plus the same pedestrian's resolved
  CV and OLS4 errors and their contrast.
- **Wrong agent:** same fields, with the error channels cycled among
  agents with available feedback in that query. Availability, own
  age and own observed fraction are unchanged.

Of 10,475 rows, 9,122 have resolved feedback. The wrong-agent control
exchanges channels on 9,008 rows. The clock and callback tests check
that modifying a cohort's outcomes cannot affect its own inputs.
This is fixed-weight selection, not the test-time parameter adaptation
studied by [T4P](https://openaccess.thecvf.com/content/CVPR2024/html/Park_T4P_Test-Time_Training_of_Trajectory_Prediction_via_Masked_Autoencoder_and_CVPR_2024_paper.html),
which is not a fitted comparator here.

## Matched Evaluation

All arms use the same query clock, number of recipients per query, exact
slot radii, and causal prefix shape rule. They each make 1,457 edits.
The three fresh variants use identical architectures, optimization, and
paired seeds. A site supplies no fit labels to the head evaluated on it.
The original joint recipient and frozen prefix outputs reproduce their
archived scores.

| Arm | Easy edits | Undefined views | Original risk passes | Joint passes |
| --- | ---: | ---: | ---: | ---: |
| Frozen recipients, prefix shape | 459 | 334 | 112 | 77 |
| Archived joint recipient head | 407 | 354 | 96 | 65 |
| Availability only | 417 | 349 | 98 | 67 |
| Aligned resolved errors | 416 | 350 | 101 | 66 |
| Wrong-agent errors | 423 | 342 | 102 | 67 |

A joint pass requires original 2% recording-risk support and positive
conservative utility against both CV and OLS4. There are 606 repeated
recording-seed views, representing 202 recordings at seven exposed
OpenDD sites. The intervals resample the seven sites 3,000 times after
averaging the three seeds. Gains below are direct paired conservative
effects in percentage points of observed CV error mass.

| Aligned compared with | Utility difference, pp [95% CI] |
| --- | ---: |
| Availability | +0.0067 [-0.0103, +0.0282] |
| Wrong agent | +0.0157 [-0.0005, +0.0348] |
| Archived joint head | +0.0172 [-0.0202, +0.0766] |
| Frozen recipients | -0.0380 [-0.0982, -0.0006] |

Candidate easy-support Brier falls from 0.17364 (availability)
to 0.16948 (aligned). The secondary paired bootstrap difference is
-0.00416 [-0.00903, +0.00067]. The difference is not reliable and
does not increase the number of selected easy references. In
particular, the aggregate improvement in prediction does not transfer
to recording-level safety and utility. There is no evidence here
that identity-aligned earlier errors rescue recipient choice.

## Reproduction And Scope

- `fresh_run`: causal trace construction, 63 native CPU Torch
  fits and 50,400 updates, exact matched readout and bootstrap.
- `cached_verified`: seven exposed OpenDD inputs, frozen count,
  recipient and path controls; zero-update replay reproduces the full
  scalar readout.
- `not_run`: DroneCrowd calibration or independent confirmation.
  Original 15/30 roles and 2% rule unchanged; no deployment,
  Stage5C, or SMC.

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_feedback_recipient.py
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_feedback_recipient.py --replay
```

Native arm64, four Torch threads, no data workers. Training took 36.7 s;
replay took 8.4 s and performed no optimizer steps. Checkpoints and
heartbeat update every 100 steps. The experiment requires existing local
verified OpenDD inputs and models. All 21 focused tests passed. The
trajectory task is native 8/12 annotation steps in dataset-local
coordinates, not a claim of metric accuracy or physical safety.

[Registration](registration.json) | [Results](analysis.json) |
[Checkpoint receipts](head_freeze.json)
