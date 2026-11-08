# EqMotion In A Focal Comoving Frame

Changing who receives an edit did not solve the
[fixed-budget recipient problem](../opendd_feedback_recipient_v1/README.md).
I tested the trajectory producer directly before adding another selector.
The question is whether the same neural architecture is a stronger
predictor when it models departure from the focal agent's constant
velocity, and whether observed neighbors help.

## Controlled Training

I used the pinned author core of
[EqMotion, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html)
with one fixed output head, no pretrained weights, width 32 and two
layers. This is a compact single-forecast adaptation, not the original
best-of-20 benchmark. The three conditions share the same sampled rows,
seeds, update budget, optimizer and masked path loss:

- **Centered with peers:** subtract the focal agent's last position.
- **Comoving with peers:** also subtract the focal last-step displacement
  times relative time from every observed agent; add the same CV rollout
  to the output. This keeps within-frame relative positions unchanged.
- **Comoving ego only:** mask all peers, with the same focal-derived
  scale and network capacity.

The comoving input is invariant to a common velocity shift and
translation before EqMotion; reconstruction adds the shift to its
output. Neither transform uses a future point. Observed peers are the
three nearest agents in the same query. I hold out each of seven
OpenDD sites in turn, use the other six for all fitting, and report
three paired seeds. The 10,475 distinct rows include 9,823 complete
future paths. This is exposed location-held development, not
independent confirmation.

## Whole-Roster Forecast Accuracy

ADE and FDE pool the available complete and endpoint rows, respectively.
Lower gains are site macro-averages in percentage points of the common CV
error mass. They charge prediction disagreement at missing steps. The
descriptive intervals use 3,000 site bootstrap draws after averaging seeds.

| Model | Complete ADE | Endpoint FDE | Conservative gain vs CV [95% CI] | Gain vs OLS4 |
| --- | ---: | ---: | ---: | ---: |
| CV | 0.556 | 1.227 | reference | +6.875 |
| OLS4 | 0.620 | 1.319 | -10.028 [-12.952, -6.496] | reference |
| Centered with peers | 0.694 | 1.331 | -12.753 [-24.248, -2.570] | -5.050 |
| Comoving with peers | 0.669 | 1.312 | -9.434 [-19.467, -1.447] | -1.786 |
| Comoving ego only | 0.561 | 1.248 | -0.460 [-2.718, +1.647] | +7.107 |

The ego-only gain over OLS4 has interval [+4.342, +9.894], but
CV is the stronger reference here. The same ego model improves
available-step error over CV by +0.803 pp on the site macro-average,
but not the lower bound or the pooled complete ADE. Its signed
easy-case error increase ranges from +2.98% to +52.70% across
seed-averaged sites. This raw predictor cannot replace CV.

The direct conservative effect of the moving frame with peers is
+0.079 pp [-2.431, +3.182] versus the centered peer model.
Adding peers in the moving frame is worse by -10.288 pp
[-18.739, -3.270]. This is a negative result for this neighborhood
model, not proof that pedestrian interaction is uninformative.

## Forecasts At The Same Edits And Radii

Each policy below retains the original 1,457 recipients, query times and
peak displacements from CV. Neural forecasts are projected to the slot
radius in the direction of their predicted deviation from CV; this is a
forced shape test, not a trained intervention policy. The original
recording risk rule stays at 2%; a joint pass also requires positive
conservative utility over both CV and OLS4. Views repeat 202 recordings
at three seeds and are not independent.

| Shape | Risk passes | Joint passes / 606 | Direct lower gain vs original [95% CI] | Two-shape hindsight capacity |
| --- | ---: | ---: | ---: | ---: |
| Original | 112 | 77 | reference | - |
| Past-prefix choice with donor shape | 124 | 90 | [prior comparison](../donor_motion_shape_v1/README.md) | 157 |
| Centered with peers | 130 | 84 | -0.089 [-0.235, +0.005] | 154 |
| Comoving with peers | 105 | 76 | -0.053 [-0.140, +0.007] | 151 |
| Comoving ego only | 117 | 85 | -0.133 [-0.317, -0.008] | 165 |

The last column allows choosing the original or new shape at each
fixed slot with outcomes known, not a causal forecast. The donor
capacity uses the original and forced donor shapes, not the prefix
decision. The small increase to 165 does not repair the 334 views
without selected easy support or make the causal neural output qualify.
No arm replaces the frozen policy, and none advances to calibration.

## Runtime And Evidence

- `fresh_run`: 63 arm64 CPU Torch fits, 50,400 updates, full predictions,
  exact fixed-slot replay against the original controls, and new
  label-aware capacity calculations. Training and readout took 15.6 min.
- `cached_verified`: input and fixed-controller receipts, unchanged
  EqMotion source bytes, and zero-update checkpoint replay.
- `not_run`: DroneCrowd calibration and confirmation. Original 15/30
  roles and 2% rule unchanged; no Stage5C, SMC, deployment, physical
  units, seconds-level prediction, or independent generalization claim.

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_comoving_eqmotion.py
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_comoving_eqmotion.py --replay
```

Four Torch threads, no data workers. Checkpoints retain model, optimizer,
sampler state and identity every 100 steps; the same command resumes.
The zero-update replay took 36.4 s and re-emitted every forecast exactly.
Twenty-three focused tests passed, including input, gradient, frame
equivariance, matched-shape replay, and confirmation-role checks.
The named workstation's data, arrays and weight files are not exported.

[Protocol](registration.json) | [Results](analysis.json) |
[Checkpoint receipts](head_freeze.json)
