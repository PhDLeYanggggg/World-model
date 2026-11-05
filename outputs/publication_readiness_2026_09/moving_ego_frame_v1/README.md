# Does An Ego-History Frame Repair Cross-Source Risk Learning?

Sixteen new forest models are freshly trained against eight hash-verified
original-frame controls. All24models reproduce by full refit, and all target
RMS values match exactly. Forecasts and original signed targets are unchanged.
This is a source-held fitting experiment, not independent confirmation or
new trajectory dynamics. No safety rule, threshold or data role changed.

## Fixed Primary

history121 uses ego-history-scaled coordinates plus the causal log scale ratio.
original120 uses the existing neighbor-inclusive scale. Both use the same
64-tree/depth8/minleaf32 learner. The121st scalar is an explicit difference;
history120 is a secondary ablation, not a posthoc replacement primary.
Positive reductions mean better. Costs are the unchanged signed-target
decision diagnostic, not ADE improvement or full deployment risk ratios.

| Context | Primary MSE reduction | Primary cost reduction | Original harmful cost | New harmful cost | Screen |
|---|---:|---:|---:|---:|---|
| 0 | +9.825349% | +0.115156% | 0 | 0 | fail |
| 1 | +0.709317% | +0.053858% | 4.2984456e-07 | 1.40287826e-06 | fail |
| 2 | +30.215682% | +0.000000% | 0 | 0 | fail |

Internal follow-up screen: **0/3**. The fixed rule requires
>=1% original decision-cost reduction and nonworse false-safe cost in>=2contexts.
It does not certify the original complete safety contract. The complete
source/arm breakdown, including secondary and negative results, follows.

## Source Breakdown

| Context/source | Arm | Known rows | Normalized MSE | Joint cost | Harmful admissions | All admissions |
|---|---|---:|---:|---:|---:|---:|
| 0/moving_source_component_048 | history120 | 1769 | 0.9065005 | 0.002585146 | 0 | 1 |
| 0/moving_source_component_048 | history121 | 1769 | 0.9072135 | 0.002576379 | 0 | 7 |
| 0/moving_source_component_048 | original120 | 1769 | 0.8995315 | 0.002584265 | 0 | 1 |
| 0/moving_source_component_067 | history120 | 3998 | 1.067595 | 0.004263685 | 0 | 0 |
| 0/moving_source_component_067 | history121 | 3998 | 1.099083 | 0.004263685 | 0 | 0 |
| 0/moving_source_component_067 | original120 | 3998 | 1.325369 | 0.004263685 | 0 | 0 |
| 1/moving_source_component_053 | history120 | 672 | 0.5999805 | 0.001545159 | 0 | 0 |
| 1/moving_source_component_053 | history121 | 672 | 0.583314 | 0.001545159 | 0 | 0 |
| 1/moving_source_component_053 | original120 | 672 | 0.7017749 | 0.001545159 | 0 | 0 |
| 1/moving_source_component_065 | history120 | 44 | 0.5492544 | 0.0009190038 | 0 | 0 |
| 1/moving_source_component_065 | history121 | 44 | 0.5287519 | 0.0009190038 | 0 | 0 |
| 1/moving_source_component_065 | original120 | 44 | 0.5090911 | 0.0009190038 | 0 | 0 |
| 1/moving_source_component_076 | history120 | 2999 | 7.640228 | 0.0006934929 | 8 | 27 |
| 1/moving_source_component_076 | history121 | 2999 | 7.620289 | 0.0006898117 | 6 | 13 |
| 1/moving_source_component_076 | original120 | 2999 | 7.583871 | 0.0006915113 | 1 | 10 |
| 2/moving_source_component_059 | history120 | 8772 | 2.027003 | 0.01180677 | 0 | 0 |
| 2/moving_source_component_059 | history121 | 8772 | 2.142409 | 0.01180677 | 0 | 0 |
| 2/moving_source_component_059 | original120 | 8772 | 4.104555 | 0.01180677 | 0 | 0 |
| 2/moving_source_component_060 | history120 | 10749 | 0.760672 | 0.002348894 | 0 | 0 |
| 2/moving_source_component_060 | history121 | 10749 | 0.7692743 | 0.002348894 | 0 | 0 |
| 2/moving_source_component_060 | original120 | 10749 | 0.6274993 | 0.002348894 | 0 | 0 |
| 2/moving_source_component_073 | history120 | 19058 | 1.282734 | 0.001437299 | 0 | 0 |
| 2/moving_source_component_073 | history121 | 19058 | 1.276722 | 0.001437299 | 0 | 0 |
| 2/moving_source_component_073 | original120 | 19058 | 1.269876 | 0.001437299 | 0 | 0 |

## Causal Scale Audit

Ratios are ego-history scale / existing context scale on all proposed causal
rows, before label-availability filtering. A ratio below1 indicates the outer
neighbor context controls the old scale. This diagnoses representation, not
pixel-to-metre or temporal calibration.

| Context/source | Proposed rows | Ratio min/Q25/median/Q75/max | Fraction below1 |
|---|---:|---|---:|
| 0/moving_source_component_048 | 1805 | 0.209229, 0.418798, 0.497735, 0.580884, 0.905316 | 100.000% |
| 0/moving_source_component_067 | 4151 | 0.19752, 0.67446, 0.751991, 0.816519, 1 | 99.301% |
| 1/moving_source_component_053 | 689 | 0.00360008, 0.19588, 0.455003, 0.681985, 1 | 99.419% |
| 1/moving_source_component_065 | 47 | 0.177345, 0.387885, 0.472646, 0.593345, 1 | 95.745% |
| 1/moving_source_component_076 | 3146 | 0.0361758, 0.403882, 0.549073, 0.703313, 1 | 97.171% |
| 2/moving_source_component_059 | 9209 | 0.168961, 0.541082, 0.683253, 0.811315, 1 | 97.285% |
| 2/moving_source_component_060 | 10997 | 0.00132674, 0.100437, 0.227822, 0.402234, 1 | 99.482% |
| 2/moving_source_component_073 | 19291 | 0.0114572, 0.350088, 0.517974, 0.65046, 1 | 98.963% |

## Runtime And Reproduction

Use native arm64 `.venv-pytorch/bin/python`, CPU4/inter-op1/workers0.
The registered forest uses n_jobs=4 despite a legacy startup banner field.

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_ego_frame.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_ego_frame.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_ego_frame.py run
.venv-pytorch/bin/python scripts/run_m3w_moving_ego_frame.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_ego_frame.py
.venv-pytorch/bin/python scripts/run_m3w_moving_ego_frame.py replay
```

Completed phases are immutable. An interrupted run resumes with `run --resume`
after verifying the old process ended. Per-cell atomic checkpoints are reused;
an unfinished forest is refitted, not resumed tree-by-tree. The private history
assets and frozen models are required; reports alone are not a standalone
dataset or complete anonymous reproduction package. No new CREATE job is needed.

| Phase | PID | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
| pilot | 43412 | 53.837 | 6920536064 |
| run | 43521 | 71.750 | 7445200896 |
| verify | 43684 | 77.568 | 7374012416 |

## Claim Boundary

Three dependent contexts from one producer seed17,48,061known and1,274unknown
proposals. Held-source scores are still exposed-development evidence. No new
C/H, reserved readout or bootstrap claim from dependent rows. Unknown futures
are not safe outcomes. Native observation8/prediction12 image coordinates are
not metric or seconds-level results. No3D/foundation/JEPA/dynamics claim.

Original2% all/easy harm, easy-net, support and empty-control NA stay unchanged.
There is no identity-fallback exemption. Original24-group primary remains NA,
not passed. No deployment promotion, Stage5C, SMC or submission-ready claim.
Exact repeats show engineering reproducibility, not independent confirmation.

See [failure analysis](failure_analysis.md), [protocol](protocol.md), and
[complete metrics](analysis.json). The compiled main paper is unchanged.
