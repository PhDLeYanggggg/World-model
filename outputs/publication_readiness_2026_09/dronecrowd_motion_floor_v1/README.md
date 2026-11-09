# DroneCrowd: A Better Motion Floor Still Fails Most Risk Checks

The [perfect-switch bound](../dronecrowd_perfect_switch_ceiling_v1/README.md)
showed that the original two-edit budget cannot catch
all-agent OLS4, even with perfect forecasts for the two edited
agents. I therefore test full-scene motion controls separately.
They are not equal-budget competitors to the sparse selector,
and this experiment does not replace the two-edit protocol.

I use the same frozen 8/12 queries in the approved **40 calibration
recordings and 15 source groups**. The 1,757,584 requested rows
include 1,755,547 with at least one future label, across 11,240
queries. The inputs are exported head-center histories in image
coordinates, not ground-plane trajectories or real-time sensor
states. The [DroneCrowd paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wen_Detection_Tracking_and_Counting_Meets_Drones_in_Crowds_A_Benchmark_CVPR_2021_paper.html)
describes the source benchmark. The 30 confirmation groups
remain sealed.

## Motion Controls

- **OLS4** estimates the slope on the last four positions and
  extrapolates from the latest position. I reproduce the frozen
  CV and OLS4 paths and the original OLS4 group result.
- **Median slope** replaces that slope with the coordinatewise
  median of the six pair slopes, following the robust estimator of
  [Sen](https://doi.org/10.1080/01621459.1968.10480934).
- **Fitted origin** keeps the OLS4 slope but extrapolates from
  the fitted line's endpoint. **Pooled origin** shares the median
  endpoint correction across the observed query; it is not a
  camera registration and leaves pair-relative predictions unchanged.
- **Past best** chooses a method per actor using four one-step
  probes within the observed history. **Past guarded** also demands
  positive mean improvement and no more than 2% harm on any
  of those historical probes; otherwise it returns CV. The
  guard is a causal feature rule, not an evaluation change or
  proof of future risk.

All coefficients are frozen before reading labels. The scorer
still uses the original CV-referenced per-recording risk and the
same conservative CV and OLS4 comparisons. A nonzero edit count
or a negative signed mean does not substitute for the original
positive-harm checks. Each deterministic method has 40 views, not
120 replicated seed views. The interval resamples the 15 source
groups 3,000 times.

## Results

| Method | Lower gain vs CV (pp) | Lower gain vs OLS4 (pp, 95% CI) | Original risk passes | Both floors and risk |
|---|---:|---:|---:|---:|
| OLS4 | +51.288 | Reference | 8/40 | Not applicable |
| Median slope | +50.366 | -0.954 [-1.690, -0.211] | 9/40 | 0/40 |
| Fitted origin | +51.679 | **+0.391 [0.247, 0.532]** | 8/40 | 8/40 |
| Pooled origin | +51.268 | -0.020 [-0.031, -0.010] | 8/40 | 2/40 |
| Past best | +47.694 | -3.611 [-4.281, -2.887] | 8/40 | 0/40 |
| Past guarded | +17.999 | -33.412 [-36.936, -30.016] | 7/40 | 0/40 |

The fitted origin improves over OLS4 in **14/15 groups**.
Mean recording ADE decreases from 1.4295 to 1.4157 px and
available endpoint FDE from 2.5462 to 2.5406 px, but the worst
recording p95 ADE slightly increases (6.64 to 6.67 px).
All fitted-origin recordings improve signed easy error. Still,
30/40 fail the positive easy-harm test and 32/40 fail overall
positive harm. The failures overlap. There is no across-recording
pass or new deployment. One-step probes did not identify a good
twelve-step substitute; the guarded variant also gives back most
of the coverage advantage.

## Can the Frozen Paths Satisfy Both Checks?

I use labels to select the path with strictly lower observed
ADE for each actor, starting from CV. An actor without future
labels always stays on CV. The original scorer then verifies
the risk and both conservative gain comparisons. This is a
constructive full-scene witness, not a deployment or a bound
at the original two-edit budget.

| Hindsight bank | Lower gain vs OLS4 (pp, 95% CI) | Risk and both floors |
|---|---:|---:|
| CV and OLS4 only | +3.538 [2.435, 4.696] | 40/40 |
| Add median slope and both origin variants | +7.151 [5.996, 8.383] | 40/40 |

The expanded witness improves in all 15 source groups and
uses 1,212,720 non-CV paths across the 1,757,584 requested
rows. Thus full-scene candidate supply is not itself the
obstacle under the original risk definition. One-step
history probes did not recover that separation between
useful edits and harmful ones. Keep fitted-origin OLS4 as a
strong full-scene development comparator, and test an
8-to-12-step gain/harm model trained on separate sources
before considering any confirmation run.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_dronecrowd_motion_floor.py freeze
.venv-pytorch/bin/python scripts/run_m3w_dronecrowd_motion_floor.py score
.venv-pytorch/bin/python scripts/run_m3w_dronecrowd_motion_floor.py replay
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_motion_witness.py
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_motion_witness.py --replay
```

Native arm64; no worker processes, new training or CREATE jobs.
Freeze took 12.22 s and score 51.70 s, with 1.33 GB maximum
RSS. The witness took 29.09 s at 1.72 GB; 182 focused tests passed.
Per-recording hashes and checkpoints support restart; replay
uses scalar receipts without reopening labels. The coefficient
cache stays in the ignored local data directory. No changes to
the manuscript, original calibration result, risk rule or roles.

For the fitted-origin failure counts, use the local scalar receipts:

```sh
jq -s 'map(.row.arms.fitted_origin.risk) |
  [map(select(.completion_all_risk_upper > 0.02)) | length] +
  [map(select(.completion_easy_risk_upper > 0.02)) | length] +
  [map(select(.known_easy_net_degradation > 0.02)) | length]' \
  data/stage_cvpr2027_experiments/dronecrowd_motion_floor_v1/*.score.json
```

[Registration](registration.json) · [Freeze](freeze.json) · [Accuracy and risk](analysis.json) · [Hindsight witness](witness.json)
