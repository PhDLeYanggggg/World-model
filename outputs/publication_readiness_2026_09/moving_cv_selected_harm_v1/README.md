# Learning harm on proposed edits

I tested whether explicitly predicting large harm on the paths a
policy actually proposes makes its decisions better. The prior
amplitude portfolio is frozen. Each legal row gets one path: the
parent's action if selected, otherwise the shape model's best
independent action. The median and 85th-percentile heads see the
same training samples, action features, and predicted gains.
Only the quantile in the loss changes. The loss is standard
[quantile regression](https://www.jstor.org/stable/1913643);
the experiment tests its effect on the forecast decision, not a
new quantile estimator or a probabilistic safety guarantee.

## Held development results

18 native arm64 PyTorch fits, 21,600 updates, three seeds, three
rotations of disjoint fit/calibration/held source groups. The readout
has 24 exposed recordings from nine source groups (72 recording/seed
views). The rerankers match the parent exactly in query edit counts
and keep each row's proposed path fixed; total displacement is not
matched.

| Policy | Edits | New recipients vs parent | Total peak mass (local pixels) | Risk passes / 72 | Worst known signed easy change |
|---|---:|---:|---:|---:|---:|
| Frozen shape portfolio | 48,915 | 0 | 367,255 | 1 | 6.31% |
| Median harm | 48,915 | 25,112 | 661,648 | 5 | 26.96% |
| 85th-percentile harm | 48,915 | 22,143 | 626,190 | 7 | 26.96% |

On the parent's nominations, the all-harm exceedance falls from
39.5% to 22.5%; signed easy exceedance falls from 52.2% to 18.3%.
Easy positive-harm exceedance barely moves (11.7% to 11.3%).
Ties count as nonexceedances; these rates are not quantile guarantees.
Observed-only risk passes are 15/10/20. The average improvement in
exceedance does not capture the worst easy change.

| Matched-count contrast | Observed-only pp | Completion-lower pp (95% CI) |
|---|---:|---:|
| Median vs parent | +0.537 | -1.208 [-3.165, -0.075] |
| 85th percentile vs parent | +0.402 | -1.222 [-3.151, -0.103] |
| 85th percentile vs median | -0.135 | -0.484 [-0.867, -0.169] |
| 85th percentile vs CV | +0.822 | -0.677 [-2.084, +0.222] |
| 85th percentile vs OLS4 | +0.221 | -55.154 [-96.572, -22.969] |

Intervals use 3,000 paired source resamples after averaging seeds.
The tail head's conservative pair-geometry difference versus
median is -0.356 local pixels [-0.684, -0.123]; this is not a
physical-safety result. The freer abstention ladder qualifies 0/9
calibration cells under the original 2% and dual-utility checks.
It cannot increase the parent query counts.

## Why changing the ranker is not enough

For each comparator separately, use the outcomes to maximize the
paired completion-lower numerator. Choose the best action per row,
then the top increments for the query count. With risk constraints
removed, this is exact for the specified bank and count. Positive
upper scores do not establish a feasible policy for both comparators.

| Menu and count | Nonpositive vs CV | Nonpositive vs OLS4 | Median upper vs OLS4 |
|---|---:|---:|---:|
| One path per row, parent exact counts | 6/72 | 68/72 | -34.63 pp |
| All analytic rays, parent exact counts | 1/72 | 49/72 | -11.85 pp |
| Full bank, parent exact counts | 1/72 | 45/72 | -11.48 pp |
| Full bank, at most parent counts | 1/72 | 45/72 | -11.48 pp |
| Full bank, free counts | 0/72 | 0/72 | +33.84 pp |

The full bank adds source-excluded neural rays to the analytic
directions; both use the same five amplitudes. These bounds separate
the effects of allowing different paths from allowing more edits.
The high-quantile loss reduces certain underestimates, but does
not improve matched utility and cannot fix a narrow action space.
I would next retain multiple paths per row and train the number of
edits jointly with action choice, comparing the resulting controller
against an equal-count independent selector. Some fit strata had as
few as one nominated row; avoid repeatedly sampling such tiny strata
at the same rate as large strata in that next fit.

## Reproduction and status

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_selected_harm.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_selected_harm_capacity.py
```

Both fits and the capacity readout are `fresh_run`. The parent and
donor artifacts are `cached_verified`. Checkpoint replay completed
with zero optimizer updates and the same frozen readout. Remove
`--replay` to train or resume. Use local arm64 PyTorch, four compute
threads and no loader workers. Checkpoints and heartbeat remain
under the ignored local experiment directory. No new CREATE job
was needed; the previous CREATE receipt is separate engineering
evidence, not an independent result.

Eight observed / twelve forecast steps, dataset-local image pixels.
No time/metric, true 3D, foundation or physical-safety claim.
Independent confirmation is `not_run`; the approved 15/30
DroneCrowd split and original 2% rule are unchanged. No deployment
change, Stage5C, SMC, or submission-ready claim.

[Protocol](protocol.md) | [Readout](readout.json) |
[Capacity](capacity.json) | [Working supplement](../evidence_manuscript_v140/README.md).
