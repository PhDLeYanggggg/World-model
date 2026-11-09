# How far can a forecast improve without changing its recipients?

I kept the earlier 54,748-edit policy's recipients and halved each
selected path's peak displacement. The reference shrinks the original
path toward CV. The two learned alternatives may choose a different
path only at the same peak, without amplifying any candidate. This
isolates direction from intervention count, recipients and peak.

I trained a pooled comparison model and three source-specific models
per fold and seed: 36 native PyTorch fits and 36,000 updates. The
train-only targets capture observed benefit, paired disagreement on
missing steps, conservative increments over CV and OLS4, risk changes
and label availability. Training anchors come from analytic forecasts,
not outcome-fitted selection in the same recording. The consensus
option requires all three train-source models to support a change.
It is an empirical decision rule, not a probabilistic guarantee.

| Held development policy | Edited rows | Original risk passes / 72 | Worst known signed easy change |
|---|---:|---:|---:|
| Frozen, full peak | 54,748 | 14 | +61.90% |
| Uniform half-peak shrinkage | 54,748 | 18 | +30.25% |
| Pooled, same half peak | 54,748 | 18 | +30.25% |
| Source consensus, same half peak | 54,748 | 18 | +30.25% |

Compared with uniform shrinkage, the pooled model changes the actual
path on 6,323 rows (coordinate difference greater than (10^{-6})
pixels). It gains +0.028 pp on observed steps, but loses **0.088 pp**
on the conservative comparison (95% source-bootstrap interval
-0.209 to -0.007). Its conservative pair-distance comparison loses
0.082 pixels (-0.185 to -0.016). Consensus selects a nondefault
candidate four times, but changes the actual path on only two rows
at that tolerance, with negligible effect. Uniform shrinkage itself
has conservative gains of -0.140 pp against CV and -55.380 pp
against OLS4. **Calibration: 0/9.** CV fallback stays unchanged.

## Why changing only directions is not enough

The half-peak menu cannot beat OLS4 in 69/72 views even with
outcomes available and with risk ignored. Allowing an arbitrary
path, point by point inside each selected row's CV-centered
radius, still rules out OLS4 gains in 67/72. At the full parent
radius the wider envelope rules out 67/72; with unlimited
displacement on those same rows, 39/72 remain excluded.

For a known step, the best possible distance is its CV error
minus the allowed radius, floored at zero. For a missing step,
the best OLS4 disagreement is its CV-to-OLS4 distance minus the
radius, also floored at zero. Unedited rows have radius zero.
This is an upper bound under the existing conservative metric;
positive bounds do not guarantee a feasible or physical path.
It distinguishes restricted recipients and displacement from
model failure. The next model comparison should allow causally
selected **new recipients** while holding total edits and the
original rule fixed, rather than continuing to tune only direction.

Source agreement here is distinct from optimizing worst-group
loss as in [Group DRO](https://openreview.net/pdf?id=ryxGuJrFvS).
It is also not the batch-RL support guarantee of
[Decision-Point Guided Safe Policy Improvement](https://proceedings.mlr.press/v258/sharma25a.html).

## Reproduction

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_fixed_recipient_support.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_fixed_recipient_envelope.py
.venv-pytorch/bin/python scripts/build_m3w_cv_fixed_recipient_support_replay.py
```

Fits and the envelope are `fresh_run`. Parent forecasts, frozen
recipients and the no-update checkpoint replay are `cached_verified`.
Remove `--replay` to train or resume after checking PID and
heartbeat. A clean anonymous NumPy archive verifies the scalar
policy table, not training or raw path geometry. These are
development results at native observation-8/prediction-12 in
dataset-local pixels. No new A or independent confirmation
outcomes were opened; the 30 DroneCrowd confirmation groups remain
sealed. No deployment, Stage5C or SMC.

[Protocol](protocol.md) | [Learning results](readout.json) |
[Arbitrary-path bound](arbitrary_envelope.json) |
[Manuscript v137](../evidence_manuscript_v137/README.md)
