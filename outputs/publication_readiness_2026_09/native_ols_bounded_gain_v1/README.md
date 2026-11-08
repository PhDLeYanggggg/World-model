# Bounding the score did not improve OLS4 replacement

I tested whether learning gain as a fraction of maximum forecast disagreement improves decisions. The bound is known from the candidate and OLS4, and applies to available-label ADE by the reverse triangle inequality. For wholly unknown rows the original conservative debit is that same disagreement. The model learns a signed fraction and a positive-harm fraction, then scales both back to the action's disagreement. No decision threshold was searched.

Eighty-one fresh fits cover three feature arms, three seeds and nine held fitting sources. Fit row IDs and sampled rows match the frozen direct-regression control. At readout, every arm makes the same 6,583 switches across seeds, with counts equal in each query. There are 69,746 agent windows, not independent agents. The contrast uses observed CV error mass as denominator; paired bootstrap resamples sources after averaging seeds (3,000 draws).

| Contrast at matched counts | Conservative gain (pp) | 95% paired source interval |
| --- | ---: | ---: |
| Direct vs. OLS4 | +0.0610 | [-0.0638, +0.1798] |
| Bounded full vs. OLS4 | -0.0047 | [-0.3607, +0.2424] |
| Bounded full vs. direct | -0.1114 | [-0.3605, +0.0460] |
| Bounded full vs. no hindcast | +0.0326 | [-0.0040, +0.0832] |
| Bounded full vs. no neighbors | -0.0847 | [-0.2122, +0.0183] |

On the same direct nominations, mean average precision for positive realized gain is 0.652 for the bounded full model versus 0.673 for direct (positive rate 0.602). At matched counts the full model reduces wholly unknown switches from 99 to 57, but positive harm rises from 0.1901 to 0.4736 points. Neither feature ablation establishes a robust contribution. Unretuned positive-score activation for the full bounded model makes 5,729 switches and has a -0.0039 point lower gain against OLS4 (95% interval [-0.3839, +0.2567]).

**Post-hoc amplitude control.** The full bounded model uses full amplitude in 3,344 of 6,583 switches, contributing 0.3250 of 0.4736 harm points (69%). Direct uses full amplitude only three times. I then held the direct arm's exact per-query amplitude counts fixed and let bounded scores change recipients and paths within each amplitude. This cuts positive harm to 0.1676 points and leaves 61 unknown switches, but the direct paired contrast remains -0.0273 points [-0.0782, +0.0227]. Its standalone lower gain against OLS4 is +0.0772 [-0.0726, +0.2156]. The control shows why count matching alone obscures a major behavioral difference; it does not establish a better selector. Standalone lower bounds are not additive because each comparison has its own disagreement debit.

Twelve related tests passed. Score and exact replay used frozen fitting-source choices and scores; the amplitude diagnostic was recomputed and reconciled to the registered aggregates. No exposed development labels, external calibration or confirmation data, new policy, metric/seconds claim, Stage5C or SMC.

```sh
runner=scripts/run_m3w_ols_bounded_gain.py
.venv-pytorch/bin/python "$runner" train
.venv-pytorch/bin/python "$runner" score
.venv-pytorch/bin/python "$runner" replay
.venv-pytorch/bin/python scripts/report_m3w_ols_bounded_amplitudes.py
```

Resume folds with `train --resume`. [Protocol](protocol.md) | [Aggregate score](analysis.json) | [Amplitude control](amplitude_matched_control.json) | [Replay receipt](replay_execution.json)
