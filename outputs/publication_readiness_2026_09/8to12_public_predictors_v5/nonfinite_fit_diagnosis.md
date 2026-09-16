# Isolated Nonfinite Fit Diagnosis

Result source: `fresh_run`. The EqMotion v5 seed17 full, hold0 and hold1 fits
completed all 10,000 updates. The hold2 fit (ETH + Hotel training; Zara held out)
failed early with `Nonfinite supervised loss`. No v5 development score was
available or used to choose this response. The paired study is incomplete.

## Reproduction

- The original checkpoint was at step0; its initial parameters were finite.
- A separate MPS replay reproduced failure on attempted step89 after 88 completed
  updates. Step50 matched the failed run's logged loss of 23,817,469,952.
- The failing batch had finite inputs and labels. Maximum absolute history,
  neighbor, baseline and target values were about 0.981, 12,487.085, 1.0 and 2.635.
- The first recorded nonfinite module output was the fixed EqMotion output head.
  It produced 72 nonfinite ego forecast coordinates.
- The same pre-failure MPS-trained weights and batch also produced 72 nonfinite
  coordinates in CPU float32. Thus this is not established as an MPS-only kernel
  error or the historical OpenMP/SHM hang.
- CPU float64 forward on those weights was finite, but its largest output was
  approximately 3.90e24. Extra precision alone does not make that forecast useful.
- A complete CPU float32 replay from the same initial state completed the first
  100 updates without a failure. CPU and MPS optimization traces differ; this
  does not prove full training is stable or that the underlying cause is fixed.

The isolated MPS/CPU replay traces, source checkpoint hash and module-level
summaries are in `nonfinite_replay_mps.json` and `nonfinite_replay_cpu.json`.
The pre-failure MPS weights and batch indices remain in the ignored local
`data/stage_cvpr2027_experiments/nonfinite_fit_diagnostic` directory. These are
diagnostic artifacts, not trained deployment models. The original training
checkpoint subsequently resumes and can be replaced by later atomic checkpoints.

## Interpretation and Next Action

Observed: float32 internal amplification/overflow with extreme relative-neighbor
inputs and finite moderate targets. The earlier fit-only context audit independently
found this input-range mismatch. The exact role of input scaling versus the
optimization trajectory still needs a controlled repair experiment; no gradient
or performance attribution is claimed solely from these magnitudes.

An explicit CPU float32 continuation of hold2 is now running at the unchanged
10,000-update budget, with the same data, seed, architecture, loss and sampler.
This is a runtime mitigation, not a claimed root-cause repair. Its checkpoints
retain runtime history. If it fails again, preserve the failed version and test
a separately versioned past-only input-conditioning change, without changing
the evaluation error scale or silently dropping the failing fold. Do not skip
this OOF producer or lower its budget to make the comparison complete.

No accuracy result, independent confirmation, metric/seconds claim, Stage5C,
SMC or deployment upgrade follows from these checks.
