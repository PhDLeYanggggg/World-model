# MovingDroneCrowd++: OLS4 exclusion certificate

I rebuilt the arbitrary-path bound from the frozen half-anchor edit budgets and aggregate query error segments. The verifier independently checks a feasible allocation and its matching dual upper bound. All per-view bounds and inverse break-even budgets agree with the archived allocation and break-even readouts.

| Budget allocation | Excluded views | Median OLS4 upper bound | Closest excluded view |
| --- | ---: | ---: | ---: |
| Fixed within each query | 67 / 72 | -29.82 pp | -1.57 pp |
| Pooled within each recording | 67 / 72 | -27.04 pp | -1.48 pp |

The 72 views are 24 previously exposed recordings from nine sources, each replayed with three seeds. The comparison is the frozen conservative utility relative to OLS4, expressed as a share of observed CV error. It does not infer the true error of an unlabeled waypoint. The five non-excluded views are not certified wins.

Among 62 positive-budget views, the median budget needed to tie OLS4 is 1.02 times the recorded budget on observed steps alone and 18.71 times under the original missing-step disagreement debit. The corresponding counts within budget are 31 and 5. The other 10 views have zero original peak budget. These are hindsight calculations, not a prediction policy or a change to the 2% risk rule.

## Reproduce

With the frozen development assets on the original machine:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/package_m3w_moving_peak_certificate.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_moving_peak_certificate.py
```

The archive is written locally to `data/stage_cvpr2027_experiments/submission_admin/moving_peak_certificate.zip`. Extract it and run `python -I -B -S verify.py` from its root. It needs no solver, model weights or dataset. It retains grouped error segments and budgets, not trajectories, frames or original identifiers.

Six focused tests passed, including comparisons with an independent linear program and the original censored bound. The exporter also ran the isolated verifier on all 72 real development views and checked deterministic ZIP bytes. [Numerical receipt and hashes](receipt.json). No independent confirmation outcomes were read.
