# Vehicle forecasts that improve on CV, before deciding when to use them

The frozen neural path in the [Songdo bank](../songdo_candidate_bank_v1/README.md)
was trained on pedestrians. I trained a small bounded CV residual on
[OpenDD](https://arxiv.org/abs/2007.08463) vehicles instead, with the same
eight observed and twelve forecast samples. Songdo was not read. I used
all seven location folds, three seeds, and the same architecture and
training batches for two losses. This compares forecast candidates,
not deployed selectors.

The final cohort has **141,783 vehicle-history rows**, 25,596 nonoverlapping
windows, and 95,587 complete future rows. Five published motorized classes
are included; separate Trailer rows are not independent targets. All 501
flight tables were inspected and 500 have a usable window. No selected
publisher object ID occurs in two locations. An initial class-name pilot
missed `Medium Vehicle` and `Heavy Vehicle`; I kept its files out of this
comparison, corrected the mapping and retrained before held-site scoring.

## Same network, one change to the training objective

Both models correct CV from the target's own observed motion. Their
12-step displacement from CV is capped at twice the observed RMS
one-step displacement, using the existing normalization floor. The
observed loss ignores missing labels. The completion loss adds the
model's deviation from CV at those steps, with the same requested-step
denominator. Training uses the other six locations for each fold.
I froze all 42 fixed-step checkpoints in `db6a13a81` before scoring.

| Candidate | Conservative gain vs CV, pp [95% location interval] | Conservative gain vs OLS4, pp of CV error | Available endpoint FDE, native units | Original risk passes |
| --- | ---: | ---: | ---: | ---: |
| CV | Reference | Not shown | 15.18 | No-edit risk undefined |
| OLS4 | Below CV at all seven sites | Reference | 16.65 | Not used as a release policy |
| Observed loss | +6.52 [3.46, 9.99] | +19.74 | 13.31 | 3 / 1,500 |
| Completion loss | **+11.54 [9.27, 13.71]** | **+22.26** | 13.72 | **14 / 1,500** |

The completion model gains over CV in all 21 site-seed comparisons
(range +5.82 to +16.84 pp). Its observed-step gain vs CV is +20.75 pp,
lower than +23.59 pp for the observed model. Its missing-step debit
against CV falls from 17.07 to 9.21 pp. The extra loss therefore trades
some observed accuracy for less exposure to missing labels. The exact
paired comparison of completion against the observed model is still
**-12.29 pp** on the full-grid lower bound, 95% interval [-13.39, -10.98].
Different pairwise lower bounds are not transitive or subtractable;
this negative lower bound does not establish the sign of actual errors
on unavailable steps. The intervals use 3,000 location draws after
averaging the three seeds; the training folds overlap.

Available-step ADE is 5.16 for CV, 3.99 for observed and 4.10 for
completion. The corresponding 95th-percentile ADEs are 12.50, 9.63 and
9.72. The completion arm improves the net easy-slice error on average,
but still fails the original full-policy 2% requirement on almost all
recording-seed views. Net gain and the three risk checks are different
quantities. No gate is relaxed, and neither network replaces CV.

**Decision:** Use the completion-trained path as a candidate to test
alongside CV and OLS4, not as a deployment policy. The next test should
hold the vehicle forecast fixed, then compare risk-aware additive,
same-family and joint selection at identical intervention budgets.
The prospective Songdo protocol must be chosen before scoring there.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_vehicle_residual.py extract
.venv-pytorch/bin/python scripts/run_m3w_opendd_vehicle_residual.py fit
.venv-pytorch/bin/python scripts/run_m3w_opendd_vehicle_residual.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_residual.py tests/test_m3w_songdo_prefix_pack.py
```

`fresh_run`: local native arm64 Torch, four threads, no data workers,
16,800 updates and held-site readout. `cached_verified`: seven source
archives and prior location lineage. Repeat `fit` to resume from the
last 100-step checkpoint, including optimizer and sampling RNG state.
The tests check exact resume, publisher classes, loss and peak bound.
Only aggregate reports and fingerprints are in Git. No Songdo,
independent confirmation, reserved data, Stage5C or SMC.
[Protocol](protocol.md) | [Freeze](freeze.json) | [Readout](readout.json).
