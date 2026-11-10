# Songdo: calibration and source-disjoint confirmation

The author selected the 40/160 option before outcomes were opened.
The three linked calibration components are A/B, C, and E; the
remaining fourteen linked components are reserved for confirmation.
The [approved manifest](../../../configs/m3w_songdo_calibrated_20261110.json)
fixes all roles. No training is performed on Songdo.
The pre-outcome registration SHA-256 is
`b03fcbd04a18ed464b66b070a7db3a462a57c5dc70bf6c7a5d5a3eee68bb7e61`.

Use the existing eight-observation, twelve-prediction native grid,
CV and OLS4, and the four frozen two-edit arms at equal displacement
radius. Calibration alone supplies the positive complete-CV-error
quartiles and the tie-complete score prefix meeting the original
three-part 2% recording rule. Report the same-budget proposal
comparison independently of the actual thresholded outputs. A CV
fallback is not counted as a risk pass, and a missing matched contrast
is not treated as zero. The frozen thresholds are not revised after
opening confirmation. Average the three seeds within flight and then
flights within each linked component before resampling components.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
protocol=configs/m3w_songdo_calibrated_20261110.json
.venv-pytorch/bin/python scripts/run_m3w_songdo_confirmation.py register --protocol "$protocol"
.venv-pytorch/bin/python scripts/run_m3w_songdo_confirmation.py calibrate --protocol "$protocol"
.venv-pytorch/bin/python scripts/run_m3w_songdo_confirmation.py confirm --protocol "$protocol"
```

The local run directory keeps the registration, per-flight receipts,
PID and progress heartbeat, cutoff and release freezes, and final
confirmation report. Repeat only the interrupted command to resume;
completed results replay without opening outcomes. No coordinate
conversion, metric claim, deployment promotion, Stage5C, or SMC.
