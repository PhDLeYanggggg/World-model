# Pretraining Replay Repair

The initial prepare phase stopped on an exact forecast replay assertion before
any new optimizer update or outcome comparison. The new runner instantiated the
frozen forecast using its comparison reference index1 (CV), not the baseline
selected by that forecast's original train-only fit (index2 in the first case).
The checkpoint stores the correct selection. The repair reads and checks that
checkpoint's original fold, seed, fitting roster, step count and baseline index.
No checkpoint, cached forecast, threshold, data role or training target changed.
The failed preflight identity remains in the private run directory. An explicit
regression test guards the distinction. Exact replay is rerun, not relaxed.
