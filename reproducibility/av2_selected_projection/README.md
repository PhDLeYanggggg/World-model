# Anonymous AV2 fixed-budget evidence

This archive contains sufficient counts to reproduce the fixed top-five
and calibrated release comparisons for the selected-projection ablation.
It contains no raw trajectories, agent or scene identifiers, checkpoint,
source names, or private paths. All views come from development folds.
Nothing in this archive is an independent confirmation result.

From the extracted directory run:

```sh
python -I -B -S verify.py
```

The command verifies checksums, sums the anonymized city-level cells,
and enumerates all 6^6 paired city bootstrap resamples. The exact
intervals may differ slightly from the experiment's sampled intervals.
It does not rerun a model or calibrate a new threshold. The top-five
count is a diagnostic with equal intervention counts, not a policy.
