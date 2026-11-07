# Forecasting Evaluation Replay

This bundle reproduces numerical aggregates for three development
comparisons: sparse forecast exchange, common-motion ablation, and
joint pair-path accuracy. It can run without the original repository
or any trajectory dataset.

## Run

Use Python 3.11 and the NumPy version pinned in `requirements.txt`.
Install the dependency in an isolated environment, then from the
extracted folder run:

```sh
python -m pip install -r requirements.txt
python replay.py
python replay.py --show
```

The replay checks file hashes and recomputes recording pass counts,
paired utility, and 3,000-draw source bootstrap intervals. Source
means are computed after averaging three forecast seeds. Each
recording and source is identified by a local alias only.

## Scope

`statistics.json` contains recording-level paired error masses and
risk ratios, plus source-by-seed query averages for the pair contrasts.
`expected.json` contains the previously frozen aggregate tables.
For utility we use the original constant-velocity error denominator.
For risk, zero or missing reference makes the ratio non-estimable;
the original 2 percent test does not add a fallback exception.

This is a replay from derived statistics, not a reconstruction of
the raw evaluation population, per-trajectory forecasts, training,
or source separation. The cohort comprises 34 previously exposed
recordings, 14 source groups, and three repeated forecast seeds.
None of these results uses independent confirmation data. Units
are native image pixels and raw annotation steps.

No trajectories, weights, images, video, source names, identity
lookup, or network dependency beyond NumPy installation are included.
