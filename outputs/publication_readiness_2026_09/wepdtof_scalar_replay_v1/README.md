# Offline replay of the frozen external comparison

I separated the external result from the code and data needed to generate
the trajectories. The local replay bundle contains only anonymous
recording-seed scalars, the resampling indices, and a standard-library
Python verifier. It reads neither source trajectories nor the reserved
confirmation data. The exported inputs do not contain risk-pass or
dual-pass flags.

## What the replay confirms

The verifier reapplies the original 2% rule to all three saved risk ratios,
then checks for strictly positive conservative gains against CV and OLS4.
For the fixed-count joint arm it recovers 9 no-edit views, 12 with undefined
easy reference, 20 other risk failures, and 4 risk passes. None of those
four also beats both baselines: **0/45 dual passes**, from 15 clips in 13
source groups and three seeds. All four fixed-count arms make 366 edits.
These checks start at the saved ratios, not at the raw errors or their
denominators. They do not verify the upstream tracking or model inference.

| Fixed joint minus | Estimate (pp) | Replayed 3,000-draw 95% interval | Exact empirical 95% interval |
|---|---:|---:|---:|
| Independent | -0.04501 | [-0.11323, +0.01214] | [-0.11472, +0.01277] |
| Common family | -0.02057 | [-0.10573, +0.05097] | [-0.10661, +0.05268] |
| No pair term | 0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] |

There are 10/13 estimable original source groups. The replay first averages
the clip-seed values within each source; a group with any unavailable view
is not assigned zero. The exact interval enumerates the source bootstrap
distribution with multinomial weights, not an exact population confidence
interval. The finite-sample RNG does not explain the absence of a supported
joint advantage. The strict policy's empty contrasts remain undefined.

## Run without the repository

The local archive is `wepdtof_scalar_replay.zip` under the submission
administration directory. Extract it and run `python -I -B -S verify.py`.
Two builds were byte-identical; an extracted copy ran outside the repository
without third-party packages. Four focused tests passed. To rebuild:

```sh
.venv-pytorch/bin/python scripts/package_m3w_wepdtof_scalar_replay.py
```

Archive SHA-256:
`a23d64df3e183862fb42f853405fd2f50eae3303494a855d9a8177d0c44a2186`.

`cached_verified`: the readout bound to the working manuscript.
`fresh_run`: offline pass reconstruction, original resampling replay and
exact bootstrap sensitivity. No new model, external evaluation, deployment
change, manuscript edit or upload. The bundle stays local.
