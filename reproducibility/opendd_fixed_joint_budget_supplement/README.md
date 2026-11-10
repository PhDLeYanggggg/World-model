# OpenDD count and movement matching

`python verify.py` checks the aggregate movement budget, paired missing-step
debits and the 3,000 location resamples, then replays the inherited evidence.
See `protocol.md` for how the masks were chosen. This anonymous package does
not include forecasts, raw trajectories or integer solver inputs. It replays
reported statistics, not fitting or selection from the original geometry.
