# Calibration Availability

Only the two declared calibration sites contribute to each threshold. No held scores select a model.

| Candidate/objective/support | Calibration views | All-fallback views | Selected cutoffs |
|---|---:|---:|---|
| dimensionless_moments_plain | 108 | 56 | {"-0.0005": 1, "-0.005": 1, "0.0": 50, "None": 56} |
| dimensionless_moments_support | 108 | 61 | {"0.0": 47, "None": 61} |
| dimensionless_excess_plain | 108 | 45 | {"0.0": 63, "None": 45} |
| dimensionless_excess_support | 108 | 45 | {"-0.0005": 3, "0.0": 60, "None": 45} |
| damped_moments_plain | 108 | 0 | {"0.0": 108} |
| damped_moments_support | 108 | 0 | {"0.0": 108} |
| damped_excess_plain | 108 | 0 | {"0.0": 108} |
| damped_excess_support | 108 | 0 | {"0.0": 108} |

Counts are dependent views, not independent calibration datasets.32 selected rows are an
operational support floor, not32 independent trajectories. The99% fitting support limit is a heuristic
diagonal-distance detector; passing it does not establish in-domain support or risk validity.
