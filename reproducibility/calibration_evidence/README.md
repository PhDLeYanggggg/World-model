# Calibration Evidence: Aggregate Reproduction

This package reconstructs paired completion-lower contrasts, their
original source-bootstrap intervals, and the original recording risk
counts from three exposed development studies: release guards,
first-query context matching, and calibration sample size.

Extract the archive in a clean directory. With Python 3.10 or newer
and its standard library:

```sh
python3 -I -S reproduce.py --verify --output replay
```

The program verifies each input, recomputes every scalar, and compares
the generated files to the included manifest. It retains all 14
source groups, three forecaster seeds, 102 recording views per arm,
and the actual 3,000 bootstrap draws. Undefined easy-risk views
remain undefined, not passed.

The evidence contains no scene or recording IDs, raw trajectories,
annotations, image data, model weights, or choice arrays. It is
identity-stripped, not certified anonymous: small-cohort aggregate
patterns could be linkable. It does not train a model, reconstruct
inputs, establish independent confirmation, or certify safety.
The protocol remains image coordinates with eight observed and
twelve future annotation steps. No physical-scale or timing claim.
