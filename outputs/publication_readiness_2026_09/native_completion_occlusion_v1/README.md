# Can the completion corridor cover known tracks?

The pair-ranking witness used a fixed-position center after the last released coordinate. I tested that choice by hiding released positions in a reconciled track ledger. The same 34 exposed recordings and 14 sources were used, with no changes to predictions, labels, policy selection or the 2% rule.

| 12-step pseudo-gap | Gaps | Whole gap at 1x | Whole gap at 2x | Mean center error (image px) |
| --- | ---: | ---: | ---: | ---: |
| Internal, interpolated | 4,078 | 87.74% | 96.42% | 23.63 |
| Terminal, fixed position | 4,026 | 10.61% | 71.34% | 251.79 |
| Same terminal gaps, anchored OLS4 | 4,026 | 80.42% | 94.10% | 61.72 |

At 1x the terminal whole-gap difference is **+69.82 percentage points** (95% paired source bootstrap: **+59.68 to +79.49**, 3,000 draws). Only the center changed, not the radius or the hidden coordinates. The ledger has 40,134 unique missing requested agent-steps; 2,800 terminal gaps reach the 12-step request limit. This makes the terminal center consequential for the earlier sensitivity analysis.

Artificially deleting released positions cannot recover actual missing coordinates. The test measures the coordinatewise corridor, not the more restrictive shared-direction witness. The strong coverage difference identifies a better-supported sensitivity center to test, not a calibrated missing-label model or an inference about the unseen trajectories. Steps and image pixels are not verified time or physical distance. No new deployment, independent outcome, Stage5C or SMC.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_completion_occlusion.py score
.venv-pytorch/bin/python scripts/audit_m3w_completion_occlusion.py replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_completion_occlusion.py \
  tests/test_m3w_pair_completion_witness.py tests/test_m3w_anchored_pair_completion.py \
  tests/test_m3w_shared_pair_completion.py
```

Ten focused tests passed; the fresh score and replay match exactly.
[Fixed protocol](protocol.md) | [Aggregate tables](analysis.json) | [Replay receipt](replay_execution.json)
