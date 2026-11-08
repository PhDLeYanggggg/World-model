# Leave-Source-Out Cost Prediction Does Not Recover Oracle Capacity

**Fresh deterministic fitting**, nine separate ridge fits, not nine
random seeds and not neural training. Each holds out one of the
nine original fitting groups and derives its training easy-error
cutoff from the other eight. The prior frozen easy cutoff is used
only for evaluation under the unchanged rule. An earlier pilot
used the prior cutoff for training too; it is not counted as
source-excluded evidence. No oracle decisions enter the model.

The features describe eight past positions, motion, the visible
query size and the ten candidate paths. The regression estimates
gain, positive harm, easy harm, all/easy reference mass and
missing-label probability. The independent selector takes positive
predicted gain; the recording-level selector tries a fixed price
grid and the union of individually predicted feasible moves. An
empty choice is not a risk pass. All group-excluded choices are
frozen before the score phase. No calibration or confirmation
labels are used.

| Policy | Switch rate | Conservative gain vs CV, pp | Original passes |
|---|---:|---:|---:|
| Frozen recent hindcast | 45.77% | +1.647 | 0/24 |
| Independent predicted gain | 70.92% | +0.243 | 0/24 |
| Predicted-risk joint allocation | 17.83% | +1.009 | 0/24 |

Direct conservative joint contrast vs OLS4 is **-11.025 pp**
[-20.031, -1.971]; vs recent hindcast it is **-4.532 pp**
[-9.522, -1.532]. These are 3,000 paired source-group resamples.
The two positive CV-relative lower scores cannot be subtracted
to recover a direct conservative contrast: the paired missing-label
debit uses different path disagreement.

The joint arm is nonempty in 20 recordings. In 16 recordings the
all-positive-harm ratio already exceeds 2% on released labels
alone. Easy risk is undefined in nine recordings, including the
four empty cases. Reduced intervention and predicted feasibility
therefore do not establish actual risk control. The corresponding
[outcome-aware capacity audit](../recipient_release_capacity_v1/README.md)
uses the same bank and verifies 18/24 at free count.

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_recipient_release_crossfit.py
.venv-pytorch/bin/python scripts/run_m3w_recipient_release_crossfit.py --replay
~~~

Native arm64; fitted coefficients and frozen actions remain local.
Exact score replay uses cached scalar scoring cells. The shared
fitting loader may reopen original fitting label arrays for identity
checks; replay does not refit or change scores. This is a source
diagnostic, not a deployable or independently confirmed model.
No Stage5C or SMC.
[Frozen predictions](freeze.json) | [Results](analysis.json).
