# Training the initial Moving forecast against both references

I kept the original risk rule and changed the forecast. I wanted to know
whether the old anchor left the decision head too far behind OLS4, and
whether training a new forecast against both OLS4 and constant velocity
(CV) could close the gap without giving up easy-case support.

I trained the same causal velocity spline with two objectives: an
OLS4-relative conservative loss, and a loss against both references
with fixed penalties for excess recording risk. The latter samples
one recording per minibatch. The penalty helps train the model; it
does not replace the exact evaluator or give a safety guarantee.

**Fresh run:** 18 native PyTorch fits, 14,400 optimizer updates, about
78 seconds, 4 CPU threads, no worker processes. Three seeds across
three rotations of the same nine development sources produced
72 recording-seed views over 24 recordings. Fitting, calibration
and held sources are disjoint within each rotation. The reused
development roster is not an independent confirmation set.
**Cached verified:** the old half-peak anchor and completion spline
were reconstructed from hashed artifacts; EqMotion and PCT were
held-only controls. Frozen replay completed with zero training updates.

| Forecast | Available-label gain vs OLS4, pp | Conservative gain vs OLS4, pp | Risk passes | Dual passes |
| --- | ---: | ---: | ---: | ---: |
| Old half-peak anchor | See readout | See readout | 18/72 | 2/72 |
| OLS4-only objective | +1.792 | -5.560 | 0/72 | 0/72 |
| Dual-reference risk objective | +1.649 | -52.699 | 3/72 | 0/72 |
| Best diagnostic calibration choice | +1.661 | -44.242 | 3/72 | 0/72 |

For the OLS4-only model the 95% source-bootstrap interval changes
from [+0.237, +3.397] pp on available labels to [-13.028, -1.216]
pp after the missing-step debit. For the dual-risk model the latter
interval is [-94.282, -21.825] pp. Gains use the same CV normalization
and original aggregation; they are not percentages of physical
displacement. None of the nine calibration cells has a strict
choice: CV fallback remains undefined for edited support and is not a
risk pass. A diagnostic choice excludes 60/72 views under the relaxed
single-query OLS4 upper bound versus 69/72 for the old anchor, but
this is at each forecast's own budget, with risk ignored by the bound.

**Same-budget attribution.** I fixed the 54,748 edits counted across
recording-seed views, including their query times, recipients and
individual peak displacement from CV. A new path could supply its
direction only if it reached the existing peak without amplification.
Relative to the old anchor, conservative change was -0.561 pp
[-1.376, -0.093] for OLS4-only, -0.00010 pp [-0.00097, +0.00065]
for dual-risk, and -0.757 pp [-2.068, -0.044] for the diagnostic
choice. None adds a dual pass. The dual-risk direction was usable
on only 1,648 of the old edits; its near-zero estimate is not
evidence that it learned a better direction. Intervals use 3,000
paired source draws after averaging the seeds.

I kept deployment unchanged. The next useful test is an outcome-aware
capacity audit across multiple queries at the existing recording
budget, with the same conservative scores and support requirements.
That separates a limit of the permitted actions from a failure to
learn their placement. This does not open Moving A or the 30
independent DroneCrowd confirmation groups. No Stage5C or SMC ran.
Motion is in dataset-local image coordinates and annotation steps.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Readout](readout.json) |
[Matched attribution](matched_directions.json) |
[Working manuscript v155](../evidence_manuscript_v155/README.md)
